"""Vectorized ManiSkill backend built on ManiSkill's native batched envs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from core.schemas.action import ActionSpec
from core.schemas.step_result import StepResult
from sim.maniskill.action_adapter import convert_action_to_maniskill
from sim.maniskill.obs_adapter import convert_maniskill_obs

import mani_skill.envs  # noqa: F401


def _slice_batched_value(value: Any, index: int, batch_size: int) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == batch_size:
            return value[index]
        return value

    if isinstance(value, Mapping):
        return {
            key: _slice_batched_value(val, index=index, batch_size=batch_size)
            for key, val in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) == batch_size:
            return _slice_batched_value(value[index], index=index, batch_size=batch_size)
        return type(value)(
            _slice_batched_value(item, index=index, batch_size=batch_size)
            for item in value
        )

    return value


def _to_float_scalar(value: Any) -> float:
    value = _slice_batched_value(value, index=0, batch_size=1)
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def _to_bool_scalar(value: Any) -> bool:
    value = _slice_batched_value(value, index=0, batch_size=1)
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


class ManiSkillVectorEnvBackend:
    """Shared-interface wrapper around one native batched ManiSkill env."""

    def __init__(
        self,
        *,
        env_id: str,
        num_envs: int,
        obs_mode: str = "state",
        control_mode: str | None = None,
        render_mode: str | None = None,
        sim_backend: str | None = None,
        render_backend: str | None = None,
        sim_freq: int | None = None,
        control_freq: int | None = None,
        sim_config: dict[str, Any] | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.env_id = env_id
        self.num_envs = int(num_envs)
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.sim_backend = sim_backend
        self.render_backend = render_backend
        self.requested_sim_freq = sim_freq
        self.requested_control_freq = control_freq
        self.requested_sim_config = dict(sim_config or {})
        self.env_kwargs = dict(env_kwargs or {})

        make_kwargs = dict(self.env_kwargs)
        make_kwargs.setdefault("num_envs", self.num_envs)
        make_kwargs.setdefault("obs_mode", obs_mode)
        if control_mode is not None:
            make_kwargs.setdefault("control_mode", control_mode)
        if render_mode is not None:
            make_kwargs.setdefault("render_mode", render_mode)
        if sim_backend is not None:
            make_kwargs.setdefault("sim_backend", sim_backend)
        if render_backend is not None:
            make_kwargs.setdefault("render_backend", render_backend)

        merged_sim_config = dict(self.requested_sim_config)
        if sim_freq is not None:
            merged_sim_config["sim_freq"] = int(sim_freq)
        if control_freq is not None:
            merged_sim_config["control_freq"] = int(control_freq)
        if merged_sim_config:
            make_kwargs.setdefault("sim_config", merged_sim_config)

        self.env = gym.make(env_id, **make_kwargs)
        self._fixed_rate_info = self._collect_fixed_rate_info()

    def _collect_fixed_rate_info(self) -> dict[str, Any]:
        unwrapped = self.env.unwrapped
        sim_freq = getattr(unwrapped, "sim_freq", None)
        control_freq = getattr(unwrapped, "control_freq", None)
        sim_steps_per_control = getattr(unwrapped, "_sim_steps_per_control", None)
        info = {
            "sim_freq": sim_freq,
            "control_freq": control_freq,
            "sim_steps_per_control": sim_steps_per_control,
            "sim_backend": self.sim_backend,
            "render_backend": self.render_backend,
        }
        if sim_freq:
            info["sim_dt"] = 1.0 / float(sim_freq)
        if control_freq:
            info["control_dt"] = 1.0 / float(control_freq)
        return info

    def _split_observations(self, raw_obs: Any, info: Any | None = None):
        observations = []
        for env_index in range(self.num_envs):
            obs_i = _slice_batched_value(raw_obs, index=env_index, batch_size=self.num_envs)
            info_i = (
                _slice_batched_value(info, index=env_index, batch_size=self.num_envs)
                if info is not None
                else {}
            )
            metadata = {
                "backend": "maniskill",
                "env_id": self.env_id,
                "env_index": env_index,
                "obs_mode": self.obs_mode,
                "control_mode": self.control_mode,
                **self._fixed_rate_info,
            }
            if info is not None:
                metadata["reset_info"] = info_i
            observations.append(convert_maniskill_obs(obs_i, metadata=metadata))
        return observations

    def _validate_env_ids(self, env_ids: Sequence[int]) -> list[int]:
        normalized = [int(env_id) for env_id in env_ids]
        if len(normalized) == 0:
            raise ValueError("env_ids must not be empty")
        for env_id in normalized:
            if env_id < 0 or env_id >= self.num_envs:
                raise IndexError(
                    f"env_id {env_id} is out of range for num_envs={self.num_envs}"
                )
        return normalized

    def reset(self, seeds: Sequence[int] | None = None):
        seed_arg = None
        if seeds is not None:
            if len(seeds) != self.num_envs:
                raise ValueError(
                    f"seeds length ({len(seeds)}) must match num_envs ({self.num_envs})"
                )
            seed_arg = [int(seed) for seed in seeds]
        raw_obs, info = self.env.reset(seed=seed_arg)
        return self._split_observations(raw_obs, info=info)

    def reset_at(
        self,
        env_ids: Sequence[int],
        seeds: Sequence[int] | None = None,
    ):
        env_ids = self._validate_env_ids(env_ids)
        seed_arg = None
        if seeds is not None:
            if len(seeds) != len(env_ids):
                raise ValueError(
                    f"seeds length ({len(seeds)}) must match env_ids length ({len(env_ids)})"
                )
            if len(env_ids) == self.num_envs:
                seed_arg = [int(seed) for seed in seeds]
            elif len(env_ids) == 1:
                seed_arg = int(seeds[0])
            else:
                raise NotImplementedError(
                    "ManiSkill partial reset with multiple explicit seeds is not supported "
                    "by this backend because ManiSkill reset expects either a scalar seed "
                    "or a full num_envs seed list."
                )

        raw_obs, info = self.env.reset(
            seed=seed_arg,
            options={"env_idx": np.asarray(env_ids, dtype=np.int64)},
        )
        observations = self._split_observations(raw_obs, info=info)
        return {env_id: observations[env_id] for env_id in env_ids}

    def step(self, actions: Sequence[ActionSpec]):
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions length ({len(actions)}) must match num_envs ({self.num_envs})"
            )
        batched_action = np.stack(
            [convert_action_to_maniskill(action) for action in actions],
            axis=0,
        )
        raw_obs, reward, terminated, truncated, info = self.env.step(batched_action)
        observations = self._split_observations(raw_obs)

        results = []
        for env_index in range(self.num_envs):
            info_i = _slice_batched_value(info, index=env_index, batch_size=self.num_envs)
            info_i = dict(info_i) if isinstance(info_i, Mapping) else {"raw_info": info_i}
            info_i.update(self._fixed_rate_info)
            results.append(
                StepResult.create(
                    observation=observations[env_index],
                    reward=_to_float_scalar(_slice_batched_value(reward, env_index, self.num_envs)),
                    terminated=_to_bool_scalar(_slice_batched_value(terminated, env_index, self.num_envs)),
                    truncated=_to_bool_scalar(_slice_batched_value(truncated, env_index, self.num_envs)),
                    info=info_i,
                )
            )
        return results

    def close(self) -> None:
        self.env.close()
