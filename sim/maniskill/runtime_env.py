"""Single-environment ManiSkill backend."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from core.schemas.action import ActionSpec
from core.schemas.step_result import StepResult
from sim.maniskill.action_adapter import convert_action_to_maniskill
from sim.maniskill.obs_adapter import convert_maniskill_obs

import mani_skill.envs  # noqa: F401


class ManiSkillEnvBackend:
    """Shared-interface wrapper around one ManiSkill environment instance."""

    def __init__(
        self,
        *,
        env_id: str,
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

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

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

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        metadata = {
            "backend": "maniskill",
            "env_id": self.env_id,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            **self._fixed_rate_info,
            "reset_info": info,
        }
        return convert_maniskill_obs(obs, metadata=metadata)

    def step(self, action: ActionSpec) -> StepResult:
        mani_action = convert_action_to_maniskill(action)
        obs, reward, terminated, truncated, info = self.env.step(mani_action)
        observation = convert_maniskill_obs(
            obs,
            metadata={
                "backend": "maniskill",
                "env_id": self.env_id,
                "obs_mode": self.obs_mode,
                "control_mode": self.control_mode,
                **self._fixed_rate_info,
            },
        )
        info = dict(info)
        info.update(self._fixed_rate_info)
        return StepResult.create(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
