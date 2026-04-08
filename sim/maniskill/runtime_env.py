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
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.env_id = env_id
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.env_kwargs = dict(env_kwargs or {})

        make_kwargs = dict(self.env_kwargs)
        make_kwargs.setdefault("obs_mode", obs_mode)
        if control_mode is not None:
            make_kwargs.setdefault("control_mode", control_mode)
        if render_mode is not None:
            make_kwargs.setdefault("render_mode", render_mode)

        self.env = gym.make(env_id, **make_kwargs)

    def reset(self, *, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        metadata = {
            "backend": "maniskill",
            "env_id": self.env_id,
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
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
            },
        )
        return StepResult.create(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def close(self) -> None:
        self.env.close()
