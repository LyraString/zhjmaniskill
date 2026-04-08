from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


robotwin_root = Path(__file__).resolve().parents[4]
if str(robotwin_root) not in sys.path:
    sys.path.insert(0, str(robotwin_root))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory


class ManiSkillVectorFPORunner:
    """Native ManiSkill vector runner used for smoke testing batched env semantics."""

    @staticmethod
    def _format_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        filtered: dict[str, np.ndarray] = {}
        for key in ("point_cloud", "agent_pos"):
            value = obs.get(key, None)
            if value is None:
                continue
            filtered[key] = np.asarray(value, dtype=np.float32)
        if "point_cloud" not in filtered or "agent_pos" not in filtered:
            raise KeyError(
                "ManiSkillVectorFPORunner requires learner-facing observation keys "
                "`point_cloud` and `agent_pos`."
            )
        return filtered

    def __init__(
        self,
        task_name,
        num_envs=2,
        action_type="pd_joint_delta_pos",
        obs_mode="pointcloud+state_dict",
        control_mode=None,
        render_mode=None,
        sim_freq=None,
        control_freq=None,
        sim_config=None,
        **backend_kwargs,
    ):
        self.task_name = task_name
        self.num_envs = int(num_envs)
        self.action_type = action_type
        self.backend = ManiSkillTaskFactory().create_vector_env(
            task_name=task_name,
            num_envs=self.num_envs,
            action_type=action_type,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            sim_freq=sim_freq,
            control_freq=control_freq,
            sim_config=sim_config,
            **backend_kwargs,
        )

    def reset(self):
        observations = self.backend.reset()
        obs_list = [self._format_obs(dict(obs.data)) for obs in observations]
        info_list = [dict(obs.metadata) for obs in observations]
        return obs_list, info_list

    def reset_at(self, env_ids):
        raise NotImplementedError(
            "ManiSkillVectorFPORunner.reset_at is intentionally deferred to stage 9B."
        )

    def step(self, actions):
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions length ({len(actions)}) must match num_envs ({self.num_envs})"
            )
        results = self.backend.step(
            [ActionSpec.from_value(action, action_type=self.action_type) for action in actions]
        )
        obs = [self._format_obs(dict(item.observation.data)) for item in results]
        rew = np.array([float(item.reward) for item in results], dtype=np.float32)
        terminated = np.array([bool(item.terminated) for item in results], dtype=np.bool_)
        truncated = np.array([bool(item.truncated) for item in results], dtype=np.bool_)
        info = [dict(item.info) for item in results]
        return obs, rew, terminated, truncated, info

    def close(self):
        self.backend.close()
