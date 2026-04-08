from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


robotwin_root = Path(__file__).resolve().parents[4]
if str(robotwin_root) not in sys.path:
    sys.path.insert(0, str(robotwin_root))

from runners.online_env_runner import OnlineEnvRunner
from sim.maniskill import ManiSkillTaskFactory


class ManiSkillFPORunner(OnlineEnvRunner):
    """Single-environment online runner for ManiSkill FPO prototypes."""

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
                "ManiSkillFPORunner requires learner-facing observation keys "
                "`point_cloud` and `agent_pos`."
            )
        return filtered

    def __init__(
        self,
        task_name,
        base_seed=0,
        action_type="pd_joint_delta_pos",
        obs_mode="pointcloud+state_dict",
        control_mode=None,
        render_mode=None,
        sim_freq=None,
        control_freq=None,
        sim_config=None,
        **backend_kwargs,
    ):
        super().__init__(
            task_factory=ManiSkillTaskFactory(),
            task_name=task_name,
            action_type=action_type,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            sim_freq=sim_freq,
            control_freq=control_freq,
            sim_config=sim_config,
            **backend_kwargs,
        )
        self.base_seed = int(base_seed)
        self.episode_id = 0

    def reset(self):
        max_seed = 2**32 - 1
        seed = (self.base_seed + self.episode_id) % max_seed
        obs = self.backend.reset(seed=seed)
        metadata = dict(obs.metadata)
        metadata.setdefault("seed", int(seed))
        metadata.setdefault("ep_id", int(self.episode_id))
        self.episode_id += 1
        return self._format_obs(dict(obs.data)), metadata

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._format_obs(obs), reward, terminated, truncated, info
