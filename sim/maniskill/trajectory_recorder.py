"""Utilities for writing ManiSkill episodes in the repo's `.npz` format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ManiSkillTrajectoryRecorder:
    """Accumulates one ManiSkill episode and writes it as `episode{idx}.npz`."""

    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.data_dir = self.save_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._point_cloud: list[np.ndarray] = []
        self._agent_pos: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    def _extract_obs_arrays(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if "point_cloud" not in obs or "agent_pos" not in obs:
            raise KeyError(
                "Recorder requires observation keys `point_cloud` and `agent_pos`."
            )
        point_cloud = np.asarray(obs["point_cloud"], dtype=np.float32)
        agent_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
        return point_cloud, agent_pos

    def start_episode(self, initial_obs: dict[str, Any]) -> None:
        self._point_cloud = []
        self._agent_pos = []
        self._actions = []
        point_cloud, agent_pos = self._extract_obs_arrays(initial_obs)
        self._point_cloud.append(point_cloud)
        self._agent_pos.append(agent_pos)

    def append_step(self, action: Any, next_obs: dict[str, Any]) -> None:
        if len(self._point_cloud) == 0:
            raise RuntimeError("start_episode(...) must be called before append_step(...).")
        point_cloud, agent_pos = self._extract_obs_arrays(next_obs)
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._point_cloud.append(point_cloud)
        self._agent_pos.append(agent_pos)

    def save_episode(self, episode_idx: int) -> Path:
        if len(self._point_cloud) == 0:
            raise RuntimeError("No episode data to save.")
        if len(self._actions) == 0:
            raise RuntimeError("Episode contains no actions.")

        out_path = self.data_dir / f"episode{int(episode_idx)}.npz"
        np.savez(
            out_path,
            point_cloud=np.asarray(self._point_cloud, dtype=np.float32),
            agent_pos=np.asarray(self._agent_pos, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.float32),
        )
        return out_path
