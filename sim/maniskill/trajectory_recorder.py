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
        self._obs_arrays: dict[str, list[np.ndarray]] = {}
        self._point_cloud: list[np.ndarray] = []
        self._agent_pos: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    def _infer_obs_keys(self, obs: dict[str, Any]) -> list[str]:
        if "agent_pos" not in obs:
            raise KeyError("Recorder requires observation key `agent_pos`.")
        if "point_cloud" in obs:
            return ["point_cloud", "agent_pos"]

        multiview_keys = [
            "head_point_cloud",
            "left_wrist_point_cloud",
            "right_wrist_point_cloud",
        ]
        if all(key in obs for key in multiview_keys):
            return [*multiview_keys, "agent_pos"]

        available = sorted(key for key in obs.keys() if str(key).endswith("point_cloud"))
        raise KeyError(
            "Recorder requires either `point_cloud` or all multiview point-cloud keys. "
            f"Available point-cloud keys: {available}"
        )

    def _extract_obs_arrays(self, obs: dict[str, Any], keys: list[str]) -> dict[str, np.ndarray]:
        return {key: np.asarray(obs[key], dtype=np.float32) for key in keys}

    def _reset_obs_storage(self, obs_keys: list[str]) -> None:
        self._obs_arrays = {key: [] for key in obs_keys}
        self._point_cloud = self._obs_arrays.get("point_cloud", [])
        self._agent_pos = self._obs_arrays["agent_pos"]

    def start_episode(self, initial_obs: dict[str, Any]) -> None:
        obs_keys = self._infer_obs_keys(initial_obs)
        self._reset_obs_storage(obs_keys)
        self._actions = []
        obs_arrays = self._extract_obs_arrays(initial_obs, obs_keys)
        for key, value in obs_arrays.items():
            self._obs_arrays[key].append(value)

    def append_step(self, action: Any, next_obs: dict[str, Any]) -> None:
        if len(self._obs_arrays) == 0:
            raise RuntimeError("start_episode(...) must be called before append_step(...).")
        obs_keys = list(self._obs_arrays.keys())
        obs_arrays = self._extract_obs_arrays(next_obs, obs_keys)
        self._actions.append(np.asarray(action, dtype=np.float32))
        for key, value in obs_arrays.items():
            self._obs_arrays[key].append(value)

    def save_episode(self, episode_idx: int) -> Path:
        if len(self._obs_arrays) == 0:
            raise RuntimeError("No episode data to save.")
        if len(self._actions) == 0:
            raise RuntimeError("Episode contains no actions.")

        out_path = self.data_dir / f"episode{int(episode_idx)}.npz"
        payload = {
            key: np.asarray(values, dtype=np.float32)
            for key, values in self._obs_arrays.items()
        }
        payload["actions"] = np.asarray(self._actions, dtype=np.float32)
        np.savez(
            out_path,
            **payload,
        )
        return out_path
