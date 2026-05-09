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
from sim.maniskill.task_profiles import resolve_runtime_control_mode, resolve_runtime_task_target


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
        task_profile=None,
        env_id=None,
        num_envs=2,
        base_seed=0,
        action_type="pd_joint_delta_pos",
        obs_mode="pointcloud+state_dict",
        control_mode=None,
        render_mode=None,
        sim_freq=None,
        control_freq=None,
        point_cloud_sampling="fast",
        sim_config=None,
        **backend_kwargs,
    ):
        runtime_task_name, resolved_profile_name = resolve_runtime_task_target(
            task_name=task_name,
            task_profile=task_profile,
            env_id=env_id,
        )
        runtime_action_type, runtime_control_mode = resolve_runtime_control_mode(
            action_type=action_type,
            control_mode=control_mode,
        )
        self.task_name = str(task_name)
        self.runtime_task_name = str(runtime_task_name)
        self.task_profile = resolved_profile_name
        self.env_id = str(env_id) if env_id not in (None, "") else self.runtime_task_name
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.action_type = runtime_action_type
        self.control_mode = runtime_control_mode
        self._episode_ids = [0 for _ in range(self.num_envs)]
        self.backend = ManiSkillTaskFactory().create_vector_env(
            task_name=runtime_task_name,
            num_envs=self.num_envs,
            action_type=runtime_action_type,
            obs_mode=obs_mode,
            control_mode=runtime_control_mode,
            render_mode=render_mode,
            sim_freq=sim_freq,
            control_freq=control_freq,
            point_cloud_sampling=point_cloud_sampling,
            sim_config=sim_config,
            **backend_kwargs,
        )

    def _seed_for_env(self, env_id: int) -> int:
        return self.base_seed + self._episode_ids[env_id] * self.num_envs + env_id

    def reset(self):
        seeds = [self._seed_for_env(env_id) for env_id in range(self.num_envs)]
        observations = self.backend.reset(seeds=seeds)
        obs_list = [self._format_obs(dict(obs.data)) for obs in observations]
        info_list = [dict(obs.metadata) for obs in observations]
        for env_id in range(self.num_envs):
            self._episode_ids[env_id] += 1
        return obs_list, info_list

    def reset_at(self, env_ids):
        env_ids = [int(env_id) for env_id in env_ids]
        if len(env_ids) == 0:
            raise ValueError("env_ids must not be empty")

        seeds = None
        if len(env_ids) == self.num_envs:
            seeds = [self._seed_for_env(env_id) for env_id in env_ids]
            observations = self.backend.reset(seeds=seeds)
            reset_out = {}
            for env_id in env_ids:
                obs = observations[env_id]
                reset_out[env_id] = (
                    self._format_obs(dict(obs.data)),
                    dict(obs.metadata),
                )
                self._episode_ids[env_id] += 1
            return reset_out

        observations = self.backend.reset_at(env_ids)
        reset_out = {}
        for env_id in env_ids:
            obs = observations[env_id]
            reset_out[env_id] = (
                self._format_obs(dict(obs.data)),
                dict(obs.metadata),
            )
            self._episode_ids[env_id] += 1
        return reset_out

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
