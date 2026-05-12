from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch


robotwin_root = Path(__file__).resolve().parents[4]
if str(robotwin_root) not in sys.path:
    sys.path.insert(0, str(robotwin_root))

import mani_skill.envs  # noqa: F401
import sim.maniskill  # noqa: F401
from sim.maniskill.obs_adapter import (
    DEFAULT_POINT_CLOUD_CROP_MAX,
    DEFAULT_POINT_CLOUD_CROP_MIN,
    DEFAULT_POINT_CLOUD_PRIMARY_CAMERA,
    ENV_POINT_CLOUD_EXCLUDED_SEG_IDS,
)
from sim.maniskill.task_profiles import (
    get_task_profile,
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
    resolve_task_profile,
)


class ManiSkillCudaFPORunner:
    """ManiSkill vector runner that keeps FPO rollout tensors on CUDA."""

    def __init__(
        self,
        task_name: str,
        *,
        shape_meta: Mapping[str, Any],
        task_profile: str | None = None,
        env_id: str | None = None,
        num_envs: int = 2,
        base_seed: int = 0,
        action_type: str | None = "pd_joint_delta_pos",
        obs_mode: str | None = "pointcloud+state_dict",
        control_mode: str | None = None,
        render_mode: str | None = None,
        reward_mode: str | None = None,
        robot_uids: str | None = None,
        sim_backend: str | None = None,
        render_backend: str | None = None,
        sim_freq: int | None = None,
        control_freq: int | None = None,
        point_cloud_sampling: str = "fast",
        sim_config: Mapping[str, Any] | None = None,
        **env_kwargs,
    ) -> None:
        runtime_task_name, resolved_profile_name = resolve_runtime_task_target(
            task_name=task_name,
            task_profile=task_profile,
            env_id=env_id,
        )
        profile = get_task_profile(resolved_profile_name) if resolved_profile_name else None
        if profile is None:
            profile = resolve_task_profile(runtime_task_name)

        if profile is not None:
            action_type = action_type or profile.get("action_type")
            obs_mode = obs_mode or profile.get("obs_mode")
            control_mode = control_mode if control_mode is not None else profile.get("control_mode")
            render_mode = render_mode if render_mode is not None else profile.get("render_mode")
            reward_mode = reward_mode if reward_mode is not None else profile.get("reward_mode")
            robot_uids = robot_uids if robot_uids is not None else profile.get("robot_uids")
            sim_freq = sim_freq if sim_freq is not None else profile.get("sim_freq")
            control_freq = control_freq if control_freq is not None else profile.get("control_freq")

        runtime_action_type, runtime_control_mode = resolve_runtime_control_mode(
            action_type=action_type,
            control_mode=control_mode,
        )
        sampling = str(point_cloud_sampling).lower()
        if sampling not in {"fast", "resize", "linspace"}:
            raise ValueError(
                "ManiSkillCudaFPORunner currently supports only fast/resize/linspace "
                f"point-cloud sampling, got `{point_cloud_sampling}`."
            )

        self.task_name = str(task_name)
        self.runtime_task_name = str(runtime_task_name)
        self.task_profile = resolved_profile_name
        self.env_id = str(env_id) if env_id not in (None, "") else self.runtime_task_name
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.action_type = runtime_action_type
        self.control_mode = runtime_control_mode
        self.point_cloud_sampling = sampling
        self._episode_ids = [0 for _ in range(self.num_envs)]

        obs_meta = shape_meta["obs"]
        self.point_cloud_points = int(obs_meta["point_cloud"]["shape"][0])
        self.point_cloud_dim = int(obs_meta["point_cloud"]["shape"][-1])
        self.agent_pos_dim = int(obs_meta["agent_pos"]["shape"][0])
        self.action_dim = int(shape_meta["action"]["shape"][0])
        if self.point_cloud_dim != 3:
            raise ValueError(f"Expected point_cloud dim 3, got {self.point_cloud_dim}.")

        make_kwargs = dict(env_kwargs)
        make_kwargs.setdefault("num_envs", self.num_envs)
        make_kwargs.setdefault("obs_mode", obs_mode or "pointcloud+state_dict")
        if runtime_control_mode is not None:
            make_kwargs.setdefault("control_mode", runtime_control_mode)
        if render_mode is not None:
            make_kwargs.setdefault("render_mode", render_mode)
        if sim_backend is not None:
            make_kwargs.setdefault("sim_backend", sim_backend)
        if render_backend is not None:
            make_kwargs.setdefault("render_backend", render_backend)
        if reward_mode is not None:
            make_kwargs.setdefault("reward_mode", reward_mode)
        if robot_uids not in (None, "none", "None"):
            make_kwargs.setdefault("robot_uids", robot_uids)

        merged_sim_config = dict(sim_config or {})
        if sim_freq is not None:
            merged_sim_config["sim_freq"] = int(sim_freq)
        if control_freq is not None:
            merged_sim_config["control_freq"] = int(control_freq)
        if merged_sim_config:
            make_kwargs.setdefault("sim_config", merged_sim_config)

        self.env = gym.make(self.env_id, **make_kwargs)
        self.device = torch.device(getattr(self.env.unwrapped, "device", "cpu"))
        self.crop_min = torch.as_tensor(
            DEFAULT_POINT_CLOUD_CROP_MIN,
            device=self.device,
            dtype=torch.float32,
        )
        self.crop_max = torch.as_tensor(
            DEFAULT_POINT_CLOUD_CROP_MAX,
            device=self.device,
            dtype=torch.float32,
        )

    def _seed_for_env(self, env_id: int) -> int:
        return self.base_seed + self._episode_ids[env_id] * self.num_envs + env_id

    def reset(self):
        seeds = [self._seed_for_env(env_id) for env_id in range(self.num_envs)]
        raw_obs, info = self.env.reset(seed=seeds)
        for env_id in range(self.num_envs):
            self._episode_ids[env_id] += 1
        return self.format_obs(raw_obs), info

    def reset_at(self, env_ids: Sequence[int]):
        env_ids = [int(env_id) for env_id in env_ids]
        if len(env_ids) == 0:
            raise ValueError("env_ids must not be empty.")

        if len(env_ids) == self.num_envs:
            seeds = [self._seed_for_env(env_id) for env_id in env_ids]
            raw_obs, info = self.env.reset(seed=seeds)
        elif len(env_ids) == 1:
            raw_obs, info = self.env.reset(
                seed=self._seed_for_env(env_ids[0]),
                options={"env_idx": torch.tensor(env_ids, device=self.device, dtype=torch.long)},
            )
        else:
            raw_obs, info = self.env.reset(
                options={"env_idx": torch.tensor(env_ids, device=self.device, dtype=torch.long)},
            )

        for env_id in env_ids:
            self._episode_ids[env_id] += 1
        return self.format_obs(raw_obs), info

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if tuple(actions.shape) != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected action shape {(self.num_envs, self.action_dim)}, "
                f"got {tuple(actions.shape)}."
            )
        raw_obs, reward, terminated, truncated, info = self.env.step(actions)
        return self.format_obs(raw_obs), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

    def format_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {
            "point_cloud": self._extract_point_cloud(raw_obs),
            "agent_pos": self._extract_agent_pos(raw_obs),
        }

    def _extract_agent_pos(self, raw_obs: Mapping[str, Any]) -> torch.Tensor:
        agent = raw_obs.get("agent")
        if isinstance(agent, Mapping) and "qpos" in agent:
            qpos = agent["qpos"].to(device=self.device, dtype=torch.float32)
            if qpos.shape[-1] != self.agent_pos_dim:
                raise ValueError(
                    f"Expected agent_pos dim {self.agent_pos_dim}, got {qpos.shape[-1]}."
                )
            return qpos

        state = raw_obs.get("state")
        if state is not None:
            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            return state.reshape(self.num_envs, -1)

        raise KeyError("Raw ManiSkill obs has neither `agent.qpos` nor `state`.")

    def _extract_point_cloud(self, raw_obs: Mapping[str, Any]) -> torch.Tensor:
        point_cloud = self._extract_point_cloud_from_sensor_data(raw_obs)
        if point_cloud is not None:
            return point_cloud

        pointcloud = raw_obs.get("pointcloud")
        if isinstance(pointcloud, Mapping) and "xyzw" in pointcloud:
            xyzw = pointcloud["xyzw"].to(device=self.device, dtype=torch.float32)
            xyz = xyzw[..., :3]
            valid = xyzw[..., 3] > 0 if xyzw.shape[-1] >= 4 else None
            return self._postprocess_batched_xyz(xyz.reshape(self.num_envs, -1, 3), valid)

        direct_point_cloud = raw_obs.get("point_cloud")
        if direct_point_cloud is not None:
            xyz = torch.as_tensor(direct_point_cloud, device=self.device, dtype=torch.float32)
            return self._postprocess_batched_xyz(xyz.reshape(self.num_envs, -1, 3), None)

        raise KeyError(
            "Raw ManiSkill obs has no supported point-cloud source. Expected "
            "`sensor_data`, `pointcloud.xyzw`, or `point_cloud`."
        )

    def _extract_point_cloud_from_sensor_data(
        self,
        raw_obs: Mapping[str, Any],
    ) -> torch.Tensor | None:
        sensor_data = raw_obs.get("sensor_data")
        sensor_param = raw_obs.get("sensor_param")
        if not isinstance(sensor_data, Mapping) or not isinstance(sensor_param, Mapping):
            return None

        camera_data = sensor_data.get(DEFAULT_POINT_CLOUD_PRIMARY_CAMERA)
        camera_param = sensor_param.get(DEFAULT_POINT_CLOUD_PRIMARY_CAMERA)
        if not isinstance(camera_data, Mapping) or not isinstance(camera_param, Mapping):
            return None
        if "position" not in camera_data or "cam2world_gl" not in camera_param:
            return None

        position = camera_data["position"].to(device=self.device, dtype=torch.float32)
        cam2world = camera_param["cam2world_gl"].to(device=self.device, dtype=torch.float32)
        xyz = position[..., :3] / 1000.0

        if position.shape[-1] >= 4:
            valid = position[..., 3] < 1.0
            segmentation = camera_data.get("segmentation")
        else:
            segmentation = camera_data.get("segmentation")
            if segmentation is None:
                valid = torch.ones(xyz.shape[:-1], device=self.device, dtype=torch.bool)
            else:
                segmentation = segmentation.to(device=self.device)
                valid = segmentation.reshape(self.num_envs, -1) != 0

        xyz_flat = xyz.reshape(self.num_envs, -1, 3)
        valid = valid.reshape(self.num_envs, -1)
        segmentation = camera_data.get("segmentation")
        if segmentation is not None:
            segmentation = segmentation.to(device=self.device).reshape(self.num_envs, -1)
            excluded_seg_ids = ENV_POINT_CLOUD_EXCLUDED_SEG_IDS.get(self.env_id, ())
            for seg_id in excluded_seg_ids:
                valid = torch.logical_and(valid, segmentation != int(seg_id))

        one = valid.to(dtype=torch.float32).unsqueeze(-1)
        points_xyzw = torch.cat([xyz_flat, one], dim=-1)
        world_xyzw = torch.bmm(points_xyzw, cam2world.transpose(1, 2))
        world_xyz = world_xyzw[..., :3]
        world_valid = world_xyzw[..., 3] > 0
        return self._postprocess_batched_xyz(world_xyz, world_valid)

    def _postprocess_batched_xyz(
        self,
        xyz: torch.Tensor,
        valid: torch.Tensor | None,
    ) -> torch.Tensor:
        xyz = xyz.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, -1, 3)
        if valid is None:
            valid = torch.ones(xyz.shape[:2], device=self.device, dtype=torch.bool)
        else:
            valid = valid.to(device=self.device, dtype=torch.bool).reshape(self.num_envs, -1)

        inside_bounds = torch.logical_and(
            xyz >= self.crop_min.view(1, 1, 3),
            xyz <= self.crop_max.view(1, 1, 3),
        ).all(dim=-1)
        keep = torch.logical_and(valid, inside_bounds)

        clouds = []
        for env_id in range(self.num_envs):
            points = xyz[env_id][keep[env_id]]
            clouds.append(self._resize_point_cloud(points))
        return torch.stack(clouds, dim=0)

    def _resize_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        target_points = self.point_cloud_points
        if points.shape[0] == 0:
            return torch.zeros((target_points, 3), device=self.device, dtype=torch.float32)

        if points.shape[0] >= target_points:
            idx = torch.linspace(
                0,
                points.shape[0] - 1,
                target_points,
                device=self.device,
            ).to(torch.long)
            return points[idx].to(dtype=torch.float32)

        pad_count = target_points - points.shape[0]
        pad = points[:1].expand(pad_count, -1)
        return torch.cat([points, pad], dim=0).to(dtype=torch.float32)
