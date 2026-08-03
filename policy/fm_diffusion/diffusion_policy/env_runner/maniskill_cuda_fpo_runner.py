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
    DEFAULT_POINT_CLOUD_SAMPLING,
    ENV_POINT_CLOUD_EXCLUDED_SEG_IDS,
)

MULTIVIEW_POINT_CLOUD_CAMERAS = {
    "head_point_cloud": "head_camera",
    "left_wrist_point_cloud": "left_wrist_camera",
    "right_wrist_point_cloud": "right_wrist_camera",
}
from sim.maniskill.task_profiles import (
    get_task_profile,
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
    resolve_task_profile,
)


def _bind_cuda_backend(backend: str | None, device_index: int, *, default: str) -> str:
    if backend is None or str(backend).lower() == "auto":
        return f"{default}:{device_index}"
    backend_name = str(backend)
    base_name = backend_name.split(":", maxsplit=1)[0]
    if base_name in {"cuda", "gpu", "physx_cuda", "sapien_cuda"}:
        return f"{base_name}:{device_index}"
    return backend_name


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
        seed_rank: int = 0,
        seed_world_size: int = 1,
        cuda_device_index: int | None = None,
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
        point_cloud_sampling: str = DEFAULT_POINT_CLOUD_SAMPLING,
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
        if sampling not in {"fps", "fast", "resize", "linspace"}:
            raise ValueError(
                "ManiSkillCudaFPORunner currently supports only fps/fast/resize/linspace "
                f"point-cloud sampling, got `{point_cloud_sampling}`."
            )

        self.task_name = str(task_name)
        self.runtime_task_name = str(runtime_task_name)
        self.task_profile = resolved_profile_name
        self.env_id = str(env_id) if env_id not in (None, "") else self.runtime_task_name
        self.num_envs = int(num_envs)
        self.base_seed = int(base_seed)
        self.seed_rank = int(seed_rank)
        self.seed_world_size = int(seed_world_size)
        if not 0 <= self.seed_rank < self.seed_world_size:
            raise ValueError(
                f"seed_rank must be in [0, {self.seed_world_size}), got {self.seed_rank}."
            )
        self.action_type = runtime_action_type
        self.control_mode = runtime_control_mode
        self.point_cloud_sampling = sampling
        self._episode_ids = [0 for _ in range(self.num_envs)]

        if cuda_device_index is not None:
            cuda_device_index = int(cuda_device_index)
            if cuda_device_index < 0:
                raise ValueError(f"cuda_device_index must be non-negative, got {cuda_device_index}.")
            sim_backend = _bind_cuda_backend(
                sim_backend,
                cuda_device_index,
                default="physx_cuda",
            )
            render_backend = _bind_cuda_backend(
                render_backend,
                cuda_device_index,
                default="sapien_cuda",
            )

        obs_meta = shape_meta["obs"]
        self.point_cloud_shapes = {
            key: tuple(meta["shape"])
            for key, meta in obs_meta.items()
            if meta.get("type") == "point_cloud"
        }
        if not self.point_cloud_shapes:
            raise ValueError("ManiSkillCudaFPORunner requires at least one point-cloud obs key.")
        self.point_cloud_keys = tuple(self.point_cloud_shapes.keys())
        self.point_cloud_points = {
            key: int(shape[0])
            for key, shape in self.point_cloud_shapes.items()
        }
        self.point_cloud_dim = {
            key: int(shape[-1])
            for key, shape in self.point_cloud_shapes.items()
        }
        self.agent_pos_dim = int(obs_meta["agent_pos"]["shape"][0])
        self.action_dim = int(shape_meta["action"]["shape"][0])
        for key, point_cloud_dim in self.point_cloud_dim.items():
            if point_cloud_dim != 3:
                raise ValueError(f"Expected {key} dim 3, got {point_cloud_dim}.")

        make_kwargs = dict(env_kwargs)
        if profile is not None:
            for key, value in profile.get("env_kwargs", {}).items():
                make_kwargs.setdefault(key, value)
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
        if cuda_device_index is not None:
            # DDP owns device placement; profile/env kwargs must not silently
            # move this rank's simulator or renderer onto another visible GPU.
            make_kwargs["sim_backend"] = sim_backend
            make_kwargs["render_backend"] = render_backend

        merged_sim_config = dict(sim_config or {})
        if sim_freq is not None:
            merged_sim_config["sim_freq"] = int(sim_freq)
        if control_freq is not None:
            merged_sim_config["control_freq"] = int(control_freq)
        if merged_sim_config:
            make_kwargs.setdefault("sim_config", merged_sim_config)

        self.env = gym.make(self.env_id, **make_kwargs)
        self.device = torch.device(getattr(self.env.unwrapped, "device", "cpu"))
        if cuda_device_index is not None:
            expected_device = torch.device(f"cuda:{cuda_device_index}")
            if self.device != expected_device:
                self.env.close()
                raise RuntimeError(
                    "ManiSkill CUDA environment was created on the wrong device: "
                    f"expected {expected_device}, got {self.device}."
                )
        action_space = getattr(self.env.unwrapped, "single_action_space", self.env.action_space)
        self.action_low = torch.as_tensor(
            action_space.low,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, self.action_dim)
        self.action_high = torch.as_tensor(
            action_space.high,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, self.action_dim)
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
        global_env_count = self.num_envs * self.seed_world_size
        global_env_id = self.seed_rank * self.num_envs + env_id
        return self.base_seed + self._episode_ids[env_id] * global_env_count + global_env_id

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

    def clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        return torch.minimum(torch.maximum(actions, self.action_low), self.action_high)

    def close(self) -> None:
        self.env.close()

    def format_obs(self, raw_obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        obs = {
            key: self._extract_point_cloud(raw_obs, obs_key=key)
            for key in self.point_cloud_keys
        }
        obs["agent_pos"] = self._extract_agent_pos(raw_obs)
        return obs

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

    def _camera_name_for_obs_key(self, obs_key: str) -> str:
        if obs_key == "point_cloud":
            return DEFAULT_POINT_CLOUD_PRIMARY_CAMERA
        return MULTIVIEW_POINT_CLOUD_CAMERAS.get(obs_key, obs_key.removesuffix("_point_cloud"))

    def _extract_point_cloud(self, raw_obs: Mapping[str, Any], *, obs_key: str) -> torch.Tensor:
        point_cloud = self._extract_point_cloud_from_sensor_data(raw_obs, obs_key=obs_key)
        if point_cloud is not None:
            return point_cloud

        pointcloud = raw_obs.get("pointcloud")
        if obs_key == "point_cloud" and isinstance(pointcloud, Mapping) and "xyzw" in pointcloud:
            xyzw = pointcloud["xyzw"].to(device=self.device, dtype=torch.float32)
            xyz = xyzw[..., :3]
            valid = xyzw[..., 3] > 0 if xyzw.shape[-1] >= 4 else None
            return self._postprocess_batched_xyz(xyz.reshape(self.num_envs, -1, 3), valid, obs_key=obs_key)

        direct_point_cloud = raw_obs.get(obs_key)
        if direct_point_cloud is not None:
            xyz = torch.as_tensor(direct_point_cloud, device=self.device, dtype=torch.float32)
            return self._postprocess_batched_xyz(xyz.reshape(self.num_envs, -1, 3), None, obs_key=obs_key)

        raise KeyError(
            "Raw ManiSkill obs has no supported point-cloud source. Expected "
            f"`sensor_data`, `pointcloud.xyzw`, or `{obs_key}`."
        )

    def _extract_point_cloud_from_sensor_data(
        self,
        raw_obs: Mapping[str, Any],
        *,
        obs_key: str,
    ) -> torch.Tensor | None:
        sensor_data = raw_obs.get("sensor_data")
        sensor_param = raw_obs.get("sensor_param")
        if not isinstance(sensor_data, Mapping) or not isinstance(sensor_param, Mapping):
            return None

        camera_name = self._camera_name_for_obs_key(obs_key)
        camera_data = sensor_data.get(camera_name)
        camera_param = sensor_param.get(camera_name)
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

        rotation = cam2world[:, :3, :3]
        translation = cam2world[:, :3, 3]
        world_xyz = torch.bmm(xyz_flat, rotation.transpose(1, 2)) + translation[:, None, :]
        return self._postprocess_batched_xyz(world_xyz, valid, obs_key=obs_key)

    def _postprocess_batched_xyz(
        self,
        xyz: torch.Tensor,
        valid: torch.Tensor | None,
        *,
        obs_key: str,
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

        if self.point_cloud_sampling in {"fast", "resize", "linspace"}:
            return self._resize_batched_point_cloud(xyz, keep, obs_key=obs_key)

        clouds = []
        for env_id in range(self.num_envs):
            points = xyz[env_id][keep[env_id]]
            clouds.append(self._sample_point_cloud(points, obs_key=obs_key))
        return torch.stack(clouds, dim=0)

    def _sample_point_cloud(self, points: torch.Tensor, *, obs_key: str) -> torch.Tensor:
        if self.point_cloud_sampling in {"fast", "resize", "linspace"}:
            return self._resize_point_cloud(points, obs_key=obs_key)

        sampled = self._farthest_point_sample(points, obs_key=obs_key)
        return self._resize_point_cloud(sampled, obs_key=obs_key)

    def _farthest_point_sample(self, points: torch.Tensor, *, obs_key: str) -> torch.Tensor:
        target_points = self.point_cloud_points[obs_key]
        num_points = points.shape[0]
        points = points.to(device=self.device, dtype=torch.float32)
        if num_points <= target_points:
            return points

        sampled = torch.empty((target_points, 3), device=self.device, dtype=torch.float32)
        sampled[0] = points[0]
        min_dist_sq = torch.sum((points - sampled[0]) ** 2, dim=1)
        farthest_idx = torch.argmax(min_dist_sq)

        for sample_idx in range(1, target_points):
            sampled[sample_idx] = points[farthest_idx]
            dist_sq = torch.sum((points - sampled[sample_idx]) ** 2, dim=1)
            min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
            farthest_idx = torch.argmax(min_dist_sq)

        return sampled

    def _resize_point_cloud(self, points: torch.Tensor, *, obs_key: str) -> torch.Tensor:
        target_points = self.point_cloud_points[obs_key]
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

    def _resize_batched_point_cloud(
        self,
        xyz: torch.Tensor,
        keep: torch.Tensor,
        *,
        obs_key: str,
    ) -> torch.Tensor:
        """Batched equivalent of per-env linspace resize used by fast sampling."""
        target_points = self.point_cloud_points[obs_key]
        xyz = xyz.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, -1, 3)
        keep = keep.to(device=self.device, dtype=torch.bool).reshape(self.num_envs, -1)

        num_source_points = xyz.shape[1]
        if num_source_points == 0:
            return torch.zeros(
                (self.num_envs, target_points, 3),
                device=self.device,
                dtype=torch.float32,
            )

        counts = keep.sum(dim=1)
        valid_env = counts > 0

        # cumulative rank lets searchsorted map compacted point ranks back to
        # original pixel order without looping over environments.
        cumulative = torch.cumsum(keep.to(dtype=torch.int32), dim=1, dtype=torch.int32).contiguous()
        point_rank = torch.arange(target_points, device=self.device, dtype=torch.int32).view(1, -1)

        counts_i32 = counts.to(dtype=torch.int32).view(-1, 1)
        enough_points = counts_i32 >= target_points

        linspace_alpha = torch.linspace(
            0.0,
            1.0,
            target_points,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        linspace_rank = ((counts.to(dtype=torch.float32).view(-1, 1) - 1.0) * linspace_alpha).to(torch.int32)

        padded_rank = torch.where(point_rank < counts_i32, point_rank, torch.zeros_like(point_rank))
        desired_rank = torch.where(enough_points, linspace_rank, padded_rank)
        desired_rank = torch.where(valid_env.view(-1, 1), desired_rank, torch.zeros_like(desired_rank))

        gather_index = torch.searchsorted(
            cumulative,
            desired_rank + 1,
            right=False,
        ).clamp_(0, num_source_points - 1)
        sampled = torch.gather(
            xyz,
            dim=1,
            index=gather_index.to(dtype=torch.long).unsqueeze(-1).expand(-1, -1, 3),
        )
        return torch.where(
            valid_env.view(-1, 1, 1),
            sampled,
            torch.zeros_like(sampled),
        )
