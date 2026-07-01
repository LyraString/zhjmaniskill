from __future__ import annotations

import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sim.maniskill.tasks.handover_block_env import (
    BLOCK_MIDDLE_POSE,
    BOX_BASE_QUAT,
    BOX_CENTER_Z,
    TARGET_CENTER_Z,
    HandoverBlockEnv,
    _quat_multiply,
)


V2_BOX_X_RANGE = (-0.12, -0.09)
V2_BOX_Y_RANGE = (0.07, 0.11)
V2_BOX_YAW_LIMIT = 0.05
V2_TARGET_X_RANGE = (0.08, 0.16)
V2_TARGET_Y_RANGE = (0.11, 0.19)
V2_BLOCK_MIDDLE_POSE = [0.02, 0.11, 0.92, *BLOCK_MIDDLE_POSE[3:]]


@register_env("HandoverBlockUR5-v2", max_episode_steps=800)
class HandoverBlockUR5V2Env(HandoverBlockEnv):
    """UR5-focused fixed-frequency handover variant.

    This keeps the current 14-D absolute joint-position action interface, but
    narrows the spawn/goal region to the shared reachable workspace. It is a
    separate env id so the old v1 behavior stays available for comparison.
    """

    block_middle_pose = V2_BLOCK_MIDDLE_POSE

    def __init__(self, *args, legacy_rng_compat: bool = False, **kwargs):
        super().__init__(*args, legacy_rng_compat=False, **kwargs)
        self.block_middle_pose = list(V2_BLOCK_MIDDLE_POSE)
        self.block_middle_pose[1] += float(self.object_y_offset)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            episode_rng = self._batched_episode_rng[env_idx]

            box_x = torch.as_tensor(
                episode_rng.uniform(*V2_BOX_X_RANGE),
                device=self.device,
                dtype=torch.float32,
            ).reshape(b)
            box_y = torch.as_tensor(
                episode_rng.uniform(*V2_BOX_Y_RANGE),
                device=self.device,
                dtype=torch.float32,
            ).reshape(b)
            yaw = torch.as_tensor(
                episode_rng.uniform(-V2_BOX_YAW_LIMIT, V2_BOX_YAW_LIMIT),
                device=self.device,
                dtype=torch.float32,
            ).reshape(b)

            box_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            box_xyz[:, 0] = box_x + float(self.box_x_offset)
            box_xyz[:, 1] = box_y + float(self.object_y_offset)
            box_xyz[:, 2] = BOX_CENTER_Z

            yaw_half = yaw / 2.0
            yaw_q = torch.stack(
                [
                    torch.cos(yaw_half),
                    torch.zeros_like(yaw_half),
                    torch.zeros_like(yaw_half),
                    torch.sin(yaw_half),
                ],
                dim=-1,
            )
            base_q = torch.tensor(BOX_BASE_QUAT, device=self.device, dtype=box_xyz.dtype).repeat(b, 1)
            box_q = _quat_multiply(base_q, yaw_q)
            box_q = box_q / torch.linalg.norm(box_q, dim=-1, keepdim=True)
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=box_q))

            target_x = torch.as_tensor(
                episode_rng.uniform(*V2_TARGET_X_RANGE),
                device=self.device,
                dtype=torch.float32,
            ).reshape(b)
            target_y = torch.as_tensor(
                episode_rng.uniform(*V2_TARGET_Y_RANGE),
                device=self.device,
                dtype=torch.float32,
            ).reshape(b)

            target_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            target_xyz[:, 0] = target_x
            target_xyz[:, 1] = target_y + float(self.object_y_offset)
            target_xyz[:, 2] = TARGET_CENTER_Z
            self.target_box.set_pose(Pose.create_from_pq(p=target_xyz))
            self._get_reward_stage()[env_idx.to(torch.long)] = 0

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs["block_middle_pose"] = torch.tensor(
                self.block_middle_pose,
                device=self.device,
                dtype=self.box.pose.raw_pose.dtype,
            ).repeat(self.num_envs, 1)
        return obs
