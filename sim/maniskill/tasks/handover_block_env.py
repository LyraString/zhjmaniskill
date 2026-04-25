from __future__ import annotations

from typing import Any
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from sim.maniskill.tasks.dual_ur5_wsg_env import DualUR5WSGEnv


BOX_HALF_SIZE = torch.tensor([0.03, 0.03, 0.1])
TARGET_HALF_SIZE = torch.tensor([0.05, 0.05, 0.005])
SUCCESS_XY_EPS = 0.03
SUCCESS_Z_EPS = 0.01
BOX_SETTLE_LINEAR_EPS = 0.05
BOX_SETTLE_ANGULAR_EPS = 0.50
RIGHT_GRIPPER_RELEASE_THRESHOLD = 0.95
RIGHT_EE_CLEAR_DISTANCE = 0.14
BLOCK_MIDDLE_POSE = [0.0, 0.1, 0.9, 0.0, 1.0, 0.0, 0.0]
BOX_CENTER_Z = 0.842
TARGET_CENTER_Z = 0.741
BOX_BASE_QUAT = [0.981, 0.0, 0.0, 0.195]
BOX_YAW_LIMIT = 0.2
BOX_STATIC_FRICTION = 2.0
BOX_DYNAMIC_FRICTION = 2.0
BOX_RESTITUTION = 0.0


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


@register_env("HandoverBlock-v1", max_episode_steps=800)
class HandoverBlockEnv(DualUR5WSGEnv):
    """Minimal ManiSkill port of RoboTwin handover_block scene and spatial success condition."""

    def __init__(self, *args, legacy_rng_compat: bool = False, **kwargs):
        self.legacy_rng_compat = bool(legacy_rng_compat)
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        box_builder = self.scene.create_actor_builder()
        box_material = sapien.physx.PhysxMaterial(
            BOX_STATIC_FRICTION,
            BOX_DYNAMIC_FRICTION,
            BOX_RESTITUTION,
        )
        box_builder.add_box_collision(half_size=BOX_HALF_SIZE.tolist(), material=box_material)
        if self.scene.can_render():
            box_builder.add_box_visual(
                half_size=BOX_HALF_SIZE.tolist(),
                material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0]),
            )
        box_builder.set_initial_pose(sapien.Pose(p=[-0.15, 0.1, BOX_CENTER_Z]))
        self.box = box_builder.build(name="box")

        target_builder = self.scene.create_actor_builder()
        target_builder.add_box_collision(half_size=TARGET_HALF_SIZE.tolist())
        if self.scene.can_render():
            target_builder.add_box_visual(
                half_size=TARGET_HALF_SIZE.tolist(),
                material=sapien.render.RenderMaterial(base_color=[0.0, 0.0, 1.0, 1.0]),
            )
        target_builder.set_initial_pose(sapien.Pose(p=[0.18, 0.175, TARGET_CENTER_Z]))
        self.target_box = target_builder.build_kinematic(name="target_box")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            episode_rng = self._batched_episode_rng[env_idx]

            if self.legacy_rng_compat:
                # RoboTwin's original handover_block consumes RNG before actor
                # placement for:
                # 1) table_z_bias uniform draw
                # 2) head-camera random direction (np.random.randn(3))
                # 3) head-camera distance jitter uniform draw
                # 4) fixed z values and zero roll/pitch ranges in rand_pose
                # Keep those dummy draws so identical seeds reproduce the old
                # box/target poses.
                episode_rng.uniform(-0.0, 0.0)
                episode_rng.randn(3)
                episode_rng.uniform(0.0, 0.0)
                box_x_np = episode_rng.uniform(-0.25, -0.05)
                box_y_np = episode_rng.uniform(0.0, 0.25)
                box_z_np = episode_rng.uniform(BOX_CENTER_Z, BOX_CENTER_Z)
                episode_rng.uniform(-0.0, 0.0)
                episode_rng.uniform(-0.0, 0.0)
                yaw_np = episode_rng.uniform(-BOX_YAW_LIMIT, BOX_YAW_LIMIT)
            else:
                box_x_np = episode_rng.uniform(-0.25, -0.05)
                box_y_np = episode_rng.uniform(0.0, 0.25)
                box_z_np = None
                yaw_np = episode_rng.uniform(-BOX_YAW_LIMIT, BOX_YAW_LIMIT)

            box_x = torch.as_tensor(box_x_np, device=self.device, dtype=torch.float32).reshape(b)
            box_y = torch.as_tensor(box_y_np, device=self.device, dtype=torch.float32).reshape(b)
            yaw = torch.as_tensor(yaw_np, device=self.device, dtype=torch.float32).reshape(b)

            box_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            box_xyz[:, 0] = box_x
            box_xyz[:, 1] = box_y
            if box_z_np is None:
                box_xyz[:, 2] = BOX_CENTER_Z
            else:
                box_xyz[:, 2] = torch.as_tensor(box_z_np, device=self.device, dtype=torch.float32).reshape(b)

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

            target_x_np = episode_rng.uniform(0.10, 0.25)
            target_y_np = episode_rng.uniform(0.15, 0.20)
            target_z_np = episode_rng.uniform(TARGET_CENTER_Z, TARGET_CENTER_Z) if self.legacy_rng_compat else None

            target_x = torch.as_tensor(target_x_np, device=self.device, dtype=torch.float32).reshape(b)
            target_y = torch.as_tensor(target_y_np, device=self.device, dtype=torch.float32).reshape(b)

            target_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            target_xyz[:, 0] = target_x
            target_xyz[:, 1] = target_y
            if target_z_np is None:
                target_xyz[:, 2] = TARGET_CENTER_Z
            else:
                target_xyz[:, 2] = torch.as_tensor(target_z_np, device=self.device, dtype=torch.float32).reshape(b)
            self.target_box.set_pose(Pose.create_from_pq(p=target_xyz))

    def evaluate(self):
        base_info = super().evaluate()

        box_bottom_center = self.box.pose.p.clone()
        box_bottom_center[:, 2] -= float(BOX_HALF_SIZE[2])

        target_top_center = self.target_box.pose.p.clone()
        target_top_center[:, 2] += float(TARGET_HALF_SIZE[2])

        xy_aligned = (
            torch.abs(box_bottom_center[:, 0] - target_top_center[:, 0]) < SUCCESS_XY_EPS
        ) & (
            torch.abs(box_bottom_center[:, 1] - target_top_center[:, 1]) < SUCCESS_XY_EPS
        )
        z_aligned = (
            torch.abs(box_bottom_center[:, 2] - target_top_center[:, 2]) < SUCCESS_Z_EPS
        )
        spatial_success = xy_aligned & z_aligned
        right_gripper_open = self._get_gripper_state_tensor(arm_index=1).squeeze(-1) > RIGHT_GRIPPER_RELEASE_THRESHOLD
        right_ee_pos = self.get_arm_end_effector_link(arm_index=1).pose.p
        right_ee_clear = (
            torch.linalg.norm(right_ee_pos - self.box.pose.p, axis=1) >= RIGHT_EE_CLEAR_DISTANCE
        )
        box_static = (
            torch.linalg.norm(self.box.linear_velocity, axis=1) <= BOX_SETTLE_LINEAR_EPS
        ) & (
            torch.linalg.norm(self.box.angular_velocity, axis=1) <= BOX_SETTLE_ANGULAR_EPS
        )
        success = base_info["robots_ready"] & spatial_success

        base_info.update(
            {
                "xy_aligned": xy_aligned,
                "z_aligned": z_aligned,
                "spatial_success": spatial_success,
                "right_gripper_open": right_gripper_open,
                "right_ee_clear": right_ee_clear,
                "box_static": box_static,
                "success": success & right_gripper_open & right_ee_clear & box_static,
            }
        )
        return base_info

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        obs.update(
            box_pose=self.box.pose.raw_pose,
            target_box_pose=self.target_box.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            box_bottom_center = self.box.pose.p.clone()
            box_bottom_center[:, 2] -= float(BOX_HALF_SIZE[2])

            target_top_center = self.target_box.pose.p.clone()
            target_top_center[:, 2] += float(TARGET_HALF_SIZE[2])

            obs.update(
                box_bottom_to_target_top=target_top_center - box_bottom_center,
                block_middle_pose=torch.tensor(
                    BLOCK_MIDDLE_POSE,
                    device=self.device,
                    dtype=self.box.pose.raw_pose.dtype,
                ).repeat(self.num_envs, 1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = super().compute_dense_reward(obs=obs, action=action, info=info)
        reward = reward + info["xy_aligned"].to(torch.float32)
        reward = reward + info["z_aligned"].to(torch.float32)
        reward = reward + 2.0 * info["spatial_success"].to(torch.float32)
        reward = reward + info["right_gripper_open"].to(torch.float32)
        reward = reward + info["right_ee_clear"].to(torch.float32)
        reward = reward + info["box_static"].to(torch.float32)
        return reward
