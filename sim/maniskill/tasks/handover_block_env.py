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
BLOCK_MIDDLE_POSE = [0.0, 0.1, 0.95, 0.0, 1.0, 0.0, 0.0]
BOX_STATIC_FRICTION = 5.0
BOX_DYNAMIC_FRICTION = 5.0
BOX_CENTER_Z = 0.842
TARGET_CENTER_Z = 0.741
BOX_BASE_QUAT = [0.981, 0.0, 0.0, 0.195]
BOX_YAW_LIMIT = 0.2
REWARD_LEFT_GRASP_DIST = 0.18
REWARD_HANDOVER_DIST = 0.16
REWARD_RIGHT_GRASP_DIST = 0.18
REWARD_TARGET_NEAR_DIST = 0.08
REWARD_LEFT_CLEAR_DIST = 0.18
REWARD_GRIPPER_CLOSED_THRESHOLD = 0.25
REWARD_BOX_LIFT_Z = BOX_CENTER_Z + 0.025
REWARD_DROP_Z = TARGET_CENTER_Z + 0.02


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

    def __init__(
        self,
        *args,
        legacy_rng_compat: bool = False,
        object_y_offset: float = 0.0,
        box_x_offset: float = 0.0,
        **kwargs,
    ):
        self.legacy_rng_compat = bool(legacy_rng_compat)
        self.object_y_offset = float(object_y_offset)
        self.box_x_offset = float(box_x_offset)
        self.success_gate_enabled = True
        self.reward_stage = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        box_physical_material = sapien.physx.PhysxMaterial(
            static_friction=BOX_STATIC_FRICTION,
            dynamic_friction=BOX_DYNAMIC_FRICTION,
            restitution=0.0,
        )
        box_builder = self.scene.create_actor_builder()
        box_builder.add_box_collision(
            half_size=BOX_HALF_SIZE.tolist(),
            material=box_physical_material,
        )
        if self.scene.can_render():
            box_builder.add_box_visual(
                half_size=BOX_HALF_SIZE.tolist(),
                material=sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 1.0]),
            )
        box_builder.set_initial_pose(
            sapien.Pose(p=[-0.15 + self.box_x_offset, 0.1 + self.object_y_offset, BOX_CENTER_Z])
        )
        self.box = box_builder.build(name="box")

        target_builder = self.scene.create_actor_builder()
        target_builder.add_box_collision(half_size=TARGET_HALF_SIZE.tolist())
        if self.scene.can_render():
            target_builder.add_box_visual(
                half_size=TARGET_HALF_SIZE.tolist(),
                material=sapien.render.RenderMaterial(base_color=[0.0, 0.0, 1.0, 1.0]),
            )
        target_builder.set_initial_pose(sapien.Pose(p=[0.18, 0.175 + self.object_y_offset, TARGET_CENTER_Z]))
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
            box_xyz[:, 1] = box_y + float(self.object_y_offset)
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
            target_xyz[:, 1] = target_y + float(self.object_y_offset)
            if target_z_np is None:
                target_xyz[:, 2] = TARGET_CENTER_Z
            else:
                target_xyz[:, 2] = torch.as_tensor(target_z_np, device=self.device, dtype=torch.float32).reshape(b)
            self.target_box.set_pose(Pose.create_from_pq(p=target_xyz))
            self._get_reward_stage()[env_idx.to(torch.long)] = 0

    def _get_reward_stage(self) -> torch.Tensor:
        if not isinstance(self.reward_stage, torch.Tensor) or self.reward_stage.shape != (self.num_envs,):
            self.reward_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        else:
            self.reward_stage = self.reward_stage.to(device=self.device, dtype=torch.long)
        return self.reward_stage

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
        success = base_info["robots_ready"] & spatial_success & right_gripper_open & right_ee_clear & box_static
        if not bool(getattr(self, "success_gate_enabled", True)):
            success = torch.zeros_like(success, dtype=torch.bool)

        base_info.update(
            {
                "xy_aligned": xy_aligned,
                "z_aligned": z_aligned,
                "spatial_success": spatial_success,
                "right_gripper_open": right_gripper_open,
                "right_ee_clear": right_ee_clear,
                "box_static": box_static,
                "success": success,
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
        stage = self._get_reward_stage()
        dtype = self.box.pose.p.dtype
        device = self.device

        box_pos = self.box.pose.p
        left_ee_pos = self.get_arm_end_effector_link(arm_index=0).pose.p
        right_ee_pos = self.get_arm_end_effector_link(arm_index=1).pose.p

        box_bottom_center = box_pos.clone()
        box_bottom_center[:, 2] -= float(BOX_HALF_SIZE[2])
        target_top_center = self.target_box.pose.p.clone()
        target_top_center[:, 2] += float(TARGET_HALF_SIZE[2])
        handover_pos = torch.tensor(
            BLOCK_MIDDLE_POSE[:3],
            device=device,
            dtype=dtype,
        ).repeat(self.num_envs, 1)

        left_to_box = torch.linalg.norm(left_ee_pos - box_pos, dim=1)
        right_to_box = torch.linalg.norm(right_ee_pos - box_pos, dim=1)
        box_to_handover = torch.linalg.norm(box_pos - handover_pos, dim=1)
        target_delta = box_bottom_center - target_top_center
        target_xy_dist = torch.linalg.norm(target_delta[:, :2], dim=1)
        target_z_dist = torch.abs(target_delta[:, 2])
        box_to_target = torch.linalg.norm(target_delta, dim=1)

        left_gripper_val = self._get_gripper_state_tensor(arm_index=0).squeeze(-1)
        right_gripper_val = self._get_gripper_state_tensor(arm_index=1).squeeze(-1)
        left_closed = left_gripper_val < REWARD_GRIPPER_CLOSED_THRESHOLD
        right_closed = right_gripper_val < REWARD_GRIPPER_CLOSED_THRESHOLD
        box_lifted = box_pos[:, 2] > REWARD_BOX_LIFT_Z
        box_dropped = box_pos[:, 2] < REWARD_DROP_Z
        left_clear = left_to_box > REWARD_LEFT_CLEAR_DIST

        left_holding = left_closed & (left_to_box < REWARD_LEFT_GRASP_DIST) & box_lifted
        handover_ready = box_to_handover < REWARD_HANDOVER_DIST
        right_holding = right_closed & (right_to_box < REWARD_RIGHT_GRASP_DIST)
        target_near = box_to_target < REWARD_TARGET_NEAR_DIST

        next_stage = stage.clone()
        next_stage = torch.where((next_stage == 0) & left_holding, torch.ones_like(next_stage), next_stage)
        next_stage = torch.where((next_stage == 1) & handover_ready, torch.full_like(next_stage, 2), next_stage)
        next_stage = torch.where((next_stage == 2) & right_holding, torch.full_like(next_stage, 3), next_stage)
        next_stage = torch.where((next_stage == 3) & target_near, torch.full_like(next_stage, 4), next_stage)
        self.reward_stage = next_stage
        stage = next_stage

        reward = 0.2 * info["robots_ready"].to(torch.float32)
        action_penalty = torch.zeros_like(reward)
        if isinstance(action, torch.Tensor):
            action_penalty = torch.mean(torch.square(action.to(device=device, dtype=torch.float32)), dim=-1)

        stage0 = stage == 0
        stage1 = stage == 1
        stage2 = stage == 2
        stage3 = stage == 3
        stage4 = stage >= 4

        left_reach_reward = torch.exp(-8.0 * left_to_box)
        left_grasp_reward = left_closed.to(torch.float32) * (left_to_box < REWARD_LEFT_GRASP_DIST).to(torch.float32)
        handover_reward = torch.exp(-6.0 * box_to_handover)
        right_reach_reward = torch.exp(-8.0 * right_to_box)
        right_grasp_reward = right_closed.to(torch.float32) * (right_to_box < REWARD_RIGHT_GRASP_DIST).to(torch.float32)
        target_reward = torch.exp(-10.0 * box_to_target)
        final_release_reward = (
            info["right_gripper_open"].to(torch.float32)
            + info["right_ee_clear"].to(torch.float32)
            + info["box_static"].to(torch.float32)
        )

        reward = reward + stage0.to(torch.float32) * (left_reach_reward + 0.5 * left_grasp_reward)
        reward = reward + stage1.to(torch.float32) * (1.0 + 2.0 * handover_reward + 0.5 * left_closed.to(torch.float32))
        reward = reward + stage2.to(torch.float32) * (
            2.0 + right_reach_reward + 0.5 * right_grasp_reward + 0.5 * left_closed.to(torch.float32)
        )
        reward = reward + stage3.to(torch.float32) * (
            3.0 + 3.0 * target_reward + 0.5 * right_closed.to(torch.float32) + 0.5 * left_clear.to(torch.float32)
        )
        reward = reward + stage4.to(torch.float32) * (4.0 + 4.0 * target_reward + final_release_reward)
        reward = reward + 10.0 * info["success"].to(torch.float32)
        reward = reward - 1.0 * box_dropped.to(torch.float32)
        reward = reward - 0.01 * action_penalty

        info.update(
            reward_stage=stage.to(torch.float32),
            reward_left_to_box=left_to_box,
            reward_right_to_box=right_to_box,
            reward_box_to_handover=box_to_handover,
            reward_box_to_target=box_to_target,
            reward_target_xy_dist=target_xy_dist,
            reward_target_z_dist=target_z_dist,
            reward_box_lifted=box_lifted,
            reward_left_holding=left_holding,
            reward_right_holding=right_holding,
        )
        return reward
