from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.types import SimConfig

from sim.maniskill.bimanual_embodiment import BimanualEmbodimentSpec


class BaseBimanualEnv(BaseEnv):
    """Minimal ManiSkill base env that validates a RoboTwin-style bimanual embodiment."""

    SUPPORTED_ROBOTS = ["none"]
    SUPPORTED_OBS_MODES = ("state_dict", "state", "none", "pointcloud+state_dict")
    SUPPORTED_REWARD_MODES = ("none", "dense", "normalized_dense", "sparse")
    ARM_DOF = 6
    GRIPPER_OPEN_THRESHOLD = 0.8
    ACTION_DIM = 14

    def __init__(
        self,
        *args,
        embodiment_spec: BimanualEmbodimentSpec,
        robot_uids="none",
        **kwargs,
    ):
        self.embodiment_spec = embodiment_spec
        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        action_low = np.asarray([-np.pi] * self.ARM_DOF + [0.0] + [-np.pi] * self.ARM_DOF + [0.0], dtype=np.float32)
        action_high = np.asarray([np.pi] * self.ARM_DOF + [1.0] + [np.pi] * self.ARM_DOF + [1.0], dtype=np.float32)
        self.single_action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.ACTION_DIM,),
            dtype=np.float32,
        )
        self.action_space = self.single_action_space
        self._orig_single_action_space = copy.deepcopy(self.single_action_space)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=200, control_freq=20)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[1.4, 0.0, 1.3], target=[0.0, -0.3, 0.8])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.8, 1.2, 1.4], target=[0.0, -0.25, 0.8])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_robot(self, arm_index: int, name: str):
        return self.embodiment_spec.build_robot(
            self.scene,
            arm_index=arm_index,
            name=name,
        )

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, altitude=0)
        table_builder = self.scene.create_actor_builder()
        table_builder.add_box_collision(half_size=[0.55, 0.4, 0.02])
        if self.scene.can_render():
            table_builder.add_box_visual(
                half_size=[0.55, 0.4, 0.02],
                material=sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.5, 1.0]),
            )
        table_builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, 0.82]))
        self.workspace_table = table_builder.build_static(name="workspace_table")
        self.left_robot = self._load_robot(arm_index=0, name="left_embodiment")
        self.right_robot = self._load_robot(arm_index=1, name="right_embodiment")
        self.left_arm_joints = self._get_arm_joints(self.left_robot, arm_index=0)
        self.right_arm_joints = self._get_arm_joints(self.right_robot, arm_index=1)
        self.left_gripper = self._get_gripper_joints(self.left_robot, arm_index=0)
        self.right_gripper = self._get_gripper_joints(self.right_robot, arm_index=1)
        self._configure_robot_drives(self.left_arm_joints, self.left_gripper, arm_index=0)
        self._configure_robot_drives(self.right_arm_joints, self.right_gripper, arm_index=1)

    def _get_arm_joints(self, robot, arm_index: int):
        return [
            robot.find_joint_by_name(joint_name)
            for joint_name in self.embodiment_spec.arm_joint_names(arm_index)
        ]

    def _get_gripper_joints(self, robot, arm_index: int):
        gripper_name = self.embodiment_spec.gripper_name(arm_index)
        joints = [(robot.find_joint_by_name(gripper_name["base"]), 1.0, 0.0)]
        for joint_name, sign, offset in gripper_name["mimic"]:
            joints.append((robot.find_joint_by_name(joint_name), float(sign), float(offset)))
        return joints

    def _configure_robot_drives(self, arm_joints, gripper_joints, arm_index: int):
        joint_stiffness, joint_damping = self.embodiment_spec.joint_drive_params(arm_index)
        gripper_stiffness, gripper_damping = self.embodiment_spec.gripper_drive_params(arm_index)
        for joint in arm_joints:
            joint.set_drive_properties(joint_stiffness, joint_damping)
        for joint, _, _ in gripper_joints:
            joint.set_drive_properties(gripper_stiffness, gripper_damping)

    def _get_gripper_state_tensor(self, arm_index: int) -> torch.Tensor:
        attr_name = "left_gripper_val" if arm_index == 0 else "right_gripper_val"
        value = getattr(self, attr_name)
        if isinstance(value, torch.Tensor):
            tensor_value = value.to(device=self.device, dtype=torch.float32)
            if tensor_value.ndim == 1:
                tensor_value = tensor_value.unsqueeze(-1)
            if tensor_value.shape == (self.num_envs, 1):
                setattr(self, attr_name, tensor_value)
                return tensor_value

        fill_value = float(value)
        tensor_value = torch.full(
            (self.num_envs, 1),
            fill_value,
            device=self.device,
            dtype=torch.float32,
        )
        setattr(self, attr_name, tensor_value)
        return tensor_value

    def _set_gripper_state_tensor(self, gripper_val: torch.Tensor, arm_index: int, env_idx: torch.Tensor | None = None):
        state = self._get_gripper_state_tensor(arm_index).clone()
        if env_idx is None:
            state[:, 0] = gripper_val.reshape(-1)
        else:
            state[env_idx, 0] = gripper_val.reshape(-1)
        if arm_index == 0:
            self.left_gripper_val = state
        else:
            self.right_gripper_val = state

    def _get_gripper_open_mask(self, arm_index: int) -> torch.Tensor:
        return self._get_gripper_state_tensor(arm_index).squeeze(-1) > self.GRIPPER_OPEN_THRESHOLD

    def _initialize_robot_state(self, robot, env_idx: torch.Tensor, arm_index: int):
        env_idx = common.to_tensor(env_idx, device=self.device).to(torch.long).reshape(-1)
        qpos = robot.get_qpos().clone()
        qvel = robot.get_qvel().clone()
        if qpos.numel() == 0:
            return
        qpos[env_idx] = 0
        qvel[env_idx] = 0

        home = self.embodiment_spec.homestate(arm_index)
        if len(home) > 0:
            home = torch.tensor(home, device=self.device, dtype=qpos.dtype)
            dof = min(home.shape[0], qpos.shape[1])
            qpos[env_idx, :dof] = home[:dof]

        robot.set_qpos(qpos)
        robot.set_qvel(qvel)
        arm_joints = self.left_arm_joints if arm_index == 0 else self.right_arm_joints
        arm_dof = min(len(arm_joints), qpos.shape[1])
        if arm_dof > 0:
            robot.set_joint_drive_targets(qpos[env_idx, :arm_dof], joints=arm_joints[:arm_dof])
            robot.set_joint_drive_velocity_targets(
                torch.zeros((len(env_idx), arm_dof), device=self.device, dtype=qvel.dtype),
                joints=arm_joints[:arm_dof],
            )

    def _set_gripper(self, gripper_val: torch.Tensor, arm_index: int, env_idx: torch.Tensor | None = None):
        if env_idx is not None:
            env_idx = common.to_tensor(env_idx, device=self.device).to(torch.long).reshape(-1)
            target_envs = len(env_idx)
        else:
            target_envs = self.num_envs

        gripper_val = common.to_tensor(gripper_val, device=self.device).to(torch.float32).reshape(-1)
        if gripper_val.numel() == 1 and target_envs > 1:
            gripper_val = gripper_val.repeat(target_envs)
        gripper_val = torch.clamp(gripper_val, 0.0, 1.0)
        gripper_scale = self.embodiment_spec.gripper_scale(arm_index)
        real_gripper_val = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])

        joints = self.left_gripper if arm_index == 0 else self.right_gripper
        self._set_gripper_state_tensor(gripper_val, arm_index=arm_index, env_idx=env_idx)

        joint_objs = [joint for joint, _, _ in joints]
        drive_targets = torch.stack(
            [real_gripper_val * sign + offset for _, sign, offset in joints],
            dim=1,
        )
        articulation = joint_objs[0].articulation
        articulation.set_joint_drive_targets(drive_targets, joints=joint_objs)
        articulation.set_joint_drive_velocity_targets(
            torch.zeros_like(drive_targets),
            joints=joint_objs,
        )

    def get_left_gripper_val(self):
        value = self._get_gripper_state_tensor(arm_index=0).squeeze(-1)
        if self.num_envs == 1:
            return float(value[0].item())
        return value

    def get_right_gripper_val(self):
        value = self._get_gripper_state_tensor(arm_index=1).squeeze(-1)
        if self.num_envs == 1:
            return float(value[0].item())
        return value

    def is_left_gripper_open(self):
        value = self._get_gripper_open_mask(arm_index=0)
        if self.num_envs == 1:
            return bool(value[0].item())
        return value

    def is_right_gripper_open(self):
        value = self._get_gripper_open_mask(arm_index=1)
        if self.num_envs == 1:
            return bool(value[0].item())
        return value

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self._initialize_robot_state(self.left_robot, env_idx, arm_index=0)
            self._initialize_robot_state(self.right_robot, env_idx, arm_index=1)
            self._set_gripper(torch.zeros(len(env_idx), device=self.device), arm_index=0, env_idx=env_idx)
            self._set_gripper(torch.zeros(len(env_idx), device=self.device), arm_index=1, env_idx=env_idx)

    def _apply_arm_action(self, robot, arm_joints, target_qpos: torch.Tensor):
        target_arm_qpos = torch.clamp(target_qpos, min=-np.pi, max=np.pi)
        robot.set_joint_drive_targets(target_arm_qpos, joints=arm_joints)
        robot.set_joint_drive_velocity_targets(
            torch.zeros_like(target_arm_qpos),
            joints=arm_joints,
        )

    def _step_action(self, action):
        if action is None:
            parsed_action = None
        else:
            parsed_action = common.to_tensor(action, device=self.device).to(torch.float32)
            if parsed_action.shape == self._orig_single_action_space.shape:
                parsed_action = common.batch(parsed_action)
            self._apply_arm_action(self.left_robot, self.left_arm_joints, parsed_action[:, 0:6])
            self._set_gripper(parsed_action[:, 6], arm_index=0)
            self._apply_arm_action(self.right_robot, self.right_arm_joints, parsed_action[:, 7:13])
            self._set_gripper(parsed_action[:, 13], arm_index=1)

        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self._before_simulation_step()
            self.scene.step()
            self._after_simulation_step()
        self._after_control_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        return parsed_action

    def _get_agent_qpos(self):
        left_arm_qpos = torch.stack(
            [joint.qpos.reshape(-1) for joint in self.left_arm_joints],
            dim=1,
        )
        right_arm_qpos = torch.stack(
            [joint.qpos.reshape(-1) for joint in self.right_arm_joints],
            dim=1,
        )
        left_gripper = torch.full(
            (self.num_envs, 1),
            0.0,
            device=self.device,
            dtype=torch.float32,
        )
        right_gripper = torch.full(
            (self.num_envs, 1),
            0.0,
            device=self.device,
            dtype=torch.float32,
        )
        left_gripper[:] = self._get_gripper_state_tensor(arm_index=0)
        right_gripper[:] = self._get_gripper_state_tensor(arm_index=1)
        return torch.cat(
            [left_arm_qpos, left_gripper, right_arm_qpos, right_gripper],
            dim=1,
        )

    def _get_obs_agent(self):
        return dict(
            qpos=self._get_agent_qpos(),
            left_gripper_val=self._get_gripper_state_tensor(arm_index=0),
            right_gripper_val=self._get_gripper_state_tensor(arm_index=1),
        )

    def get_state_dict(self):
        return self.scene.get_sim_state()

    def evaluate(self):
        left_root = self.left_robot.get_root_pose().p
        right_root = self.right_robot.get_root_pose().p
        x_separation = torch.abs(right_root[:, 0] - left_root[:, 0])
        robots_ready = torch.logical_and(
            torch.isfinite(left_root).all(dim=1),
            torch.logical_and(
                torch.isfinite(right_root).all(dim=1),
                x_separation > self.embodiment_spec.arms_distance * 0.9,
            ),
        )
        return {
            "success": robots_ready,
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            "robots_ready": robots_ready,
            "x_separation": x_separation,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_root_pose=self.left_robot.get_root_pose().raw_pose,
            right_root_pose=self.right_robot.get_root_pose().raw_pose,
            robots_ready=info["robots_ready"],
        )
        if "state" in self.obs_mode:
            obs.update(
                left_qpos=self.left_robot.get_qpos(),
                right_qpos=self.right_robot.get_qpos(),
                workspace_table_pose=self.workspace_table.pose.raw_pose,
                left_gripper_val=self._get_gripper_state_tensor(arm_index=0),
                right_gripper_val=self._get_gripper_state_tensor(arm_index=1),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["robots_ready"].to(torch.float32)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
