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
    TABLE_LENGTH = 1.2
    TABLE_WIDTH = 0.7
    TABLE_HEIGHT = 0.74
    TABLE_THICKNESS = 0.05
    WALL_HALF_SIZE = [3.0, 0.6, 1.5]
    WALL_POSE = [0.0, 1.0, 1.5]
    HEAD_CAMERA_FOVY_DEG = 37.0
    HEAD_CAMERA_WIDTH = 320
    HEAD_CAMERA_HEIGHT = 240
    HEAD_CAMERA_NEAR = 0.1
    HEAD_CAMERA_FAR = 100.0

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
        static_camera_list = self.embodiment_spec.left_config.get("static_camera_list", [])
        if len(static_camera_list) == 0:
            pose = sapien_utils.look_at(eye=[-0.032, -0.45, 1.35], target=[0.0, -0.15, 0.95])
            return [
                CameraConfig(
                    "head_camera",
                    pose,
                    self.HEAD_CAMERA_WIDTH,
                    self.HEAD_CAMERA_HEIGHT,
                    np.deg2rad(self.HEAD_CAMERA_FOVY_DEG),
                    self.HEAD_CAMERA_NEAR,
                    self.HEAD_CAMERA_FAR,
                )
            ]

        camera_info = static_camera_list[0]
        cam_pos = np.asarray(camera_info["position"], dtype=np.float32)
        cam_forward = np.asarray(camera_info.get("forward", -cam_pos), dtype=np.float32)
        cam_forward = cam_forward / np.linalg.norm(cam_forward)
        cam_left = np.asarray(
            camera_info.get("left", [-cam_forward[1], cam_forward[0], 0.0]),
            dtype=np.float32,
        )
        cam_left = cam_left / np.linalg.norm(cam_left)
        cam_up = np.cross(cam_forward, cam_left)
        cam_up = cam_up / np.linalg.norm(cam_up)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = np.stack([cam_forward, cam_left, cam_up], axis=1)
        transform[:3, 3] = cam_pos
        pose = sapien.Pose(transform)
        return [
            CameraConfig(
                camera_info.get("name", "head_camera"),
                pose,
                self.HEAD_CAMERA_WIDTH,
                self.HEAD_CAMERA_HEIGHT,
                np.deg2rad(self.HEAD_CAMERA_FOVY_DEG),
                self.HEAD_CAMERA_NEAR,
                self.HEAD_CAMERA_FAR,
            )
        ]

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
        wall_builder = self.scene.create_actor_builder()
        wall_builder.add_box_collision(half_size=self.WALL_HALF_SIZE)
        if self.scene.can_render():
            wall_builder.add_box_visual(
                half_size=self.WALL_HALF_SIZE,
                material=sapien.render.RenderMaterial(base_color=[1.0, 0.9, 0.9, 1.0]),
            )
        wall_builder.set_initial_pose(sapien.Pose(p=self.WALL_POSE))
        self.workspace_wall = wall_builder.build_static(name="workspace_wall")

        table_builder = self.scene.create_actor_builder()
        tabletop_pose = sapien.Pose([0.0, 0.0, -self.TABLE_THICKNESS / 2])
        tabletop_half_size = [self.TABLE_LENGTH / 2, self.TABLE_WIDTH / 2, self.TABLE_THICKNESS / 2]
        table_builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size)
        if self.scene.can_render():
            table_builder.add_box_visual(
                pose=tabletop_pose,
                half_size=tabletop_half_size,
                material=sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0]),
            )

        leg_spacing = 0.1
        leg_half_size = [self.TABLE_THICKNESS / 2, self.TABLE_THICKNESS / 2, self.TABLE_HEIGHT / 2 - 0.002]
        leg_z = -self.TABLE_HEIGHT / 2 - 0.002
        for i in (-1, 1):
            for j in (-1, 1):
                leg_pose = sapien.Pose(
                    [
                        i * (self.TABLE_LENGTH / 2 - leg_spacing / 2),
                        j * (self.TABLE_WIDTH / 2 - leg_spacing / 2),
                        leg_z,
                    ]
                )
                table_builder.add_box_collision(pose=leg_pose, half_size=leg_half_size)
                if self.scene.can_render():
                    table_builder.add_box_visual(
                        pose=leg_pose,
                        half_size=leg_half_size,
                        material=sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0]),
                    )

        table_builder.set_initial_pose(sapien.Pose(p=[0.0, 0.0, self.TABLE_HEIGHT]))
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

    def get_arm_robot(self, arm_index: int):
        return self.left_robot if arm_index == 0 else self.right_robot

    def get_arm_joint_names(self, arm_index: int) -> list[str]:
        return list(self.embodiment_spec.arm_joint_names(arm_index))

    def get_arm_link_names(self, arm_index: int) -> list[str]:
        robot = self.get_arm_robot(arm_index)
        names: list[str] = []
        for link in robot.get_links():
            if hasattr(link, "get_name"):
                names.append(link.get_name())
            else:
                names.append(str(link.name))
        return names

    def get_arm_joint_positions(self, arm_index: int) -> np.ndarray:
        joints = self.left_arm_joints if arm_index == 0 else self.right_arm_joints
        return np.asarray([float(joint.qpos.reshape(-1)[0].item()) for joint in joints], dtype=np.float64)

    def get_arm_full_joint_positions(self, arm_index: int) -> np.ndarray:
        robot = self.get_arm_robot(arm_index)
        qpos = robot.get_qpos()
        if hasattr(qpos, "detach") and hasattr(qpos, "cpu") and hasattr(qpos, "numpy"):
            qpos = qpos.detach().cpu().numpy()
        return np.asarray(qpos, dtype=np.float64).reshape(-1)

    def get_arm_home_joint_positions(self, arm_index: int) -> np.ndarray:
        return np.asarray(self.embodiment_spec.homestate(arm_index), dtype=np.float64)

    def _to_numpy_array(self, value, final_shape: tuple[int, ...]) -> np.ndarray:
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64).reshape(final_shape)

    def get_arm_root_pose_array(self, arm_index: int) -> np.ndarray:
        root_pose = self.get_arm_robot(arm_index).get_root_pose()
        return np.concatenate(
            [
                self._to_numpy_array(root_pose.p, (-1, 3))[0],
                self._to_numpy_array(root_pose.q, (-1, 4))[0],
            ],
            axis=0,
        ).astype(np.float64, copy=False)

    def get_arm_move_group(self, arm_index: int) -> str:
        config = self.embodiment_spec.left_config if arm_index == 0 else self.embodiment_spec.right_config
        move_group = config.get("move_group", "ee_link")
        if isinstance(move_group, (list, tuple)):
            return str(move_group[min(arm_index, len(move_group) - 1)])
        return str(move_group)

    def get_arm_end_effector_link(self, arm_index: int):
        move_group = self.get_arm_move_group(arm_index)
        for link in self.get_arm_robot(arm_index).get_links():
            link_name = link.get_name() if hasattr(link, "get_name") else str(link.name)
            if link_name == move_group:
                return link
        raise KeyError(f"Could not find end-effector link `{move_group}` for arm {arm_index}.")

    def get_arm_end_effector_pose_array(self, arm_index: int) -> np.ndarray:
        ee_link = self.get_arm_end_effector_link(arm_index)
        return np.concatenate(
            [
                self._to_numpy_array(ee_link.pose.p, (-1, 3))[0],
                self._to_numpy_array(ee_link.pose.q, (-1, 4))[0],
            ],
            axis=0,
        ).astype(np.float64, copy=False)

    def get_arm_planner_assets(self, arm_index: int) -> dict[str, str]:
        config = self.embodiment_spec.left_config if arm_index == 0 else self.embodiment_spec.right_config
        robot_dir = self.embodiment_spec.left_robot_dir if arm_index == 0 else self.embodiment_spec.right_robot_dir
        urdf_path = (robot_dir / config["urdf_path"]).resolve()
        srdf_rel_path = config.get("srdf_path")
        srdf_path = (robot_dir / srdf_rel_path).resolve() if srdf_rel_path else urdf_path.with_suffix(".srdf")
        return {
            "urdf_path": str(urdf_path),
            "srdf_path": str(srdf_path),
            "package_dir": str(robot_dir.resolve()),
            "move_group": self.get_arm_move_group(arm_index),
        }

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
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()

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
