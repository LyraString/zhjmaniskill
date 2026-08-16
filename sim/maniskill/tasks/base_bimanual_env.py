from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.agents.controllers.pd_ee_pose import (
    PDEEPoseController,
    PDEEPoseControllerConfig,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    quaternion_to_matrix,
)
from mani_skill.utils.structs.types import SimConfig

from sim.maniskill.bimanual_embodiment import BimanualEmbodimentSpec


class _StrictPDEEPoseController(PDEEPoseController):
    """PDEE controller that never hides an IK failure as a zero joint target."""

    def set_action(self, action):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        previous_pose = self._target_pose if self.config.use_target else self.ee_pose_at_base
        self._target_pose = self.compute_target_pose(previous_pose, action)

        # ManiSkill's CUDA differential IK currently performs one Jacobian update
        # even when solver_iterations is configured. Iterate here against the same
        # absolute target so a bounded Cartesian action is solved accurately.
        solver_config = dict(self.config.delta_solver_config)
        solver_iterations = int(solver_config.pop("solver_iterations", 1))
        q0 = self.articulation.get_qpos()
        target_qpos = None
        for _ in range(max(1, solver_iterations)):
            target_qpos = self.kinematics.compute_ik(
                pose=self._target_pose,
                q0=q0,
                is_delta_pose=False,
                current_pose=None,
                solver_config=solver_config,
            )
            if target_qpos is None:
                raise RuntimeError("PDEE inverse kinematics failed.")
            if not torch.isfinite(target_qpos).all():
                raise RuntimeError("PDEE inverse kinematics returned a non-finite joint target.")
            q0 = target_qpos

        self._target_qpos = target_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)


class BaseBimanualEnv(BaseEnv):
    """Minimal ManiSkill base env that validates a RoboTwin-style bimanual embodiment."""

    SUPPORTED_ROBOTS = ["none"]
    SUPPORTED_OBS_MODES = ("state_dict", "state", "none", "pointcloud+state_dict")
    SUPPORTED_REWARD_MODES = ("none", "dense", "normalized_dense", "sparse")
    ARM_DOF = 6
    GRIPPER_OPEN_THRESHOLD = 0.8
    ACTION_DIM = 14
    JOINT_POSITION_CONTROL_MODE = "bimanual_pd_joint_pos"
    DELTA_POSE_CONTROL_MODE = "bimanual_pd_ee_delta_pose"
    DELTA_POSITION_LIMIT = 0.015
    DELTA_ROTATION_LIMIT = 0.10
    ABSOLUTE_EE_POSITION_LIMIT = 2.0
    ABSOLUTE_EE_ROTATION_LIMIT = np.pi
    DELTA_JOINT_STIFFNESS_SCALE = 8.0
    DELTA_JOINT_DAMPING_SCALE = 3.0
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
    WRIST_CAMERA_FOVY_DEG = 60.0

    def __init__(
        self,
        *args,
        embodiment_spec: BimanualEmbodimentSpec,
        robot_uids="none",
        **kwargs,
    ):
        self.enable_wrist_cameras = bool(kwargs.pop("enable_wrist_cameras", False))
        self.clamp_gripper_drive_targets = bool(
            kwargs.pop("clamp_gripper_drive_targets", False)
        )
        self.disable_arm_self_collisions = bool(
            kwargs.pop("disable_arm_self_collisions", False)
        )
        self.workspace_table_y = float(kwargs.pop("workspace_table_y", 0.0))
        self.expert_grasp_variant = str(kwargs.pop("expert_grasp_variant", ""))
        self.embodiment_spec = embodiment_spec
        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0
        self._manual_gripper_drive_override: dict[int, float] = {}
        self.left_delta_pose_controller = None
        self.right_delta_pose_controller = None
        self.delta_controller_failure_count = 0
        self.last_delta_control_debug: dict[str, Any] = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        if self._control_mode in (None, "", "continuous", self.JOINT_POSITION_CONTROL_MODE):
            action_low = np.asarray(
                [-np.pi] * self.ARM_DOF + [0.0] + [-np.pi] * self.ARM_DOF + [0.0],
                dtype=np.float32,
            )
            action_high = np.asarray(
                [np.pi] * self.ARM_DOF + [1.0] + [np.pi] * self.ARM_DOF + [1.0],
                dtype=np.float32,
            )
        elif self._control_mode == self.DELTA_POSE_CONTROL_MODE:
            arm_low = [
                -self.ABSOLUTE_EE_POSITION_LIMIT,
                -self.ABSOLUTE_EE_POSITION_LIMIT,
                -self.ABSOLUTE_EE_POSITION_LIMIT,
                -self.ABSOLUTE_EE_ROTATION_LIMIT,
                -self.ABSOLUTE_EE_ROTATION_LIMIT,
                -self.ABSOLUTE_EE_ROTATION_LIMIT,
                0.0,
            ]
            arm_high = [
                self.ABSOLUTE_EE_POSITION_LIMIT,
                self.ABSOLUTE_EE_POSITION_LIMIT,
                self.ABSOLUTE_EE_POSITION_LIMIT,
                self.ABSOLUTE_EE_ROTATION_LIMIT,
                self.ABSOLUTE_EE_ROTATION_LIMIT,
                self.ABSOLUTE_EE_ROTATION_LIMIT,
                1.0,
            ]
            action_low = np.asarray(arm_low + arm_low, dtype=np.float32)
            action_high = np.asarray(arm_high + arm_high, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported bimanual control mode: {self._control_mode}")
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
        sensor_configs = []
        static_camera_list = self.embodiment_spec.left_config.get("static_camera_list", [])
        if len(static_camera_list) == 0:
            pose = sapien_utils.look_at(eye=[-0.032, -0.45, 1.35], target=[0.0, -0.15, 0.95])
            sensor_configs.append(
                CameraConfig(
                    "head_camera",
                    pose,
                    self.HEAD_CAMERA_WIDTH,
                    self.HEAD_CAMERA_HEIGHT,
                    np.deg2rad(self.HEAD_CAMERA_FOVY_DEG),
                    self.HEAD_CAMERA_NEAR,
                    self.HEAD_CAMERA_FAR,
                )
            )
        else:
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
            sensor_configs.append(
                CameraConfig(
                    camera_info.get("name", "head_camera"),
                    pose,
                    self.HEAD_CAMERA_WIDTH,
                    self.HEAD_CAMERA_HEIGHT,
                    np.deg2rad(self.HEAD_CAMERA_FOVY_DEG),
                    self.HEAD_CAMERA_NEAR,
                    self.HEAD_CAMERA_FAR,
                )
            )

        if self.enable_wrist_cameras:
            wrist_pose = sapien.Pose()
            sensor_configs.extend(
                [
                    CameraConfig(
                        "left_wrist_camera",
                        wrist_pose,
                        self.HEAD_CAMERA_WIDTH,
                        self.HEAD_CAMERA_HEIGHT,
                        np.deg2rad(self.WRIST_CAMERA_FOVY_DEG),
                        self.HEAD_CAMERA_NEAR,
                        self.HEAD_CAMERA_FAR,
                        mount=self.get_arm_link_by_name(arm_index=0, link_name="camera"),
                    ),
                    CameraConfig(
                        "right_wrist_camera",
                        wrist_pose,
                        self.HEAD_CAMERA_WIDTH,
                        self.HEAD_CAMERA_HEIGHT,
                        np.deg2rad(self.WRIST_CAMERA_FOVY_DEG),
                        self.HEAD_CAMERA_NEAR,
                        self.HEAD_CAMERA_FAR,
                        mount=self.get_arm_link_by_name(arm_index=1, link_name="camera"),
                    ),
                ]
            )

        return sensor_configs

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

        table_builder.set_initial_pose(sapien.Pose(p=[0.0, self.workspace_table_y, self.TABLE_HEIGHT]))
        self.workspace_table = table_builder.build_static(name="workspace_table")
        self.left_robot = self._load_robot(arm_index=0, name="left_embodiment")
        self.right_robot = self._load_robot(arm_index=1, name="right_embodiment")
        if self.disable_arm_self_collisions:
            # Use distinct bits so each arm keeps colliding with the other arm,
            # the objects, and the scene while ignoring only its own mesh overlap.
            for link in self.left_robot.get_links():
                link.set_collision_group_bit(group=2, bit_idx=28, bit=1)
            for link in self.right_robot.get_links():
                link.set_collision_group_bit(group=2, bit_idx=27, bit=1)
        self.left_arm_joints = self._get_arm_joints(self.left_robot, arm_index=0)
        self.right_arm_joints = self._get_arm_joints(self.right_robot, arm_index=1)
        self.left_gripper = self._get_gripper_joints(self.left_robot, arm_index=0)
        self.right_gripper = self._get_gripper_joints(self.right_robot, arm_index=1)
        self._configure_robot_drives(self.left_arm_joints, self.left_gripper, arm_index=0)
        self._configure_robot_drives(self.right_arm_joints, self.right_gripper, arm_index=1)
        if self._control_mode == self.DELTA_POSE_CONTROL_MODE:
            self.left_delta_pose_controller = self._build_delta_pose_controller(arm_index=0)
            self.right_delta_pose_controller = self._build_delta_pose_controller(arm_index=1)

    def _build_delta_pose_controller(self, arm_index: int) -> PDEEPoseController:
        stiffness, damping = self.embodiment_spec.joint_drive_params(arm_index)
        assets = self.get_arm_planner_assets(arm_index)
        config = PDEEPoseControllerConfig(
            joint_names=self.get_arm_joint_names(arm_index),
            pos_lower=-self.ABSOLUTE_EE_POSITION_LIMIT,
            pos_upper=self.ABSOLUTE_EE_POSITION_LIMIT,
            rot_lower=-self.ABSOLUTE_EE_ROTATION_LIMIT,
            rot_upper=self.ABSOLUTE_EE_ROTATION_LIMIT,
            stiffness=stiffness * self.DELTA_JOINT_STIFFNESS_SCALE,
            damping=damping * self.DELTA_JOINT_DAMPING_SCALE,
            ee_link=self.get_arm_move_group(arm_index),
            urdf_path=assets["urdf_path"],
            frame="root_translation:root_aligned_body_rotation",
            use_delta=False,
            use_target=False,
            interpolate=False,
            normalize_action=False,
            delta_solver_config={
                "type": "levenberg_marquardt",
                "alpha": 1.0,
                "solver_iterations": 8,
            },
        )
        controller = _StrictPDEEPoseController(
            config=config,
            articulation=self.get_arm_robot(arm_index),
            control_freq=self._control_freq,
            sim_freq=self._sim_freq,
            scene=self.scene,
        )
        controller.set_drive_property()
        return controller

    def prepare_action_chunk(self, action_chunk, env_ids=None):
        """Decode one anchor-relative policy chunk into absolute EEF targets.

        The left and right EEF anchors are sampled exactly once when this method
        is called. Every pose in the returned chunk is fixed in its arm's root
        frame and can be sent directly to the absolute EEF controller.
        """

        return_numpy = isinstance(action_chunk, np.ndarray)
        chunk = common.to_tensor(action_chunk, device=self.device).to(torch.float32)
        squeeze_batch = False
        if chunk.ndim == 2:
            if self.num_envs != 1:
                raise ValueError(
                    "Unbatched action chunks are only valid for a single environment."
                )
            chunk = chunk.unsqueeze(0)
            squeeze_batch = True
        if chunk.ndim != 3 or chunk.shape[-1] != self.ACTION_DIM:
            raise ValueError(
                "Expected anchor-relative action chunk shape [B, K, 14] or [K, 14], "
                f"got {tuple(chunk.shape)}."
            )
        if chunk.shape[1] <= 0:
            raise ValueError("Action chunk must contain at least one target.")
        if not torch.isfinite(chunk).all():
            raise ValueError("Anchor-relative action chunk contains non-finite values.")

        if env_ids is None:
            if chunk.shape[0] != self.num_envs:
                raise ValueError(
                    f"Expected chunk batch {self.num_envs}, got {chunk.shape[0]}."
                )
            selected_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            selected_env_ids = torch.as_tensor(
                env_ids,
                device=self.device,
                dtype=torch.long,
            ).flatten()
            if selected_env_ids.numel() != chunk.shape[0]:
                raise ValueError(
                    "env_ids length must match the action chunk batch dimension."
                )
            if torch.any(selected_env_ids < 0) or torch.any(
                selected_env_ids >= self.num_envs
            ):
                raise IndexError("env_ids contains an out-of-range environment index.")

        absolute_targets = chunk.clone()
        arm_specs = (
            (0, self.left_delta_pose_controller),
            (7, self.right_delta_pose_controller),
        )
        for offset, controller in arm_specs:
            if controller is None:
                raise RuntimeError("Absolute EEF controller is not initialized.")
            anchor_pose = controller.ee_pose_at_base.raw_pose[selected_env_ids]
            anchor_position = anchor_pose[:, :3]
            anchor_rotation = quaternion_to_matrix(anchor_pose[:, 3:])

            delta_position = chunk[:, :, offset : offset + 3]
            delta_euler = chunk[:, :, offset + 3 : offset + 6]
            delta_rotation = euler_angles_to_matrix(delta_euler, "XYZ")
            target_rotation = torch.matmul(
                delta_rotation,
                anchor_rotation[:, None, :, :],
            )

            absolute_targets[:, :, offset : offset + 3] = (
                anchor_position[:, None, :] + delta_position
            )
            absolute_targets[:, :, offset + 3 : offset + 6] = matrix_to_euler_angles(
                target_rotation,
                "XYZ",
            )

        if squeeze_batch:
            absolute_targets = absolute_targets[0]
        if return_numpy:
            return absolute_targets.detach().cpu().numpy()
        return absolute_targets

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

    def set_manual_gripper_drive_override(self, arm_index: int, gripper_val: float | None) -> None:
        if gripper_val is None:
            self._manual_gripper_drive_override.pop(int(arm_index), None)
            return
        self._manual_gripper_drive_override[int(arm_index)] = float(gripper_val)

    def clear_manual_gripper_drive_overrides(self) -> None:
        self._manual_gripper_drive_override.clear()

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

    def get_arm_link_by_name(self, arm_index: int, link_name: str):
        for link in self.get_arm_robot(arm_index).get_links():
            current_name = link.get_name() if hasattr(link, "get_name") else str(link.name)
            if current_name == str(link_name):
                return link
        raise KeyError(f"Could not find link `{link_name}` for arm {arm_index}.")

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
        current_qpos = robot.get_qpos()
        current_qvel = robot.get_qvel()
        if current_qpos.numel() == 0:
            return
        qpos = torch.zeros(
            (len(env_idx), current_qpos.shape[1]),
            device=self.device,
            dtype=current_qpos.dtype,
        )
        qvel = torch.zeros(
            (len(env_idx), current_qvel.shape[1]),
            device=self.device,
            dtype=current_qvel.dtype,
        )

        home = self.embodiment_spec.homestate(arm_index)
        if len(home) > 0:
            home = torch.tensor(home, device=self.device, dtype=qpos.dtype)
            dof = min(home.shape[0], qpos.shape[1])
            qpos[:, :dof] = home[:dof]

        robot.set_qpos(qpos)
        robot.set_qvel(qvel)
        arm_joints = self.left_arm_joints if arm_index == 0 else self.right_arm_joints
        arm_dof = min(len(arm_joints), qpos.shape[1])
        if arm_dof > 0:
            robot.set_joint_drive_targets(qpos[:, :arm_dof], joints=arm_joints[:arm_dof])
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
        drive_gripper_val = gripper_val
        override = self._manual_gripper_drive_override.get(int(arm_index))
        if override is not None:
            drive_gripper_val = torch.minimum(
                drive_gripper_val,
                torch.full_like(drive_gripper_val, float(override)),
            )
        closed_drive_value, closed_drive_threshold = self.embodiment_spec.gripper_closed_drive_params(arm_index)
        if closed_drive_value is not None:
            closed_drive_tensor = torch.full_like(drive_gripper_val, float(closed_drive_value))
            drive_gripper_val = torch.where(
                gripper_val <= float(closed_drive_threshold),
                torch.minimum(drive_gripper_val, closed_drive_tensor),
                drive_gripper_val,
            )
        gripper_scale = self.embodiment_spec.gripper_scale(arm_index)
        real_gripper_val = gripper_scale[0] + drive_gripper_val * (gripper_scale[1] - gripper_scale[0])

        joints = self.left_gripper if arm_index == 0 else self.right_gripper
        self._set_gripper_state_tensor(gripper_val, arm_index=arm_index, env_idx=env_idx)

        joint_objs = [joint for joint, _, _ in joints]
        drive_targets = torch.stack(
            [real_gripper_val * sign + offset for _, sign, offset in joints],
            dim=1,
        )
        if self.clamp_gripper_drive_targets:
            clamped_targets = []
            for column, (joint, _, _) in enumerate(joints):
                limits = joint.get_limits().to(
                    device=drive_targets.device,
                    dtype=drive_targets.dtype,
                )
                if limits.ndim == 1:
                    limits = limits.unsqueeze(0)
                if env_idx is not None and limits.shape[0] == self.num_envs:
                    limits = limits[env_idx]
                if limits.shape[0] == 1 and target_envs > 1:
                    limits = limits.expand(target_envs, -1)
                clamped_targets.append(
                    torch.clamp(
                        drive_targets[:, column],
                        min=limits[:, 0],
                        max=limits[:, 1],
                    )
                )
            drive_targets = torch.stack(clamped_targets, dim=1)
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
            if self.left_delta_pose_controller is not None:
                self.left_delta_pose_controller.reset()
                self.right_delta_pose_controller.reset()
                self.delta_controller_failure_count = 0
                self.last_delta_control_debug = {}

    def _arm_joint_limits(self, robot, arm_joints, target_qpos: torch.Tensor):
        qlimits = robot.get_qlimits().to(device=target_qpos.device, dtype=target_qpos.dtype)
        if qlimits.ndim == 2:
            qlimits = qlimits.unsqueeze(0)
        arm_dof = len(arm_joints)
        if qlimits.shape[1] < arm_dof:
            return None
        limits = qlimits[:, :arm_dof, :]
        if limits.shape[0] == 1 and target_qpos.shape[0] > 1:
            limits = limits.expand(target_qpos.shape[0], -1, -1)
        return limits[:, :, 0], limits[:, :, 1]

    def _apply_arm_action(self, robot, arm_joints, target_qpos: torch.Tensor):
        joint_limits = self._arm_joint_limits(robot, arm_joints, target_qpos)
        if joint_limits is None:
            target_arm_qpos = target_qpos
        else:
            lower, upper = joint_limits
            target_arm_qpos = torch.minimum(torch.maximum(target_qpos, lower), upper)
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
            if parsed_action.shape != (self.num_envs, self.ACTION_DIM):
                raise ValueError(
                    f"Expected action shape {(self.num_envs, self.ACTION_DIM)}, got {tuple(parsed_action.shape)}"
                )
            if not torch.isfinite(parsed_action).all():
                raise ValueError("Bimanual action contains non-finite values.")

            if self._control_mode == self.DELTA_POSE_CONTROL_MODE:
                low = torch.as_tensor(
                    self.single_action_space.low,
                    device=self.device,
                    dtype=parsed_action.dtype,
                )
                high = torch.as_tensor(
                    self.single_action_space.high,
                    device=self.device,
                    dtype=parsed_action.dtype,
                )
                if torch.any(parsed_action < low) or torch.any(parsed_action > high):
                    raise ValueError(
                        "Absolute EEF target exceeds action-space bounds; targets are not clipped implicitly."
                    )
                left_action = parsed_action[:, 0:6]
                right_action = parsed_action[:, 7:13]
                self.last_delta_control_debug = {
                    "action": parsed_action.detach().clone(),
                    "left_start_pose": self.left_delta_pose_controller.ee_pose_at_base.raw_pose.detach().clone(),
                    "right_start_pose": self.right_delta_pose_controller.ee_pose_at_base.raw_pose.detach().clone(),
                }
                try:
                    self.left_delta_pose_controller.set_action(left_action)
                    self.right_delta_pose_controller.set_action(right_action)
                except Exception:
                    self.delta_controller_failure_count += 1
                    raise
                target_qpos = (
                    self.left_delta_pose_controller._target_qpos,
                    self.right_delta_pose_controller._target_qpos,
                )
                if any(value is None or not torch.isfinite(value).all() for value in target_qpos):
                    self.delta_controller_failure_count += 1
                    raise RuntimeError("Absolute EEF controller produced an invalid joint target.")
                left_target_pose = self.left_delta_pose_controller._target_pose.raw_pose
                right_target_pose = self.right_delta_pose_controller._target_pose.raw_pose
                self.last_delta_control_debug.update(
                    {
                        "left_target_pose": left_target_pose.detach().clone(),
                        "right_target_pose": right_target_pose.detach().clone(),
                        "left_target_qpos": target_qpos[0].detach().clone(),
                        "right_target_qpos": target_qpos[1].detach().clone(),
                    }
                )
                self._set_gripper(parsed_action[:, 6], arm_index=0)
                self._set_gripper(parsed_action[:, 13], arm_index=1)
            else:
                self._apply_arm_action(self.left_robot, self.left_arm_joints, parsed_action[:, 0:6])
                self._set_gripper(parsed_action[:, 6], arm_index=0)
                self._apply_arm_action(self.right_robot, self.right_arm_joints, parsed_action[:, 7:13])
                self._set_gripper(parsed_action[:, 13], arm_index=1)
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()

        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            if self.left_delta_pose_controller is not None:
                self.left_delta_pose_controller.before_simulation_step()
                self.right_delta_pose_controller.before_simulation_step()
            self.scene.step()
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
