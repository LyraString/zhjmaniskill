from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.structs.types import SimConfig

from sim.maniskill.bimanual_embodiment import BimanualEmbodimentSpec


class BaseBimanualEnv(BaseEnv):
    """Minimal ManiSkill base env that validates a RoboTwin-style bimanual embodiment."""

    SUPPORTED_ROBOTS = ["none"]
    SUPPORTED_OBS_MODES = ("state_dict", "state", "none")
    SUPPORTED_REWARD_MODES = ("none", "dense", "normalized_dense", "sparse")

    def __init__(
        self,
        *args,
        embodiment_spec: BimanualEmbodimentSpec,
        robot_uids="none",
        **kwargs,
    ):
        self.embodiment_spec = embodiment_spec
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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

    def _initialize_robot_state(self, robot, env_idx: torch.Tensor, arm_index: int):
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self._initialize_robot_state(self.left_robot, env_idx, arm_index=0)
            self._initialize_robot_state(self.right_robot, env_idx, arm_index=1)

    def _get_obs_agent(self):
        return dict()

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
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["robots_ready"].to(torch.float32)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
