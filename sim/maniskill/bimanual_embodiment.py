from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import sapien
import yaml


def load_embodiment_config(robot_dir: Path | str) -> dict:
    robot_dir = Path(robot_dir)
    with open(robot_dir / "config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_arm_value(values: Sequence, arm_index: int):
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return values
    if len(values) == 0:
        return values
    index = arm_index if len(values) > arm_index else 0
    return values[index]


def build_collision_only_urdf(urdf_path: Path) -> Path:
    collision_only_path = urdf_path.with_name(f"{urdf_path.stem}_collision_only.urdf")
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            link.remove(visual)
    tree.write(collision_only_path, encoding="utf-8", xml_declaration=True)
    return collision_only_path


@dataclass
class BimanualEmbodimentSpec:
    left_robot_dir: Path
    right_robot_dir: Path
    left_config: dict
    right_config: dict
    arms_distance: float

    @classmethod
    def from_robot_dirs(
        cls,
        *,
        left_robot_dir: Path | str,
        right_robot_dir: Path | str,
        arms_distance: float,
    ) -> "BimanualEmbodimentSpec":
        left_robot_dir = Path(left_robot_dir)
        right_robot_dir = Path(right_robot_dir)
        return cls(
            left_robot_dir=left_robot_dir,
            right_robot_dir=right_robot_dir,
            left_config=load_embodiment_config(left_robot_dir),
            right_config=load_embodiment_config(right_robot_dir),
            arms_distance=float(arms_distance),
        )

    def _robot_dir(self, arm_index: int) -> Path:
        return self.left_robot_dir if arm_index == 0 else self.right_robot_dir

    def _config(self, arm_index: int) -> dict:
        return self.left_config if arm_index == 0 else self.right_config

    def urdf_path(self, arm_index: int) -> Path:
        config = self._config(arm_index)
        return self._robot_dir(arm_index) / config["urdf_path"]

    def package_dir(self, arm_index: int) -> str:
        return str(self._robot_dir(arm_index))

    def robot_pose(self, arm_index: int) -> sapien.Pose:
        config = self._config(arm_index)
        pose_values = _select_arm_value(config.get("robot_pose", []), arm_index)
        x_offset = -self.arms_distance / 2 if arm_index == 0 else self.arms_distance / 2
        position = [
            float(pose_values[0] + x_offset),
            float(pose_values[1]),
            float(pose_values[2]),
        ]
        return sapien.Pose(p=position, q=pose_values[3:])

    def homestate(self, arm_index: int):
        config = self._config(arm_index)
        return _select_arm_value(config.get("homestate", []), arm_index)

    def arm_joint_names(self, arm_index: int):
        config = self._config(arm_index)
        return list(_select_arm_value(config.get("arm_joints_name", []), arm_index))

    def gripper_name(self, arm_index: int):
        config = self._config(arm_index)
        return _select_arm_value(config.get("gripper_name", []), arm_index)

    def gripper_scale(self, arm_index: int):
        config = self._config(arm_index)
        return list(config.get("gripper_scale", [0.0, 1.0]))

    def gripper_bias(self, arm_index: int):
        config = self._config(arm_index)
        return float(config.get("gripper_bias", 0.0))

    def joint_drive_params(self, arm_index: int):
        config = self._config(arm_index)
        return (
            float(config.get("joint_stiffness", 1000.0)),
            float(config.get("joint_damping", 200.0)),
        )

    def gripper_drive_params(self, arm_index: int):
        config = self._config(arm_index)
        return (
            float(config.get("gripper_stiffness", config.get("gripper_stiffnes", 1000.0))),
            float(config.get("gripper_damping", 200.0)),
        )

    def gripper_closed_drive_params(self, arm_index: int):
        config = self._config(arm_index)
        closed_drive_value = config.get("gripper_closed_drive_value")
        if closed_drive_value is None:
            return None, None
        return (
            float(closed_drive_value),
            float(config.get("gripper_closed_drive_threshold", 0.2)),
        )

    def build_robot(self, scene, *, arm_index: int, name: str):
        loader = scene.create_urdf_loader()
        loader.name = name
        loader.fix_root_link = True
        material = self._config(arm_index).get("material")
        if isinstance(material, dict):
            loader.set_material(
                float(material.get("static_friction", 0.3)),
                float(material.get("dynamic_friction", 0.3)),
                float(material.get("restitution", 0.0)),
            )

        urdf_path = self.urdf_path(arm_index)
        if not scene.can_render():
            urdf_path = build_collision_only_urdf(urdf_path)

        parsed = loader.parse(str(urdf_path), package_dir=self.package_dir(arm_index))
        builder = parsed["articulation_builders"][0]
        builder.initial_pose = self.robot_pose(arm_index)
        return builder.build()
