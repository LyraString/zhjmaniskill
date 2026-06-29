from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from mani_skill.utils.registration import register_env

from sim.maniskill.bimanual_embodiment import BimanualEmbodimentSpec
from sim.maniskill.tasks.base_bimanual_env import BaseBimanualEnv


REPO_ROOT = Path(__file__).resolve().parents[3]
UR5_WSG_DIR = REPO_ROOT / "assets" / "embodiments" / "ur5-wsg"
DEFAULT_ARMS_DISTANCE = 0.8


@register_env("DualUR5WSG-v1", max_episode_steps=100)
class DualUR5WSGEnv(BaseBimanualEnv):
    """Thin validation env that instantiates the generic bimanual embodiment layer with UR5-WSG."""

    def __init__(
        self,
        *args,
        robot_uids="none",
        arms_distance: float = DEFAULT_ARMS_DISTANCE,
        robot_pose_override=None,
        homestate_override=None,
        static_camera_list_override=None,
        **kwargs,
    ):
        embodiment_spec = BimanualEmbodimentSpec.from_robot_dirs(
            left_robot_dir=UR5_WSG_DIR,
            right_robot_dir=UR5_WSG_DIR,
            arms_distance=arms_distance,
        )
        for config in (embodiment_spec.left_config, embodiment_spec.right_config):
            if robot_pose_override is not None:
                config["robot_pose"] = deepcopy(robot_pose_override)
            if homestate_override is not None:
                config["homestate"] = deepcopy(homestate_override)
            if static_camera_list_override is not None:
                config["static_camera_list"] = deepcopy(static_camera_list_override)
        super().__init__(
            *args,
            embodiment_spec=embodiment_spec,
            robot_uids=robot_uids,
            **kwargs,
        )
