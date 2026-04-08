"""Task factory for the legacy RoboTwin runtime."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from core.interfaces.env_backend import EnvBackend
from core.interfaces.task_factory import TaskFactory
from core.interfaces.vector_env_backend import VectorEnvBackend
from legacy.robotwin.runtime_env import RoboTwinEnvBackend


class RoboTwinTaskFactory(TaskFactory):
    """Factory that wraps legacy RoboTwin tasks behind the shared interfaces."""

    def __init__(self) -> None:
        self.robotwin_root = Path(__file__).resolve().parents[2]
        if str(self.robotwin_root) not in sys.path:
            sys.path.insert(0, str(self.robotwin_root))

    def create_env(self, task_name: str, **kwargs: Any) -> EnvBackend:
        setting = str(kwargs.pop("setting"))
        head_camera_type = str(kwargs.pop("head_camera_type"))
        base_seed = int(kwargs.pop("base_seed", 0))
        action_type = str(kwargs.pop("action_type", "qpos"))

        task = self._instantiate_task(task_name)
        task_args = self._build_task_args(
            task_name=task_name,
            setting=setting,
            head_camera_type=head_camera_type,
            action_type=action_type,
        )

        return RoboTwinEnvBackend(
            task=task,
            task_name=task_name,
            task_args=task_args,
            base_seed=base_seed,
            action_type=action_type,
        )

    def create_vector_env(
        self,
        task_name: str,
        num_envs: int,
        **kwargs: Any,
    ) -> VectorEnvBackend:
        raise NotImplementedError(
            "Legacy RoboTwin vector environment adapter is not part of stage 3."
        )

    def _instantiate_task(self, task_name: str) -> Any:
        envs_module = importlib.import_module(f"envs.{task_name}")
        env_class = getattr(envs_module, task_name)
        return env_class()

    def _get_embodiment_config(self, robot_file: Path) -> Dict[str, Any]:
        with open(robot_file / "config.yml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_task_args(
        self,
        *,
        task_name: str,
        setting: str,
        head_camera_type: str,
        action_type: str,
    ) -> Dict[str, Any]:
        from envs import CONFIGS_PATH

        task_cfg_path = self.robotwin_root / "task_config" / f"{setting}.yml"
        with open(task_cfg_path, "r", encoding="utf-8") as f:
            args = yaml.safe_load(f)

        args["task_name"] = task_name
        args["eval_mode"] = True
        args["save_data"] = False
        args["render_freq"] = 0
        args["collect_data"] = False
        args.setdefault("need_curobo", action_type == "ee")
        args.setdefault("need_topp", action_type == "qpos")

        renderer_cfg = args.setdefault("renderer", {})
        renderer_cfg.setdefault("device", "cuda:0")
        renderer_cfg.setdefault("camera_shader_dir", "rt")
        renderer_cfg.setdefault("ray_tracing_samples_per_pixel", 1)
        renderer_cfg.setdefault("ray_tracing_path_depth", 1)
        renderer_cfg.setdefault("ray_tracing_denoiser", None)
        renderer_cfg.setdefault("enable_observer_camera", False)
        renderer_cfg.setdefault("enable_world_cameras", False)

        args["camera"]["head_camera_type"] = head_camera_type

        embodiment_type = args.get("embodiment")
        embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
        with open(embodiment_config_path, "r", encoding="utf-8") as f:
            emb_types = yaml.safe_load(f)

        def get_embodiment_file(emb_name: str) -> Path:
            robot_file = emb_types[emb_name]["file_path"]
            if robot_file is None:
                raise ValueError("No embodiment file")
            robot_path = Path(robot_file)
            if not robot_path.is_absolute():
                robot_path = (self.robotwin_root / robot_path).resolve()
            return robot_path

        if len(embodiment_type) == 1:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["dual_arm_embodied"] = True
        elif len(embodiment_type) == 3:
            args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
            args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
            args["embodiment_dis"] = embodiment_type[2]
            args["dual_arm_embodied"] = False
        else:
            raise ValueError("embodiment items should be either 1 or 3")

        args["left_embodiment_config"] = self._get_embodiment_config(args["left_robot_file"])
        args["right_embodiment_config"] = self._get_embodiment_config(args["right_robot_file"])
        return args
