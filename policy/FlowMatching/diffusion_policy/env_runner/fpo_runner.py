import torch
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

import sys


class FPORunner:

    def __init__(
        self,
        task_name,
        setting,
        head_camera_type,
        base_seed=0,
        action_type="qpos",
    ):
        self.task_name = task_name
        self.setting = setting
        self.head_camera_type = head_camera_type
        self.base_seed = int(base_seed)
        self.action_type = action_type

        self.ep_id = 0
        self._task = None
        self._args = None

        self._build_once()

    def _class_decorator(self, task_name):
        envs_module = importlib.import_module(f"envs.{task_name}")
        env_class = getattr(envs_module, task_name)
        
        return env_class()

    def _get_embodiment_config(self, robot_file):
        with open(os.path.join(robot_file, "config.yml"), 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    # 初始化并构建仿真任务
    def _build_once(self):
        robotwin_root = Path(__file__).resolve().parents[4]
        if str(robotwin_root) not in sys.path:
            sys.path.insert(0, str(robotwin_root))

        # 记录当前工作目录并更改
        prev_cwd = os.getcwd()
        try:
            os.chdir(str(robotwin_root))

            from envs import CONFIGS_PATH

            # 读取task_config
            task_cfg_path = os.path.join(str(robotwin_root), "task_config", f"{self.setting}.yml")
            with open(task_cfg_path, "r", encoding="utf-8") as f:
                args = yaml.safe_load(f)

            self._args = args
        finally:
            os.chdir(prev_cwd)

        # 覆盖参数
        args["task_name"] = self.task_name
        args["eval_mode"] = True
        args["save_data"] = False
        args["render_freq"] = 0
        args["collect_data"] = False
        args.setdefault("need_curobo", self.action_type == "ee")
        args.setdefault("need_topp", self.action_type == "qpos")

        # 在线 FPO 默认使用低显存渲染配置；如 task_config 显式提供则以 task_config 为准。
        renderer_cfg = args.setdefault("renderer", {})
        renderer_cfg.setdefault("device", "cuda:0")
        renderer_cfg.setdefault("camera_shader_dir", "rt")
        renderer_cfg.setdefault("ray_tracing_samples_per_pixel", 1)
        renderer_cfg.setdefault("ray_tracing_path_depth", 1)
        renderer_cfg.setdefault("ray_tracing_denoiser", None)
        renderer_cfg.setdefault("enable_observer_camera", False)
        renderer_cfg.setdefault("enable_world_cameras", False)

        # 覆盖相机类型
        args["camera"]["head_camera_type"] = self.head_camera_type

        # 补 embodiment 字段
        embodiment_type = args.get("embodiment")
        embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
        with open(embodiment_config_path, 'r', encoding='utf-8') as f:
            emb_types = yaml.safe_load(f)

        def get_embodiment_file(emb_name):
            robot_file = emb_types[emb_name]["file_path"]
            if robot_file is None:
                raise ValueError("No embodiment file")
            
            robot_file = Path(robot_file)
            if not robot_file.is_absolute():
                robot_file = (robotwin_root / robot_file).resolve()

            return robot_file
        
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

        # 实例化任务
        self._task = self._class_decorator(self.task_name)
        self._args = args

    # 对齐策略的输入观测格式
    def _convert_obs_for_policy(self, raw_obs):
        head_cam = raw_obs["observation"]["head_camera"]["rgb"]
        head_cam = np.moveaxis(head_cam, -1, 0).astype(np.float32) / 255.0

        agent_pos = raw_obs["joint_action"]["vector"].astype(np.float32)

        return{
            "head_cam": head_cam,
            "agent_pos": agent_pos,
        }
    
    # 重置环境,开始一个新 episode
    def reset(self):
        MAX_SEED = 2**32 - 1
        seed = ((100000 * (1 + self.base_seed)) + self.ep_id) % MAX_SEED
        self._task.setup_demo(now_ep_num=self.ep_id, seed=seed, is_test=True, **self._args)
        raw_obs = self._task.get_obs()
        obs = self._convert_obs_for_policy(raw_obs)
        self.ep_id += 1
        info = {"seed": seed}

        return obs, info
    
    def step(self, action):
        self._task.take_action(action, action_type=self.action_type)

        raw_obs = self._task.get_obs()
        obs = self._convert_obs_for_policy(raw_obs)

        terminated = bool(self._task.eval_success)
        truncated = bool((self._task.take_action_cnt >= self._task.step_lim) and (not terminated))

        # 先设置稀疏奖励，后续可以根据需要修改
        reward = 1.0 if terminated else 0.0

        info = {
            "success": terminated,
            "step": int(self._task.take_action_cnt),
            "step_lim": int(self._task.step_lim),
        }

        return obs, reward, terminated, truncated, info
    
    # 关闭环境
    def close(self):
        if self._task is not None:
            self._task.close_env(clear_cache=False)

    # def stack_last_n_obs(self, all_obs, n_steps):
    #     assert len(all_obs) > 0
    #     all_obs = list(all_obs)
    #     if isinstance(all_obs[0], np.ndarray):
    #         result = np.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    #         start_idx = -min(n_steps, len(all_obs))
    #         result[start_idx:] = np.array(all_obs[start_idx:])
    #         if n_steps > len(all_obs):
    #             # pad
    #             result[:start_idx] = result[start_idx]
    #     elif isinstance(all_obs[0], torch.Tensor):
    #         result = torch.zeros((n_steps, ) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    #         start_idx = -min(n_steps, len(all_obs))
    #         result[start_idx:] = torch.stack(all_obs[start_idx:])
    #         if n_steps > len(all_obs):
    #             # pad
    #             result[:start_idx] = result[start_idx]
    #     else:
    #         raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
    #     return result

    # def reset_obs(self):
    #     self.obs.clear()

    # def update_obs(self, current_obs):
    #     self.obs.append(current_obs)

    # def get_n_steps_obs(self):
    #     assert len(self.obs) > 0, "no observation is recorded, please update obs first"

    #     result = dict()
    #     for key in self.obs[0].keys():
    #         result[key] = self.stack_last_n_obs([obs[key] for obs in self.obs], self.n_obs_steps)

    #     return result

    # def get_action(self, policy: BaseImagePolicy, observaton=None):
    #     device, dtype = policy.device, policy.dtype
    #     if observaton is not None:
    #         self.obs.append(observaton)  # update
    #     obs = self.get_n_steps_obs()

    #     # create obs dict
    #     np_obs_dict = dict(obs)
    #     # device transfer
    #     obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
    #     # run policy
    #     with torch.no_grad():
    #         obs_dict_input = {}  # flush unused keys
    #         obs_dict_input["head_cam"] = obs_dict["head_cam"].unsqueeze(0)
    #         # obs_dict_input['front_cam'] = obs_dict['front_cam'].unsqueeze(0)
    #         obs_dict_input["left_cam"] = obs_dict["left_cam"].unsqueeze(0)
    #         obs_dict_input["right_cam"] = obs_dict["right_cam"].unsqueeze(0)
    #         obs_dict_input["agent_pos"] = obs_dict["agent_pos"].unsqueeze(0)

    #         action_dict = policy.predict_action(obs_dict_input)

    #     # device_transfer
    #     np_action_dict = dict_apply(action_dict, lambda x: x.detach().to("cpu").numpy())
    #     action = np_action_dict["action"].squeeze(0)[:self.n_action_steps]
    #     return action
