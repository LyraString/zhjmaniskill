import numpy as np
import torch
import hydra
import dill
import sys, os

# 确保导入 FlowMatching 目录下的模块
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
# 将 FlowMatching 目录添加到路径最前面，确保优先级
sys.path.insert(0, parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps):
        self.policy = self.get_policy(ckpt_file, None, "cuda:0")
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

        # 为兼容 eval_policy 的接口，提供 obs_cache 代理到 runner.obs
        # eval_policy.py 会检查 len(model.obs_cache)
        self.obs_cache = self.runner.obs

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def reset_obs(self):
        self.runner.reset_obs()
        # 同步清空兼容字段
        self.obs_cache = self.runner.obs

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
