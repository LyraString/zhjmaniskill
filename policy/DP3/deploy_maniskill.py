"""RoboTwin DP3 adapter for the ManiSkill VR runtime."""
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

DP3_ROOT = Path(__file__).resolve().parent / "3D-Diffusion-Policy"
if str(DP3_ROOT) not in sys.path:
    sys.path.insert(0, str(DP3_ROOT))
from diffusion_policy_3d.env_runner.robot_runner import RobotRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy_3d.common.maniskill_pointcloud import encode_raw_observation


def encode_obs(observation):
    """Accept only a cloud already processed with this checkpoint's raw settings."""
    return {key: np.asarray(observation[key], dtype=np.float32)
            for key in ("point_cloud", "agent_pos")}


def encode_env_obs(env, observation, model):
    return encode_raw_observation(env.last_raw_obs, model.preprocessing,
                                  device=model.policy.device)


class ManiSkillDP3:
    def __init__(self, policy, n_obs_steps, n_action_steps, preprocessing):
        self.policy = policy
        self.preprocessing = preprocessing
        self.env_runner = RobotRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    @property
    def obs_cache(self):
        return self.env_runner.obs

    def update_obs(self, observation):
        self.env_runner.update_obs(observation)

    def get_action(self):
        return self.env_runner.get_action(self.policy)


def get_model(usr_args):
    with open(usr_args["ckpt_path"], "rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    steps = int(usr_args.get("n_action_steps", cfg.n_action_steps))
    if not 1 <= steps <= int(cfg.horizon) - int(cfg.n_obs_steps) + 1:
        raise ValueError("n_action_steps exceeds the executable prediction horizon.")
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key])
    policy.n_action_steps = steps
    policy.to(usr_args.get("device", "cuda:0")).eval()
    return ManiSkillDP3(policy, int(cfg.n_obs_steps), steps,
                       OmegaConf.to_container(cfg.task.preprocessing, resolve=True))


def reset_model(model):
    model.env_runner.reset_obs()
