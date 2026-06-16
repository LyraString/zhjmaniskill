import dill
import numpy as np
import torch

from .dp_model import DP


def encode_obs(observation):
    agent_pos = np.asarray(observation["agent_pos"], dtype=np.float32)
    encoded = {
        "agent_pos": agent_pos,
    }
    for key in (
        "point_cloud",
        "head_point_cloud",
        "left_wrist_point_cloud",
        "right_wrist_point_cloud",
    ):
        if key in observation:
            encoded[key] = np.asarray(observation[key], dtype=np.float32)
    return encoded


def get_model(usr_args):
    ckpt_file = usr_args["ckpt_path"]
    payload = torch.load(open(ckpt_file, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    return DP(
        ckpt_file,
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
    )


def reset_model(model):
    model.reset_obs()
