import dill
import numpy as np
import torch

from .dp_model import DP


def encode_obs(observation):
    point_cloud = np.asarray(observation["point_cloud"], dtype=np.float32)
    agent_pos = np.asarray(observation["agent_pos"], dtype=np.float32)
    return {
        "point_cloud": point_cloud,
        "agent_pos": agent_pos,
    }


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
