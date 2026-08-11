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
    with open(ckpt_file, "rb") as file:
        payload = torch.load(file, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    n_action_steps = int(usr_args.get("n_action_steps", cfg.n_action_steps))
    if n_action_steps != 1:
        raise ValueError(
            "ResFiT v0 only supports n_action_steps=1; "
            f"got {n_action_steps}."
        )
    if "action_low" not in usr_args or "action_high" not in usr_args:
        raise ValueError("ResFiT deployment requires environment action bounds.")

    return DP(
        ckpt_file,
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=n_action_steps,
        action_low=usr_args["action_low"],
        action_high=usr_args["action_high"],
    )


def reset_model(model):
    model.reset_obs()
