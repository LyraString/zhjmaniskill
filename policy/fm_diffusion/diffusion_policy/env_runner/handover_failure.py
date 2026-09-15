import math

import torch


class HandoverFailureDetector:
    """Training-only failure rules, with independent state for each environment."""

    reasons = ("left_empty_grasp", "right_empty_grasp", "box_toppled")

    def __init__(self, num_envs, device, cfg):
        self.confirm_steps = int(cfg.confirm_steps)
        self.closed_threshold = float(cfg.closed_threshold)
        self.open_threshold = float(cfg.open_threshold)
        self.tilt_cost_limit = 1 - math.cos(math.radians(float(cfg.box_tilt_deg)))
        self.armed = torch.zeros((num_envs, 2), dtype=torch.bool, device=device)
        self.counts = torch.zeros((num_envs, 3), dtype=torch.long, device=device)

    def reset(self, env_ids):
        self.armed[env_ids] = False
        self.counts[env_ids] = 0

    def step(self, actions, info):
        gripper = torch.stack((info["reward_left_gripper_state"],
                               info["reward_right_gripper_state"]), dim=-1)
        grasped = torch.stack((info["reward_left_grasped"],
                               info["reward_right_grasped"]), dim=-1).bool()
        command = actions[:, [6, 13]]
        # Initial closed grippers are not failed attempts. Arm after opening or grasping.
        self.armed |= (gripper > self.open_threshold) | grasped
        stage = info["reward_stage"]
        active = torch.stack((stage < 3, stage < 4), dim=-1)
        empty = (self.armed & active & ~grasped
                 & (gripper < self.closed_threshold)
                 & (command < self.closed_threshold))
        toppled = info["reward_box_tilt_cost"] > self.tilt_cost_limit
        candidates = torch.cat((empty, toppled[:, None]), dim=-1)
        candidates &= ~info["success"].bool()[:, None]
        self.counts = torch.where(candidates, self.counts + 1, 0)
        return self.counts >= self.confirm_steps
