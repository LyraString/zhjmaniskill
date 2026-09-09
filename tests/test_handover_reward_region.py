import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sim.maniskill.task_profiles import get_task_profile
from sim.maniskill.tasks.handover_block_env import BLOCK_MIDDLE_POSE, HandoverBlockEnv
from sim.maniskill.tasks.dual_ur5_wsg_env import DualUR5WSGEnv
from sim.maniskill.tasks.handover_block_ur5_v2_env import HandoverBlockUR5V2Env


class HandoverRewardRegionTest(unittest.TestCase):
    def test_profile_parameters_reach_environment(self):
        cfg = get_task_profile('handover_block_ur5_v2_vr_robotwin_clean')['env_kwargs']
        with patch.object(DualUR5WSGEnv, '__init__', return_value=None):
            env = HandoverBlockUR5V2Env(**cfg)
            legacy = HandoverBlockUR5V2Env()
        self.assertEqual(env.handover_reward_center, cfg['handover_reward_center'])
        self.assertEqual(env.handover_reward_plateau_radius, .16)
        self.assertEqual(legacy.handover_reward_center, BLOCK_MIDDLE_POSE[:3])
        self.assertEqual(legacy.handover_reward_plateau_radius, 0)

    def evaluate(self, offsets, center, radius):
        pos = torch.tensor(center).repeat(len(offsets), 1)
        pos[:, 0] += torch.tensor(offsets)
        n = len(pos)
        env = SimpleNamespace(
            device='cpu', num_envs=n, handover_reward_center=center,
            handover_reward_plateau_radius=radius,
            box=SimpleNamespace(pose=SimpleNamespace(p=pos)),
            target_box=SimpleNamespace(pose=SimpleNamespace(p=pos + 1)),
            _get_reward_stage=lambda: torch.ones(n, dtype=torch.long),
            get_arm_end_effector_link=lambda arm_index: SimpleNamespace(pose=SimpleNamespace(p=pos + .3)),
            _get_gripper_state_tensor=lambda arm_index: torch.zeros(n, 1),
        )
        info = {key: torch.zeros(n, dtype=torch.bool)
                for key in ['robots_ready', 'right_gripper_open', 'box_static', 'success']}
        reward = HandoverBlockEnv.compute_dense_reward(env, None, None, info)
        return reward, info

    def test_vr_region_distance_and_stage_gate(self):
        cfg = get_task_profile('handover_block_ur5_v2_vr_robotwin_clean')['env_kwargs']
        self.assertEqual(cfg['handover_reward_center'], [-.023, -.092, 1.011])
        self.assertEqual(cfg['handover_reward_plateau_radius'], .16)
        reward, info = self.evaluate([0, .08, .15, .17, .26], cfg['handover_reward_center'], .16)
        torch.testing.assert_close(info['reward_box_to_handover_region'], torch.tensor([0, 0, 0, .01, .10]))
        torch.testing.assert_close(info['reward_stage'], torch.tensor([2., 2., 2., 1., 1.]))
        torch.testing.assert_close(reward[:3], reward[0].expand(3))
        self.assertGreater(float(reward[3]), float(reward[4]))

    def test_default_retains_point_reward(self):
        offsets = [.2, .3]
        reward, info = self.evaluate(offsets, BLOCK_MIDDLE_POSE[:3], 0)
        expected = torch.tensor(offsets)
        torch.testing.assert_close(info['reward_box_to_handover_region'], expected)
        torch.testing.assert_close(reward, 1.5 + 2 * torch.exp(-6 * expected))

    def test_actual_fpo_reward_uses_plateau_and_legacy_fallback(self):
        path = Path(__file__).resolve().parents[1] / 'policy/fm_diffusion/diffusion_policy/workspace/fpo_robotworkspace.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == 'compute_handover_fpo_reward')
        # Run the production nested function without constructing a training workspace.
        function.body = [ast.Global(names=n.names) if isinstance(n, ast.Nonlocal) else n for n in function.body]
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        ns = {'torch': torch, 'num_envs': 1, 'device': 'cpu',
              'info_tensor': lambda info, key, dtype=torch.float32: torch.as_tensor(info[key], dtype=dtype)}
        exec(compile(module, str(path), 'exec'), ns)
        ns.update(fpo_reward_stage=torch.ones(1), fpo_reward_potential=torch.ones(1),
                  fpo_reward_has_potential=torch.ones(1, dtype=torch.bool),
                  fpo_reward_success_awarded=torch.zeros(1, dtype=torch.bool))
        info = {'reward_stage': [1.], 'reward_left_to_box': [.1], 'reward_right_to_box': [.2],
                'reward_box_to_handover': [.08], 'reward_box_to_handover_region': [0.],
                'reward_box_to_target': [.3], 'success': [False]}
        reward, fallback = ns['compute_handover_fpo_reward'](torch.zeros(1), info)
        torch.testing.assert_close(reward, torch.tensor([-.02]))
        self.assertFalse(bool(fallback.any()))
        del info['reward_box_to_handover_region']
        reward, _ = ns['compute_handover_fpo_reward'](torch.zeros(1), info)
        torch.testing.assert_close(reward, 3 * (torch.exp(torch.tensor([-.48])) - 1) - .02)


if __name__ == '__main__':
    unittest.main()
