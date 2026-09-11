from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'policy/fm_diffusion'))
from diffusion_policy.workspace.fpo_robotworkspace import (
    RobotWorkspace, ONLINE_ROLLOUT_LOG_KEYS, ONLINE_ROLLOUT_DIAG_KEYS,
)


class HandoverFPOLoggingTest(unittest.TestCase):
    def test_reward_components_and_completed_episodes_in_real_collector(self):
        workspace = RobotWorkspace.__new__(RobotWorkspace)
        workspace.cfg = OmegaConf.create({'training': {
            'timesteps_per_batch': 10, 'max_timesteps_per_episode': 3,
            'num_fpo_samples': 1, 'show_rollout_tqdm': False,
            'gamma': .99, 'lmbda': .99,
        }})
        workspace.model = SimpleNamespace(
            n_obs_steps=1, n_action_steps=1,
            normalizer={'action': SimpleNamespace(normalize=lambda x: x)},
            sample_action_with_info=lambda *a, **kw: {
                'action': torch.zeros(2, 1, 14), 'global_cond': torch.zeros(2, 1),
                'loss_eps': torch.zeros(2, 1, 1, 14), 'loss_t': torch.zeros(2, 1),
                'naction_pred': torch.zeros(2, 1, 14),
                'cond_mask': torch.zeros(2, 1, 14, dtype=torch.bool),
            },
            compute_cfm_loss_with_fixed_noise_t=lambda *a, **kw: torch.zeros(2),
        )
        workspace.critic = lambda x: torch.zeros(2, 1)
        workspace._uses_absolute_eef_chunk_runtime = lambda cfg: False
        workspace._build_critic_input_from_obs = lambda obs: torch.zeros(2, 1)
        workspace._build_online_critic_input = lambda base, stage, env: base
        obs = {'agent_pos': torch.zeros(2, 14)}
        steps = torch.zeros(2, dtype=torch.long)

        def reset():
            steps.zero_()
            return obs, {}

        def reset_at(ids):
            steps[ids] = 0
            return obs, {}

        def step(action):
            steps.add_(1)
            success = torch.tensor([steps[0] == 2, False])
            stage = torch.tensor([4 if success[0] else 0, 0])
            info = {
                'reward_stage': stage, 'success': success,
                'reward_box_upright_penalty': torch.full((2,), .01),
                **{k: torch.ones(2) for k in (
                    'reward_left_to_box', 'reward_right_to_box',
                    'reward_box_to_handover', 'reward_box_to_target',
                )},
            }
            return obs, torch.zeros(2), success, steps >= 3, info

        rollout = workspace.collect_rollout_maniskill_cuda_vector(
            SimpleNamespace(reset=reset, reset_at=reset_at, step=step), torch.device('cpu'), 2,
        )
        # Five steps per environment: two successes, one timeout, two unfinished tails.
        self.assertEqual(rollout['rollout_completed_episodes'], 3)
        self.assertEqual(rollout['rollout_successful_episodes'], 2)
        for stage in range(1, 5):
            self.assertEqual(rollout[f'rollout_stage_{stage}_events'], 2)
        self.assertAlmostEqual(rollout['rollout_reward_component_stage_mean'], 10.6, places=5)
        self.assertAlmostEqual(rollout['rollout_reward_component_success_mean'], 24., places=5)
        component_sum = sum(rollout[f'rollout_reward_component_{key}_mean']
                            for key in ('stage', 'progress', 'success', 'time', 'upright'))
        self.assertAlmostEqual(rollout['reward_mean'], component_sum, places=5)
        self.assertFalse(set(ONLINE_ROLLOUT_LOG_KEYS) & set(ONLINE_ROLLOUT_DIAG_KEYS))


if __name__ == '__main__':
    unittest.main()
