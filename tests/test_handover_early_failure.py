from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'policy/fm_diffusion'))
from diffusion_policy.env_runner.handover_failure import HandoverFailureDetector
from diffusion_policy.workspace.fpo_robotworkspace import RobotWorkspace


def failure_config():
    return OmegaConf.create(dict(enabled=True, penalty=30., confirm_steps=2,
                                 closed_threshold=.25, open_threshold=.95, box_tilt_deg=60.))


def observation_info(n):
    return {
        'reward_stage': torch.zeros(n), 'success': torch.zeros(n, dtype=torch.bool),
        'reward_left_gripper_state': torch.ones(n),
        'reward_right_gripper_state': torch.ones(n),
        'reward_left_grasped': torch.zeros(n, dtype=torch.bool),
        'reward_right_grasped': torch.zeros(n, dtype=torch.bool),
        'reward_box_tilt_cost': torch.zeros(n),
    }


class HandoverEarlyFailureTest(unittest.TestCase):
    def test_initial_closed_grippers_and_normal_approach(self):
        detector = HandoverFailureDetector(2, 'cpu', failure_config())
        info = observation_info(2)
        info['reward_left_gripper_state'][0] = 0
        info['reward_right_gripper_state'][0] = 0
        for _ in range(5):
            self.assertFalse(detector.step(torch.zeros(2, 14), info).any())

    def test_both_grasp_failures_contact_and_release(self):
        detector = HandoverFailureDetector(4, 'cpu', failure_config())
        info = observation_info(4)
        actions = torch.zeros(4, 14)
        detector.step(actions, info)
        info['reward_left_gripper_state'][:] = 0
        info['reward_right_gripper_state'][:] = 0
        info['reward_left_grasped'][:] = torch.tensor([False, True, True, False])
        info['reward_right_grasped'][:] = torch.tensor([True, False, True, False])
        # Left withdrawal / right final release are not grasp attempts.
        info['reward_stage'][3] = 4
        self.assertFalse(detector.step(actions, info).any())
        result = detector.step(actions, info)
        torch.testing.assert_close(result, torch.tensor([
            [True, False, False], [False, True, False],
            [False, False, False], [False, False, False],
        ]))
        detector.reset([0])
        self.assertFalse(detector.step(actions, info)[0].any())
        actions[1, 13] = 1  # Opening command while fingers are still physically closed.
        self.assertFalse(detector.step(actions, info)[1].any())

    def test_tilt_confirmation_success_priority_and_partial_reset(self):
        detector = HandoverFailureDetector(3, 'cpu', failure_config())
        info = observation_info(3)
        info['reward_box_tilt_cost'][:] = torch.tensor([1., 1., .2])
        actions = torch.zeros(3, 14)
        self.assertFalse(detector.step(actions, info).any())
        detector.reset([0])
        torch.testing.assert_close(detector.step(actions, info)[:, 2],
                                   torch.tensor([False, True, False]))
        info['success'][:] = True
        self.assertFalse(detector.step(actions, info).any())
        info['success'][:] = False
        info['reward_box_tilt_cost'][:] = 0
        self.assertFalse(detector.step(actions, info).any())

    def test_real_collector_resets_only_failed_env_and_does_not_bootstrap(self):
        workspace = RobotWorkspace.__new__(RobotWorkspace)
        workspace.cfg = OmegaConf.create({'training': {
            'timesteps_per_batch': 12, 'max_timesteps_per_episode': 20,
            'num_fpo_samples': 1, 'show_rollout_tqdm': False,
            'gamma': .99, 'lmbda': .99, 'early_failure': failure_config(),
        }})
        workspace.model = SimpleNamespace(
            n_obs_steps=1, n_action_steps=4,
            normalizer={'action': SimpleNamespace(normalize=lambda x: x)},
            sample_action_with_info=lambda *a, **kw: {
                'action': torch.zeros(2, 4, 14), 'global_cond': torch.zeros(2, 1),
                'loss_eps': torch.zeros(2, 1, 4, 14), 'loss_t': torch.zeros(2, 1),
                'naction_pred': torch.zeros(2, 4, 14),
                'cond_mask': torch.zeros(2, 4, 14, dtype=torch.bool),
            },
            compute_cfm_loss_with_fixed_noise_t=lambda *a, **kw: torch.zeros(2),
        )
        workspace.critic = lambda x: torch.full((2,), 5.).unsqueeze(-1)
        workspace._uses_absolute_eef_chunk_runtime = lambda cfg: False
        workspace._build_critic_input_from_obs = lambda obs: torch.zeros(2, 1)
        workspace._build_online_critic_input = lambda base, stage, env: base
        obs = {'agent_pos': torch.zeros(2, 14)}
        steps = torch.zeros(2, dtype=torch.long)
        resets = []

        def reset():
            steps.zero_()
            return obs, {}

        def reset_at(ids):
            resets.append(list(ids))
            steps[ids] = 0
            return obs, {}

        def step(action):
            steps.add_(1)
            info = observation_info(2)
            info['reward_left_gripper_state'][0] = float(steps[0] == 1)
            info.update({k: torch.ones(2) for k in (
                'reward_left_to_box', 'reward_right_to_box',
                'reward_box_to_handover', 'reward_box_to_target',
            )})
            # Failure coincides with a truncation; failure must take precedence.
            return (obs, torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                    torch.tensor([steps[0] == 3, False]), info)

        rollout = workspace.collect_rollout_maniskill_cuda_vector(
            SimpleNamespace(reset=reset, reset_at=reset_at, step=step), torch.device('cpu'), 2,
        )
        self.assertEqual(resets, [[0], [0]])
        self.assertEqual(steps.tolist(), [0, 6])
        self.assertEqual(rollout['rollout_completed_episodes'], 2)
        self.assertEqual(rollout['rollout_successful_episodes'], 0)
        self.assertEqual(rollout['rollout_early_failures'], 2)
        self.assertEqual(rollout['rollout_failure_left_empty_grasp'], 2)
        self.assertAlmostEqual(rollout['rollout_reward_component_failure_mean'], -5.)
        # A failure inside the four-step chunk stops its remaining stale actions.
        self.assertEqual(rollout['rollout_chunk_steps_mean'], 3.)
        returns = rollout['returns'].reshape(6, 2)
        torch.testing.assert_close(returns[[2, 5], 0], torch.tensor([-30.02, -30.02]))
        self.assertGreater(returns[-1, 1].item(), 0)  # Unfinished env still bootstraps.


if __name__ == '__main__':
    unittest.main()
