import unittest

import torch
from pytorch3d.transforms import euler_angles_to_matrix

from diffusion_policy.common.online_replay_buffer import OnlineReplayBuffer
from diffusion_policy.model.rl.q_critic import TwinQCritic
from diffusion_policy.model.rl.residual_actor import (
    ResidualActor,
    compose_bimanual_delta_pose_action,
)


ACTION_LOW = torch.tensor(
    [-0.015, -0.015, -0.015, -0.10, -0.10, -0.10, 0.0] * 2
)
ACTION_HIGH = torch.tensor(
    [0.015, 0.015, 0.015, 0.10, 0.10, 0.10, 1.0] * 2
)


class ResfitCoreTest(unittest.TestCase):
    def test_zero_initialized_actor_preserves_base_action(self):
        actor = ResidualActor(
            feature_dim=32,
            residual_scale=[0.003, 0.003, 0.003, 0.02, 0.02, 0.02, 0.1] * 2,
        )
        feature = torch.randn(5, 32)
        base_action = ACTION_LOW + torch.rand(5, 14) * (ACTION_HIGH - ACTION_LOW)

        residual = actor(feature, base_action)
        composed = compose_bimanual_delta_pose_action(
            base_action,
            residual,
            ACTION_LOW,
            ACTION_HIGH,
        )

        torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0, atol=0)
        torch.testing.assert_close(composed, base_action, rtol=0, atol=0)

    def test_rotation_is_root_aligned_composition_and_action_is_clipped(self):
        base_action = torch.zeros(1, 14)
        residual = torch.zeros_like(base_action)
        base_action[0, 3:6] = torch.tensor([0.04, -0.03, 0.02])
        residual[0, 3:6] = torch.tensor([-0.02, 0.01, 0.03])
        base_action[0, 0] = 0.014
        residual[0, 0] = 0.005
        base_action[0, 6] = 0.95
        residual[0, 6] = 0.10

        composed = compose_bimanual_delta_pose_action(
            base_action,
            residual,
            ACTION_LOW,
            ACTION_HIGH,
        )

        expected_rotation = (
            euler_angles_to_matrix(residual[:, 3:6], "XYZ")
            @ euler_angles_to_matrix(base_action[:, 3:6], "XYZ")
        )
        actual_rotation = euler_angles_to_matrix(composed[:, 3:6], "XYZ")
        torch.testing.assert_close(actual_rotation, expected_rotation, rtol=1e-5, atol=1e-6)
        self.assertEqual(composed[0, 0].item(), ACTION_HIGH[0].item())
        self.assertEqual(composed[0, 6].item(), ACTION_HIGH[6].item())

    def test_zero_rotation_residual_keeps_gradient(self):
        base_action = torch.zeros(1, 14)
        base_action[0, 3:6] = torch.tensor([0.04, -0.03, 0.02])
        residual = torch.zeros_like(base_action, requires_grad=True)

        composed = compose_bimanual_delta_pose_action(
            base_action,
            residual,
            ACTION_LOW,
            ACTION_HIGH,
        )
        composed[0, 3:6].sum().backward()

        self.assertIsNotNone(residual.grad)
        self.assertTrue(torch.all(torch.isfinite(residual.grad[0, 3:6])))
        self.assertGreater(torch.linalg.vector_norm(residual.grad[0, 3:6]).item(), 0.0)

    def test_twin_q_output_shape(self):
        critic = TwinQCritic(feature_dim=32, action_dim=14, hidden_dims=(64, 32))
        q1, q2 = critic(torch.randn(7, 32), torch.randn(7, 14))
        self.assertEqual(q1.shape, (7, 1))
        self.assertEqual(q2.shape, (7, 1))
        self.assertNotEqual(q1.data_ptr(), q2.data_ptr())

    def test_online_replay_buffer_wraps_and_samples(self):
        buffer = OnlineReplayBuffer(capacity=5, feature_dim=3, action_dim=2)

        def transitions(start: int, count: int):
            values = torch.arange(start, start + count, dtype=torch.float32)
            return {
                "obs_feature": values[:, None].repeat(1, 3),
                "base_action": values[:, None].repeat(1, 2),
                "action": (values + 0.5)[:, None].repeat(1, 2),
                "reward": values,
                "next_obs_feature": (values + 1)[:, None].repeat(1, 3),
                "next_base_action": (values + 1)[:, None].repeat(1, 2),
                "terminated": torch.zeros(count),
            }

        buffer.add_batch(**transitions(0, 3))
        buffer.add_batch(**transitions(3, 4))
        self.assertEqual(len(buffer), 5)
        self.assertEqual(buffer.position, 2)

        sample = buffer.sample(8, generator=torch.Generator().manual_seed(0))
        self.assertEqual(set(sample), set(OnlineReplayBuffer._FIELD_DIMS))
        self.assertEqual(sample["obs_feature"].shape, (8, 3))
        self.assertEqual(sample["action"].shape, (8, 2))
        self.assertTrue(torch.all(sample["reward"] >= 2))
        self.assertTrue(torch.all(sample["reward"] <= 6))


if __name__ == "__main__":
    unittest.main()
