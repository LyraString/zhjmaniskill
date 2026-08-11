import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix

from diffusion_policy.common.online_replay_buffer import OnlineReplayBuffer
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.model.rl.q_critic import TwinQCritic
from diffusion_policy.model.rl.resfit_td3 import (
    compute_resfit_td_target,
    resfit_td3_update,
)
from diffusion_policy.model.rl.residual_actor import (
    ResidualActor,
    compose_bimanual_delta_pose_action,
)
from diffusion_policy.policy.FPO_image_policy import FPOImagePolicy
from diffusion_policy.policy.resfit_policy import ResFiTPolicy
from diffusion_policy.workspace.resfit_robotworkspace import ResFiTWorkspace


ACTION_LOW = torch.tensor(
    [-0.015, -0.015, -0.015, -0.10, -0.10, -0.10, 0.0] * 2
)
ACTION_HIGH = torch.tensor(
    [0.015, 0.015, 0.015, 0.10, 0.10, 0.10, 1.0] * 2
)


class _ToyObsEncoder(nn.Module):
    def forward(self, obs_dict):
        return obs_dict["encoded"]

    @staticmethod
    def output_shape():
        return (8,)


class _ToyBasePolicy(nn.Module):
    def __init__(self, n_action_steps=1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.n_action_steps = int(n_action_steps)

    def predict_action_with_feature(self, obs_dict, *, generator=None):
        del generator
        batch_size = obs_dict["encoded"].shape[0]
        action = torch.zeros(batch_size, self.n_action_steps, 14)
        action[:, :, 0] = 0.01 * self.weight
        return {
            "action": action,
            "obs_feature": obs_dict["encoded"],
        }


class _ToyRolloutPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_policy = SimpleNamespace(n_obs_steps=1, n_action_steps=4)
        self.base_predict_count = 0
        self.residual_actor = ResidualActor(
            feature_dim=4,
            residual_scale=[0.003, 0.003, 0.003, 0.02, 0.02, 0.02, 0.1] * 2,
        )
        self.register_buffer("action_low", ACTION_LOW.clone())
        self.register_buffer("action_high", ACTION_HIGH.clone())

    def predict_base_action_chunk(self, obs_history):
        self.base_predict_count += 1
        feature = obs_history["encoded"][:, -1]
        base_action = torch.zeros(
            feature.shape[0],
            self.base_policy.n_action_steps,
            14,
        )
        return {
            "base_action": base_action,
            "obs_feature": feature,
        }

    def predict_action(self, obs_history):
        result = self.predict_base_action_chunk(obs_history)
        base_action = result["base_action"]
        feature = result["obs_feature"]
        feature_sequence = feature[:, None, :].expand(-1, base_action.shape[1], -1)
        residual_action = self.residual_actor(feature_sequence, base_action)
        action = compose_bimanual_delta_pose_action(
            base_action,
            residual_action,
            self.action_low,
            self.action_high,
        )
        return {
            "action": action,
            "base_action": base_action,
            "residual_action": residual_action,
            "obs_feature": feature,
        }


class _ToyContinuousRunner:
    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.reset_count = 0
        self.step_count = 0

    def reset(self):
        self.reset_count += 1
        return {"encoded": torch.zeros(self.num_envs, 4)}, {}

    def step(self, action):
        self.step_count += 1
        obs = {
            "encoded": torch.full(
                (self.num_envs, 4),
                float(self.step_count),
            )
        }
        reward = torch.ones(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        info = {
            "reward_stage": torch.zeros(self.num_envs),
            "success": done,
        }
        return obs, reward, done, done.clone(), info

    def reset_at(self, env_ids):
        raise AssertionError(f"Unexpected reset_at for continuing envs: {env_ids}")


class _ToyPartialDoneRunner(_ToyContinuousRunner):
    def step(self, action):
        del action
        self.step_count += 1
        obs = {"encoded": torch.ones(self.num_envs, 4)}
        reward = torch.ones(self.num_envs)
        terminated = torch.tensor([True, False])
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        info = {
            "reward_stage": torch.zeros(self.num_envs),
            "success": terminated,
        }
        return obs, reward, terminated, truncated, info

    def reset_at(self, env_ids):
        self.reset_count += 1
        if env_ids != [0]:
            raise AssertionError(f"Expected only env 0 to reset, got {env_ids}.")
        obs = {"encoded": torch.zeros(self.num_envs, 4)}
        obs["encoded"][0] = 100.0
        return obs, {}


class ResfitCoreTest(unittest.TestCase):
    @staticmethod
    def _make_toy_action_dit_policy():
        policy = FPOImagePolicy(
            shape_meta={"action": {"shape": [14]}},
            obs_encoder=_ToyObsEncoder(),
            horizon=3,
            n_action_steps=1,
            n_obs_steps=1,
            num_inference_steps=2,
            obs_as_global_cond=True,
            denoiser_type="action_dit",
            action_dit={
                "view_feature_dim": 4,
                "num_views": 1,
                "view_token_counts": [1],
                "use_spatial_token_embedding": False,
                "num_arms": 2,
                "proprio_dim_per_arm": 2,
                "condition_depth": 1,
                "condition_num_heads": 4,
                "hidden_dim": 16,
                "depth": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
        )
        identity = SingleFieldLinearNormalizer.create_identity()
        policy.normalizer["encoded"] = identity
        policy.normalizer["action"] = identity
        return policy.eval()

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

    def test_feature_api_preserves_base_policy_action(self):
        policy = self._make_toy_action_dit_policy()
        obs = {"encoded": torch.randn(2, 1, 8)}

        torch.manual_seed(123)
        original = policy.predict_action(obs)
        torch.manual_seed(123)
        with_feature = policy.predict_action_with_feature(obs)

        torch.testing.assert_close(
            with_feature["action"],
            original["action"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            with_feature["action_pred"],
            original["action_pred"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(with_feature["obs_feature"].shape, (2, 16))

    def test_resfit_policy_freezes_base_and_starts_as_identity(self):
        base_policy = _ToyBasePolicy()
        residual_actor = ResidualActor(
            feature_dim=4,
            residual_scale=[0.003, 0.003, 0.003, 0.02, 0.02, 0.02, 0.1] * 2,
        )
        policy = ResFiTPolicy(
            base_policy=base_policy,
            residual_actor=residual_actor,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
        )
        policy.train()

        result = policy.predict_action({"encoded": torch.randn(3, 4)})

        self.assertEqual(policy.device, ACTION_LOW.device)
        self.assertEqual(policy.dtype, ACTION_LOW.dtype)
        self.assertFalse(base_policy.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in base_policy.parameters()))
        self.assertEqual(result["action"].shape, (3, 1, 14))
        self.assertEqual(result["base_action"].shape, (3, 1, 14))
        self.assertEqual(result["obs_feature"].shape, (3, 4))
        torch.testing.assert_close(
            result["action"],
            result["base_action"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result["residual_action"],
            torch.zeros_like(result["residual_action"]),
            rtol=0,
            atol=0,
        )
        result["action"].sum().backward()
        self.assertIsNone(base_policy.weight.grad)
        actor_output_layer = residual_actor.network[-1]
        self.assertIsNotNone(actor_output_layer.weight.grad)
        self.assertGreater(torch.linalg.vector_norm(actor_output_layer.weight.grad).item(), 0.0)

    def test_resfit_policy_corrects_all_action_chunk_slots(self):
        policy = ResFiTPolicy(
            base_policy=_ToyBasePolicy(n_action_steps=4),
            residual_actor=ResidualActor(
                feature_dim=4,
                residual_scale=[0.003, 0.003, 0.003, 0.02, 0.02, 0.02, 0.1] * 2,
            ),
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
        )

        result = policy.predict_action({"encoded": torch.randn(3, 4)})

        self.assertEqual(result["action"].shape, (3, 4, 14))
        self.assertEqual(result["base_action"].shape, (3, 4, 14))
        self.assertEqual(result["residual_action"].shape, (3, 4, 14))
        torch.testing.assert_close(
            result["action"],
            result["base_action"],
            rtol=0,
            atol=0,
        )

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

    def test_td_target_bootstraps_time_limit_but_not_termination(self):
        reward = torch.tensor([[1.0], [1.0]])
        terminated = torch.tensor([[1.0], [0.0]])
        next_q = torch.tensor([[5.0], [5.0]])

        target = compute_resfit_td_target(reward, terminated, next_q, gamma=0.9)

        torch.testing.assert_close(
            target,
            torch.tensor([[1.0], [5.5]]),
            rtol=0,
            atol=0,
        )

    def test_td3_update_delays_actor_and_keeps_values_finite(self):
        torch.manual_seed(7)
        actor = ResidualActor(
            feature_dim=4,
            residual_scale=[0.003, 0.003, 0.003, 0.02, 0.02, 0.02, 0.1] * 2,
        )
        critic = TwinQCritic(feature_dim=4, action_dim=14, hidden_dims=(32, 16))
        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)
        for module in (target_actor, target_critic):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        batch_size = 8
        base_action = ACTION_LOW + torch.rand(batch_size, 14) * (
            ACTION_HIGH - ACTION_LOW
        )
        batch = {
            "obs_feature": torch.randn(batch_size, 4),
            "base_action": base_action,
            "action": base_action.clone(),
            "reward": torch.randn(batch_size, 1),
            "next_obs_feature": torch.randn(batch_size, 4),
            "next_base_action": base_action.clone(),
            "terminated": torch.zeros(batch_size, 1),
        }
        actor_before = tuple(parameter.detach().clone() for parameter in actor.parameters())

        first = resfit_td3_update(
            residual_actor=actor,
            critic=critic,
            target_residual_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            batch=batch,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
            gamma=0.99,
            tau=0.005,
            critic_update_step=1,
            policy_delay=2,
            target_noise_fraction=0.0,
            target_noise_clip_fraction=0.0,
        )
        self.assertFalse(first["actor_updated"])
        for before, after in zip(actor_before, actor.parameters(), strict=True):
            torch.testing.assert_close(after, before, rtol=0, atol=0)

        second = resfit_td3_update(
            residual_actor=actor,
            critic=critic,
            target_residual_actor=target_actor,
            target_critic=target_critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            batch=batch,
            action_low=ACTION_LOW,
            action_high=ACTION_HIGH,
            gamma=0.99,
            tau=0.005,
            critic_update_step=2,
            policy_delay=2,
            target_noise_fraction=0.0,
            target_noise_clip_fraction=0.0,
        )
        self.assertTrue(second["actor_updated"])
        self.assertTrue(torch.isfinite(torch.tensor(second["critic_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(second["actor_loss"])))
        actor_change = max(
            float((after.detach() - before).abs().amax().item())
            for before, after in zip(actor_before, actor.parameters(), strict=True)
        )
        self.assertGreater(actor_change, 0.0)

    def test_rollout_state_continues_across_collection_iterations(self):
        workspace = ResFiTWorkspace.__new__(ResFiTWorkspace)
        policy = _ToyRolloutPolicy()
        runner = _ToyContinuousRunner()
        replay = OnlineReplayBuffer(capacity=16, feature_dim=4, action_dim=14)
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                num_envs=runner.num_envs,
                exploration_noise_fraction=0.0,
                exploration_noise_clip_fraction=0.0,
            )
        )

        rollout_state = workspace._initialize_rollout_state(policy, runner, cfg)
        first = workspace._collect_online_transitions(
            policy,
            runner,
            replay,
            cfg,
            rollout_state,
            rollout_steps=2,
        )
        with torch.no_grad():
            policy.residual_actor.network[-1].bias.fill_(1.0)
        second = workspace._collect_online_transitions(
            policy,
            runner,
            replay,
            cfg,
            rollout_state,
            rollout_steps=2,
        )

        self.assertEqual(runner.reset_count, 1)
        self.assertEqual(runner.step_count, 4)
        self.assertEqual(len(replay), 8)
        self.assertEqual(first["transitions"], 4)
        self.assertEqual(second["transitions"], 4)
        self.assertEqual(policy.base_predict_count, 2)
        self.assertEqual(rollout_state["chunk_index"], 0)
        torch.testing.assert_close(
            replay._storage["action"][:8],
            torch.zeros(8, 14),
        )
        self.assertGreater(
            float(rollout_state["current_result"]["residual_action"].abs().amax()),
            0.0,
        )
        torch.testing.assert_close(
            rollout_state["episode_length"],
            torch.full((runner.num_envs,), 4, dtype=torch.long),
        )

    def test_partial_reset_only_replaces_finished_env_action_chunk(self):
        workspace = ResFiTWorkspace.__new__(ResFiTWorkspace)
        policy = _ToyRolloutPolicy()
        runner = _ToyPartialDoneRunner()
        replay = OnlineReplayBuffer(capacity=8, feature_dim=4, action_dim=14)
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                num_envs=runner.num_envs,
                exploration_noise_fraction=0.0,
                exploration_noise_clip_fraction=0.0,
            )
        )

        rollout_state = workspace._initialize_rollout_state(policy, runner, cfg)
        workspace._collect_online_transitions(
            policy,
            runner,
            replay,
            cfg,
            rollout_state,
            rollout_steps=1,
        )

        self.assertEqual(policy.base_predict_count, 3)
        self.assertEqual(rollout_state["chunk_index"], 0)
        torch.testing.assert_close(
            replay._storage["next_obs_feature"][:2],
            torch.ones(2, 4),
        )
        torch.testing.assert_close(
            rollout_state["current_result"]["obs_feature"],
            torch.tensor([[100.0] * 4, [1.0] * 4]),
        )


if __name__ == "__main__":
    unittest.main()
