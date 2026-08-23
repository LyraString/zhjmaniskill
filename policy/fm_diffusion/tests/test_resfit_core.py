import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf
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
from diffusion_policy.model.rl.residual_velocity_transformer import (
    ResidualVelocityTransformer,
)
from diffusion_policy.policy.FPO_image_policy import FPOImagePolicy
from diffusion_policy.policy.resfit_policy import ResFiTPolicy
from diffusion_policy.workspace.fpo_robotworkspace import RobotWorkspace as FPOWorkspace
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


class _ForwardProxy(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.module(*args, **kwargs)


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
        self.prepared_policy_chunks = []
        self.prepared_absolute_chunks = []
        self.stepped_actions = []
        self.action_low = ACTION_LOW.clone()
        self.action_high = ACTION_HIGH.clone()
        self._reset_anchors()

    def _reset_anchors(self):
        env_offset = torch.arange(self.num_envs, dtype=torch.float32)[:, None]
        self.left_anchor = torch.full((self.num_envs, 3), 10.0) + env_offset
        self.right_anchor = torch.full((self.num_envs, 3), 20.0) + env_offset

    def reset(self):
        self.reset_count += 1
        self._reset_anchors()
        return {"encoded": torch.zeros(self.num_envs, 4)}, {}

    def set_policy_action_bounds(self, action_low, action_high):
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32).flatten()
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32).flatten()

    def clip_actions(self, action):
        action = torch.as_tensor(action, dtype=torch.float32)
        return torch.maximum(torch.minimum(action, self.action_high), self.action_low)

    def prepare_action_chunk(self, action_chunk, env_ids=None):
        action_chunk = torch.as_tensor(action_chunk, dtype=torch.float32)
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        env_ids = torch.as_tensor(env_ids, dtype=torch.long)
        absolute_chunk = action_chunk.clone()
        absolute_chunk[:, :, :3] += self.left_anchor[env_ids, None]
        absolute_chunk[:, :, 7:10] += self.right_anchor[env_ids, None]
        self.prepared_policy_chunks.append(action_chunk.clone())
        self.prepared_absolute_chunks.append(absolute_chunk.clone())
        return absolute_chunk

    def _record_step_action(self, action):
        action = torch.as_tensor(action, dtype=torch.float32)
        self.stepped_actions.append(action.clone())
        self.left_anchor.copy_(action[:, :3])
        self.right_anchor.copy_(action[:, 7:10])

    def step(self, action):
        self._record_step_action(action)
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
        self._record_step_action(action)
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
        self.left_anchor[0] = 100.0
        self.right_anchor[0] = 200.0
        obs = {"encoded": torch.zeros(self.num_envs, 4)}
        obs["encoded"][0] = 100.0
        return obs, {}


class _ToyActionStats:
    def __init__(self, action_low, action_high):
        self.action_low = action_low
        self.action_high = action_high

    def get_input_stats(self):
        return {"min": self.action_low, "max": self.action_high}


class _ToyNormalizer:
    def __init__(self, action_low, action_high):
        self.action = _ToyActionStats(action_low, action_high)

    def __getitem__(self, key):
        if key != "action":
            raise KeyError(key)
        return self.action


class ResfitCoreTest(unittest.TestCase):
    @staticmethod
    def _make_toy_action_dit_policy(
        velocity_residual=None,
        freeze_base_velocity=False,
    ):
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
            velocity_residual=velocity_residual,
            freeze_base_velocity=freeze_base_velocity,
        )
        identity = SingleFieldLinearNormalizer.create_identity()
        policy.normalizer["encoded"] = identity
        policy.normalizer["action"] = identity
        return policy.eval()

    def test_residual_fpo_starts_as_exact_base_velocity(self):
        torch.manual_seed(7)
        base_policy = self._make_toy_action_dit_policy()
        residual_policy = self._make_toy_action_dit_policy(
            velocity_residual=ResidualVelocityTransformer(
                action_dim=14,
                horizon=3,
                condition_dim=16,
                hidden_dim=32,
                depth=2,
                num_heads=4,
                dropout=0.0,
                residual_scale=0.1,
            ),
            freeze_base_velocity=True,
        )
        FPOWorkspace._load_actor_state_with_optional_residual(
            residual_policy,
            base_policy.state_dict(),
            "model",
        )

        sample = torch.randn(4, 3, 14)
        timestep = torch.randint(0, 1000, (4,))
        global_cond = torch.randn(4, 8)
        base_velocity = base_policy._predict_velocity(
            sample,
            timestep,
            global_cond=global_cond,
        )
        combined_velocity = residual_policy._predict_velocity(
            sample,
            timestep,
            global_cond=global_cond,
        )

        torch.testing.assert_close(
            combined_velocity,
            base_velocity,
            rtol=0,
            atol=0,
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in residual_policy.model.parameters())
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in residual_policy.velocity_residual.parameters()
            )
        )

        combined_velocity.sum().backward()
        self.assertIsNotNone(
            residual_policy.velocity_residual.output_projection.bias.grad
        )
        self.assertGreater(
            residual_policy.velocity_residual.output_projection.bias.grad.abs().sum().item(),
            0.0,
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in residual_policy.model.parameters())
        )

    def test_residual_fpo_rejects_partial_residual_checkpoint(self):
        policy = self._make_toy_action_dit_policy(
            velocity_residual=ResidualVelocityTransformer(
                action_dim=14,
                horizon=3,
                condition_dim=16,
                hidden_dim=32,
                depth=1,
                num_heads=4,
                dropout=0.0,
            ),
            freeze_base_velocity=True,
        )
        state_dict = policy.state_dict()
        removed_key = next(
            key for key in state_dict if key.startswith("velocity_residual.")
        )
        state_dict.pop(removed_key)

        with self.assertRaisesRegex(RuntimeError, "Incompatible model checkpoint"):
            FPOWorkspace._load_actor_state_with_optional_residual(
                policy,
                state_dict,
                "model",
            )

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

    def test_fpo_prepares_unclipped_policy_chunk_once(self):
        runner = _ToyContinuousRunner()
        action_chunk = torch.zeros(runner.num_envs, 3, 14)
        action_chunk[0, 1, 0] = 0.2
        action_chunk[:, :, 6] = 0.5
        action_chunk[:, :, 13] = 0.5

        absolute_chunk = FPOWorkspace._prepare_absolute_action_chunk(
            runner,
            action_chunk,
        )

        self.assertEqual(len(runner.prepared_policy_chunks), 1)
        torch.testing.assert_close(
            runner.prepared_policy_chunks[0],
            action_chunk,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            absolute_chunk,
            runner.prepared_absolute_chunks[0],
            rtol=0,
            atol=0,
        )
        self.assertAlmostEqual(
            runner.prepared_policy_chunks[0][0, 1, 0].item(),
            0.2,
        )

    def test_fpo_checkpoint_validator_rejects_old_delta_semantics(self):
        checkpoint_cfg = OmegaConf.create(
            {
                "n_action_steps": 4,
                "task": {
                    "maniskill": {
                        "control_mode": "bimanual_pd_ee_delta_pose",
                        "action_semantics": "chunk_anchor_relative_target",
                        "rotation_representation": "euler_xyz",
                        "translation_frame": "root_aligned",
                        "gripper_semantics": "absolute",
                    }
                }
            }
        )
        FPOWorkspace._validate_maniskill_checkpoint_action_semantics(
            {"cfg": checkpoint_cfg},
            expected_action_steps=4,
        )

        checkpoint_cfg.task.maniskill.action_semantics = "delta_pose_actual"
        checkpoint_cfg.task.maniskill.translation_frame = "root_world_aligned"
        with self.assertRaisesRegex(RuntimeError, "action semantics mismatch"):
            FPOWorkspace._validate_maniskill_checkpoint_action_semantics(
                {"cfg": checkpoint_cfg},
                expected_action_steps=4,
            )

    def test_resfit_action_bounds_come_from_base_normalizer(self):
        workspace = ResFiTWorkspace.__new__(ResFiTWorkspace)
        action_low = torch.linspace(-0.3, -0.1, 14)
        action_high = torch.linspace(0.1, 0.4, 14)
        base_policy = SimpleNamespace(
            normalizer=_ToyNormalizer(action_low, action_high),
        )
        runner = _ToyContinuousRunner()

        configured_low, configured_high = workspace._configure_policy_action_bounds(
            base_policy,
            runner,
        )

        torch.testing.assert_close(configured_low, action_low, rtol=0, atol=0)
        torch.testing.assert_close(configured_high, action_high, rtol=0, atol=0)
        torch.testing.assert_close(runner.action_low, action_low, rtol=0, atol=0)
        torch.testing.assert_close(runner.action_high, action_high, rtol=0, atol=0)

    def test_resfit_checkpoint_validator_requires_chunk_anchor_semantics(self):
        shape_meta = {
            "obs": {"agent_pos": {"shape": [14], "type": "low_dim"}},
            "action": {"shape": [14]},
        }
        cfg = OmegaConf.create(
            {
                "n_obs_steps": 3,
                "n_action_steps": 4,
                "horizon": 8,
                "shape_meta": shape_meta,
                "training": {"base_policy_source": "ema_model"},
            }
        )
        checkpoint_cfg = OmegaConf.create(
            {
                "n_obs_steps": 3,
                "n_action_steps": 4,
                "horizon": 8,
                "shape_meta": shape_meta,
                "policy": {"denoiser_type": "action_dit"},
                "task": {
                    "maniskill": {
                        "control_mode": "bimanual_pd_ee_delta_pose",
                        "action_semantics": "chunk_anchor_relative_target",
                        "rotation_representation": "euler_xyz",
                        "translation_frame": "root_aligned",
                        "gripper_semantics": "absolute",
                        "control_freq": 20,
                    }
                },
            }
        )
        payload = {
            "cfg": checkpoint_cfg,
            "state_dicts": {
                "ema_model": {"normalizer.params_dict.action.scale": torch.ones(14)}
            },
        }

        ResFiTWorkspace._validate_base_checkpoint(payload, cfg)

        old_semantics_payload = copy.deepcopy(payload)
        old_semantics_payload["cfg"].task.maniskill.action_semantics = "delta_pose_actual"
        old_semantics_payload["cfg"].task.maniskill.translation_frame = "root_world_aligned"
        with self.assertRaisesRegex(RuntimeError, "Base checkpoint semantics mismatch"):
            ResFiTWorkspace._validate_base_checkpoint(old_semantics_payload, cfg)

        ResFiTWorkspace._validate_resume_checkpoint(payload, cfg)
        with self.assertRaisesRegex(RuntimeError, "resume checkpoint semantics mismatch"):
            ResFiTWorkspace._validate_resume_checkpoint(old_semantics_payload, cfg)

    def test_resfit_identity_rollout_steps_absolute_targets(self):
        workspace = ResFiTWorkspace.__new__(ResFiTWorkspace)
        policy = _ToyRolloutPolicy()
        runner = _ToyContinuousRunner()
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                identity_rollout_steps=2,
                num_envs=runner.num_envs,
            ),
            residual_actor=SimpleNamespace(feature_dim=4),
        )

        summary = workspace._run_identity_rollout(policy, runner, cfg)

        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["identity_error_max"], 0.0)
        self.assertEqual(len(runner.prepared_policy_chunks), 2)
        self.assertEqual(len(runner.stepped_actions), 2)
        for prepared, stepped in zip(
            runner.prepared_absolute_chunks,
            runner.stepped_actions,
            strict=True,
        ):
            torch.testing.assert_close(stepped, prepared[:, 0], rtol=0, atol=0)

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

    def test_td3_update_uses_distributed_forward_modules(self):
        torch.manual_seed(11)
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
        actor_forward = _ForwardProxy(actor)
        critic_forward = _ForwardProxy(critic)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        base_action = ACTION_LOW + torch.rand(8, 14) * (ACTION_HIGH - ACTION_LOW)
        batch = {
            "obs_feature": torch.randn(8, 4),
            "base_action": base_action,
            "action": base_action.clone(),
            "reward": torch.randn(8, 1),
            "next_obs_feature": torch.randn(8, 4),
            "next_base_action": base_action.clone(),
            "terminated": torch.zeros(8, 1),
        }

        result = resfit_td3_update(
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
            policy_delay=1,
            target_noise_fraction=0.0,
            target_noise_clip_fraction=0.0,
            residual_actor_forward=actor_forward,
            critic_forward=critic_forward,
        )

        self.assertTrue(result["actor_updated"])
        self.assertEqual(actor_forward.calls, 1)
        self.assertEqual(critic_forward.calls, 1)

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
        self.assertEqual(len(runner.prepared_policy_chunks), 2)
        self.assertEqual(rollout_state["chunk_index"], 0)
        torch.testing.assert_close(
            replay._storage["action"][:8],
            torch.zeros(8, 14),
        )
        torch.testing.assert_close(
            torch.stack(runner.stepped_actions, dim=1),
            runner.prepared_absolute_chunks[0],
            rtol=0,
            atol=0,
        )
        self.assertNotEqual(
            runner.stepped_actions[0][0, 0].item(),
            replay._storage["action"][0, 0].item(),
        )
        self.assertGreater(
            float(rollout_state["current_result"]["residual_action"].abs().amax()),
            0.0,
        )
        torch.testing.assert_close(
            rollout_state["current_result"]["absolute_action"],
            runner.prepared_absolute_chunks[-1],
            rtol=0,
            atol=0,
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
        self.assertEqual(len(runner.prepared_policy_chunks), 2)
        self.assertEqual(rollout_state["chunk_index"], 0)
        torch.testing.assert_close(
            replay._storage["next_obs_feature"][:2],
            torch.ones(2, 4),
        )
        torch.testing.assert_close(
            replay._storage["action"][:2],
            torch.zeros(2, 14),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            runner.stepped_actions[0],
            runner.prepared_absolute_chunks[0][:, 0],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            runner.prepared_absolute_chunks[1][0, :, :3],
            torch.full((4, 3), 100.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rollout_state["current_result"]["absolute_action"],
            runner.prepared_absolute_chunks[1],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            rollout_state["current_result"]["obs_feature"],
            torch.tensor([[100.0] * 4, [1.0] * 4]),
        )


if __name__ == "__main__":
    unittest.main()
