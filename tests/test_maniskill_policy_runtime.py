from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FM_ROOT = REPO_ROOT / "policy" / "fm_diffusion"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))

from diffusion_policy.workspace.robotworkspace import RobotWorkspace as FMWorkspace
from policy.fm_diffusion.diffusion_policy.env_runner.maniskill_cuda_fpo_runner import (
    ManiSkillCudaFPORunner,
)
from script.eval_maniskill_policy import (
    DELTA_POSE_CONTROL_MODE,
    _append_action_trace,
    _clip_legacy_action_to_space,
    _new_action_trace,
    _prepare_runtime_action_chunk,
    _validate_action_in_space,
    _validate_checkpoint_action_semantics,
    build_parser,
)


class ManiSkillPolicyRuntimeTest(unittest.TestCase):
    def test_delta_chunk_is_prepared_exactly_once(self) -> None:
        policy_chunk = np.zeros((3, 14), dtype=np.float32)
        policy_chunk[:, 0] = [2.0, 5.0, 3.0]
        target_chunk = policy_chunk.copy()
        target_chunk[:, 0] += 10.0
        env = SimpleNamespace(
            prepare_action_chunk=mock.Mock(return_value=target_chunk),
        )

        result = _prepare_runtime_action_chunk(
            env,
            policy_chunk,
            control_mode=DELTA_POSE_CONTROL_MODE,
        )

        env.prepare_action_chunk.assert_called_once()
        np.testing.assert_array_equal(
            env.prepare_action_chunk.call_args.args[0],
            policy_chunk,
        )
        np.testing.assert_allclose(result[:, 0], [12.0, 15.0, 13.0])
        np.testing.assert_allclose(policy_chunk[:, 0], [2.0, 5.0, 3.0])

    def test_non_delta_chunk_preserves_legacy_path(self) -> None:
        chunk = np.arange(28, dtype=np.float32).reshape(2, 14)
        env = SimpleNamespace(prepare_action_chunk=mock.Mock())

        result = _prepare_runtime_action_chunk(
            env,
            chunk,
            control_mode="bimanual_pd_joint_pos",
        )

        env.prepare_action_chunk.assert_not_called()
        np.testing.assert_array_equal(result, chunk)
        self.assertIsNot(result, chunk)

    def test_absolute_target_fails_instead_of_silent_clip(self) -> None:
        action_space = gym.spaces.Box(
            low=np.full(14, -2.0, dtype=np.float32),
            high=np.full(14, 2.0, dtype=np.float32),
            dtype=np.float32,
        )
        target = np.zeros(14, dtype=np.float32)
        target[0] = 2.5

        with self.assertRaisesRegex(ValueError, "absolute environment action space"):
            _validate_action_in_space(target, action_space)

        self.assertEqual(float(target[0]), 2.5)
        legacy = _clip_legacy_action_to_space(target, action_space)
        self.assertEqual(float(legacy[0]), 2.0)

    def test_delta_checkpoint_semantics_are_required(self) -> None:
        expected = {
            "action_semantics": "chunk_anchor_relative_target",
            "rotation_representation": "euler_xyz",
            "translation_frame": "root_aligned",
            "gripper_semantics": "absolute",
        }
        cfg = SimpleNamespace(
            task=SimpleNamespace(maniskill=expected),
        )
        _validate_checkpoint_action_semantics(
            cfg,
            control_mode=DELTA_POSE_CONTROL_MODE,
        )

        old_cfg = SimpleNamespace(
            task=SimpleNamespace(
                maniskill={**expected, "action_semantics": "delta_pose_actual"}
            )
        )
        with self.assertRaisesRegex(ValueError, "Checkpoint action semantics"):
            _validate_checkpoint_action_semantics(
                old_cfg,
                control_mode=DELTA_POSE_CONTROL_MODE,
            )

        _validate_checkpoint_action_semantics(
            old_cfg,
            control_mode="bimanual_pd_joint_pos",
        )

    def test_flow_matching_resume_rejects_old_semantics_or_chunk_length(self) -> None:
        expected = SimpleNamespace(
            control_mode=DELTA_POSE_CONTROL_MODE,
            action_semantics="chunk_anchor_relative_target",
            rotation_representation="euler_xyz",
            translation_frame="root_aligned",
            gripper_semantics="absolute",
        )
        cfg = SimpleNamespace(
            n_action_steps=4,
            task=SimpleNamespace(maniskill=expected),
        )
        checkpoint_cfg = SimpleNamespace(
            n_action_steps=4,
            task=SimpleNamespace(maniskill=expected),
        )
        FMWorkspace._validate_maniskill_resume_action_semantics(
            cfg,
            {"cfg": checkpoint_cfg},
        )

        old_checkpoint_cfg = SimpleNamespace(
            n_action_steps=1,
            task=SimpleNamespace(
                maniskill=SimpleNamespace(
                    **{
                        **vars(expected),
                        "action_semantics": "delta_pose_actual",
                    }
                )
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "resume checkpoint"):
            FMWorkspace._validate_maniskill_resume_action_semantics(
                cfg,
                {"cfg": old_checkpoint_cfg},
            )

    def test_action_trace_separates_policy_and_target_domains(self) -> None:
        trace = _new_action_trace()
        policy_action = np.zeros(14, dtype=np.float32)
        policy_action[0] = 0.05
        target_action = policy_action.copy()
        target_action[0] = 0.45

        _append_action_trace(
            trace,
            episode_idx=1,
            step=2,
            target_action=target_action,
            info={},
            raw_policy_action=policy_action,
        )

        np.testing.assert_array_equal(trace["raw_policy_action"][0], policy_action)
        np.testing.assert_array_equal(trace["target_action"][0], target_action)
        self.assertAlmostEqual(trace["policy_left_xyz_norm"][0], 0.05)
        self.assertAlmostEqual(trace["target_left_xyz_norm"][0], 0.45)

    def test_legacy_hold_options_are_removed(self) -> None:
        parser = build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--delta-pose-hold-mode", option_strings)

    def test_cuda_runner_separates_policy_and_target_bounds(self) -> None:
        runner = object.__new__(ManiSkillCudaFPORunner)
        runner.device = torch.device("cpu")
        runner.action_dim = 14
        runner.action_low = None
        runner.action_high = None
        runner.target_action_low = torch.full((14,), -2.0)
        runner.target_action_high = torch.full((14,), 2.0)
        policy_low = torch.linspace(-0.06, -0.01, 14)
        policy_high = torch.linspace(0.01, 0.06, 14)

        with self.assertRaisesRegex(RuntimeError, "not configured"):
            runner.clip_actions(torch.zeros((1, 14)))
        runner.set_policy_action_bounds(policy_low, policy_high)

        torch.testing.assert_close(runner.action_low, policy_low)
        torch.testing.assert_close(runner.action_high, policy_high)
        torch.testing.assert_close(runner.target_action_low, torch.full((14,), -2.0))
        clipped = runner.clip_actions(torch.full((1, 14), 1.0))
        torch.testing.assert_close(clipped[0], policy_high)

    def test_cuda_runner_prepares_the_full_batched_chunk_once(self) -> None:
        runner = object.__new__(ManiSkillCudaFPORunner)
        runner.device = torch.device("cpu")
        runner.action_dim = 14
        runner.num_envs = 2
        runner.target_action_low = torch.tensor([-2.0] * 6 + [0.0] + [-2.0] * 6 + [0.0])
        runner.target_action_high = torch.tensor([2.0] * 6 + [1.0] + [2.0] * 6 + [1.0])
        chunk = torch.zeros((2, 4, 14), dtype=torch.float32)
        expected = chunk + 1.0
        prepare = mock.Mock(return_value=expected)
        runner.env = SimpleNamespace(
            unwrapped=SimpleNamespace(prepare_action_chunk=prepare),
        )

        result = runner.prepare_action_chunk(chunk)

        prepare.assert_called_once_with(chunk, env_ids=None)
        torch.testing.assert_close(result, expected)

    def test_cuda_runner_clamps_only_prepared_gripper_targets(self) -> None:
        runner = object.__new__(ManiSkillCudaFPORunner)
        runner.device = torch.device("cpu")
        runner.action_dim = 14
        runner.num_envs = 1
        runner.target_action_low = torch.tensor([-2.0] * 6 + [0.0] + [-2.0] * 6 + [0.0])
        runner.target_action_high = torch.tensor([2.0] * 6 + [1.0] + [2.0] * 6 + [1.0])
        chunk = torch.zeros((1, 2, 14), dtype=torch.float32)
        targets = chunk.clone()
        targets[..., 6] = -0.1
        targets[..., 13] = 1.1
        runner.env = SimpleNamespace(
            unwrapped=SimpleNamespace(
                prepare_action_chunk=mock.Mock(return_value=targets),
            ),
        )

        result = runner.prepare_action_chunk(chunk)

        torch.testing.assert_close(result[..., 6], torch.zeros((1, 2)))
        torch.testing.assert_close(result[..., 13], torch.ones((1, 2)))
        torch.testing.assert_close(targets[..., 6], torch.full((1, 2), -0.1))
        torch.testing.assert_close(targets[..., 13], torch.full((1, 2), 1.1))

        bad_targets = targets.clone()
        bad_targets[..., 0] = 2.1
        runner.env.unwrapped.prepare_action_chunk.return_value = bad_targets
        with self.assertRaisesRegex(ValueError, "Cartesian target"):
            runner.prepare_action_chunk(chunk)


if __name__ == "__main__":
    unittest.main()
