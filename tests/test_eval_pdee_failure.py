from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from script import eval_maniskill_policy as evaluator


class EvalPDEEFailureTest(unittest.TestCase):
    def run_case(self, directory, error, *, completed_before_failure=0, fail_all=False):
        directory = Path(directory)
        checkpoint = directory / "test.ckpt"
        checkpoint.touch()
        args = evaluator.build_parser().parse_args([
            "flowmatching", str(checkpoint), "HandoverBlockUR5-v2",
            "--task-profile", "handover_block_ur5_v2_vr_robotwin_clean",
            "--episodes", "2", "--seed", "999", "--max-episode-steps", "10",
            "--save-video-dir", str(directory / "videos"),
            "--save-action-trace", str(directory / "actions.npz"),
        ])
        cfg = OmegaConf.create({
            "n_action_steps": 4, "horizon": 8, "n_obs_steps": 3,
            "task": {
                "name": "test",
                "shape_meta": {"obs": {"agent_pos": {"shape": [14]}}, "action": {"shape": [14]}},
                "maniskill": {
                    "control_mode": evaluator.DELTA_POSE_CONTROL_MODE,
                    "action_type": "bimanual_pd_ee_delta_pose",
                    "action_semantics": "chunk_anchor_relative_target",
                    "rotation_representation": "euler_xyz",
                    "translation_frame": "root_aligned",
                    "gripper_semantics": "absolute",
                },
            },
        })
        observation = SimpleNamespace(data={"agent_pos": np.zeros(14, dtype=np.float32)})
        seeds, calls = [], []
        step_in_episode = 0

        def reset(*, seed):
            nonlocal step_in_episode
            seeds.append(seed)
            step_in_episode = 0
            return observation

        def step(action):
            nonlocal step_in_episode
            calls.append((seeds[-1], step_in_episode))
            if (fail_all or seeds[-1] == 999) and step_in_episode == completed_before_failure:
                raise error
            step_in_episode += 1
            success = seeds[-1] == 1000 and not fail_all
            return SimpleNamespace(
                observation=observation, reward=1.0, terminated=success, truncated=False,
                info={"success": success, "reward_stage": 4 if success else 0},
            )

        env = SimpleNamespace(
            reset=mock.Mock(side_effect=reset), step=mock.Mock(side_effect=step),
            close=mock.Mock(), prepare_action_chunk=lambda action: action.copy(),
            action_space=gym.spaces.Box(-2.0, 2.0, shape=(14,), dtype=np.float32),
            env=SimpleNamespace(render=lambda: np.zeros((2, 2, 3), dtype=np.uint8)),
        )
        model = SimpleNamespace(obs_cache=[])
        model.update_obs = lambda obs: model.obs_cache.append(obs)
        model.get_action = lambda: np.zeros((4, 14), dtype=np.float32)
        adapter = SimpleNamespace(
            get_model=lambda config: model,
            reset_model=mock.Mock(side_effect=lambda model: model.obs_cache.clear()),
            encode_obs=lambda obs: obs,
        )
        video_lengths = []
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(evaluator.torch, "load", return_value={"cfg": cfg}))
            stack.enter_context(mock.patch.object(evaluator, "ManiSkillTaskFactory", return_value=SimpleNamespace(create_env=lambda **kw: env)))
            stack.enter_context(mock.patch.object(evaluator, "_actual_eef_pose_root_wxyz", return_value=np.array([[0, 0, 0, 1, 0, 0, 0]] * 2, dtype=np.float32)))
            stack.enter_context(mock.patch.object(evaluator.imageio, "mimsave", side_effect=lambda path, frames, **kw: video_lengths.append(len(frames))))
            trace_save = stack.enter_context(mock.patch.object(evaluator, "_save_action_trace"))
            stack.enter_context(mock.patch.object(evaluator.importlib, "import_module", return_value=adapter))
            stack.enter_context(redirect_stdout(output))
            try:
                evaluator.run_evaluation(args)
            finally:
                self.assertEqual(env.close.call_count, 1)
                self.last_seeds = seeds
        report = json.loads((directory / "videos/evaluation_failures.json").read_text())
        return seeds, calls, video_lengths, report, output.getvalue(), trace_save.call_args.args[0], adapter

    def test_first_or_mid_chunk_ik_failure_is_counted_and_next_seed_runs(self):
        for completed in (0, 2):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as tmp:
                seeds, calls, videos, report, output, trace, adapter = self.run_case(
                    tmp, RuntimeError("PDEE inverse kinematics failed."),
                    completed_before_failure=completed,
                )
                self.assertEqual(seeds, [999, 1000])
                self.assertEqual(calls, [(999, i) for i in range(completed + 1)] + [(1000, 0)])
                self.assertEqual(videos, [completed + 1, 2])
                self.assertEqual(adapter.reset_model.call_count, 2)
                self.assertEqual(report["evaluated_episodes"], 2)
                self.assertEqual(len(report["ik_failures"]), 1)
                failure = report["ik_failures"][0]
                self.assertEqual(failure["seed"], 999)
                self.assertEqual(failure["completed_steps"], completed)
                self.assertEqual(len(failure["target_action"]), 14)
                self.assertIn("success_rate=0.5000 pdee_ik_failures=1", output)
                self.assertEqual(trace["episode"], [0] * completed + [1])

    def test_all_first_step_failures_finish_with_zero_success_and_empty_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            seeds, calls, videos, report, output, trace, _ = self.run_case(
                tmp, RuntimeError("PDEE inverse kinematics failed."), fail_all=True,
            )
            self.assertEqual(seeds, [999, 1000])
            self.assertEqual(videos, [1, 1])
            self.assertEqual(len(report["ik_failures"]), 2)
            self.assertEqual(trace["episode"], [])
            self.assertIn("success_rate=0.0000 pdee_ik_failures=2", output)

    def test_nonfinite_ik_joint_target_is_an_episode_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_case(tmp, RuntimeError("PDEE inverse kinematics returned a non-finite joint target."))
            self.assertEqual(result[0], [999, 1000])

    def test_unrelated_runtime_or_value_errors_still_propagate(self):
        for error in (RuntimeError("CUDA out of memory"), RuntimeError("PDEE inverse kinematics internal error"), ValueError("bad action")):
            with self.subTest(error=str(error)), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(type(error)) as raised:
                    self.run_case(tmp, error)
                self.assertIs(raised.exception, error)
                self.assertEqual(self.last_seeds, [999])


if __name__ == "__main__":
    unittest.main()
