from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.handover.eval_smolvla import (
    DEFAULT_TASK_PROFILE,
    build_parser,
    case_task_instruction,
    evaluation_cases,
    reset_case_metadata,
    reset_task_profile,
    save_results,
    validate_args,
)
from sim.maniskill.tasks.handover_block_ur5_v2_env import HandoverBlockUR5V2Env


class SmolVLAEvalTaskInstructionTest(unittest.TestCase):
    def test_reset_uses_environment_task_without_a_dataset(self) -> None:
        args = build_parser().parse_args(["--evaluation-mode=reset", "--num-episodes=2"])
        args.task_profile = reset_task_profile(args.dataset, args.task_profile)
        metadata = reset_case_metadata(args.task_profile)
        env = SimpleNamespace(env=SimpleNamespace(unwrapped=HandoverBlockUR5V2Env))

        self.assertIsNone(args.dataset)
        self.assertEqual(args.task_profile, DEFAULT_TASK_PROFILE)
        self.assertEqual(metadata["control_mode"], "bimanual_pd_ee_delta_pose")
        self.assertEqual(metadata["sim_freq_hz"], 200)
        self.assertEqual(metadata["control_freq_hz"], 20)
        self.assertEqual(
            case_task_instruction(args, metadata, env),
            HandoverBlockUR5V2Env.task_description,
        )
        self.assertEqual(
            [case["seed"] for case in evaluation_cases(args)],
            [1000, 1001],
        )

    def test_explicit_instruction_overrides_environment_and_recording(self) -> None:
        env = SimpleNamespace(env=SimpleNamespace(unwrapped=HandoverBlockUR5V2Env))
        for mode in ("heldout", "reset"):
            args = build_parser().parse_args(
                [f"--evaluation-mode={mode}", "--task-instruction=Move the block"]
            )
            self.assertEqual(
                case_task_instruction(args, {"task_instruction": "Recorded task"}, env),
                "Move the block",
            )

    def test_heldout_keeps_recorded_instruction(self) -> None:
        args = build_parser().parse_args([])
        env = SimpleNamespace(env=SimpleNamespace(unwrapped=HandoverBlockUR5V2Env))
        self.assertEqual(
            case_task_instruction(args, {"task_instruction": "Recorded task"}, env),
            "Recorded task",
        )

    def test_explicit_dataset_keeps_recorded_task_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "episodes" / "episode_000000"
            episode.mkdir(parents=True)
            (episode / "episode.json").write_text(
                json.dumps({"task_profile": "handover_block_ur5_v2_vr_robotwin_clean_near001"})
            )
            (episode / "data.hdf5").touch()
            self.assertEqual(
                reset_task_profile(Path(directory), None),
                "handover_block_ur5_v2_vr_robotwin_clean_near001",
            )

    def test_empty_instruction_is_rejected(self) -> None:
        args = build_parser().parse_args(["--task-instruction= "])
        with self.assertRaisesRegex(ValueError, "Task instruction must not be empty"):
            validate_args(args)

    def test_heldout_profile_override_is_rejected(self) -> None:
        args = build_parser().parse_args(["--task-profile=handover_block_ur5_v2_vr_robotwin_clean_near001"])
        with self.assertRaisesRegex(ValueError, "profiles come from the recorded episodes"):
            validate_args(args)

    def test_result_records_instruction_and_task_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(
                ["--evaluation-mode=reset", "--task-instruction=Move the block"]
            )
            args.task_profile = reset_task_profile(None, None)
            args.output = Path(directory) / "results.json"
            row = {
                "case_id": "seed_001000",
                "task_instruction": "Move the block",
                "success": False,
                "steps": 0,
                "max_stage": 0,
                "constrained_action_steps": 0,
                "error": None,
            }
            case = {"case_id": "seed_001000", "episode": None, "seed": 1000}
            save_results(args, [row], [case])
            result = json.loads(args.output.read_text())

            self.assertIsNone(result["dataset"])
            self.assertEqual(result["task_profile"], DEFAULT_TASK_PROFILE)
            self.assertEqual(result["task_instruction"], "Move the block")
            self.assertEqual(result["rows"][0]["task_instruction"], "Move the block")


if __name__ == "__main__":
    unittest.main()
