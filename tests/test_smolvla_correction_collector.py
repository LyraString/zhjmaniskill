from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.handover.collect_smolvla_corrections import (
    CollectorCommand,
    CollectorMode,
    CollectorState,
    build_parser,
    initialize_takeover_mapper,
    triggers_match_grippers,
)
from sim.maniskill.teleop.pose_mapper import BimanualXRMapper
from sim.maniskill.teleop.xr_client import XRState


class CollectorStateTest(unittest.TestCase):
    def test_policy_takeover_and_success_flow(self) -> None:
        state = CollectorState()

        self.assertEqual(
            state.handle_buttons({"B": True}), CollectorCommand.START_POLICY
        )
        self.assertEqual(state.mode, CollectorMode.POLICY)
        self.assertEqual(
            state.handle_buttons({"A": True}), CollectorCommand.PREPARE_TAKEOVER
        )
        self.assertEqual(state.mode, CollectorMode.TAKEOVER_READY)
        self.assertEqual(
            state.handle_buttons({"B": True}, trigger_aligned=False),
            CollectorCommand.TRIGGER_NOT_ALIGNED,
        )
        self.assertEqual(state.mode, CollectorMode.TAKEOVER_READY)
        self.assertEqual(
            state.handle_buttons({"B": True}, trigger_aligned=True),
            CollectorCommand.START_TELEOP,
        )
        self.assertEqual(state.mode, CollectorMode.TELEOP)
        self.assertEqual(state.handle_success(), CollectorCommand.SAVE_CORRECTION)

    def test_autonomous_success_is_not_a_correction(self) -> None:
        state = CollectorState()
        state.handle_buttons({"B": True})

        self.assertEqual(
            state.handle_success(), CollectorCommand.AUTONOMOUS_SUCCESS
        )

    def test_x_retries_and_y_skips_from_every_active_mode(self) -> None:
        for mode in CollectorMode:
            with self.subTest(mode=mode):
                state = CollectorState(mode)
                self.assertEqual(
                    state.handle_buttons({"X": True}),
                    CollectorCommand.RESET_CURRENT_SEED,
                )
                self.assertEqual(
                    state.handle_buttons({"Y": True}),
                    CollectorCommand.SKIP_CURRENT_SEED,
                )


class TriggerAlignmentTest(unittest.TestCase):
    def test_trigger_semantics_match_current_gripper_commands(self) -> None:
        self.assertTrue(
            triggers_match_grippers(
                np.asarray([0.0, 1.0]),
                np.asarray([1.0, 0.0]),
                tolerance=0.01,
            )
        )

    def test_takeover_anchor_does_not_move_eef_on_first_action(self) -> None:
        mapper = BimanualXRMapper(xr_to_world_rotation=np.eye(3))
        current_ee = np.asarray(
            [
                [0.2, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0],
                [-0.2, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        roots = np.asarray(
            [
                [0.0, -0.6, 0.7, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.6, 0.7, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        xr_state = XRState(
            timestamp_ns=1,
            controller_pose_xyzw=np.asarray(
                [
                    [0.3, 1.2, -0.4, 0.0, 0.0, 0.0, 1.0],
                    [-0.2, 0.7, 0.8, 0.0, 0.0, 0.0, 1.0],
                ]
            ),
            grip=np.ones(2),
            trigger=np.asarray([0.0, 1.0]),
            axis=np.zeros((2, 2)),
            buttons=np.zeros(8, dtype=np.bool_),
        )

        desired = initialize_takeover_mapper(
            mapper,
            xr_state,
            current_ee_pose_root=current_ee,
            root_pose_world=roots,
        )
        action = mapper.build_delta_action(
            current_ee_pose_root=current_ee,
            trigger=xr_state.trigger,
        )

        np.testing.assert_allclose(desired, current_ee, atol=1e-7)
        np.testing.assert_allclose(action[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], 0.0, atol=1e-7)
        np.testing.assert_allclose(action[[6, 13]], [1.0, 0.0], atol=1e-7)
        self.assertFalse(
            triggers_match_grippers(
                np.asarray([0.0, 0.0]),
                np.asarray([1.0, 0.0]),
                tolerance=0.01,
            )
        )


class CollectorArgumentTest(unittest.TestCase):
    def test_collection_defaults(self) -> None:
        args = build_parser().parse_args(["smolvla_corrections_round01"])

        self.assertEqual(args.default_seed, 2000)
        self.assertEqual(args.execution_horizon, 10)
        self.assertEqual(args.max_translation_step, 0.015)
        self.assertEqual(args.max_rotation_step, 0.10)
        self.assertEqual(args.camera_preview, "multiview")


if __name__ == "__main__":
    unittest.main()
