from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill.runtime_env import ManiSkillEnvBackend
from sim.maniskill.task_profiles import apply_task_profile
from sim.maniskill.teleop.canonical_dataset import (
    CanonicalTeleopDatasetRecorder,
    DEFAULT_CAMERA_NAMES,
)
from sim.maniskill.teleop.pose_mapper import (
    DEFAULT_XR_TO_WORLD_ROTATION,
    BimanualXRMapper,
)
from sim.maniskill.teleop.xr_client import XRState
from script.teleop_maniskill_vr import (
    _MultiviewCameraPreview,
    _is_recoverable_pdee_ik_failure,
    _render_human_if_open,
    build_parser,
)
from script.convert_vr_teleop_dataset import build_parser as build_converter_parser
from script.convert_vr_teleop_dataset import convert_dataset_to_zarr


def _xr_state(
    *,
    left_position=(0.0, 0.0, 0.0),
    left_rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    grip=(0.0, 0.0),
    trigger=(0.0, 0.0),
) -> XRState:
    poses = np.asarray(
        [
            [*left_position, *left_rotation_xyzw],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return XRState(
        timestamp_ns=1,
        controller_pose_xyzw=poses,
        grip=np.asarray(grip),
        trigger=np.asarray(trigger),
        axis=np.zeros((2, 2)),
        buttons=np.zeros(8, dtype=np.bool_),
    )


class BimanualXRMapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = BimanualXRMapper(xr_to_world_rotation=np.eye(3))
        self.root_pose = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        self.ee_pose = np.asarray(
            [
                [0.2, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        self.mapper.reset(self.ee_pose)

    def test_clutch_translation_is_bounded_and_release_holds_target(self) -> None:
        self.mapper.update_targets(
            _xr_state(grip=(1.0, 0.0)),
            root_pose_world=self.root_pose,
            current_ee_pose_root=self.ee_pose,
        )
        desired = self.mapper.update_targets(
            _xr_state(left_position=(0.03, 0.0, 0.0), grip=(1.0, 0.0)),
            root_pose_world=self.root_pose,
            current_ee_pose_root=self.ee_pose,
        )
        self.assertAlmostEqual(float(desired[0, 0]), 0.23, places=6)
        action = self.mapper.build_delta_action(
            current_ee_pose_root=self.ee_pose,
            trigger=np.asarray([0.0, 1.0]),
        )
        self.assertAlmostEqual(float(np.linalg.norm(action[:3])), 0.015, places=6)
        self.assertEqual(float(action[6]), 1.0)
        self.assertEqual(float(action[13]), 0.0)

        advanced_pose = self.ee_pose.copy()
        advanced_pose[0, 0] += 0.015
        held = self.mapper.update_targets(
            _xr_state(left_position=(0.5, 0.0, 0.0), grip=(0.0, 0.0)),
            root_pose_world=self.root_pose,
            current_ee_pose_root=advanced_pose,
        )
        self.assertAlmostEqual(float(held[0, 0]), 0.23, places=6)
        held_action = self.mapper.build_delta_action(
            current_ee_pose_root=advanced_pose,
            trigger=np.zeros(2),
        )
        self.assertAlmostEqual(float(held_action[0]), 0.015, places=6)

    def test_rotation_delta_uses_radial_limit(self) -> None:
        self.mapper.update_targets(
            _xr_state(grip=(1.0, 0.0)),
            root_pose_world=self.root_pose,
            current_ee_pose_root=self.ee_pose,
        )
        controller_quat = Rotation.from_euler("z", 0.2).as_quat()
        self.mapper.update_targets(
            _xr_state(left_rotation_xyzw=controller_quat, grip=(1.0, 0.0)),
            root_pose_world=self.root_pose,
            current_ee_pose_root=self.ee_pose,
        )
        action = self.mapper.build_delta_action(
            current_ee_pose_root=self.ee_pose,
            trigger=np.zeros(2),
        )
        self.assertAlmostEqual(float(np.linalg.norm(action[3:6])), 0.10, places=6)

    def test_default_mapping_matches_maniskill_task_axes(self) -> None:
        np.testing.assert_allclose(
            DEFAULT_XR_TO_WORLD_ROTATION @ np.asarray([1.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            DEFAULT_XR_TO_WORLD_ROTATION @ np.asarray([0.0, 0.0, -1.0]),
            [0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            DEFAULT_XR_TO_WORLD_ROTATION @ np.asarray([0.0, 1.0, 0.0]),
            [0.0, 0.0, 1.0],
        )


class ManiSkillRawStepTest(unittest.TestCase):
    def test_raw_step_updates_observation_without_learner_conversion(self) -> None:
        backend = ManiSkillEnvBackend.__new__(ManiSkillEnvBackend)
        backend.env = MagicMock()
        backend._last_raw_obs = None
        backend._fixed_rate_info = {"sim_freq": 200, "control_freq": 20}
        raw_obs = {"sensor_data": {"head_camera": {}}}
        backend.env.step.return_value = (
            raw_obs,
            np.float32(0.5),
            np.bool_(False),
            np.bool_(True),
            {"success": False},
        )
        action = ActionSpec.from_value(np.zeros(14, dtype=np.float32))

        with patch(
            "sim.maniskill.runtime_env.convert_action_to_maniskill",
            return_value=np.ones(14, dtype=np.float32),
        ) as convert_action:
            result = backend.step_raw(action)

        convert_action.assert_called_once_with(action)
        np.testing.assert_array_equal(
            backend.env.step.call_args.args[0],
            np.ones(14, dtype=np.float32),
        )
        self.assertIs(result[0], raw_obs)
        self.assertIs(backend.last_raw_obs, raw_obs)
        self.assertEqual(result[1:4], (0.5, False, True))
        self.assertEqual(result[4]["control_freq"], 20)


class TeleopIKRecoveryTest(unittest.TestCase):
    def test_only_explicit_pdee_ik_failures_are_recoverable(self) -> None:
        self.assertTrue(
            _is_recoverable_pdee_ik_failure(
                RuntimeError("PDEE inverse kinematics failed.")
            )
        )
        self.assertTrue(
            _is_recoverable_pdee_ik_failure(
                RuntimeError(
                    "PDEE inverse kinematics returned a non-finite joint target."
                )
            )
        )
        self.assertFalse(
            _is_recoverable_pdee_ik_failure(RuntimeError("dataset writer failed"))
        )


class HumanViewerLifecycleTest(unittest.TestCase):
    def test_closed_viewer_is_not_rendered_again(self) -> None:
        env = MagicMock()
        base_env = MagicMock()
        base_env._viewer.closed = True

        self.assertFalse(_render_human_if_open(env, base_env))
        env.render.assert_not_called()

    def test_viewer_closed_by_render_is_reported(self) -> None:
        env = MagicMock()
        base_env = MagicMock()
        base_env._viewer.closed = False

        def close_viewer() -> None:
            base_env._viewer.closed = True

        env.render.side_effect = close_viewer
        self.assertFalse(_render_human_if_open(env, base_env))


class TeleopArgumentDefaultsTest(unittest.TestCase):
    def test_default_session_duration_and_operator_view(self) -> None:
        args = build_parser().parse_args(["test_dataset"])

        self.assertEqual(args.max_episode_steps, 72_000)
        self.assertEqual(tuple(args.viewer_eye), (0.0, -0.45, 1.90))
        self.assertEqual(tuple(args.viewer_target), (0.0, -0.05, 0.75))
        self.assertEqual(args.camera_preview, "multiview")
        self.assertEqual(args.camera_preview_every, 2)
        self.assertEqual(args.camera_preview_scale, 1.5)
        self.assertIsNone(args.seed)
        self.assertEqual(
            args.task_profile,
            "handover_block_ur5_v2_vr_robotwin_clean",
        )

    def test_explicit_seed_overrides_automatic_seed_selection(self) -> None:
        args = build_parser().parse_args(["test_dataset", "--seed", "123"])

        self.assertEqual(args.seed, 123)

    def test_robotwin_clean_profile_preserves_original_task_ranges(self) -> None:
        env_id, runtime, profile_name = apply_task_profile(
            "handover_block_ur5_v2_vr_robotwin_clean",
            {},
        )

        self.assertEqual(env_id, "HandoverBlockUR5-v2")
        self.assertEqual(profile_name, "handover_block_ur5_v2_vr_robotwin_clean")
        self.assertEqual(runtime["control_mode"], "bimanual_pd_ee_delta_pose")
        self.assertEqual(runtime["wrist_camera_fovy_deg"], 37.0)
        self.assertEqual(runtime["initial_gripper_value"], 1.0)
        self.assertEqual(runtime["box_x_range_override"], [-0.25, -0.05])
        self.assertEqual(runtime["box_y_range_override"], [0.0, 0.25])
        self.assertEqual(runtime["box_yaw_limit_override"], 0.2)
        self.assertEqual(runtime["target_x_range_override"], [0.1, 0.25])
        self.assertEqual(runtime["target_y_range_override"], [0.15, 0.2])


class MultiviewCameraPreviewTest(unittest.TestCase):
    def test_three_rgb_views_are_shown_side_by_side_in_bgr_order(self) -> None:
        cv2_module = MagicMock()
        cv2_module.WINDOW_NORMAL = 0
        cv2_module.FONT_HERSHEY_SIMPLEX = 0
        cv2_module.LINE_AA = 0
        cv2_module.INTER_LINEAR = 0
        raw_obs = {
            "sensor_data": {
                camera_name: {
                    "rgb": np.full((1, 2, 3, 3), index, dtype=np.uint8)
                }
                for index, camera_name in enumerate(
                    ("head_camera", "left_wrist_camera", "right_wrist_camera")
                )
            }
        }
        raw_obs["sensor_data"]["head_camera"]["rgb"][0, 0, 0] = [1, 2, 3]
        preview = _MultiviewCameraPreview(scale=1.0, cv2_module=cv2_module)

        self.assertTrue(preview.show(raw_obs))

        displayed = cv2_module.imshow.call_args.args[1]
        self.assertEqual(displayed.shape, (2, 9, 3))
        np.testing.assert_array_equal(displayed[0, 0], [3, 2, 1])
        self.assertEqual(cv2_module.putText.call_count, 3)
        cv2_module.waitKey.assert_called_once_with(1)


def _observation(frame_value: int) -> dict:
    cameras = {}
    for camera_index, camera_name in enumerate(DEFAULT_CAMERA_NAMES):
        cameras[camera_name] = {
            "rgb": np.full((4, 6, 3), frame_value + camera_index, dtype=np.uint8),
            "position_mm": np.full(
                (4, 6, 3), frame_value + camera_index, dtype=np.int16
            ),
            "segmentation_id": np.full((4, 6), camera_index, dtype=np.int16),
            "intrinsic_cv": np.eye(3, dtype=np.float32),
            "extrinsic_cv": np.zeros((3, 4), dtype=np.float32),
            "cam2world_gl": np.eye(4, dtype=np.float32),
        }
    return {
        "host_monotonic_ns": np.int64(frame_value),
        "cameras": cameras,
        "robot": {
            "agent_qpos": np.zeros(14, dtype=np.float32),
            "ee_pose_root_wxyz": np.zeros((2, 7), dtype=np.float32),
        },
        "task": {"box_pose": np.zeros(7, dtype=np.float32)},
    }


class CanonicalTeleopDatasetRecorderTest(unittest.TestCase):
    def test_next_seed_uses_only_completed_episode_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = CanonicalTeleopDatasetRecorder(Path(tmpdir) / "dataset")
            self.assertEqual(recorder.next_seed(), 0)

            for episode_index, seed in ((0, 3), (4, 11)):
                episode_dir = recorder.episodes_dir / f"episode_{episode_index:06d}"
                episode_dir.mkdir()
                (episode_dir / "episode.json").write_text(
                    json.dumps({"seed": seed}),
                    encoding="utf-8",
                )
            malformed_dir = recorder.episodes_dir / "episode_000005"
            malformed_dir.mkdir()
            (malformed_dir / "episode.json").write_text("not json", encoding="utf-8")
            partial_dir = recorder.episodes_dir / ".episode_000006.partial"
            partial_dir.mkdir()
            (partial_dir / "episode.json").write_text(
                json.dumps({"seed": 999}),
                encoding="utf-8",
            )

            self.assertEqual(recorder.next_seed(), 12)

    def test_streamed_episode_preserves_geometry_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = CanonicalTeleopDatasetRecorder(
                Path(tmpdir) / "dataset",
                fps=20,
                rgb_crf=18,
                rgb_preset="ultrafast",
            )
            recorder.start_episode(_observation(10), metadata={"seed": 3})
            recorder.append_step(
                action={
                    "control": {
                        "delta_pose_actual": np.zeros(14, dtype=np.float32),
                    }
                },
                transition={"reward": np.float32(1.0), "success": np.bool_(False)},
                next_observation=_observation(20),
            )
            episode_dir = recorder.save_episode(final_metadata={"success": False})

            with h5py.File(episode_dir / "data.hdf5", "r") as h5_file:
                self.assertEqual(int(h5_file.attrs["observation_count"]), 2)
                self.assertEqual(int(h5_file.attrs["action_count"]), 1)
                saved = np.asarray(
                    h5_file[
                        "observations/cameras/head_camera/position_mm"
                    ]
                )
                self.assertEqual(saved.dtype, np.int16)
                np.testing.assert_array_equal(saved[0], _observation(10)["cameras"]["head_camera"]["position_mm"])
                np.testing.assert_array_equal(saved[1], _observation(20)["cameras"]["head_camera"]["position_mm"])

            for camera_name in DEFAULT_CAMERA_NAMES:
                with av.open(str(episode_dir / f"{camera_name}.mp4")) as container:
                    self.assertEqual(sum(1 for _ in container.decode(video=0)), 2)

    def test_direct_zarr_conversion_preserves_observation_action_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "canonical"
            recorder = CanonicalTeleopDatasetRecorder(
                dataset_dir,
                fps=20,
                rgb_crf=18,
                rgb_preset="ultrafast",
            )
            initial_observation = _observation(10)
            next_observation = _observation(20)
            next_observation["robot"]["agent_qpos"][:] = 1.0
            action = np.zeros(14, dtype=np.float32)
            action[0] = 0.01
            recorder.start_episode(
                initial_observation,
                metadata={
                    "seed": 7,
                    "task_profile": "handover_block_ur5_v2_vr_robotwin_clean",
                    "sim_freq_hz": 200,
                    "control_freq_hz": 20,
                    "action_semantics": "actual_relative_delta_pose_euler_xyz",
                },
            )
            recorder.append_step(
                action={"control": {"delta_pose_actual": action}},
                transition={"reward": np.float32(1.0), "success": np.bool_(True)},
                next_observation=next_observation,
            )
            recorder.save_episode(final_metadata={"success": True})

            output_path = Path(tmpdir) / "converted.zarr"
            args = build_converter_parser().parse_args(
                [
                    str(dataset_dir),
                    "test",
                    "--output-format",
                    "zarr",
                    "--output-path",
                    str(output_path),
                ]
            )
            convert_dataset_to_zarr(args)

            root = zarr.open_group(str(output_path), mode="r")
            self.assertEqual(root["data/state"].shape, (1, 14))
            self.assertEqual(root["data/action"].shape, (1, 14))
            np.testing.assert_allclose(root["data/state"][0], 0.0)
            np.testing.assert_allclose(root["data/action"][0], action)
            np.testing.assert_array_equal(root["meta/episode_ends"][:], [1])
            np.testing.assert_array_equal(root["meta/episode_seeds"][:], [7])
            self.assertEqual(
                root.attrs["task_profile"],
                "handover_block_ur5_v2_vr_robotwin_clean",
            )

    def test_direct_zarr_conversion_can_trim_initial_hold_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "canonical"
            recorder = CanonicalTeleopDatasetRecorder(
                dataset_dir,
                fps=20,
                rgb_crf=18,
                rgb_preset="ultrafast",
            )
            metadata = {
                "seed": 9,
                "task_profile": "handover_block_ur5_v2_vr_robotwin_clean",
                "sim_freq_hz": 200,
                "control_freq_hz": 20,
                "action_semantics": "actual_relative_delta_pose_euler_xyz",
            }
            recorder.start_episode(_observation(10), metadata=metadata)
            for step, translation in enumerate((0.001, 0.006, 0.007), start=1):
                action = np.zeros(14, dtype=np.float32)
                action[0] = translation
                next_observation = _observation(10 + step)
                next_observation["robot"]["agent_qpos"][:] = float(step)
                recorder.append_step(
                    action={"control": {"delta_pose_actual": action}},
                    transition={"reward": np.float32(0.0), "success": np.bool_(False)},
                    next_observation=next_observation,
                )
            recorder.save_episode(final_metadata={"success": True})

            output_path = Path(tmpdir) / "trimmed.zarr"
            args = build_converter_parser().parse_args(
                [
                    str(dataset_dir),
                    "test",
                    "--output-format",
                    "zarr",
                    "--output-path",
                    str(output_path),
                    "--trim-idle-prefix",
                ]
            )
            convert_dataset_to_zarr(args)

            root = zarr.open_group(str(output_path), mode="r")
            self.assertEqual(root["data/state"].shape, (2, 14))
            np.testing.assert_allclose(root["data/state"][0], 1.0)
            np.testing.assert_allclose(root["data/action"][:, 0], [0.006, 0.007])
            np.testing.assert_array_equal(root["meta/episode_ends"][:], [2])
            np.testing.assert_array_equal(root["meta/episode_trim_starts"][:], [1])
            self.assertTrue(root.attrs["idle_prefix_trimmed"])


if __name__ == "__main__":
    unittest.main()
