from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from sim.maniskill.tasks.base_bimanual_env import BaseBimanualEnv


class _CountingPose:
    def __init__(self, raw_pose: torch.Tensor) -> None:
        self._raw_pose = raw_pose
        self.read_count = 0

    @property
    def raw_pose(self) -> torch.Tensor:
        self.read_count += 1
        return self._raw_pose


class AbsolutePoseSemanticsTest(unittest.TestCase):
    def _fake_env(self, left_pose: torch.Tensor, right_pose: torch.Tensor):
        left = _CountingPose(left_pose)
        right = _CountingPose(right_pose)
        env = SimpleNamespace(
            ACTION_DIM=14,
            num_envs=int(left_pose.shape[0]),
            device=torch.device("cpu"),
            left_delta_pose_controller=SimpleNamespace(ee_pose_at_base=left),
            right_delta_pose_controller=SimpleNamespace(ee_pose_at_base=right),
        )
        return env, left, right

    def test_chunk_targets_are_fixed_to_one_anchor_snapshot(self) -> None:
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        env, left_pose, right_pose = self._fake_env(
            torch.cat([torch.tensor([[10.0, 0.0, 0.0]]), identity[None]], dim=1),
            torch.cat([torch.tensor([[20.0, 0.0, 0.0]]), identity[None]], dim=1),
        )
        chunk = np.zeros((3, 14), dtype=np.float32)
        chunk[:, 0] = [2.0, 5.0, 3.0]
        chunk[:, 7] = [-2.0, -5.0, -3.0]
        chunk[:, 6] = [0.0, 0.5, 1.0]
        chunk[:, 13] = [1.0, 0.5, 0.0]

        targets = BaseBimanualEnv.prepare_action_chunk(env, chunk)

        np.testing.assert_allclose(targets[:, 0], [12.0, 15.0, 13.0])
        np.testing.assert_allclose(targets[:, 7], [18.0, 15.0, 17.0])
        np.testing.assert_allclose(targets[:, 6], chunk[:, 6])
        np.testing.assert_allclose(targets[:, 13], chunk[:, 13])
        self.assertEqual(left_pose.read_count, 1)
        self.assertEqual(right_pose.read_count, 1)

        left_pose._raw_pose[0, 0] = 100.0
        right_pose._raw_pose[0, 0] = 200.0
        np.testing.assert_allclose(targets[:, 0], [12.0, 15.0, 13.0])
        np.testing.assert_allclose(targets[:, 7], [18.0, 15.0, 17.0])

    def test_rotation_is_root_aligned_delta_times_anchor(self) -> None:
        left_anchor_euler = torch.tensor([[0.2, -0.1, 0.3]])
        right_anchor_euler = torch.tensor([[-0.1, 0.3, -0.2]])
        left_anchor_rotation = euler_angles_to_matrix(left_anchor_euler, "XYZ")
        right_anchor_rotation = euler_angles_to_matrix(right_anchor_euler, "XYZ")
        left_anchor = torch.cat(
            [torch.zeros((1, 3)), matrix_to_quaternion(left_anchor_rotation)],
            dim=1,
        )
        right_anchor = torch.cat(
            [torch.ones((1, 3)), matrix_to_quaternion(right_anchor_rotation)],
            dim=1,
        )
        env, _, _ = self._fake_env(left_anchor, right_anchor)
        chunk = torch.zeros((1, 2, 14), dtype=torch.float32)
        chunk[0, :, 3:6] = torch.tensor(
            [[0.1, 0.05, -0.02], [-0.04, 0.02, 0.08]]
        )
        chunk[0, :, 10:13] = torch.tensor(
            [[-0.03, 0.06, 0.01], [0.02, -0.05, 0.04]]
        )

        targets = BaseBimanualEnv.prepare_action_chunk(env, chunk)

        left_target_rotation = euler_angles_to_matrix(targets[0, :, 3:6], "XYZ")
        right_target_rotation = euler_angles_to_matrix(targets[0, :, 10:13], "XYZ")
        expected_left = torch.matmul(
            euler_angles_to_matrix(chunk[0, :, 3:6], "XYZ"),
            left_anchor_rotation[:, None][0],
        )
        expected_right = torch.matmul(
            euler_angles_to_matrix(chunk[0, :, 10:13], "XYZ"),
            right_anchor_rotation[:, None][0],
        )
        torch.testing.assert_close(left_target_rotation, expected_left)
        torch.testing.assert_close(right_target_rotation, expected_right)

    def test_selected_vector_envs_use_their_own_anchors(self) -> None:
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, -1)
        left_anchor = torch.cat(
            [
                torch.tensor(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
                ),
                identity,
            ],
            dim=1,
        )
        right_anchor = torch.cat(
            [
                torch.tensor(
                    [[11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]]
                ),
                identity,
            ],
            dim=1,
        )
        env, _, _ = self._fake_env(left_anchor, right_anchor)
        chunk = torch.zeros((2, 1, 14), dtype=torch.float32)
        chunk[:, 0, 0] = torch.tensor([0.5, -0.5])
        chunk[:, 0, 7] = torch.tensor([1.0, -1.0])

        targets = BaseBimanualEnv.prepare_action_chunk(
            env,
            chunk,
            env_ids=[2, 0],
        )

        torch.testing.assert_close(targets[:, 0, 0], torch.tensor([3.5, 0.5]))
        torch.testing.assert_close(targets[:, 0, 7], torch.tensor([14.0, 10.0]))

    def test_absolute_controller_configuration(self) -> None:
        controller_instance = mock.Mock()
        embodiment = SimpleNamespace(
            joint_drive_params=lambda _arm_index: (1000.0, 200.0)
        )
        env = SimpleNamespace(
            ABSOLUTE_EE_POSITION_LIMIT=2.0,
            ABSOLUTE_EE_ROTATION_LIMIT=np.pi,
            DELTA_JOINT_STIFFNESS_SCALE=8.0,
            DELTA_JOINT_DAMPING_SCALE=3.0,
            embodiment_spec=embodiment,
            get_arm_planner_assets=lambda _arm_index: {"urdf_path": "robot.urdf"},
            get_arm_joint_names=lambda _arm_index: ["joint"],
            get_arm_move_group=lambda _arm_index: "ee_link",
            get_arm_robot=lambda _arm_index: object(),
            _control_freq=20,
            _sim_freq=200,
            scene=object(),
        )
        with mock.patch(
            "sim.maniskill.tasks.base_bimanual_env._StrictPDEEPoseController",
            return_value=controller_instance,
        ) as controller_cls:
            result = BaseBimanualEnv._build_delta_pose_controller(env, arm_index=0)

        config = controller_cls.call_args.kwargs["config"]
        self.assertFalse(config.use_delta)
        self.assertFalse(config.use_target)
        self.assertEqual(config.frame, "root_translation:root_aligned_body_rotation")
        self.assertEqual(config.pos_lower, -2.0)
        self.assertEqual(config.pos_upper, 2.0)
        self.assertEqual(config.rot_lower, -np.pi)
        self.assertEqual(config.rot_upper, np.pi)
        controller_instance.set_drive_property.assert_called_once_with()
        self.assertIs(result, controller_instance)


if __name__ == "__main__":
    unittest.main()
