from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import (
    ManiSkillTaskFactory,
    convert_action_to_maniskill,
    convert_maniskill_obs,
)


class ManiSkillAdapterSmokeTest(unittest.TestCase):
    def test_convert_action_to_maniskill(self) -> None:
        action = ActionSpec.from_value([1, 2, 3], action_type="continuous")
        action_np = convert_action_to_maniskill(action)
        self.assertIsInstance(action_np, np.ndarray)
        self.assertEqual(action_np.dtype, np.float32)
        np.testing.assert_allclose(action_np, np.array([1, 2, 3], dtype=np.float32))

    def test_convert_maniskill_obs_aliases_point_cloud(self) -> None:
        raw_obs = {
            "pointcloud": np.array(
                [[1, 2, 3, 9, 9, 9], [4, 5, 6, 9, 9, 9]],
                dtype=np.float32,
            ),
            "state": np.array([0.1, 0.2], dtype=np.float32),
        }
        obs = convert_maniskill_obs(raw_obs, metadata={"env_id": "PickCube-v1"})
        self.assertIn("pointcloud", obs.data)
        self.assertIn("point_cloud", obs.data)
        np.testing.assert_allclose(
            obs.data["point_cloud"],
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        )
        self.assertEqual(obs.metadata["env_id"], "PickCube-v1")

    def test_task_factory_passes_fixed_rate_kwargs(self) -> None:
        factory = ManiSkillTaskFactory()
        with patch("sim.maniskill.task_factory.ManiSkillEnvBackend") as backend_cls:
            backend_cls.return_value = object()
            factory.create_env(
                task_name="PickCube-v1",
                action_type="continuous",
                obs_mode="state",
                sim_freq=200,
                control_freq=40,
                sim_config={"spacing": 3},
                render_mode=None,
            )

        backend_cls.assert_called_once_with(
            env_id="PickCube-v1",
            obs_mode="state",
            control_mode=None,
            render_mode=None,
            sim_freq=200,
            control_freq=40,
            sim_config={"spacing": 3},
            env_kwargs={},
        )


if __name__ == "__main__":
    unittest.main()
