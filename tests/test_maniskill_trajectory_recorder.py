from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill import ManiSkillTrajectoryRecorder


class ManiSkillTrajectoryRecorderTest(unittest.TestCase):
    def test_save_episode_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ManiSkillTrajectoryRecorder(Path(tmpdir) / "pickcube_smoke")
            recorder.start_episode(
                {
                    "point_cloud": np.ones((512, 3), dtype=np.float32),
                    "agent_pos": np.array([0.0, 0.1], dtype=np.float32),
                }
            )
            recorder.append_step(
                np.array([1.0, 1.1, 1.2], dtype=np.float32),
                {
                    "point_cloud": np.full((512, 3), 2.0, dtype=np.float32),
                    "agent_pos": np.array([1.0, 1.1], dtype=np.float32),
                },
            )
            recorder.append_step(
                np.array([2.0, 2.1, 2.2], dtype=np.float32),
                {
                    "point_cloud": np.full((512, 3), 3.0, dtype=np.float32),
                    "agent_pos": np.array([2.0, 2.1], dtype=np.float32),
                },
            )

            out_path = recorder.save_episode(0)
            with np.load(out_path, allow_pickle=False) as data:
                self.assertEqual(data["point_cloud"].shape, (3, 512, 3))
                self.assertEqual(data["agent_pos"].shape, (3, 2))
                self.assertEqual(data["actions"].shape, (2, 3))
                np.testing.assert_array_equal(
                    data["agent_pos"],
                    np.array(
                        [
                            [0.0, 0.1],
                            [1.0, 1.1],
                            [2.0, 2.1],
                        ],
                        dtype=np.float32,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
