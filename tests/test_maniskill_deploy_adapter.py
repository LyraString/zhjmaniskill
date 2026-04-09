from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.FPO import deploy_maniskill as fpo_deploy
from policy.FlowMatching import deploy_maniskill as fm_deploy


class ManiSkillDeployAdapterTest(unittest.TestCase):
    def test_flowmatching_encode_obs(self) -> None:
        obs = fm_deploy.encode_obs(
            {
                "point_cloud": np.ones((512, 3), dtype=np.float32),
                "agent_pos": np.arange(9, dtype=np.float32),
            }
        )
        self.assertEqual(sorted(obs.keys()), ["agent_pos", "point_cloud"])
        self.assertEqual(obs["point_cloud"].shape, (512, 3))
        self.assertEqual(obs["agent_pos"].shape, (9,))

    def test_fpo_encode_obs(self) -> None:
        obs = fpo_deploy.encode_obs(
            {
                "point_cloud": np.ones((512, 3), dtype=np.float32),
                "agent_pos": np.arange(9, dtype=np.float32),
            }
        )
        self.assertEqual(sorted(obs.keys()), ["agent_pos", "point_cloud"])
        self.assertEqual(obs["point_cloud"].shape, (512, 3))
        self.assertEqual(obs["agent_pos"].shape, (9,))


if __name__ == "__main__":
    unittest.main()
