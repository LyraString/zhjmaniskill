from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.FPO import deploy_maniskill as fpo_deploy
from policy.FlowMatching import deploy_maniskill as fm_deploy
from policy.ResFiT import deploy_maniskill as resfit_deploy


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

    def test_resfit_encode_obs(self) -> None:
        obs = resfit_deploy.encode_obs(
            {
                "head_point_cloud": np.ones((1024, 3), dtype=np.float32),
                "left_wrist_point_cloud": np.ones((512, 3), dtype=np.float32),
                "right_wrist_point_cloud": np.ones((512, 3), dtype=np.float32),
                "agent_pos": np.arange(14, dtype=np.float32),
            }
        )
        self.assertEqual(
            sorted(obs.keys()),
            [
                "agent_pos",
                "head_point_cloud",
                "left_wrist_point_cloud",
                "right_wrist_point_cloud",
            ],
        )
        self.assertEqual(obs["head_point_cloud"].shape, (1024, 3))
        self.assertEqual(obs["agent_pos"].shape, (14,))

    def test_action_step_override_updates_policy_and_runner(self) -> None:
        cfg = SimpleNamespace(n_obs_steps=3, n_action_steps=1)
        for deploy in (fm_deploy, fpo_deploy):
            with self.subTest(deploy=deploy.__name__), tempfile.NamedTemporaryFile() as ckpt:
                loaded_model = SimpleNamespace(policy=SimpleNamespace(n_action_steps=1))
                with (
                    mock.patch.object(deploy.torch, "load", return_value={"cfg": cfg}),
                    mock.patch.object(deploy, "DP", return_value=loaded_model) as dp_cls,
                ):
                    model = deploy.get_model(
                        {
                            "ckpt_path": ckpt.name,
                            "n_action_steps": 4,
                        }
                    )

                dp_cls.assert_called_once_with(
                    ckpt.name,
                    n_obs_steps=3,
                    n_action_steps=4,
                )
                self.assertEqual(model.policy.n_action_steps, 4)

    def test_resfit_get_model_receives_environment_action_bounds(self) -> None:
        cfg = SimpleNamespace(n_obs_steps=3, n_action_steps=1)
        action_low = np.full(14, -1.0, dtype=np.float32)
        action_high = np.full(14, 1.0, dtype=np.float32)
        with tempfile.NamedTemporaryFile() as ckpt:
            loaded_model = SimpleNamespace()
            with (
                mock.patch.object(
                    resfit_deploy.torch,
                    "load",
                    return_value={"cfg": cfg},
                ),
                mock.patch.object(
                    resfit_deploy,
                    "DP",
                    return_value=loaded_model,
                ) as dp_cls,
            ):
                model = resfit_deploy.get_model(
                    {
                        "ckpt_path": ckpt.name,
                        "n_action_steps": 4,
                        "action_low": action_low,
                        "action_high": action_high,
                    }
                )

        dp_cls.assert_called_once_with(
            ckpt.name,
            n_obs_steps=3,
            n_action_steps=4,
            action_low=action_low,
            action_high=action_high,
        )
        self.assertIs(model, loaded_model)


if __name__ == "__main__":
    unittest.main()
