"""Raw point-cloud units, filtering and offline/online input consistency."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy/DP3/3D-Diffusion-Policy"))
from diffusion_policy_3d.common.maniskill_pointcloud import encode_raw_observation
from policy.DP3 import deploy_maniskill
from script.eval_maniskill_policy import _encode_policy_observation


class RawVRPointCloudTest(unittest.TestCase):
    def setUp(self):
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [0.1, 0.2, 0.8]
        self.raw = {
            "sensor_data": {"head_camera": {
                "position": np.array([[[0, 0, 0], [100, 0, 0], [2000, 0, 0],
                                       [0, 500, 0]]], dtype=np.int16),
                "segmentation": np.array([[1, 1, 1, 0]]),
            }},
            "sensor_param": {"head_camera": {"cam2world_gl": transform}},
            "agent": {"qpos": np.arange(14, dtype=np.float32)},
        }
        self.pre = {"cameras": ["head_camera"], "points": 3,
                    "crop_min": [-0.6, -0.35, 0.7401], "crop_max": [0.6, 0.35, 2]}

    def test_mm_world_transform_crop_mask_and_zero_padding(self):
        obs = encode_raw_observation(self.raw, self.pre, device="cpu")
        actual = sorted(map(tuple, obs["point_cloud"]))
        expected = sorted([(0., 0., 0.), (.1, .2, .8), (.2, .2, .8)])
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_online_uses_raw_cameras_not_preprocessed_observation(self):
        expected = encode_raw_observation(self.raw, self.pre, device="cpu")
        # Online sensors have a leading environment dimension and tensors.
        online = {
            "sensor_data": {"head_camera": {k: torch.as_tensor(v).unsqueeze(0)
                for k, v in self.raw["sensor_data"]["head_camera"].items()}},
            "sensor_param": {"head_camera": {"cam2world_gl": torch.as_tensor(
                self.raw["sensor_param"]["head_camera"]["cam2world_gl"]).unsqueeze(0)}},
            "agent": {"qpos": torch.as_tensor(self.raw["agent"]["qpos"]).unsqueeze(0)},
        }
        result = _encode_policy_observation(deploy_maniskill,
            SimpleNamespace(last_raw_obs=online), SimpleNamespace(data={}),
            SimpleNamespace(preprocessing=self.pre, policy=SimpleNamespace(device="cpu")))
        for key in expected:
            np.testing.assert_array_equal(result[key], expected[key])

    def test_empty_cloud_has_correct_shape(self):
        self.raw["sensor_data"]["head_camera"]["segmentation"][:] = 0
        result = encode_raw_observation(self.raw, self.pre, device="cpu")
        np.testing.assert_array_equal(result["point_cloud"], np.zeros((3, 3)))

    def test_adapter_exposes_and_resets_observation_cache(self):
        model = deploy_maniskill.ManiSkillDP3(
            policy=None,
            n_obs_steps=3,
            n_action_steps=6,
            preprocessing=self.pre,
        )
        self.assertEqual(len(model.obs_cache), 0)
        model.update_obs({"point_cloud": np.zeros((3, 3)), "agent_pos": np.zeros(14)})
        self.assertEqual(len(model.obs_cache), 1)
        deploy_maniskill.reset_model(model)
        self.assertEqual(len(model.obs_cache), 0)


if __name__ == "__main__":
    unittest.main()
