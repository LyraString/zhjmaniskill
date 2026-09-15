from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.maniskill.obs_adapter import convert_maniskill_obs


class EmptyCameraTest(unittest.TestCase):
    def setUp(self):
        self.views = [
            {'camera': f'{name}_camera', 'key': f'{name}_point_cloud', 'points': 4}
            for name in ('head', 'left_wrist', 'right_wrist')
        ]
        position = np.zeros((1, 2, 2, 4), dtype=np.float32)
        position[..., 2] = 800.0
        self.raw = {
            'sensor_data': {v['camera']: {'position': position.copy()} for v in self.views},
            'sensor_param': {v['camera']: {'cam2world_gl': np.eye(4, dtype=np.float32)[None]}
                             for v in self.views},
        }
        self.metadata = {'env_id': 'HandoverBlockUR5-v2', 'point_cloud_views': self.views,
                         'point_cloud_sampling': 'fast'}

    def check_empty_left(self):
        data = convert_maniskill_obs(self.raw, self.metadata).data
        self.assertEqual(set(data), {v['key'] for v in self.views})
        self.assertEqual(data['left_wrist_point_cloud'].shape, (4, 3))
        self.assertEqual(data['left_wrist_point_cloud'].dtype, np.float32)
        np.testing.assert_array_equal(data['left_wrist_point_cloud'], np.zeros((4, 3)))
        np.testing.assert_allclose(data['head_point_cloud'], np.tile([0, 0, .8], (4, 1)))
        np.testing.assert_array_equal(data['head_point_cloud'], data['right_wrist_point_cloud'])

    def test_no_valid_depth_keeps_view_and_recovers_next_frame(self):
        original = copy.deepcopy(self.raw)
        self.raw['sensor_data']['left_wrist_camera']['position'][..., 3] = 1
        self.check_empty_left()
        data = convert_maniskill_obs(original, self.metadata).data
        np.testing.assert_array_equal(data['left_wrist_point_cloud'], data['head_point_cloud'])

    def test_no_segmented_depth_keeps_view(self):
        images = self.raw['sensor_data']['left_wrist_camera']
        images['position'] = images['position'][..., :3]
        images['segmentation'] = np.zeros((1, 2, 2, 1), dtype=np.int32)
        self.check_empty_left()

    def test_all_points_cropped_keeps_view(self):
        self.raw['sensor_data']['left_wrist_camera']['position'][..., 0] = 2000
        self.check_empty_left()

    def test_missing_camera_is_not_silently_padded(self):
        del self.raw['sensor_data']['left_wrist_camera']
        with self.assertRaisesRegex(ValueError, 'left_wrist_camera'):
            convert_maniskill_obs(self.raw, self.metadata)

    def test_missing_position_or_extrinsics_is_not_silently_padded(self):
        for container, key in [('sensor_data', 'position'), ('sensor_param', 'cam2world_gl')]:
            with self.subTest(key=key):
                raw = copy.deepcopy(self.raw)
                del raw[container]['left_wrist_camera'][key]
                with self.assertRaisesRegex(ValueError, 'left_wrist_camera'):
                    convert_maniskill_obs(raw, self.metadata)


if __name__ == '__main__':
    unittest.main()
