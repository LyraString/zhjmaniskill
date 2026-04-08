from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.converters import convert_robotwin_hdf5_dir_to_zarr


class RobotwinHdf5ConverterTest(unittest.TestCase):
    def test_convert_robotwin_hdf5_dir_to_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            load_dir = tmp_path / "robotwin"
            (load_dir / "data").mkdir(parents=True)
            save_dir = tmp_path / "converted.zarr"

            vector = np.array(
                [
                    [0.0, 0.1],
                    [1.0, 1.1],
                    [2.0, 2.1],
                ],
                dtype=np.float32,
            )
            pointcloud = np.array(
                [
                    [[0, 1, 2, 9, 9, 9], [3, 4, 5, 9, 9, 9]],
                    [[6, 7, 8, 9, 9, 9], [9, 10, 11, 9, 9, 9]],
                    [[12, 13, 14, 9, 9, 9], [15, 16, 17, 9, 9, 9]],
                ],
                dtype=np.float32,
            )

            with h5py.File(load_dir / "data" / "episode0.hdf5", "w") as root:
                root.create_dataset("/joint_action/vector", data=vector)
                root.create_dataset("/pointcloud", data=pointcloud)

            out_path = convert_robotwin_hdf5_dir_to_zarr(
                load_dir=load_dir,
                save_dir=save_dir,
                num_episodes=1,
            )

            root = zarr.open(str(out_path), mode="r")
            np.testing.assert_allclose(root["data"]["state"][:], vector[:2])
            np.testing.assert_allclose(root["data"]["action"][:], vector[1:])
            np.testing.assert_allclose(
                root["data"]["point_cloud"][:],
                pointcloud[:2, :, :3],
            )
            np.testing.assert_array_equal(root["meta"]["episode_ends"][:], np.array([2]))


if __name__ == "__main__":
    unittest.main()
