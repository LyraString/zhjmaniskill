from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.converters import convert_maniskill_episodes_to_zarr
from data.converters import convert_maniskill_npz_dir_to_zarr


class ManiSkillConverterTest(unittest.TestCase):
    def test_convert_maniskill_episodes_to_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "maniskill_pickcube.zarr"

            episodes = [
                {
                    "observations": {
                        "point_cloud": np.arange(3 * 512 * 4, dtype=np.float32).reshape(3, 512, 4),
                        "agent_pos": np.array(
                            [
                                [0.0, 0.1],
                                [1.0, 1.1],
                                [2.0, 2.1],
                            ],
                            dtype=np.float32,
                        ),
                    },
                    "actions": np.array(
                        [
                            [10.0, 10.1, 10.2],
                            [11.0, 11.1, 11.2],
                        ],
                        dtype=np.float32,
                    ),
                },
                {
                    "observations": {
                        "point_cloud": np.arange(2 * 512 * 3, dtype=np.float32).reshape(2, 512, 3),
                        "agent_pos": np.array(
                            [
                                [3.0, 3.1],
                                [4.0, 4.1],
                            ],
                            dtype=np.float32,
                        ),
                    },
                    "actions": np.array(
                        [
                            [12.0, 12.1, 12.2],
                        ],
                        dtype=np.float32,
                    ),
                },
            ]

            out_path = convert_maniskill_episodes_to_zarr(
                episodes=episodes,
                save_dir=save_dir,
            )

            root = zarr.open(str(out_path), mode="r")

            self.assertEqual(root["data"]["point_cloud"].shape, (3, 512, 3))
            self.assertEqual(root["data"]["state"].shape, (3, 2))
            self.assertEqual(root["data"]["action"].shape, (3, 3))

            np.testing.assert_allclose(
                root["data"]["state"][:],
                np.array(
                    [
                        [0.0, 0.1],
                        [1.0, 1.1],
                        [3.0, 3.1],
                    ],
                    dtype=np.float32,
                ),
            )
            np.testing.assert_allclose(
                root["data"]["action"][:],
                np.array(
                    [
                        [10.0, 10.1, 10.2],
                        [11.0, 11.1, 11.2],
                        [12.0, 12.1, 12.2],
                    ],
                    dtype=np.float32,
                ),
            )
            np.testing.assert_array_equal(
                root["meta"]["episode_ends"][:],
                np.array([2, 3], dtype=np.int64),
            )

    def test_convert_maniskill_npz_dir_to_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            load_dir = tmp_path / "maniskill"
            (load_dir / "data").mkdir(parents=True)
            save_dir = tmp_path / "maniskill_pickcube.zarr"

            np.savez(
                load_dir / "data" / "episode0.npz",
                point_cloud=np.arange(3 * 512 * 4, dtype=np.float32).reshape(3, 512, 4),
                agent_pos=np.array(
                    [
                        [0.0, 0.1],
                        [1.0, 1.1],
                        [2.0, 2.1],
                    ],
                    dtype=np.float32,
                ),
                actions=np.array(
                    [
                        [10.0, 10.1, 10.2],
                        [11.0, 11.1, 11.2],
                    ],
                    dtype=np.float32,
                ),
            )

            out_path = convert_maniskill_npz_dir_to_zarr(
                load_dir=load_dir,
                save_dir=save_dir,
                num_episodes=1,
            )

            root = zarr.open(str(out_path), mode="r")
            self.assertEqual(root["data"]["point_cloud"].shape, (2, 512, 3))
            self.assertEqual(root["data"]["state"].shape, (2, 2))
            self.assertEqual(root["data"]["action"].shape, (2, 3))
            np.testing.assert_array_equal(
                root["meta"]["episode_ends"][:],
                np.array([2], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
