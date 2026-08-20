from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from pytorch3d.transforms import euler_angles_to_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
FM_ROOT = REPO_ROOT / "policy" / "fm_diffusion"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))

from diffusion_policy.dataset.robot_pointcloud_dataset import (
    RobotMultiViewPointCloudChunkAnchorDataset,
    _absolute_actions_to_anchor_relative,
)


class ManiSkillAnchorActionDatasetTest(unittest.TestCase):
    def _write_dataset(self, path: Path):
        episode_lengths = (6, 7)
        num_steps = sum(episode_lengths)
        home = np.asarray(
            [-1.5447, -1.5447, -1.5447, -1.5794, 1.5794, 0.0],
            dtype=np.float32,
        )
        state = np.zeros((num_steps, 14), dtype=np.float32)
        action = np.zeros((num_steps, 14), dtype=np.float32)
        start = 0
        for episode_index, length in enumerate(episode_lengths):
            for local_step in range(length):
                index = start + local_step
                state[index, :6] = home
                state[index, 7:13] = home
                state[index, 0] += 0.01 * local_step
                state[index, 8] -= 0.008 * local_step
                state[index, 6] = float(local_step % 2)
                state[index, 13] = float((local_step + 1) % 2)

                scale = float(local_step + 1 + 2 * episode_index)
                action[index, 0:3] = [0.001 * scale, -0.0005, 0.00025]
                action[index, 3:6] = [0.002, -0.001 * scale, 0.001]
                action[index, 6] = float(local_step % 2)
                action[index, 7:10] = [-0.00075, 0.0004 * scale, -0.0002]
                action[index, 10:13] = [-0.001, 0.0015, -0.001 * scale]
                action[index, 13] = float((local_step + 1) % 2)
            start += length

        root = zarr.group(str(path))
        data = root.create_group("data")
        meta = root.create_group("meta")
        data.create_dataset("state", data=state, chunks=(8, 14))
        data.create_dataset("action", data=action, chunks=(8, 14))
        data.create_dataset(
            "head_point_cloud",
            data=np.zeros((num_steps, 4, 3), dtype=np.float32),
            chunks=(4, 4, 3),
        )
        for key in ("left_wrist_point_cloud", "right_wrist_point_cloud"):
            data.create_dataset(
                key,
                data=np.zeros((num_steps, 2, 3), dtype=np.float32),
                chunks=(4, 2, 3),
            )
        meta.create_dataset(
            "episode_ends",
            data=np.cumsum(episode_lengths, dtype=np.int64),
        )
        return state, action

    def _make_dataset(self, path: Path, *, val_ratio: float = 0.0):
        return RobotMultiViewPointCloudChunkAnchorDataset(
            zarr_path=str(path),
            horizon=4,
            obs_steps=2,
            pad_before=1,
            pad_after=2,
            batch_size=2,
            val_ratio=val_ratio,
            normalize_point_clouds=False,
        )

    def test_absolute_targets_are_relative_to_one_chunk_anchor(self):
        anchor = torch.zeros((2, 12), dtype=torch.float32)
        anchor[0, 0] = 10.0
        anchor[1, 0] = 20.0
        anchor[:, 3:] = torch.eye(3).reshape(1, 9)
        targets = torch.zeros((3, 14), dtype=torch.float32)
        targets[:, 0] = torch.tensor([12.0, 15.0, 13.0])
        targets[:, 7] = torch.tensor([18.0, 15.0, 17.0])
        targets[:, 6] = torch.tensor([0.0, 0.5, 1.0])
        targets[:, 13] = torch.tensor([1.0, 0.5, 0.0])

        relative = _absolute_actions_to_anchor_relative(targets, anchor)

        torch.testing.assert_close(relative[:, 0], torch.tensor([2.0, 5.0, 3.0]))
        torch.testing.assert_close(relative[:, 7], torch.tensor([-2.0, -5.0, -3.0]))
        torch.testing.assert_close(relative[:, 6], targets[:, 6])
        torch.testing.assert_close(relative[:, 13], targets[:, 13])

    def test_scalar_batch_padding_and_normalizer_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "dataset.zarr"
            _, recorded_action = self._write_dataset(zarr_path)
            dataset = self._make_dataset(zarr_path)

            np.testing.assert_array_equal(
                zarr.open(str(zarr_path), mode="r")["data/action"][:],
                recorded_action,
            )

            checked_indices = [0, 1, 5, 6, len(dataset) - 1]
            expected_sources = {
                0: [0, 0, 1, 2],
                1: [0, 1, 2, 3],
                5: [4, 5, 5, 5],
                6: [6, 6, 7, 8],
                len(dataset) - 1: [11, 12, 12, 12],
            }
            for sample_index in checked_indices:
                source_indices = dataset._source_indices(sample_index)
                np.testing.assert_array_equal(
                    source_indices[0], expected_sources[sample_index]
                )
                anchor_global_index = source_indices[0, dataset.action_anchor_index]
                sample = dataset[np.int64(sample_index)]
                label = sample["action"]
                torch.testing.assert_close(
                    label[dataset.action_anchor_index],
                    torch.from_numpy(recorded_action[anchor_global_index]),
                    atol=2.0e-5,
                    rtol=2.0e-5,
                )

                anchor = torch.from_numpy(
                    dataset._actual_eef_pose[anchor_global_index]
                )
                absolute = torch.from_numpy(
                    dataset.replay_buffer["action"][source_indices[0]]
                )
                for arm_index, offset in enumerate((0, 7)):
                    recovered_position = (
                        anchor[arm_index, :3][None]
                        + label[:, offset : offset + 3]
                    )
                    torch.testing.assert_close(
                        recovered_position,
                        absolute[:, offset : offset + 3],
                        atol=2.0e-5,
                        rtol=2.0e-5,
                    )
                    anchor_rotation = anchor[arm_index, 3:].reshape(3, 3)
                    recovered_rotation = torch.matmul(
                        euler_angles_to_matrix(
                            label[:, offset + 3 : offset + 6], "XYZ"
                        ),
                        anchor_rotation,
                    )
                    absolute_rotation = euler_angles_to_matrix(
                        absolute[:, offset + 3 : offset + 6], "XYZ"
                    )
                    torch.testing.assert_close(
                        recovered_rotation,
                        absolute_rotation,
                        atol=2.0e-5,
                        rtol=2.0e-5,
                    )

            batch_indices = np.asarray([0, 6], dtype=np.int64)
            batch = dataset[batch_indices]
            for batch_row, sample_index in enumerate(batch_indices):
                torch.testing.assert_close(
                    batch["action"][batch_row],
                    dataset[int(sample_index)]["action"],
                    atol=2.0e-5,
                    rtol=2.0e-5,
                )
            processed = dataset.postprocess(batch, torch.device("cpu"))
            self.assertEqual(tuple(processed["action"].shape), (2, 4, 14))
            self.assertEqual(tuple(processed["obs"]["agent_pos"].shape), (2, 2, 14))

            labels = dataset._materialize_anchor_relative_actions()
            normalizer = dataset.get_normalizer()
            stats = normalizer["action"].get_input_stats()
            flat_labels = torch.from_numpy(labels.reshape(-1, 14))
            torch.testing.assert_close(stats["min"], flat_labels.min(dim=0).values)
            torch.testing.assert_close(stats["max"], flat_labels.max(dim=0).values)
            torch.testing.assert_close(stats["mean"], flat_labels.mean(dim=0))
            torch.testing.assert_close(stats["std"], flat_labels.std(dim=0))
            normalized = normalizer["action"].normalize(flat_labels)
            restored = normalizer["action"].unnormalize(normalized)
            torch.testing.assert_close(restored, flat_labels, atol=1.0e-6, rtol=1.0e-6)

    def test_validation_view_has_independent_batch_buffers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            zarr_path = Path(tmp_dir) / "dataset.zarr"
            self._write_dataset(zarr_path)
            dataset = self._make_dataset(zarr_path, val_ratio=0.5)
            val_dataset = dataset.get_validation_dataset()

            self.assertGreater(len(dataset), 0)
            self.assertGreater(len(val_dataset), 0)
            for key in dataset.buffers:
                self.assertIsNot(dataset.buffers[key], val_dataset.buffers[key])
                self.assertIsNot(
                    dataset.buffers_torch[key], val_dataset.buffers_torch[key]
                )

            indices = np.asarray([0, 1], dtype=np.int64)
            train_batch = dataset[indices]
            train_state = train_batch["state"].clone()
            val_dataset[indices]
            torch.testing.assert_close(train_batch["state"], train_state)

    def test_training_configs_share_terminal_anchor_semantics(self):
        terminal_name = (
            "HandoverBlockUR5-v2-"
            "delta_mv_actual_place_terminal_300_dynamic_v1-300.zarr"
        )
        expected_zarr = (
            REPO_ROOT / "policy" / "FlowMatching" / "data" / terminal_name
        ).resolve()
        task_paths = (
            REPO_ROOT
            / "policy"
            / "FlowMatching"
            / "diffusion_policy"
            / "config"
            / "task"
            / "maniskill_handover_block_ur5_v2_multiview_delta_pose_actual_task.yaml",
            REPO_ROOT
            / "policy"
            / "FPO"
            / "diffusion_policy"
            / "config"
            / "task"
            / "maniskill_handover_block_ur5_v2_multiview_delta_pose_actual_task.yaml",
        )
        for task_path in task_paths:
            config = yaml.safe_load(task_path.read_text(encoding="utf-8"))
            maniskill = config["maniskill"]
            dataset = config["dataset"]
            self.assertEqual(
                maniskill["action_semantics"], "chunk_anchor_relative_target"
            )
            self.assertEqual(maniskill["translation_frame"], "root_aligned")
            self.assertEqual(
                dataset["_target_"],
                "diffusion_policy.dataset.robot_pointcloud_dataset."
                "RobotMultiViewPointCloudChunkAnchorDataset",
            )
            self.assertEqual(dataset["obs_steps"], "${n_obs_steps}")
            self.assertEqual(
                (task_path.parents[3] / maniskill["zarr_path"]).resolve(),
                expected_zarr,
            )

        resfit_path = (
            REPO_ROOT
            / "policy"
            / "ResFiT"
            / "diffusion_policy"
            / "config"
            / "maniskill_resfit_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch.yaml"
        )
        resfit = yaml.safe_load(resfit_path.read_text(encoding="utf-8"))
        self.assertEqual(
            resfit["task"]["maniskill"]["action_semantics"],
            "chunk_anchor_relative_target",
        )
        self.assertEqual(
            resfit["task"]["maniskill"]["translation_frame"], "root_aligned"
        )
        chunk_configs = (
            *(
                REPO_ROOT / "policy" / "FlowMatching" / "diffusion_policy" / "config" / name
                for name in (
                    "maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual.yaml",
                    "maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual_action_dit_large.yaml",
                    "maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch.yaml",
                )
            ),
            *(
                REPO_ROOT / "policy" / "FPO" / "diffusion_policy" / "config" / name
                for name in (
                    "maniskill_fpo_handover_ur5_v2_multiview_delta_pose_actual.yaml",
                    "maniskill_fpo_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch.yaml",
                )
            ),
            resfit_path,
        )
        for config_path in chunk_configs:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["n_action_steps"], 4, config_path)


if __name__ == "__main__":
    unittest.main()
