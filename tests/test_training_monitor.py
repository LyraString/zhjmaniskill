from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


FM_ROOT = Path(__file__).resolve().parents[1] / "policy" / "fm_diffusion"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))

from diffusion_policy.common.training_monitor import (
    action_monitor_metrics,
    evaluate_action_monitor,
    gradient_norm,
    make_monitor_batch,
)


METRIC_ARGS = dict(
    n_obs_steps=3, n_action_steps=4,
    translation_threshold_m=0.002, rotation_threshold_rad=0.02,
    gripper_threshold=0.5,
)


class TrainingMonitorTest(unittest.TestCase):
    def metrics(self, pred, target, valid=None):
        if valid is None:
            valid = torch.ones(target.shape[:2], dtype=torch.bool)
        return action_monitor_metrics(pred, target, valid, **METRIC_ARGS)

    def test_executed_window_padding_and_first_anchor_units(self):
        target = torch.zeros(1, 8, 14)
        target[0, 2:6, 0] = torch.tensor([0.004, 0.008, 0.012, 0.016])
        pred = target.clone()
        pred[:, :2] = 500  # Unexecuted prefix.
        pred[:, 5:] = 500  # Padded target and unexecuted suffix.
        pred[0, 3, 0] += 0.006
        valid = torch.ones(1, 8, dtype=torch.bool)
        valid[:, 5] = False
        m = self.metrics(pred, target, valid)
        self.assertEqual(m["valid_action_slots"], 3)
        self.assertAlmostEqual(m["left/position_error_mm"], 2, places=5)
        self.assertAlmostEqual(m["left/first_target_offset_demo_mm"], 4, places=5)
        self.assertEqual(m["right/position_error_mm"], 0)

    def test_missed_motion_is_conditional_per_arm_and_includes_rotation(self):
        target = torch.zeros(3, 8, 14)
        target[0, 2, 0] = 0.01
        target[1, 2, 3] = 0.1
        pred = torch.zeros_like(target)
        pred[1, 2, 3] = 0.1
        pred[2, 2, 6] = 1.0  # Gripper activity is not arm pose motion.
        m = self.metrics(pred, target)
        self.assertEqual(m["left/demo_active_count"], 2)
        self.assertAlmostEqual(m["left/missed_motion_rate"], 0.5)
        self.assertEqual(m["right/demo_active_count"], 0)
        self.assertNotIn("right/missed_motion_rate", m)
        self.assertAlmostEqual(m["both/near_zero_pose_demo_rate"], 1 / 3, places=6)
        self.assertAlmostEqual(m["both/near_zero_pose_pred_rate"], 2 / 3, places=6)

    def test_rotation_uses_so3_distance_with_periodic_euler_angles(self):
        target = torch.zeros(2, 8, 14)
        pred = target.clone()
        pred[..., 3] = 2 * torch.pi
        pred[..., 10] = torch.pi / 2
        m = self.metrics(pred, target)
        self.assertLess(m["left/rotation_error_deg"], 0.0001)
        self.assertAlmostEqual(m["right/rotation_error_deg"], 90, places=4)
        pred[..., 10] = torch.pi
        self.assertAlmostEqual(self.metrics(pred, target)["right/rotation_error_deg"], 180, places=4)

    def test_gripper_recall_exposes_majority_class_prediction(self):
        target = torch.zeros(100, 8, 14)
        target[:99, :, 6] = 1
        pred = torch.zeros_like(target)
        pred[..., 6] = 1
        m = self.metrics(pred, target)
        self.assertAlmostEqual(m["left/gripper_accuracy"], 0.99, places=6)
        self.assertEqual(m["left/gripper_open_recall"], 1)
        self.assertEqual(m["left/gripper_closed_recall"], 0)
        self.assertEqual(m["left/gripper_balanced_accuracy"], 0.5)
        self.assertNotIn("right/gripper_open_recall", m)
        self.assertNotIn("right/gripper_balanced_accuracy", m)

    def test_empty_mask_and_invalid_contract(self):
        action = torch.zeros(2, 8, 14)
        self.assertEqual(self.metrics(action, action, torch.zeros(2, 8, dtype=torch.bool)), {})
        with self.assertRaisesRegex(ValueError, "pose target"):
            self.metrics(action[..., :7], action[..., :7])
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.metrics(action[:, :3], action[:, :3])

    def test_fixed_probe_does_not_change_rng_or_module_modes(self):
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Dropout()

            def predict_action(self, obs):
                value = random.random() + np.random.rand()
                return {"action_pred": torch.rand_like(obs["template"]) * value * 0.01}

        policy = Policy().train()
        policy.encoder.eval()
        target = torch.zeros(2, 8, 14)
        probe = ({"action": target, "obs": {"template": target}}, torch.ones(2, 8, dtype=torch.bool))
        cpu_state, py_state, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
        first = evaluate_action_monitor(policy, probe, seed=3, **METRIC_ARGS)
        second = evaluate_action_monitor(policy, probe, seed=3, **METRIC_ARGS)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(cpu_state, torch.get_rng_state()))
        self.assertEqual(py_state, random.getstate())
        new_np_state = np.random.get_state()
        np.testing.assert_array_equal(np_state[1], new_np_state[1])
        self.assertEqual(np_state[2:], new_np_state[2:])
        self.assertTrue(policy.training)
        self.assertFalse(policy.encoder.training)

    def test_probe_spans_split_and_records_real_frames(self):
        class Dataset:
            sampler = SimpleNamespace(indices=np.tile([0, 6, 2, 8], (10, 1)))

            def __len__(self):
                return 10

            def __getitem__(self, index):
                return {"action": torch.full((8, 14), float(index))}

            def postprocess(self, batch, device):
                return {"action": batch["action"].to(device), "obs": {}}

        batch, valid = make_monitor_batch(Dataset(), "cpu", 4)
        torch.testing.assert_close(batch["action"][:, 0, 0], torch.tensor([0., 3., 6., 9.]))
        self.assertFalse(valid[:, :2].any())
        self.assertTrue(valid[:, 2:].all())

    def test_gradient_norm_does_not_clip_or_modify_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3., 4.])
        self.assertEqual(gradient_norm([parameter]), 5)
        torch.testing.assert_close(parameter.grad, torch.tensor([3., 4.]))
        self.assertEqual(gradient_norm([]), 0)

    def test_workspace_logs_metrics_without_changing_training_weights(self):
        import hydra
        from omegaconf import OmegaConf
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from diffusion_policy.common.pytorch_util import dict_apply
        from diffusion_policy.dataset.base_dataset import BaseImageDataset
        from diffusion_policy.workspace.robotworkspace import RobotWorkspace

        class Dataset(BaseImageDataset):
            action_anchor_index = 2
            sampler = SimpleNamespace(indices=np.tile([0, 8, 0, 8], (8, 1)))

            def __len__(self):
                return 8

            def __getitem__(self, idx):
                x = torch.as_tensor(idx, dtype=torch.float32)
                return {
                    "obs": {"agent_pos": x[..., None].expand(*x.shape, 14) * 0.01},
                    "action": x[..., None, None].expand(*x.shape, 8, 14) * 0.001,
                }

            def postprocess(self, batch, device):
                return dict_apply(batch, lambda x: x.to(device))

            def get_normalizer(self):
                return None

            def get_validation_dataset(self):
                return Dataset()

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.obs_encoder = torch.nn.Identity()
                self.model = torch.nn.Sequential(torch.nn.Linear(14, 112), torch.nn.Dropout(0.1))

            def set_normalizer(self, normalizer):
                pass

            def predict_action(self, obs):
                x = self.model(obs["agent_pos"]).reshape(-1, 8, 14)
                return {"action_pred": x + 0.01 * torch.randn_like(x)}

            def compute_loss(self, batch):
                return (self.predict_action(batch["obs"])["action_pred"] - batch["action"]).square().mean()

        config_dir = FM_ROOT.parent / "FlowMatching" / "diffusion_policy" / "config"
        with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = hydra.compose(config_name="maniskill_fm_vr")
        cfg.policy = OmegaConf.create({"_target_": "monitor_test_policy"})
        cfg.task.dataset = OmegaConf.create({"_target_": "monitor_test_dataset"})
        cfg.optimizer.fused = False
        cfg.training.device = "cpu"
        cfg.training.torch_compile.enabled = False
        cfg.training.amp.enabled = False
        cfg.training.num_epochs = 2
        cfg.training.lr_warmup_steps = 0
        cfg.training.checkpoint_every = 100
        cfg.training.val_every = cfg.training.sample_every = 1
        cfg.training.action_monitor.max_samples = 4
        cfg.dataloader.batch_size = cfg.val_dataloader.batch_size = 2
        instantiate = hydra.utils.instantiate

        def instantiate_for_test(config, *args, **kwargs):
            if config.get("_target_") == "monitor_test_policy":
                return Policy()
            if config.get("_target_") == "monitor_test_dataset":
                return Dataset()
            return instantiate(config, *args, **kwargs)

        states = []
        with tempfile.TemporaryDirectory() as tmp:
            for enabled in (False, True):
                cfg.training.action_monitor.enabled = enabled
                cfg.training.monitor_grad_every = 1 if enabled else 0
                output = Path(tmp) / str(enabled)
                output.mkdir()
                with patch("hydra.utils.instantiate", side_effect=instantiate_for_test), patch(
                    "diffusion_policy.workspace.robotworkspace._query_gpu_system_metrics", return_value={},
                ):
                    workspace = RobotWorkspace(cfg, output_dir=str(output))
                    workspace.run()
                states.append({k: v.clone() for k, v in workspace.model.state_dict().items()})
                if enabled:
                    events = EventAccumulator(str(output / "tensorboard"))
                    events.Reload()
                    tags = events.Tags()["scalars"]
                    for tag in (
                        "action/train/left/position_error_mm",
                        "action/val/right/rotation_error_deg",
                        "action/val/both/near_zero_pose_demo_rate",
                        "train/grad_norm", "performance/train_samples_per_second",
                    ):
                        self.assertIn(tag, tags)
                        self.assertTrue(np.isfinite(events.Scalars(tag)[-1].value))
                    self.assertEqual([v.step for v in events.Scalars("action/train/samples")], [0, 1])
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
