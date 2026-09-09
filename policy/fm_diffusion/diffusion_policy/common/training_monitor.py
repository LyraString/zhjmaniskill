"""Read-only diagnostics for root-aligned bimanual chunk-anchor targets."""

from contextlib import contextmanager
import random

import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix
from torch.utils.data import default_collate


def make_monitor_batch(dataset, device, max_samples):
    """Cover a split evenly, independent of its training sampling weights."""
    if max_samples <= 0:
        raise ValueError("action_monitor.max_samples must be positive.")
    if len(dataset) == 0:
        return None
    indices = np.linspace(0, len(dataset) - 1, min(max_samples, len(dataset)), dtype=int)
    samples = [dataset[int(index)] for index in indices]
    batch = dataset.postprocess(default_collate(samples), device)
    source = dataset.sampler.indices[indices]
    steps = np.arange(batch["action"].shape[1])[None, :]
    valid = (steps >= source[:, 2:3]) & (steps < source[:, 3:4])
    return batch, torch.as_tensor(valid, device=device)


def _rotation_angle(matrix):
    # atan2 avoids the large near-zero error of acos in float32.
    skew = torch.stack((
        matrix[..., 2, 1] - matrix[..., 1, 2],
        matrix[..., 0, 2] - matrix[..., 2, 0],
        matrix[..., 1, 0] - matrix[..., 0, 1],
    ), dim=-1)
    sin_angle = torch.linalg.vector_norm(skew, dim=-1) * 0.5
    cos_angle = (matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    return torch.atan2(sin_angle, cos_angle)


def action_monitor_metrics(
    prediction, target, valid_mask, *, n_obs_steps, n_action_steps,
    translation_threshold_m, rotation_threshold_rad, gripper_threshold,
):
    """Compare executed target slots; detect near-zero poses only at the anchor.

    Later targets share the first observation anchor, so their magnitudes are
    not per-control-step motion. These metrics never measure actual EE speed.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or target.shape[-1] != 14:
        raise ValueError("Expected matching [batch, horizon, 14] pose target tensors.")
    if valid_mask.shape != target.shape[:2]:
        raise ValueError("valid_mask must match batch and horizon dimensions.")
    start = int(n_obs_steps) - 1
    end = start + int(n_action_steps)
    if start < 0 or n_action_steps <= 0 or end > target.shape[1]:
        raise ValueError("Executed action window lies outside the predicted horizon.")
    if translation_threshold_m <= 0 or rotation_threshold_rad <= 0:
        raise ValueError("Near-zero pose thresholds must be positive.")
    if not 0.0 < gripper_threshold < 1.0:
        raise ValueError("gripper_threshold must be between zero and one.")
    pred = prediction[:, start:end].detach().float()
    demo = target[:, start:end].detach().float()
    valid = valid_mask[:, start:end].bool()
    if not bool(valid.any()):
        return {}
    first_valid = valid[:, 0]
    metrics = {"samples": float(first_valid.sum()), "valid_action_slots": float(valid.sum())}
    pred_static_arms, demo_static_arms = [], []
    for name, offset in (("left", 0), ("right", 7)):
        pred_pos = pred[..., offset:offset + 3]
        demo_pos = demo[..., offset:offset + 3]
        pred_rot = euler_angles_to_matrix(pred[..., offset + 3:offset + 6], "XYZ")
        demo_rot = euler_angles_to_matrix(demo[..., offset + 3:offset + 6], "XYZ")
        distance = torch.linalg.vector_norm(pred_pos - demo_pos, dim=-1)[valid] * 1000
        angle = _rotation_angle(pred_rot @ demo_rot.transpose(-1, -2))[valid]
        values = {
            "position_error_mm": distance.mean(),
            "position_error_p95_mm": torch.quantile(distance, 0.95),
            "rotation_error_deg": torch.rad2deg(angle).mean(),
        }
        pred_open = pred[..., offset + 6][valid] >= gripper_threshold
        demo_open = demo[..., offset + 6][valid] >= gripper_threshold
        correct = pred_open == demo_open
        values["gripper_accuracy"] = correct.float().mean()
        # Overall accuracy can hide missed open/close events in imbalanced data.
        recalls = []
        for label, mask in (("open", demo_open), ("closed", ~demo_open)):
            values[f"gripper_{label}_count"] = mask.sum()
            if bool(mask.any()):
                recall = correct[mask].float().mean()
                values[f"gripper_{label}_recall"] = recall
                recalls.append(recall)
        if len(recalls) == 2:
            values["gripper_balanced_accuracy"] = torch.stack(recalls).mean()

        pred_offset = torch.linalg.vector_norm(pred_pos[:, 0], dim=-1)[first_valid]
        demo_offset = torch.linalg.vector_norm(demo_pos[:, 0], dim=-1)[first_valid]
        pred_angle = _rotation_angle(pred_rot[:, 0])[first_valid]
        demo_angle = _rotation_angle(demo_rot[:, 0])[first_valid]
        pred_static = (pred_offset < translation_threshold_m) & (pred_angle < rotation_threshold_rad)
        demo_static = (demo_offset < translation_threshold_m) & (demo_angle < rotation_threshold_rad)
        pred_static_arms.append(pred_static)
        demo_static_arms.append(demo_static)
        if pred_offset.numel():
            values.update({
                "first_target_offset_pred_mm": pred_offset.mean() * 1000,
                "first_target_offset_demo_mm": demo_offset.mean() * 1000,
                "first_target_angle_pred_deg": torch.rad2deg(pred_angle).mean(),
                "first_target_angle_demo_deg": torch.rad2deg(demo_angle).mean(),
                "demo_active_count": (~demo_static).sum(),
            })
            if bool((~demo_static).any()):
                values["missed_motion_rate"] = pred_static[~demo_static].float().mean()
        metrics.update({f"{name}/{key}": float(value) for key, value in values.items()})
    if bool(first_valid.any()):
        metrics["both/near_zero_pose_pred_rate"] = float((pred_static_arms[0] & pred_static_arms[1]).float().mean())
        metrics["both/near_zero_pose_demo_rate"] = float((demo_static_arms[0] & demo_static_arms[1]).float().mean())
    return metrics


@contextmanager
def monitor_rng(seed, device):
    """Repeat diagnostic noise without consuming the training RNG streams."""
    device = torch.device(device)
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def evaluate_action_monitor(policy, probe, *, seed, **metric_kwargs):
    if probe is None:
        return {}
    batch, valid_mask = probe
    modes = [(module, module.training) for module in policy.modules()]
    try:
        policy.eval()
        with torch.no_grad(), monitor_rng(seed, batch["action"].device):
            result = policy.predict_action(batch["obs"])
            return action_monitor_metrics(
                result["action_pred"], batch["action"], valid_mask, **metric_kwargs,
            )
    finally:
        for module, training in modes:
            module.training = training


def gradient_norm(parameters):
    norms = [torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0
