import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import dill
import imageio.v2 as imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory, apply_task_profile
from sim.maniskill.task_profiles import resolve_runtime_control_mode, resolve_runtime_task_target


ADAPTER_MODULES = {
    "flowmatching": "policy.FlowMatching.deploy_maniskill",
    "fpo": "policy.FPO.deploy_maniskill",
    "resfit": "policy.ResFiT.deploy_maniskill",
}


FINAL_SUCCESS_FIELDS = (
    ("spatial", "spatial_success"),
    ("xy", "xy_aligned"),
    ("z", "z_aligned"),
    ("right_open", "right_gripper_open"),
    ("right_clear", "right_ee_clear"),
    ("box_static", "box_static"),
)


MULTIVIEW_POINT_CLOUD_SPECS = {
    "head_point_cloud": "head_camera",
    "left_wrist_point_cloud": "left_wrist_camera",
    "right_wrist_point_cloud": "right_wrist_camera",
}

DELTA_POSE_CONTROL_MODE = "bimanual_pd_ee_delta_pose"


def _shape_meta_obs_items(ckpt_cfg) -> list[tuple[str, object]]:
    obs_meta = ckpt_cfg.task.shape_meta.obs
    if isinstance(obs_meta, Mapping):
        return list(obs_meta.items())
    return [(str(key), obs_meta[key]) for key in obs_meta.keys()]


def _shape_meta_shape(meta) -> list[int]:
    if isinstance(meta, Mapping):
        shape = meta.get("shape")
    else:
        shape = getattr(meta, "shape")
    return [int(dim) for dim in shape]


def _shape_meta_obs_keys(ckpt_cfg) -> list[str]:
    return [key for key, _meta in _shape_meta_obs_items(ckpt_cfg)]


def _checkpoint_multiview_point_cloud_views(ckpt_cfg) -> list[dict[str, object]]:
    views = []
    for key, meta in _shape_meta_obs_items(ckpt_cfg):
        if key not in MULTIVIEW_POINT_CLOUD_SPECS:
            continue
        shape = _shape_meta_shape(meta)
        if len(shape) != 2 or int(shape[-1]) != 3:
            raise ValueError(
                f"Invalid multiview point-cloud shape for `{key}` in checkpoint: {shape}"
            )
        views.append(
            {
                "key": key,
                "camera": MULTIVIEW_POINT_CLOUD_SPECS[key],
                "points": int(shape[0]),
            }
        )
    return views


def _adapt_runtime_for_checkpoint_obs(ckpt_cfg, runtime_kwargs: dict) -> None:
    views = _checkpoint_multiview_point_cloud_views(ckpt_cfg)
    if not views:
        return

    runtime_kwargs["point_cloud_views"] = views
    runtime_kwargs["enable_wrist_cameras"] = True


def _adapt_runtime_for_checkpoint_action(ckpt_cfg, runtime_kwargs: dict) -> None:
    task_cfg = getattr(ckpt_cfg, "task", None)
    maniskill_cfg = getattr(task_cfg, "maniskill", None)
    checkpoint_control_mode = getattr(maniskill_cfg, "control_mode", None)
    if runtime_kwargs.get("control_mode") in (None, "") and checkpoint_control_mode not in (None, ""):
        runtime_kwargs["control_mode"] = str(checkpoint_control_mode)
    checkpoint_action_type = getattr(maniskill_cfg, "action_type", None)
    if runtime_kwargs.get("action_type") in (None, "") and checkpoint_action_type not in (None, ""):
        runtime_kwargs["action_type"] = str(checkpoint_action_type)


def _validate_checkpoint_action_semantics(ckpt_cfg, *, control_mode: str) -> None:
    """Reject old per-step delta checkpoints on the absolute-target runtime."""

    if str(control_mode) != DELTA_POSE_CONTROL_MODE:
        return
    task_cfg = getattr(ckpt_cfg, "task", None)
    maniskill_cfg = getattr(task_cfg, "maniskill", None)
    expected = {
        "action_semantics": "chunk_anchor_relative_target",
        "rotation_representation": "euler_xyz",
        "translation_frame": "root_aligned",
        "gripper_semantics": "absolute",
    }
    actual = {
        key: (
            maniskill_cfg.get(key, None)
            if isinstance(maniskill_cfg, Mapping)
            else getattr(maniskill_cfg, key, None)
        )
        for key in expected
    }
    mismatches = {
        key: {"expected": expected_value, "actual": actual[key]}
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    }
    if mismatches:
        raise ValueError(
            "Checkpoint action semantics are incompatible with chunk-anchored absolute "
            f"EEF execution: {mismatches}. Retrain or use a checkpoint whose saved Hydra "
            "config declares the new action semantics."
        )


def _prepare_runtime_action_chunk(env, action_chunk, *, control_mode: str):
    """Prepare a policy chunk once before any of its targets are executed."""

    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] < 1:
        raise ValueError(f"Expected a non-empty [K, A] action chunk, got {chunk.shape}.")
    if not np.isfinite(chunk).all():
        raise ValueError("Policy action chunk contains NaN or Inf values.")
    if str(control_mode) != DELTA_POSE_CONTROL_MODE:
        return chunk.copy()

    prepared = np.asarray(env.prepare_action_chunk(chunk), dtype=np.float32)
    if prepared.shape != chunk.shape:
        raise ValueError(
            "Prepared target chunk shape differs from the policy chunk: "
            f"policy={chunk.shape}, target={prepared.shape}."
        )
    if not np.isfinite(prepared).all():
        raise ValueError("Prepared absolute EEF target chunk contains NaN or Inf values.")
    return prepared


def _validate_action_in_space(action, action_space) -> np.ndarray:
    """Clamp absolute gripper commands while keeping pose targets strict."""

    action_np = np.asarray(action, dtype=np.float32).copy()
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if action_np.shape != low.shape or action_np.shape != high.shape:
        raise ValueError(
            "Action/action-space shape mismatch: "
            f"action={action_np.shape}, low={low.shape}, high={high.shape}."
        )
    if not np.isfinite(action_np).all():
        raise ValueError("Controller target contains NaN or Inf values.")

    # Gripper outputs are continuous network predictions for bounded absolute
    # commands, so harmless numerical overshoot should saturate at the actuator
    # boundary. Cartesian pose targets remain fail-fast below.
    gripper_dims = (6, 13)
    action_np[list(gripper_dims)] = np.clip(
        action_np[list(gripper_dims)],
        low[list(gripper_dims)],
        high[list(gripper_dims)],
    )
    invalid = (action_np < low) | (action_np > high)
    if np.any(invalid):
        indices = np.flatnonzero(invalid).tolist()
        raise ValueError(
            "Prepared controller target exceeds the absolute environment action space; "
            f"dims={indices}, target={action_np[invalid].tolist()}, "
            f"low={low[invalid].tolist()}, high={high[invalid].tolist()}."
        )
    return action_np


def _clip_legacy_action_to_space(action, action_space) -> np.ndarray:
    """Preserve the pre-existing clipping behavior for non-EEF controllers."""

    action_np = np.asarray(action, dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if action_np.shape != low.shape or action_np.shape != high.shape:
        raise ValueError(
            "Action/action-space shape mismatch: "
            f"action={action_np.shape}, low={low.shape}, high={high.shape}."
        )
    if not np.isfinite(action_np).all():
        raise ValueError("Policy action contains NaN or Inf values.")
    return np.clip(action_np, low, high)


def _new_action_trace():
    return {
        "episode": [],
        "step": [],
        "policy_inference": [],
        "chunk_step": [],
        "raw_policy_action": [],
        "target_action": [],
        "actual_ee_pose_root_before_wxyz": [],
        "actual_ee_pose_root_after_wxyz": [],
        "target_left_xyz_step_norm": [],
        "target_right_xyz_step_norm": [],
        "actual_left_xyz_step_norm": [],
        "actual_right_xyz_step_norm": [],
        "target_left_xyz_error_before": [],
        "target_right_xyz_error_before": [],
        "target_left_xyz_error_after": [],
        "target_right_xyz_error_after": [],
        "target_left_xyz_progress": [],
        "target_right_xyz_progress": [],
        "policy_left_xyz_norm": [],
        "policy_left_rot_norm": [],
        "policy_left_gripper": [],
        "policy_right_xyz_norm": [],
        "policy_right_rot_norm": [],
        "policy_right_gripper": [],
        "target_left_xyz_norm": [],
        "target_left_rot_norm": [],
        "target_left_gripper": [],
        "target_right_xyz_norm": [],
        "target_right_rot_norm": [],
        "target_right_gripper": [],
        "reward_stage": [],
        "box_to_target": [],
        "spatial_success": [],
        "xy_aligned": [],
        "z_aligned": [],
        "right_gripper_open": [],
        "right_ee_clear": [],
        "box_static": [],
        "success": [],
    }


def _actual_eef_pose_root_wxyz(env) -> np.ndarray:
    """Read both actual EEF poses in the frame used by absolute PDEE targets."""

    base_env = env.env.unwrapped
    controllers = (
        base_env.left_delta_pose_controller,
        base_env.right_delta_pose_controller,
    )
    poses = []
    for controller in controllers:
        if controller is None:
            raise RuntimeError("PDEE action tracing requires both EEF controllers.")
        pose = controller.ee_pose_at_base.raw_pose
        if hasattr(pose, "detach") and hasattr(pose, "cpu"):
            pose = pose.detach().cpu().numpy()
        pose = np.asarray(pose, dtype=np.float32).reshape(-1, 7)
        if pose.shape[0] != 1 or not np.isfinite(pose).all():
            raise ValueError(f"Expected one finite EEF pose, got {pose.shape}.")
        poses.append(pose[0].copy())
    return np.stack(poses, axis=0)


def _action_or_nan(action) -> np.ndarray:
    if action is None:
        return np.full((14,), np.nan, dtype=np.float32)
    action_np = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_np.shape[0] == 14:
        return action_np.astype(np.float32, copy=True)
    padded = np.full((14,), np.nan, dtype=np.float32)
    padded[: min(14, action_np.shape[0])] = action_np[: min(14, action_np.shape[0])]
    return padded


def _eef_pose_or_nan(pose) -> np.ndarray:
    if pose is None:
        return np.full((2, 7), np.nan, dtype=np.float32)
    pose_np = np.asarray(pose, dtype=np.float32)
    if pose_np.shape != (2, 7) or not np.isfinite(pose_np).all():
        raise ValueError(f"Expected finite EEF poses with shape (2, 7), got {pose_np.shape}.")
    return pose_np.copy()


def _append_action_trace(
    trace: dict,
    episode_idx: int,
    step: int,
    target_action: np.ndarray,
    info: Mapping,
    *,
    raw_policy_action,
    policy_inference: int = 0,
    chunk_step: int = 0,
    actual_ee_pose_root_before_wxyz=None,
    actual_ee_pose_root_after_wxyz=None,
) -> None:
    policy_np = _action_or_nan(raw_policy_action)
    target_np = _action_or_nan(target_action)
    actual_before = _eef_pose_or_nan(actual_ee_pose_root_before_wxyz)
    actual_after = _eef_pose_or_nan(actual_ee_pose_root_after_wxyz)

    trace["episode"].append(int(episode_idx))
    trace["step"].append(int(step))
    trace["policy_inference"].append(int(policy_inference))
    trace["chunk_step"].append(int(chunk_step))
    trace["raw_policy_action"].append(policy_np)
    trace["target_action"].append(target_np)
    trace["actual_ee_pose_root_before_wxyz"].append(actual_before)
    trace["actual_ee_pose_root_after_wxyz"].append(actual_after)

    previous_target = None
    if len(trace["target_action"]) >= 2 and trace["episode"][-2] == int(episode_idx):
        previous_target = trace["target_action"][-2]
    for arm_name, arm_index, offset in (("left", 0, 0), ("right", 1, 7)):
        target_position = target_np[offset : offset + 3]
        before_position = actual_before[arm_index, :3]
        after_position = actual_after[arm_index, :3]
        error_before = float(np.linalg.norm(target_position - before_position))
        error_after = float(np.linalg.norm(target_position - after_position))
        trace[f"actual_{arm_name}_xyz_step_norm"].append(
            float(np.linalg.norm(after_position - before_position))
        )
        trace[f"target_{arm_name}_xyz_error_before"].append(error_before)
        trace[f"target_{arm_name}_xyz_error_after"].append(error_after)
        trace[f"target_{arm_name}_xyz_progress"].append(error_before - error_after)
        target_step = np.nan
        if previous_target is not None:
            target_step = float(
                np.linalg.norm(target_position - previous_target[offset : offset + 3])
            )
        trace[f"target_{arm_name}_xyz_step_norm"].append(target_step)

    for prefix, value in (("policy", policy_np), ("target", target_np)):
        trace[f"{prefix}_left_xyz_norm"].append(float(np.linalg.norm(value[0:3])))
        trace[f"{prefix}_left_rot_norm"].append(float(np.linalg.norm(value[3:6])))
        trace[f"{prefix}_left_gripper"].append(float(value[6]))
        trace[f"{prefix}_right_xyz_norm"].append(float(np.linalg.norm(value[7:10])))
        trace[f"{prefix}_right_rot_norm"].append(float(np.linalg.norm(value[10:13])))
        trace[f"{prefix}_right_gripper"].append(float(value[13]))
    trace["reward_stage"].append(_to_float_scalar_or_none(info.get("reward_stage")))
    trace["box_to_target"].append(_to_float_scalar_or_none(info.get("reward_box_to_target")))
    for key in (
        "spatial_success",
        "xy_aligned",
        "z_aligned",
        "right_gripper_open",
        "right_ee_clear",
        "box_static",
        "success",
    ):
        value = _to_bool_scalar_or_none(info.get(key))
        trace[key].append(np.nan if value is None else float(bool(value)))


def _save_action_trace(trace: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, values in trace.items():
        if key in {
            "raw_policy_action",
            "target_action",
            "actual_ee_pose_root_before_wxyz",
            "actual_ee_pose_root_after_wxyz",
        }:
            arrays[key] = (
                np.stack(values, axis=0).astype(np.float32)
                if len(values) > 0
                else np.zeros((0, 0), dtype=np.float32)
            )
        else:
            arrays[key] = np.asarray(
                [np.nan if value is None else value for value in values],
                dtype=(
                    np.int64
                    if key in {"episode", "step", "policy_inference", "chunk_step"}
                    else np.float32
                ),
            )
    np.savez_compressed(output_path, **arrays)

    if arrays["target_action"].size == 0:
        print(f"action_trace_saved path={output_path} steps=0")
        return

    summary = {
        "path": str(output_path),
        "steps": int(arrays["target_action"].shape[0]),
        "policy_left_xyz_norm_mean": float(np.nanmean(arrays["policy_left_xyz_norm"])),
        "policy_left_xyz_norm_max": float(np.nanmax(arrays["policy_left_xyz_norm"])),
        "policy_right_xyz_norm_mean": float(np.nanmean(arrays["policy_right_xyz_norm"])),
        "policy_right_xyz_norm_max": float(np.nanmax(arrays["policy_right_xyz_norm"])),
        "target_left_xyz_norm_mean": float(np.nanmean(arrays["target_left_xyz_norm"])),
        "target_right_xyz_norm_mean": float(np.nanmean(arrays["target_right_xyz_norm"])),
        "policy_right_gripper_mean": float(np.nanmean(arrays["policy_right_gripper"])),
    }
    for arm_name in ("left", "right"):
        actual_step = arrays[f"actual_{arm_name}_xyz_step_norm"]
        target_step = arrays[f"target_{arm_name}_xyz_step_norm"]
        target_error = arrays[f"target_{arm_name}_xyz_error_after"]
        target_progress = arrays[f"target_{arm_name}_xyz_progress"]
        summary.update(
            {
                f"actual_{arm_name}_xyz_step_mean_m": float(np.nanmean(actual_step)),
                f"actual_{arm_name}_xyz_step_p95_m": float(np.nanpercentile(actual_step, 95)),
                f"target_{arm_name}_xyz_step_mean_m": float(np.nanmean(target_step)),
                f"target_{arm_name}_xyz_step_p95_m": float(np.nanpercentile(target_step, 95)),
                f"target_{arm_name}_xyz_error_after_mean_m": float(np.nanmean(target_error)),
                f"target_{arm_name}_xyz_error_after_p95_m": float(np.nanpercentile(target_error, 95)),
                f"target_{arm_name}_xyz_progress_mean_m": float(np.nanmean(target_progress)),
            }
        )
    stage = arrays["reward_stage"]
    stage4_mask = np.isfinite(stage) & (np.rint(stage).astype(np.int64) == 4)
    if bool(np.any(stage4_mask)):
        summary.update(
            {
                "stage4_steps": int(np.count_nonzero(stage4_mask)),
                "stage4_policy_right_xyz_norm_mean": float(
                    np.nanmean(arrays["policy_right_xyz_norm"][stage4_mask])
                ),
                "stage4_policy_right_xyz_norm_max": float(
                    np.nanmax(arrays["policy_right_xyz_norm"][stage4_mask])
                ),
                "stage4_policy_right_gripper_mean": float(
                    np.nanmean(arrays["policy_right_gripper"][stage4_mask])
                ),
                "stage4_right_open_frac": float(np.nanmean(arrays["right_gripper_open"][stage4_mask])),
                "stage4_right_clear_frac": float(np.nanmean(arrays["right_ee_clear"][stage4_mask])),
                "stage4_spatial_frac": float(np.nanmean(arrays["spatial_success"][stage4_mask])),
            }
        )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"action_trace_saved path={output_path} summary={summary_path} steps={summary['steps']}")


def _to_bool_scalar(value) -> bool:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def _to_bool_scalar_or_none(value):
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return None
    return bool(array[0])


def _to_float_scalar_or_none(value):
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return None
    return float(array[0])


def _new_handover_diag():
    return {
        "has_stage": False,
        "max_stage": -1,
        "final_stage": -1,
        "left_hold_sum": 0.0,
        "right_hold_sum": 0.0,
        "hold_count": 0,
        "box_to_target": None,
        "final_success": {label: None for label, _info_key in FINAL_SUCCESS_FIELDS},
    }


def _update_handover_diag(diag, info) -> None:
    stage = _to_float_scalar_or_none(info.get("reward_stage"))
    if stage is None:
        return

    stage_id = int(round(stage))
    clamped_stage_id = min(4, max(0, stage_id))
    diag["has_stage"] = True
    diag["max_stage"] = max(diag["max_stage"], clamped_stage_id)
    diag["final_stage"] = clamped_stage_id

    left_hold = _to_float_scalar_or_none(info.get("reward_left_holding"))
    right_hold = _to_float_scalar_or_none(info.get("reward_right_holding"))
    if left_hold is not None or right_hold is not None:
        diag["left_hold_sum"] += 0.0 if left_hold is None else left_hold
        diag["right_hold_sum"] += 0.0 if right_hold is None else right_hold
        diag["hold_count"] += 1

    box_to_target = _to_float_scalar_or_none(info.get("reward_box_to_target"))
    if box_to_target is not None:
        diag["box_to_target"] = box_to_target

    for label, info_key in FINAL_SUCCESS_FIELDS:
        value = _to_bool_scalar_or_none(info.get(info_key))
        if value is not None:
            diag["final_success"][label] = value


def _classify_handover_failure(diag, success: bool) -> str:
    if success:
        return "success"
    if not diag["has_stage"]:
        return "unknown"
    max_stage = int(diag["max_stage"])
    if max_stage < 1:
        return "failed_left_grasp"
    if max_stage < 3:
        return "failed_handover"
    if max_stage == 3:
        return "failed_right_carry"
    return "failed_place_or_release"


def _format_optional_float(value, precision: int = 3) -> str:
    if value is None:
        return "nan"
    return f"{float(value):.{precision}f}"


def _format_optional_bool(value) -> str:
    if value is None:
        return "nan"
    return str(int(bool(value)))


def _resolve_n_action_steps(ckpt_cfg, override: int | None) -> int:
    checkpoint_steps = int(ckpt_cfg.n_action_steps)
    if override is None:
        return checkpoint_steps
    if override < 1:
        raise ValueError(f"--n-action-steps must be >= 1, got {override}.")

    horizon = int(ckpt_cfg.horizon)
    n_obs_steps = int(ckpt_cfg.n_obs_steps)
    available_steps = horizon - (n_obs_steps - 1)
    if override > available_steps:
        raise ValueError(
            "--n-action-steps exceeds the executable part of the predicted horizon: "
            f"requested={override}, available={available_steps}, "
            f"horizon={horizon}, n_obs_steps={n_obs_steps}."
        )
    return int(override)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a ManiSkill checkpoint in a ManiSkill env.")
    parser.add_argument("policy_kind", choices=sorted(ADAPTER_MODULES.keys()))
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("task_name", type=str, help="ManiSkill env id or the env id associated with --task-profile")
    parser.add_argument("--task-profile", type=str, default=None, help="Optional registered ManiSkill task profile")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-type", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default=None)
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--sim-backend", type=str, default=None)
    parser.add_argument("--render-backend", type=str, default=None)
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--robot-uids", type=str, default=None)
    parser.add_argument("--sim-freq", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=None)
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override how many predicted actions are executed before the next policy inference.",
    )
    parser.add_argument("--save-video-dir", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--save-action-trace",
        type=str,
        default=None,
        help="Optional .npz path for per-step action/stage diagnostics. Default is disabled.",
    )
    return parser


def _maybe_capture_frame(env_backend, frames: list[np.ndarray]) -> None:
    frame = env_backend.env.render()
    if frame is None:
        return
    if hasattr(frame, "detach") and hasattr(frame, "cpu"):
        frame = frame.detach().cpu()
    frame = np.asarray(frame)
    # Raw ManiSkill rgb_array render returns a batched image of shape
    # (num_envs, H, W, C) even when num_envs == 1.
    if frame.ndim == 4 and frame.shape[0] >= 1:
        frame = frame[0]
    if frame.ndim != 3:
        return
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    frames.append(frame)


def _is_pdee_ik_failure(error: RuntimeError) -> bool:
    return str(error) in {
        "PDEE inverse kinematics failed.",
        "PDEE inverse kinematics returned a non-finite joint target.",
    }


def run_evaluation(args) -> None:
    adapter = importlib.import_module(ADAPTER_MODULES[args.policy_kind])
    with open(args.ckpt_path, "rb") as checkpoint_file:
        payload = torch.load(checkpoint_file, pickle_module=dill, map_location="cpu")
    ckpt_cfg = payload["cfg"]
    n_action_steps = _resolve_n_action_steps(ckpt_cfg, args.n_action_steps)
    runtime_env_id = args.task_name
    if args.task_profile is not None and args.task_name == args.task_profile:
        runtime_env_id = None
    runtime_task_name, resolved_profile_name = resolve_runtime_task_target(
        task_name=args.task_name,
        task_profile=args.task_profile,
        env_id=runtime_env_id,
    )
    runtime_kwargs = {
        "action_type": args.action_type,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "render_mode": args.render_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "reward_mode": args.reward_mode,
        "robot_uids": args.robot_uids,
        "sim_freq": args.sim_freq,
        "control_freq": args.control_freq,
        "max_episode_steps": args.max_episode_steps,
    }
    _adapt_runtime_for_checkpoint_obs(ckpt_cfg, runtime_kwargs)
    _adapt_runtime_for_checkpoint_action(ckpt_cfg, runtime_kwargs)
    _resolved_action_type, resolved_control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["control_mode"] = resolved_control_mode
    if args.save_video_dir is not None and runtime_kwargs["render_mode"] in (None, ""):
        runtime_kwargs["render_mode"] = "rgb_array"
    profile_runtime_key = resolved_profile_name or runtime_task_name
    _, resolved_runtime, _profile_name = apply_task_profile(profile_runtime_key, runtime_kwargs)
    effective_control_mode = str(
        resolved_runtime.get("control_mode") or resolved_control_mode or ""
    )
    _validate_checkpoint_action_semantics(
        ckpt_cfg,
        control_mode=effective_control_mode,
    )

    env = ManiSkillTaskFactory().create_env(
        task_name=runtime_task_name,
        **resolved_runtime,
    )
    try:
        model = adapter.get_model(
            {
                "ckpt_path": args.ckpt_path,
                "n_action_steps": n_action_steps,
            }
        )
    except BaseException:
        env.close()
        raise
    print(
        "action_chunk "
        f"predicted_horizon={int(ckpt_cfg.horizon)} "
        f"executed_steps={n_action_steps} "
        f"checkpoint_executed_steps={int(ckpt_cfg.n_action_steps)}"
    )

    video_dir = Path(args.save_video_dir) if args.save_video_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    returns = []
    lengths = []
    successes = []
    handover_diags = []
    ik_failures = []
    action_trace = _new_action_trace() if args.save_action_trace is not None else None

    try:
        for episode_idx in range(args.episodes):
            obs = env.reset(seed=args.seed + episode_idx)
            adapter.reset_model(model)
            encoded_obs = adapter.encode_obs(dict(obs.data))
            missing_obs_keys = [
                key for key in _shape_meta_obs_keys(ckpt_cfg) if key not in encoded_obs
            ]
            if missing_obs_keys:
                raise KeyError(
                    "Runtime observation is missing checkpoint-required obs keys: "
                    f"{missing_obs_keys}. "
                    "Check the task profile or checkpoint observation schema."
                )
            expected_obs_dim = int(ckpt_cfg.task.shape_meta.obs.agent_pos.shape[0])
            actual_obs_dim = int(np.asarray(encoded_obs["agent_pos"]).shape[-1])
            expected_action_dim = int(ckpt_cfg.task.shape_meta.action.shape[0])
            actual_action_dim = int(env.action_space.shape[0])
            if expected_obs_dim != actual_obs_dim or expected_action_dim != actual_action_dim:
                ckpt_task_name = str(getattr(ckpt_cfg.task, "name", "<unknown>"))
                raise ValueError(
                    "Checkpoint/env shape mismatch: "
                    f"checkpoint task={ckpt_task_name} "
                    f"agent_pos={expected_obs_dim} action={expected_action_dim}, "
                    f"runtime task={runtime_task_name} "
                    f"agent_pos={actual_obs_dim} action={actual_action_dim}. "
                    "Use a checkpoint trained for this task, or evaluate against the task the checkpoint was trained on."
                )
            episode_return = 0.0
            episode_length = 0
            success = False
            episode_failure = None
            frames = []
            if video_dir is not None:
                _maybe_capture_frame(env, frames)
            result = None
            handover_diag = _new_handover_diag()
            policy_inference = 0
            while episode_length < args.max_episode_steps:
                if len(model.obs_cache) == 0:
                    model.update_obs(encoded_obs)
                raw_policy_chunk = np.asarray(model.get_action(), dtype=np.float32)
                target_chunk = _prepare_runtime_action_chunk(
                    env,
                    raw_policy_chunk,
                    control_mode=effective_control_mode,
                )
                if effective_control_mode == DELTA_POSE_CONTROL_MODE:
                    target_chunk = np.stack(
                        [
                            _validate_action_in_space(target, env.action_space)
                            for target in target_chunk
                        ],
                        axis=0,
                    )
                else:
                    target_chunk = np.stack(
                        [
                            _clip_legacy_action_to_space(action, env.action_space)
                            for action in target_chunk
                        ],
                        axis=0,
                    )

                for chunk_step, (raw_policy_action, target_action) in enumerate(
                    zip(raw_policy_chunk, target_chunk)
                ):
                    actual_ee_before = None
                    if action_trace is not None and effective_control_mode == DELTA_POSE_CONTROL_MODE:
                        actual_ee_before = _actual_eef_pose_root_wxyz(env)
                    try:
                        result = env.step(
                            ActionSpec.from_value(
                                target_action,
                                action_type=resolved_runtime["action_type"],
                            )
                        )
                    except RuntimeError as error:
                        if effective_control_mode != DELTA_POSE_CONTROL_MODE or not _is_pdee_ik_failure(error):
                            raise
                        episode_failure = "pdee_ik_failure"
                        ik_failures.append({
                            "episode": episode_idx,
                            "seed": args.seed + episode_idx,
                            "completed_steps": episode_length,
                            "policy_inference": policy_inference,
                            "chunk_step": chunk_step,
                            "reason": episode_failure,
                            "error": str(error),
                            "raw_policy_action": raw_policy_action.tolist(),
                            "target_action": target_action.tolist(),
                        })
                        print(
                            f"episode_failure episode={episode_idx} seed={args.seed + episode_idx} "
                            f"completed_steps={episode_length} reason={episode_failure} error={error}"
                        )
                        # A controller may already have updated one arm. Do not step
                        # again until the next episode has reset the whole environment.
                        break
                    if action_trace is not None:
                        actual_ee_after = None
                        if effective_control_mode == DELTA_POSE_CONTROL_MODE:
                            actual_ee_after = _actual_eef_pose_root_wxyz(env)
                        _append_action_trace(
                            action_trace,
                            episode_idx=episode_idx,
                            step=episode_length,
                            target_action=target_action,
                            info=result.info,
                            raw_policy_action=raw_policy_action,
                            policy_inference=policy_inference,
                            chunk_step=chunk_step,
                            actual_ee_pose_root_before_wxyz=actual_ee_before,
                            actual_ee_pose_root_after_wxyz=actual_ee_after,
                        )
                    if _to_bool_scalar(result.info.get("success", False)):
                        success = True

                    episode_return += float(result.reward)
                    episode_length += 1
                    _update_handover_diag(handover_diag, result.info)
                    encoded_obs = adapter.encode_obs(dict(result.observation.data))
                    model.update_obs(encoded_obs)
                    if video_dir is not None:
                        _maybe_capture_frame(env, frames)
                    if success:
                        break
                    if result.terminated or result.truncated or episode_length >= args.max_episode_steps:
                        break

                policy_inference += 1

                if episode_failure is not None:
                    break
                if success or result.terminated or result.truncated or episode_length >= args.max_episode_steps:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(float(success))
            handover_failure = episode_failure or _classify_handover_failure(handover_diag, success)
            if handover_diag["has_stage"]:
                handover_diags.append({**handover_diag, "failure": handover_failure})
            if video_dir is not None and len(frames) > 0:
                episode_dir = video_dir
                episode_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(
                    episode_dir / f"episode{episode_idx}.mp4",
                    frames,
                    fps=int(args.video_fps),
                )
            ep_line = (
                f"episode={episode_idx} return={episode_return:.4f} "
                f"length={episode_length} success={int(success)}"
            )
            print(ep_line)
            if handover_diag["has_stage"]:
                hold_count = max(1, int(handover_diag["hold_count"]))
                print(
                    "diag "
                    f"max_stage={handover_diag['max_stage']} "
                    f"final_stage={handover_diag['final_stage']} "
                    f"failure={handover_failure} "
                    f"left_hold={handover_diag['left_hold_sum'] / hold_count:.3f} "
                    f"right_hold={handover_diag['right_hold_sum'] / hold_count:.3f} "
                    f"box_to_target={_format_optional_float(handover_diag['box_to_target'])}"
                )
                if handover_failure == "failed_place_or_release":
                    final_success = handover_diag["final_success"]
                    print(
                        "final_diag "
                        f"spatial={_format_optional_bool(final_success['spatial'])} "
                        f"xy={_format_optional_bool(final_success['xy'])} "
                        f"z={_format_optional_bool(final_success['z'])} "
                        f"right_open={_format_optional_bool(final_success['right_open'])} "
                        f"right_clear={_format_optional_bool(final_success['right_clear'])} "
                        f"box_static={_format_optional_bool(final_success['box_static'])}"
                    )
    finally:
        env.close()

    print(
        "summary "
        f"mean_return={float(np.mean(returns)):.4f} "
        f"mean_length={float(np.mean(lengths)):.2f} "
        f"success_rate={float(np.mean(successes)):.4f} "
        f"pdee_ik_failures={len(ik_failures)}"
    )
    if len(handover_diags) > 0:
        diag_count = float(len(handover_diags))
        stage_buckets = [0, 0, 0, 0, 0]
        failure_counts = {
            "failed_left_grasp": 0,
            "failed_handover": 0,
            "failed_right_carry": 0,
            "failed_place_or_release": 0,
            "pdee_ik_failure": 0,
            "unknown": 0,
        }
        for diag in handover_diags:
            stage_buckets[int(diag["max_stage"])] += 1
            failure = str(diag["failure"])
            if failure in failure_counts:
                failure_counts[failure] += 1
        print(
            "diag_summary "
            f"stage0={stage_buckets[0] / diag_count:.3f} "
            f"stage1={stage_buckets[1] / diag_count:.3f} "
            f"stage2={stage_buckets[2] / diag_count:.3f} "
            f"stage3={stage_buckets[3] / diag_count:.3f} "
            f"stage4={stage_buckets[4] / diag_count:.3f} "
            f"failed_left_grasp={failure_counts['failed_left_grasp'] / diag_count:.3f} "
            f"failed_handover={failure_counts['failed_handover'] / diag_count:.3f} "
            f"failed_right_carry={failure_counts['failed_right_carry'] / diag_count:.3f} "
            f"failed_place={failure_counts['failed_place_or_release'] / diag_count:.3f} "
            f"pdee_ik_failure={failure_counts['pdee_ik_failure'] / diag_count:.3f} "
            f"unknown={failure_counts['unknown'] / diag_count:.3f}"
        )
        failed_place_diags = [
            diag for diag in handover_diags if str(diag["failure"]) == "failed_place_or_release"
        ]
        if len(failed_place_diags) > 0:
            failed_place_count = float(len(failed_place_diags))
            final_success_sums = {label: 0.0 for label, _info_key in FINAL_SUCCESS_FIELDS}
            for diag in failed_place_diags:
                final_success = diag["final_success"]
                for label, _info_key in FINAL_SUCCESS_FIELDS:
                    final_success_sums[label] += float(bool(final_success.get(label, False)))
            print(
                "final_diag_summary "
                f"failed_place_n={len(failed_place_diags)} "
                f"spatial={final_success_sums['spatial'] / failed_place_count:.3f} "
                f"xy={final_success_sums['xy'] / failed_place_count:.3f} "
                f"z={final_success_sums['z'] / failed_place_count:.3f} "
                f"right_open={final_success_sums['right_open'] / failed_place_count:.3f} "
                f"right_clear={final_success_sums['right_clear'] / failed_place_count:.3f} "
                f"box_static={final_success_sums['box_static'] / failed_place_count:.3f}"
            )
    if action_trace is not None:
        _save_action_trace(action_trace, args.save_action_trace)
    failure_path = None
    if video_dir is not None:
        failure_path = video_dir / "evaluation_failures.json"
    elif args.save_action_trace is not None:
        failure_path = Path(args.save_action_trace).with_suffix(".failures.json")
    if failure_path is not None:
        failure_path.write_text(
            json.dumps({"evaluated_episodes": len(successes), "ik_failures": ik_failures}, indent=2),
            encoding="utf-8",
        )


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
