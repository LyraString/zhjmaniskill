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
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory, apply_task_profile
from sim.maniskill.task_profiles import resolve_runtime_control_mode, resolve_runtime_task_target


ADAPTER_MODULES = {
    "flowmatching": "policy.FlowMatching.deploy_maniskill",
    "fpo": "policy.FPO.deploy_maniskill",
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
DELTA_POSE_ARM_SPECS = (
    ("left", slice(0, 6), 6, "left_delta_pose_controller"),
    ("right", slice(7, 13), 13, "right_delta_pose_controller"),
)


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


def _new_action_clip_stats():
    return {
        "steps": 0,
        "clipped_steps": 0,
        "dims": 0,
        "clipped_dims": 0,
        "abs_sum": 0.0,
        "abs_max": 0.0,
    }


def _clip_action_to_space(action, action_space, stats: dict):
    action_np = np.asarray(action, dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    clipped = np.clip(action_np, low, high)
    diff = np.abs(clipped - action_np)

    stats["steps"] += 1
    stats["dims"] += int(diff.size)
    clipped_mask = diff > 0.0
    clipped_dims = int(np.count_nonzero(clipped_mask))
    stats["clipped_dims"] += clipped_dims
    if clipped_dims > 0:
        stats["clipped_steps"] += 1
        stats["abs_sum"] += float(diff[clipped_mask].sum())
        stats["abs_max"] = max(stats["abs_max"], float(diff.max()))

    return clipped


def _quaternion_inverse_wxyz(quat: torch.Tensor) -> torch.Tensor:
    norm_sq = torch.sum(quat * quat).clamp_min(1e-8)
    return torch.cat([quat[0:1], -quat[1:]], dim=0) / norm_sq


def _normalize_pose_wxyz(pose: torch.Tensor) -> torch.Tensor:
    pose = pose.to(dtype=torch.float32)
    quat = pose[3:7]
    quat = quat / torch.linalg.norm(quat).clamp_min(1e-8)
    return torch.cat([pose[0:3], quat], dim=0)


def _target_pose_from_delta(current_pose: torch.Tensor, delta_action: torch.Tensor) -> torch.Tensor:
    delta_pos = delta_action[0:3]
    delta_rot = delta_action[3:6].reshape(1, 3)
    delta_quat = matrix_to_quaternion(euler_angles_to_matrix(delta_rot, "XYZ"))[0]
    target_pos = current_pose[0:3] + delta_pos
    target_quat = quaternion_multiply(
        delta_quat.reshape(1, 4),
        current_pose[3:7].reshape(1, 4),
    )[0]
    return _normalize_pose_wxyz(torch.cat([target_pos, target_quat], dim=0))


def _delta_action_from_target_pose(current_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    delta_pos = target_pose[0:3] - current_pose[0:3]
    delta_quat = quaternion_multiply(
        target_pose[3:7].reshape(1, 4),
        _quaternion_inverse_wxyz(current_pose[3:7]).reshape(1, 4),
    )[0]
    delta_rot = matrix_to_euler_angles(
        quaternion_to_matrix(delta_quat.reshape(1, 4)),
        "XYZ",
    )[0]
    return torch.cat([delta_pos, delta_rot], dim=0)


class DeltaPoseHoldFilter:
    """Eval-only postprocessor for suppressing open-arm delta-pose drift."""

    def __init__(
        self,
        *,
        mode: str,
        gripper_threshold: float,
        pos_threshold: float,
        rot_threshold: float,
    ) -> None:
        self.mode = mode
        self.gripper_threshold = float(gripper_threshold)
        self.pos_threshold = float(pos_threshold)
        self.rot_threshold = float(rot_threshold)
        self.anchors = {arm_name: None for arm_name, *_rest in DELTA_POSE_ARM_SPECS}
        self.stats = {
            "steps": 0,
            "arm_steps": 0,
            "held": 0,
            "committed": 0,
            "closed_passthrough": 0,
            "unavailable": 0,
        }
        self.last_decisions = {arm_name: "off" for arm_name, *_rest in DELTA_POSE_ARM_SPECS}

    def reset_episode(self) -> None:
        self.anchors = {arm_name: None for arm_name, *_rest in DELTA_POSE_ARM_SPECS}
        self.last_decisions = {arm_name: "off" for arm_name, *_rest in DELTA_POSE_ARM_SPECS}

    def _read_current_pose(self, env_backend, controller_attr: str) -> torch.Tensor | None:
        inner_env = getattr(env_backend, "env", env_backend)
        unwrapped = getattr(inner_env, "unwrapped", inner_env)
        controller = getattr(unwrapped, controller_attr, None)
        ee_pose = getattr(controller, "ee_pose_at_base", None)
        raw_pose = getattr(ee_pose, "raw_pose", None)
        if raw_pose is None:
            return None
        pose = raw_pose.detach()
        if pose.ndim == 2:
            pose = pose[0]
        elif pose.ndim > 2:
            pose = pose.reshape(-1, pose.shape[-1])[0]
        if pose.shape[-1] != 7:
            return None
        return _normalize_pose_wxyz(pose)

    def apply(self, action, env_backend) -> np.ndarray:
        action_np = np.asarray(action, dtype=np.float32).copy()
        if action_np.shape != (14,):
            raise ValueError(
                f"Delta-pose hold filter expects a 14D bimanual action, got {action_np.shape}."
            )

        self.stats["steps"] += 1
        for arm_name, action_slice, gripper_idx, controller_attr in DELTA_POSE_ARM_SPECS:
            self.last_decisions[arm_name] = "passthrough"
            self.stats["arm_steps"] += 1
            gripper_cmd = float(action_np[gripper_idx])
            current_pose = None
            if gripper_cmd <= self.gripper_threshold:
                current_pose = self._read_current_pose(env_backend, controller_attr)
                if current_pose is not None:
                    self.anchors[arm_name] = current_pose
                self.last_decisions[arm_name] = "closed_passthrough"
                self.stats["closed_passthrough"] += 1
                continue

            delta_np = action_np[action_slice]
            pos_norm = float(np.linalg.norm(delta_np[0:3]))
            rot_norm = float(np.linalg.norm(delta_np[3:6]))
            is_small_delta = pos_norm <= self.pos_threshold and rot_norm <= self.rot_threshold

            if self.mode == "deadband":
                if is_small_delta:
                    action_np[action_slice] = 0.0
                    self.last_decisions[arm_name] = "held"
                    self.stats["held"] += 1
                else:
                    self.last_decisions[arm_name] = "committed"
                    self.stats["committed"] += 1
                continue

            current_pose = self._read_current_pose(env_backend, controller_attr)
            if current_pose is None:
                self.last_decisions[arm_name] = "unavailable"
                self.stats["unavailable"] += 1
                continue
            if self.anchors[arm_name] is None:
                self.anchors[arm_name] = current_pose

            delta_action = torch.as_tensor(delta_np, device=current_pose.device, dtype=torch.float32)
            if is_small_delta:
                held_delta = _delta_action_from_target_pose(current_pose, self.anchors[arm_name])
                action_np[action_slice] = held_delta.detach().cpu().numpy().astype(np.float32)
                self.last_decisions[arm_name] = "held"
                self.stats["held"] += 1
            else:
                self.anchors[arm_name] = _target_pose_from_delta(current_pose, delta_action)
                self.last_decisions[arm_name] = "committed"
                self.stats["committed"] += 1

        return action_np


def _new_action_trace():
    return {
        "episode": [],
        "step": [],
        "action": [],
        "raw_action": [],
        "filtered_action": [],
        "clip_delta": [],
        "left_xyz_norm": [],
        "left_rot_norm": [],
        "left_gripper": [],
        "right_xyz_norm": [],
        "right_rot_norm": [],
        "right_gripper": [],
        "raw_left_gripper": [],
        "raw_right_gripper": [],
        "filtered_left_gripper": [],
        "filtered_right_gripper": [],
        "left_filter_decision": [],
        "right_filter_decision": [],
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


FILTER_DECISION_CODES = {
    "off": 0,
    "passthrough": 1,
    "closed_passthrough": 2,
    "held": 3,
    "committed": 4,
    "unavailable": 5,
}


def _action_or_nan(action) -> np.ndarray:
    if action is None:
        return np.full((14,), np.nan, dtype=np.float32)
    action_np = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_np.shape[0] == 14:
        return action_np.astype(np.float32, copy=True)
    padded = np.full((14,), np.nan, dtype=np.float32)
    padded[: min(14, action_np.shape[0])] = action_np[: min(14, action_np.shape[0])]
    return padded


def _append_action_trace(
    trace: dict,
    episode_idx: int,
    step: int,
    action: np.ndarray,
    info: Mapping,
    *,
    raw_action=None,
    filtered_action=None,
    filter_decisions: Mapping[str, str] | None = None,
) -> None:
    action_np = np.asarray(action, dtype=np.float32).reshape(-1)
    raw_action_np = _action_or_nan(raw_action)
    filtered_action_np = _action_or_nan(filtered_action)
    clipped_action_np = _action_or_nan(action)
    if action_np.shape[0] >= 14:
        left_xyz = action_np[0:3]
        left_rot = action_np[3:6]
        left_gripper = float(action_np[6])
        right_xyz = action_np[7:10]
        right_rot = action_np[10:13]
        right_gripper = float(action_np[13])
    else:
        left_xyz = np.zeros(3, dtype=np.float32)
        left_rot = np.zeros(3, dtype=np.float32)
        left_gripper = float("nan")
        right_xyz = np.zeros(3, dtype=np.float32)
        right_rot = np.zeros(3, dtype=np.float32)
        right_gripper = float("nan")

    trace["episode"].append(int(episode_idx))
    trace["step"].append(int(step))
    trace["action"].append(action_np.astype(np.float32, copy=True))
    trace["raw_action"].append(raw_action_np)
    trace["filtered_action"].append(filtered_action_np)
    trace["clip_delta"].append(clipped_action_np - filtered_action_np)
    trace["left_xyz_norm"].append(float(np.linalg.norm(left_xyz)))
    trace["left_rot_norm"].append(float(np.linalg.norm(left_rot)))
    trace["left_gripper"].append(left_gripper)
    trace["right_xyz_norm"].append(float(np.linalg.norm(right_xyz)))
    trace["right_rot_norm"].append(float(np.linalg.norm(right_rot)))
    trace["right_gripper"].append(right_gripper)
    trace["raw_left_gripper"].append(float(raw_action_np[6]))
    trace["raw_right_gripper"].append(float(raw_action_np[13]))
    trace["filtered_left_gripper"].append(float(filtered_action_np[6]))
    trace["filtered_right_gripper"].append(float(filtered_action_np[13]))
    filter_decisions = filter_decisions or {}
    trace["left_filter_decision"].append(
        FILTER_DECISION_CODES.get(str(filter_decisions.get("left", "off")), -1)
    )
    trace["right_filter_decision"].append(
        FILTER_DECISION_CODES.get(str(filter_decisions.get("right", "off")), -1)
    )
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
        if key in {"action", "raw_action", "filtered_action", "clip_delta"}:
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
                    if key in {"episode", "step", "left_filter_decision", "right_filter_decision"}
                    else np.float32
                ),
            )
    np.savez_compressed(output_path, **arrays)

    if arrays["action"].size == 0:
        print(f"action_trace_saved path={output_path} steps=0")
        return

    summary = {
        "path": str(output_path),
        "steps": int(arrays["action"].shape[0]),
        "left_xyz_norm_mean": float(np.nanmean(arrays["left_xyz_norm"])),
        "left_xyz_norm_max": float(np.nanmax(arrays["left_xyz_norm"])),
        "right_xyz_norm_mean": float(np.nanmean(arrays["right_xyz_norm"])),
        "right_xyz_norm_max": float(np.nanmax(arrays["right_xyz_norm"])),
        "left_rot_norm_mean": float(np.nanmean(arrays["left_rot_norm"])),
        "right_rot_norm_mean": float(np.nanmean(arrays["right_rot_norm"])),
        "right_gripper_mean": float(np.nanmean(arrays["right_gripper"])),
    }
    stage = arrays["reward_stage"]
    stage4_mask = np.isfinite(stage) & (np.rint(stage).astype(np.int64) == 4)
    if bool(np.any(stage4_mask)):
        summary.update(
            {
                "stage4_steps": int(np.count_nonzero(stage4_mask)),
                "stage4_right_xyz_norm_mean": float(np.nanmean(arrays["right_xyz_norm"][stage4_mask])),
                "stage4_right_xyz_norm_max": float(np.nanmax(arrays["right_xyz_norm"][stage4_mask])),
                "stage4_right_gripper_mean": float(np.nanmean(arrays["right_gripper"][stage4_mask])),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a ManiSkill checkpoint in a ManiSkill env.")
    parser.add_argument("policy_kind", choices=sorted(ADAPTER_MODULES.keys()))
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("task_name", type=str, help="ManiSkill env id or the env id associated with --task-profile")
    parser.add_argument("--task-profile", type=str, default=None, help="Optional registered ManiSkill task profile")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=50)
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
    parser.add_argument("--save-video-dir", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--save-action-trace",
        type=str,
        default=None,
        help="Optional .npz path for per-step action/stage diagnostics. Default is disabled.",
    )
    parser.add_argument(
        "--delta-pose-hold-mode",
        choices=("off", "deadband", "anchor"),
        default="off",
        help="Eval-only filter for suppressing open-arm drift in delta-pose control.",
    )
    parser.add_argument("--delta-pose-hold-gripper-threshold", type=float, default=0.95)
    parser.add_argument("--delta-pose-hold-pos-threshold", type=float, default=0.002)
    parser.add_argument("--delta-pose-hold-rot-threshold", type=float, default=0.02)
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


def run_evaluation(args) -> None:
    adapter = importlib.import_module(ADAPTER_MODULES[args.policy_kind])
    payload = torch.load(open(args.ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    ckpt_cfg = payload["cfg"]
    model = adapter.get_model({"ckpt_path": args.ckpt_path})
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

    env = ManiSkillTaskFactory().create_env(
        task_name=runtime_task_name,
        **resolved_runtime,
    )
    delta_pose_hold_filter = None
    if args.delta_pose_hold_mode != "off":
        runtime_action_type = str(resolved_runtime.get("action_type") or "")
        runtime_control_mode = str(resolved_runtime.get("control_mode") or resolved_control_mode or "")
        if DELTA_POSE_CONTROL_MODE in (runtime_action_type, runtime_control_mode):
            delta_pose_hold_filter = DeltaPoseHoldFilter(
                mode=args.delta_pose_hold_mode,
                gripper_threshold=args.delta_pose_hold_gripper_threshold,
                pos_threshold=args.delta_pose_hold_pos_threshold,
                rot_threshold=args.delta_pose_hold_rot_threshold,
            )
        else:
            print(
                "delta_pose_hold_filter disabled "
                f"mode={args.delta_pose_hold_mode} "
                f"action_type={runtime_action_type} "
                f"control_mode={runtime_control_mode}"
            )

    video_dir = Path(args.save_video_dir) if args.save_video_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    returns = []
    lengths = []
    successes = []
    handover_diags = []
    action_clip_stats = _new_action_clip_stats()
    action_trace = _new_action_trace() if args.save_action_trace is not None else None

    try:
        for episode_idx in range(args.episodes):
            obs = env.reset(seed=args.seed + episode_idx)
            if delta_pose_hold_filter is not None:
                delta_pose_hold_filter.reset_episode()
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
            frames = []
            if video_dir is not None:
                _maybe_capture_frame(env, frames)
            result = None
            handover_diag = _new_handover_diag()
            while episode_length < args.max_episode_steps:
                if len(model.obs_cache) == 0:
                    model.update_obs(encoded_obs)
                action_chunk = model.get_action()

                for action in action_chunk:
                    raw_action = np.asarray(action, dtype=np.float32).copy()
                    filtered_action = raw_action
                    filter_decisions = None
                    if delta_pose_hold_filter is not None:
                        filtered_action = delta_pose_hold_filter.apply(raw_action, env)
                        filter_decisions = dict(delta_pose_hold_filter.last_decisions)
                    action = _clip_action_to_space(filtered_action, env.action_space, action_clip_stats)
                    result = env.step(
                        ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                    )
                    if action_trace is not None:
                        _append_action_trace(
                            action_trace,
                            episode_idx=episode_idx,
                            step=episode_length,
                            action=action,
                            info=result.info,
                            raw_action=raw_action,
                            filtered_action=filtered_action,
                            filter_decisions=filter_decisions,
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

                if success or result.terminated or result.truncated or episode_length >= args.max_episode_steps:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(float(success))
            handover_failure = _classify_handover_failure(handover_diag, success)
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
        f"success_rate={float(np.mean(successes)):.4f}"
    )
    if action_clip_stats["steps"] > 0:
        clipped_dims = max(1, int(action_clip_stats["clipped_dims"]))
        print(
            "action_clip_summary "
            f"step_frac={action_clip_stats['clipped_steps'] / action_clip_stats['steps']:.6f} "
            f"dim_frac={action_clip_stats['clipped_dims'] / max(1, action_clip_stats['dims']):.6f} "
            f"abs_mean={action_clip_stats['abs_sum'] / clipped_dims:.6f} "
            f"abs_max={action_clip_stats['abs_max']:.6f}"
        )
    if delta_pose_hold_filter is not None:
        stats = delta_pose_hold_filter.stats
        print(
            "delta_pose_hold_summary "
            f"mode={delta_pose_hold_filter.mode} "
            f"steps={stats['steps']} "
            f"arm_steps={stats['arm_steps']} "
            f"held={stats['held']} "
            f"committed={stats['committed']} "
            f"closed_passthrough={stats['closed_passthrough']} "
            f"unavailable={stats['unavailable']} "
            f"gripper_threshold={delta_pose_hold_filter.gripper_threshold:.4f} "
            f"pos_threshold={delta_pose_hold_filter.pos_threshold:.6f} "
            f"rot_threshold={delta_pose_hold_filter.rot_threshold:.6f}"
        )
    if len(handover_diags) > 0:
        diag_count = float(len(handover_diags))
        stage_buckets = [0, 0, 0, 0, 0]
        failure_counts = {
            "failed_left_grasp": 0,
            "failed_handover": 0,
            "failed_right_carry": 0,
            "failed_place_or_release": 0,
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


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
