from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import dill
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory, apply_task_profile
from sim.maniskill.task_profiles import resolve_runtime_control_mode, resolve_runtime_task_target
from sim.maniskill.tasks.handover_block_env import BOX_HALF_SIZE, TARGET_HALF_SIZE


ADAPTER_MODULES = {
    "flowmatching": "policy.FlowMatching.deploy_maniskill",
    "fpo": "policy.FPO.deploy_maniskill",
}

TASK_CACHE_ATTRS = ("reward_stage", "left_gripper_val", "right_gripper_val")
TARGET_STAGES = (2, 3, 4)
DEFAULT_OUTPUT_PATH = "tmp/handover/counterfactual_coupling_labels.npz"
FEATURE_NAMES = (
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_ee_qw",
    "left_ee_qx",
    "left_ee_qy",
    "left_ee_qz",
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_ee_qw",
    "right_ee_qx",
    "right_ee_qy",
    "right_ee_qz",
    "box_x",
    "box_y",
    "box_z",
    "box_qw",
    "box_qx",
    "box_qy",
    "box_qz",
    "target_x",
    "target_y",
    "target_z",
    "target_qw",
    "target_qx",
    "target_qy",
    "target_qz",
    "left_gripper",
    "right_gripper",
    "box_minus_left_ee_x",
    "box_minus_left_ee_y",
    "box_minus_left_ee_z",
    "box_minus_right_ee_x",
    "box_minus_right_ee_y",
    "box_minus_right_ee_z",
    "target_top_minus_box_bottom_x",
    "target_top_minus_box_bottom_y",
    "target_top_minus_box_bottom_z",
    "left_to_box_dist",
    "right_to_box_dist",
    "box_to_target_dist",
    "ee_to_ee_dist",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate counterfactual coupling labels for HandoverBlockUR5-v2."
    )
    parser.add_argument("policy_kind", choices=sorted(ADAPTER_MODULES.keys()))
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("task_name", type=str)
    parser.add_argument("--task-profile", default="handover_block_ur5_v2_smoke")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--probe-points-per-episode", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--identity-threshold", type=float, default=0.002)
    parser.add_argument("--coupled-threshold", type=float, default=0.02)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--action-type", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default=None)
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--sim-backend", type=str, default="physx_cuda")
    parser.add_argument("--render-backend", type=str, default="sapien_cuda")
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--robot-uids", type=str, default=None)
    parser.add_argument("--sim-freq", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=None)
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_env_vector(value: Any, width: int | None = None) -> np.ndarray:
    array = _to_numpy(value)
    if width is None:
        array = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(1, -1)
    else:
        array = array.reshape(-1, width)
    return np.asarray(array[0], dtype=np.float64)


def _scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    array = _to_numpy(value).reshape(-1)
    if array.size == 0:
        return None
    return float(array[0])


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _save_snapshot(unwrapped_env) -> dict[str, Any]:
    attrs = {}
    for attr in TASK_CACHE_ATTRS:
        if hasattr(unwrapped_env, attr):
            attrs[attr] = _clone_value(getattr(unwrapped_env, attr))
    return {
        "sim_state": copy.deepcopy(unwrapped_env.get_state_dict()),
        "attrs": attrs,
    }


def _restore_snapshot(unwrapped_env, snapshot: dict[str, Any]) -> None:
    unwrapped_env.set_state_dict(copy.deepcopy(snapshot["sim_state"]))
    for attr, value in snapshot.get("attrs", {}).items():
        setattr(unwrapped_env, attr, _clone_value(value))


def _pose_array(pose) -> np.ndarray:
    return np.concatenate(
        [
            _first_env_vector(pose.p, width=3),
            _first_env_vector(pose.q, width=4),
        ],
        axis=0,
    ).astype(np.float64, copy=False)


def _box_pos(unwrapped_env) -> np.ndarray:
    return _first_env_vector(unwrapped_env.box.pose.p, width=3)


def _stage_from_env_or_info(unwrapped_env, info: dict[str, Any] | None) -> int:
    stage = None
    if hasattr(unwrapped_env, "reward_stage"):
        stage = _scalar_or_none(getattr(unwrapped_env, "reward_stage"))
    if stage is None and isinstance(info, dict):
        stage = _scalar_or_none(info.get("reward_stage"))
    if stage is None:
        return -1
    return int(np.clip(round(stage), 0, 4))


def _compact_info(info: dict[str, Any] | None) -> dict[str, float | None]:
    info = info or {}
    keys = (
        "reward_stage",
        "reward_left_holding",
        "reward_right_holding",
        "reward_box_to_handover",
        "reward_box_to_target",
        "reward_target_xy_dist",
        "reward_target_z_dist",
    )
    return {key: _scalar_or_none(info.get(key)) for key in keys}


def _make_action_sequence(action_chunk: np.ndarray, horizon: int) -> np.ndarray:
    action_chunk = np.asarray(action_chunk, dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk shape [T, Da], got {action_chunk.shape}.")
    if action_chunk.shape[0] <= 0:
        raise ValueError("Action chunk is empty.")
    repeats = int(math.ceil(float(horizon) / float(action_chunk.shape[0])))
    return np.concatenate([action_chunk] * repeats, axis=0)[:horizon].astype(np.float32, copy=False)


def _clip_actions(action_seq: np.ndarray, action_space) -> np.ndarray:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return action_seq.astype(np.float32, copy=False)
    return np.clip(action_seq, low, high).astype(np.float32, copy=False)


def _add_noise(
    action_seq: np.ndarray,
    *,
    branch: str,
    noise_std: float,
    rng: np.random.Generator,
    action_space,
) -> np.ndarray:
    noisy = np.array(action_seq, dtype=np.float32, copy=True)
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=noisy.shape).astype(np.float32)
    if branch == "left_noise":
        mask = np.zeros_like(noisy)
        mask[:, :7] = 1.0
    elif branch == "right_noise":
        mask = np.zeros_like(noisy)
        mask[:, 7:14] = 1.0
    elif branch == "both_noise":
        mask = np.ones_like(noisy)
    else:
        raise ValueError(f"Unknown noisy branch: {branch}")
    return _clip_actions(noisy + noise * mask, action_space)


def _rollout_action_sequence(env, action_seq: np.ndarray, action_type: str | None) -> dict[str, Any]:
    positions = []
    terminated = False
    truncated = False
    for action in action_seq:
        result = env.step(ActionSpec.from_value(action, action_type=action_type))
        positions.append(_box_pos(env.env.unwrapped))
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)
        if terminated or truncated:
            break
    return {
        "box_positions": np.asarray(positions, dtype=np.float64),
        "steps": int(len(positions)),
        "terminated": terminated,
        "truncated": truncated,
    }


def _trajectory_max_dev(reference: np.ndarray, candidate: np.ndarray) -> float:
    steps = min(reference.shape[0], candidate.shape[0])
    if steps <= 0:
        return float("nan")
    return float(np.linalg.norm(reference[:steps] - candidate[:steps], axis=1).max())


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _frac(values: list[bool]) -> float:
    if not values:
        return float("nan")
    return float(np.mean([float(value) for value in values]))


def _stage_histogram(stages: np.ndarray) -> dict[str, int]:
    histogram = {str(idx): 0 for idx in range(5)}
    histogram["unknown"] = 0
    for stage in stages.reshape(-1).tolist():
        stage_int = int(stage)
        if 0 <= stage_int <= 4:
            histogram[str(stage_int)] += 1
        else:
            histogram["unknown"] += 1
    return histogram


def _extract_features(unwrapped_env, info: dict[str, Any] | None) -> np.ndarray:
    left_ee_pose = _pose_array(unwrapped_env.get_arm_end_effector_link(arm_index=0).pose)
    right_ee_pose = _pose_array(unwrapped_env.get_arm_end_effector_link(arm_index=1).pose)
    box_pose = _pose_array(unwrapped_env.box.pose)
    target_pose = _pose_array(unwrapped_env.target_box.pose)

    left_ee_pos = left_ee_pose[:3]
    right_ee_pos = right_ee_pose[:3]
    box_pos = box_pose[:3]
    box_bottom = box_pos.copy()
    box_bottom[2] -= float(BOX_HALF_SIZE[2])
    target_top = target_pose[:3].copy()
    target_top[2] += float(TARGET_HALF_SIZE[2])

    left_gripper = _scalar_or_none(unwrapped_env._get_gripper_state_tensor(arm_index=0))
    right_gripper = _scalar_or_none(unwrapped_env._get_gripper_state_tensor(arm_index=1))
    if left_gripper is None or right_gripper is None:
        raise ValueError("Could not read gripper state.")

    box_minus_left = box_pos - left_ee_pos
    box_minus_right = box_pos - right_ee_pos
    target_top_minus_box_bottom = target_top - box_bottom
    distances = np.asarray(
        [
            np.linalg.norm(box_minus_left),
            np.linalg.norm(box_minus_right),
            np.linalg.norm(target_top_minus_box_bottom),
            np.linalg.norm(right_ee_pos - left_ee_pos),
        ],
        dtype=np.float64,
    )

    feature = np.concatenate(
        [
            left_ee_pose,
            right_ee_pose,
            box_pose,
            target_pose,
            np.asarray([left_gripper, right_gripper], dtype=np.float64),
            box_minus_left,
            box_minus_right,
            target_top_minus_box_bottom,
            distances,
        ],
        axis=0,
    ).astype(np.float32)
    if feature.shape[0] != len(FEATURE_NAMES):
        raise RuntimeError(f"Feature shape mismatch: {feature.shape[0]} vs {len(FEATURE_NAMES)}.")
    return feature


def _counterfactual_from_snapshot(
    *,
    env,
    snapshot: dict[str, Any],
    action_seq: np.ndarray,
    action_type: str | None,
    noise_std: float,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    unwrapped = env.env.unwrapped

    _restore_snapshot(unwrapped, snapshot)
    baseline = _rollout_action_sequence(env, action_seq, action_type)

    _restore_snapshot(unwrapped, snapshot)
    identity = _rollout_action_sequence(env, action_seq, action_type)

    baseline_pos = baseline["box_positions"]
    identity_pos = identity["box_positions"]
    result: dict[str, float | int] = {
        "baseline_steps": int(baseline["steps"]),
        "identity_steps": int(identity["steps"]),
        "restore_identity_max_dev": _trajectory_max_dev(baseline_pos, identity_pos),
    }
    for branch in ("left_noise", "right_noise", "both_noise"):
        noisy_actions = _add_noise(
            action_seq,
            branch=branch,
            noise_std=noise_std,
            rng=rng,
            action_space=env.action_space,
        )
        _restore_snapshot(unwrapped, snapshot)
        branch_rollout = _rollout_action_sequence(env, noisy_actions, action_type)
        result[f"{branch}_steps"] = int(branch_rollout["steps"])
        result[f"{branch}_max_dev"] = _trajectory_max_dev(baseline_pos, branch_rollout["box_positions"])
    return result


def _build_env(args):
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
        "point_cloud_sampling": "fast",
        "max_episode_steps": int(args.max_episode_steps)
        + int(args.probe_points_per_episode) * int(args.horizon) * 5
        + 10,
    }
    _resolved_action_type, resolved_control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["control_mode"] = resolved_control_mode
    profile_runtime_key = resolved_profile_name or runtime_task_name
    _, resolved_runtime, _profile_name = apply_task_profile(profile_runtime_key, runtime_kwargs)
    env = ManiSkillTaskFactory().create_env(task_name=runtime_task_name, **resolved_runtime)
    return env, resolved_runtime, runtime_task_name


def _load_policy(args):
    adapter = importlib.import_module(ADAPTER_MODULES[args.policy_kind])
    payload = torch.load(open(args.ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    model = adapter.get_model({"ckpt_path": args.ckpt_path})
    return adapter, model, payload["cfg"]


def _state_restore_support(unwrapped_env) -> tuple[bool, str | None]:
    if not hasattr(unwrapped_env, "get_state_dict"):
        return False, "missing get_state_dict"
    if not hasattr(unwrapped_env, "set_state_dict"):
        return False, "missing set_state_dict"
    try:
        snapshot = _save_snapshot(unwrapped_env)
        _restore_snapshot(unwrapped_env, snapshot)
    except Exception as exc:  # noqa: BLE001 - report label generation feasibility, do not crash.
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _save_dataset(output_path: Path, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
        summary_json=np.asarray(json.dumps(summary, ensure_ascii=False, default=_json_default), dtype=str),
    )


def generate_labels(args) -> dict[str, Any]:
    adapter, model, ckpt_cfg = _load_policy(args)
    env, resolved_runtime, runtime_task_name = _build_env(args)
    rng = np.random.default_rng(int(args.seed))
    target_stages = set(TARGET_STAGES)

    rows: list[dict[str, Any]] = []
    attempted_probe_count = 0
    skipped_identity_count = 0

    try:
        obs = env.reset(seed=int(args.seed))
        supported, reason = _state_restore_support(env.env.unwrapped)
        if not supported:
            summary = {
                "state_restore_supported": False,
                "state_restore_error": reason,
                "recommended_next_step": "debug_state_restore_before_labeling",
            }
            _save_dataset(Path(args.output_path), {"features": np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)}, summary)
            return summary

        expected_obs_dim = int(ckpt_cfg.task.shape_meta.obs.agent_pos.shape[0])
        expected_action_dim = int(ckpt_cfg.task.shape_meta.action.shape[0])
        actual_obs_dim = int(np.asarray(obs.data["agent_pos"]).shape[-1])
        actual_action_dim = int(env.action_space.shape[0])
        if expected_obs_dim != actual_obs_dim or expected_action_dim != actual_action_dim:
            raise ValueError(
                "Checkpoint/env shape mismatch: "
                f"checkpoint agent_pos={expected_obs_dim} action={expected_action_dim}, "
                f"runtime agent_pos={actual_obs_dim} action={actual_action_dim}."
            )

        for episode_idx in range(int(args.episodes)):
            if episode_idx > 0:
                obs = env.reset(seed=int(args.seed) + episode_idx)
            adapter.reset_model(model)
            encoded_obs = adapter.encode_obs(dict(obs.data))
            model.update_obs(encoded_obs)
            episode_length = 0
            probe_count = 0
            probe_interval = max(1, int(args.max_episode_steps) // max(1, int(args.probe_points_per_episode)))
            next_probe_at = 0
            last_info: dict[str, Any] | None = None

            while episode_length < int(args.max_episode_steps):
                if len(model.obs_cache) == 0:
                    model.update_obs(encoded_obs)
                action_chunk = np.asarray(model.get_action(), dtype=np.float32)
                stage = _stage_from_env_or_info(env.env.unwrapped, last_info)

                if (
                    probe_count < int(args.probe_points_per_episode)
                    and episode_length >= next_probe_at
                    and stage in target_stages
                ):
                    snapshot = _save_snapshot(env.env.unwrapped)
                    feature = _extract_features(env.env.unwrapped, last_info)
                    info_at_probe = _compact_info(last_info)
                    box_pos_at_probe = _box_pos(env.env.unwrapped).astype(np.float32)
                    action_seq = _make_action_sequence(action_chunk, int(args.horizon))
                    cf = _counterfactual_from_snapshot(
                        env=env,
                        snapshot=snapshot,
                        action_seq=action_seq,
                        action_type=resolved_runtime.get("action_type"),
                        noise_std=float(args.noise_std),
                        rng=rng,
                    )
                    _restore_snapshot(env.env.unwrapped, snapshot)
                    attempted_probe_count += 1
                    probe_count += 1
                    next_probe_at += probe_interval

                    identity_dev = float(cf["restore_identity_max_dev"])
                    identity_pass = bool(
                        np.isfinite(identity_dev)
                        and identity_dev < float(args.identity_threshold)
                    )
                    if not identity_pass:
                        skipped_identity_count += 1
                        continue

                    left_dev = float(cf["left_noise_max_dev"])
                    right_dev = float(cf["right_noise_max_dev"])
                    both_dev = float(cf["both_noise_max_dev"])
                    rows.append(
                        {
                            "feature": feature,
                            "stage": int(stage),
                            "episode": int(episode_idx),
                            "step": int(episode_length),
                            "box_pos": box_pos_at_probe,
                            "info": info_at_probe,
                            "identity_dev": identity_dev,
                            "left_dev": left_dev,
                            "right_dev": right_dev,
                            "both_dev": both_dev,
                            "left_label": int(left_dev > float(args.coupled_threshold)),
                            "right_label": int(right_dev > float(args.coupled_threshold)),
                            "both_label": int(both_dev > float(args.coupled_threshold)),
                            "baseline_steps": int(cf["baseline_steps"]),
                        }
                    )

                done = False
                for action in action_chunk:
                    result = env.step(
                        ActionSpec.from_value(action, action_type=resolved_runtime.get("action_type"))
                    )
                    last_info = dict(result.info)
                    encoded_obs = adapter.encode_obs(dict(result.observation.data))
                    model.update_obs(encoded_obs)
                    episode_length += 1
                    done = bool(result.terminated or result.truncated)
                    if done or episode_length >= int(args.max_episode_steps):
                        break
                if done:
                    break
    finally:
        env.close()

    if rows:
        features = np.stack([row["feature"] for row in rows], axis=0).astype(np.float32)
        stages = np.asarray([row["stage"] for row in rows], dtype=np.int64)
        episodes = np.asarray([row["episode"] for row in rows], dtype=np.int64)
        steps = np.asarray([row["step"] for row in rows], dtype=np.int64)
        box_pos = np.stack([row["box_pos"] for row in rows], axis=0).astype(np.float32)
        identity_dev = np.asarray([row["identity_dev"] for row in rows], dtype=np.float32)
        left_dev = np.asarray([row["left_dev"] for row in rows], dtype=np.float32)
        right_dev = np.asarray([row["right_dev"] for row in rows], dtype=np.float32)
        both_dev = np.asarray([row["both_dev"] for row in rows], dtype=np.float32)
        left_label = np.asarray([row["left_label"] for row in rows], dtype=np.int64)
        right_label = np.asarray([row["right_label"] for row in rows], dtype=np.int64)
        both_label = np.asarray([row["both_label"] for row in rows], dtype=np.int64)
        baseline_steps = np.asarray([row["baseline_steps"] for row in rows], dtype=np.int64)
    else:
        features = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        stages = np.zeros((0,), dtype=np.int64)
        episodes = np.zeros((0,), dtype=np.int64)
        steps = np.zeros((0,), dtype=np.int64)
        box_pos = np.zeros((0, 3), dtype=np.float32)
        identity_dev = np.zeros((0,), dtype=np.float32)
        left_dev = np.zeros((0,), dtype=np.float32)
        right_dev = np.zeros((0,), dtype=np.float32)
        both_dev = np.zeros((0,), dtype=np.float32)
        left_label = np.zeros((0,), dtype=np.int64)
        right_label = np.zeros((0,), dtype=np.int64)
        both_label = np.zeros((0,), dtype=np.int64)
        baseline_steps = np.zeros((0,), dtype=np.int64)

    valid_sample_count = int(features.shape[0])
    attempted_probe_frac = (
        float(valid_sample_count / attempted_probe_count)
        if attempted_probe_count > 0
        else 0.0
    )
    label_has_signal = bool(
        valid_sample_count > 0
        and any(
            0.0 < float(labels.mean()) < 1.0
            for labels in (left_label, right_label, both_label)
        )
    )
    if valid_sample_count <= 0:
        recommended_next_step = "rerun_with_more_episodes_or_probe_points"
    elif attempted_probe_count > 0 and attempted_probe_frac < 0.75:
        recommended_next_step = "debug_state_restore_before_labeling"
    elif label_has_signal:
        recommended_next_step = "train_coupling_net_v0"
    else:
        recommended_next_step = "adjust_threshold_noise_or_horizon"

    summary = {
        "state_restore_supported": True,
        "task_name": runtime_task_name,
        "task_profile": args.task_profile,
        "policy_kind": args.policy_kind,
        "ckpt_path": str(args.ckpt_path),
        "output_path": str(args.output_path),
        "episodes": int(args.episodes),
        "attempted_probe_count": int(attempted_probe_count),
        "valid_sample_count": valid_sample_count,
        "skipped_identity_count": int(skipped_identity_count),
        "valid_sample_frac": attempted_probe_frac,
        "feature_dim": int(features.shape[1]),
        "target_stages": list(TARGET_STAGES),
        "horizon": int(args.horizon),
        "noise_std": float(args.noise_std),
        "identity_threshold": float(args.identity_threshold),
        "coupled_threshold": float(args.coupled_threshold),
        "stage_histogram": _stage_histogram(stages),
        "left_dev_mean": _mean(left_dev.tolist()),
        "right_dev_mean": _mean(right_dev.tolist()),
        "both_dev_mean": _mean(both_dev.tolist()),
        "left_label_frac": _mean(left_label.astype(np.float32).tolist()),
        "right_label_frac": _mean(right_label.astype(np.float32).tolist()),
        "both_label_frac": _mean(both_label.astype(np.float32).tolist()),
        "recommended_next_step": recommended_next_step,
    }

    arrays = {
        "features": features,
        "stage": stages,
        "episode": episodes,
        "step": steps,
        "box_pos": box_pos,
        "identity_dev": identity_dev,
        "left_dev": left_dev,
        "right_dev": right_dev,
        "both_dev": both_dev,
        "left_label": left_label,
        "right_label": right_label,
        "both_label": both_label,
        "baseline_steps": baseline_steps,
    }
    _save_dataset(Path(args.output_path), arrays, summary)
    return summary


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    payload = generate_labels(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
