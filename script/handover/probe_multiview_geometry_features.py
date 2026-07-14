#!/usr/bin/env python3
"""Probe whether frozen multiview features retain near-target geometry."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn as nn
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
FM_DIFFUSION_ROOT = REPO_ROOT / "policy/fm_diffusion"
for import_root in (REPO_ROOT, FM_DIFFUSION_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from diffusion_policy.env_runner.maniskill_cuda_fpo_runner import (  # noqa: E402
    ManiSkillCudaFPORunner,
)
from script.eval_maniskill_policy import (  # noqa: E402
    ADAPTER_MODULES,
    DELTA_POSE_CONTROL_MODE,
    _adapt_runtime_for_checkpoint_action,
    _adapt_runtime_for_checkpoint_obs,
    _shape_meta_obs_items,
)
from sim.maniskill import apply_task_profile  # noqa: E402
from sim.maniskill.task_profiles import (  # noqa: E402
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
)
from sim.maniskill.tasks.handover_block_env import (  # noqa: E402
    BOX_HALF_SIZE,
    TARGET_HALF_SIZE,
)


VIEW_KEYS = (
    "head_point_cloud",
    "left_wrist_point_cloud",
    "right_wrist_point_cloud",
)
DELTA_POSE_ARM_SPECS = (
    (slice(0, 6), 6, "left_delta_pose_controller"),
    (slice(7, 13), 13, "right_delta_pose_controller"),
)
MLP_HIDDEN_DIM = 128


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect near-target states in CUDA vector environments and fit frozen "
            "512D-feature geometry probes."
        )
    )
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("task_name", type=str)
    parser.add_argument("--task-profile", type=str, default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=90,
        help="Number of valid near-target episodes to collect.",
    )
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--val-episodes", type=int, default=15)
    parser.add_argument("--test-episodes", type=int, default=15)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument(
        "--max-rollout-episodes",
        type=int,
        default=270,
        help="Hard cap on completed rollout attempts, including invalid episodes.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=1500)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--near-target-distance", type=float, default=0.08)
    parser.add_argument("--max-samples-per-episode", type=int, default=64)
    parser.add_argument("--min-samples-per-episode", type=int, default=8)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--probe-epochs", type=int, default=400)
    parser.add_argument("--probe-min-epochs", type=int, default=40)
    parser.add_argument("--probe-patience", type=int, default=40)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--sim-backend", type=str, default="physx_cuda")
    parser.add_argument("--render-backend", type=str, default="sapien_cuda")
    parser.add_argument("--point-cloud-sampling", type=str, default="fast")
    parser.add_argument(
        "--delta-pose-hold-mode",
        choices=("off", "deadband", "anchor"),
        default="anchor",
    )
    parser.add_argument("--delta-pose-hold-gripper-threshold", type=float, default=0.95)
    parser.add_argument("--delta-pose-hold-pos-threshold", type=float, default=0.002)
    parser.add_argument("--delta-pose-hold-rot-threshold", type=float, default=0.02)
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(REPO_ROOT / "tmp/handover/multiview_geometry_probe_summary.json"),
    )
    return parser


def _shape_meta_shape(meta) -> tuple[int, ...]:
    shape = meta["shape"] if isinstance(meta, Mapping) else meta.shape
    return tuple(int(dim) for dim in shape)


def _checkpoint_obs_shapes(ckpt_cfg) -> dict[str, tuple[int, ...]]:
    return {
        str(key): _shape_meta_shape(meta)
        for key, meta in _shape_meta_obs_items(ckpt_cfg)
    }


def _resolve_runtime(args, ckpt_cfg):
    runtime_env_id = args.task_name
    if args.task_profile is not None and args.task_name == args.task_profile:
        runtime_env_id = None
    runtime_task_name, resolved_profile_name = resolve_runtime_task_target(
        task_name=args.task_name,
        task_profile=args.task_profile,
        env_id=runtime_env_id,
    )
    runtime_kwargs = {
        "action_type": None,
        "obs_mode": None,
        "control_mode": None,
        "render_mode": None,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "reward_mode": None,
        "robot_uids": None,
        "sim_freq": None,
        "control_freq": None,
        "max_episode_steps": args.max_episode_steps,
        "point_cloud_sampling": args.point_cloud_sampling,
    }
    _adapt_runtime_for_checkpoint_obs(ckpt_cfg, runtime_kwargs)
    _adapt_runtime_for_checkpoint_action(ckpt_cfg, runtime_kwargs)
    action_type, control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["action_type"] = action_type
    runtime_kwargs["control_mode"] = control_mode
    profile_key = resolved_profile_name or runtime_task_name
    runtime_task_name, resolved_runtime, resolved_profile_name = apply_task_profile(
        profile_key,
        runtime_kwargs,
    )
    return runtime_task_name, resolved_profile_name, resolved_runtime


def _create_vector_environment(args, ckpt_cfg):
    runtime_task_name, resolved_profile_name, runtime = _resolve_runtime(args, ckpt_cfg)
    shape_meta = OmegaConf.to_container(ckpt_cfg.shape_meta, resolve=True)
    env = ManiSkillCudaFPORunner(
        task_name=runtime_task_name,
        shape_meta=shape_meta,
        task_profile=resolved_profile_name,
        env_id=runtime_task_name,
        num_envs=args.num_envs,
        base_seed=args.seed,
        action_type=runtime["action_type"],
        obs_mode=runtime["obs_mode"],
        control_mode=runtime["control_mode"],
        render_mode=runtime["render_mode"],
        reward_mode=runtime["reward_mode"],
        robot_uids=runtime["robot_uids"],
        sim_backend=runtime["sim_backend"],
        render_backend=runtime["render_backend"],
        sim_freq=runtime["sim_freq"],
        control_freq=runtime["control_freq"],
        point_cloud_sampling=runtime["point_cloud_sampling"],
        max_episode_steps=args.max_episode_steps,
    )
    return env, runtime


def _unwrap_compiled_module(module):
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


@torch.inference_mode()
def _extract_visual_features(policy, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    obs_encoder = _unwrap_compiled_module(policy.obs_encoder)
    if not all(hasattr(obs_encoder, name) for name in ("pointnet", "fusion", "view_keys")):
        raise TypeError(
            "This probe requires MultiViewPointCloudObsEncoder with pointnet/fusion/view_keys."
        )
    view_keys = tuple(str(key) for key in obs_encoder.view_keys)
    if view_keys != VIEW_KEYS:
        raise ValueError(f"Unexpected multiview key order: {view_keys}.")

    normalized_views = policy.normalizer.normalize({key: obs[key] for key in view_keys})
    view_features = [obs_encoder.pointnet(normalized_views[key]) for key in view_keys]
    return obs_encoder.fusion(torch.cat(view_features, dim=-1)).float()


@torch.inference_mode()
def _geometry_labels_and_stages(env) -> tuple[torch.Tensor, torch.Tensor]:
    task_env = env.env.unwrapped
    box_center = task_env.box.pose.p.to(device=env.device, dtype=torch.float32)
    target_center = task_env.target_box.pose.p.to(device=env.device, dtype=torch.float32)
    if box_center.shape != (env.num_envs, 3) or target_center.shape != box_center.shape:
        raise ValueError(
            "Unexpected vector box/target pose shapes: "
            f"box={tuple(box_center.shape)}, target={tuple(target_center.shape)}."
        )

    box_bottom = box_center.clone()
    target_top = target_center.clone()
    box_bottom[:, 2] -= float(BOX_HALF_SIZE[2])
    target_top[:, 2] += float(TARGET_HALF_SIZE[2])
    labels = target_top - box_bottom
    stages = task_env._get_reward_stage().to(device=env.device, dtype=torch.long).reshape(-1)
    return labels, stages


def _normalize_pose_wxyz(pose: torch.Tensor) -> torch.Tensor:
    quat = pose[..., 3:7]
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat([pose[..., 0:3], quat], dim=-1)


def _quaternion_inverse_wxyz(quat: torch.Tensor) -> torch.Tensor:
    norm_sq = torch.sum(quat * quat, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cat([quat[..., 0:1], -quat[..., 1:]], dim=-1) / norm_sq


def _target_pose_from_delta(current_pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    delta_quat = matrix_to_quaternion(euler_angles_to_matrix(delta[:, 3:6], "XYZ"))
    target_quat = quaternion_multiply(delta_quat, current_pose[:, 3:7])
    return _normalize_pose_wxyz(
        torch.cat([current_pose[:, 0:3] + delta[:, 0:3], target_quat], dim=-1)
    )


def _delta_from_target_pose(current_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    delta_quat = quaternion_multiply(
        target_pose[:, 3:7],
        _quaternion_inverse_wxyz(current_pose[:, 3:7]),
    )
    delta_rot = matrix_to_euler_angles(quaternion_to_matrix(delta_quat), "XYZ")
    return torch.cat([target_pose[:, 0:3] - current_pose[:, 0:3], delta_rot], dim=-1)


class _BatchedDeltaPoseHoldFilter:
    """Vector equivalent of the eval-only delta-pose hold filter."""

    def __init__(
        self,
        *,
        mode: str,
        num_envs: int,
        device: torch.device,
        gripper_threshold: float,
        pos_threshold: float,
        rot_threshold: float,
    ) -> None:
        self.mode = str(mode)
        self.num_envs = int(num_envs)
        self.device = device
        self.gripper_threshold = float(gripper_threshold)
        self.pos_threshold = float(pos_threshold)
        self.rot_threshold = float(rot_threshold)
        self.anchors = {
            controller_attr: torch.zeros((num_envs, 7), device=device, dtype=torch.float32)
            for _action_slice, _gripper_idx, controller_attr in DELTA_POSE_ARM_SPECS
        }
        self.anchor_valid = {
            controller_attr: torch.zeros(num_envs, device=device, dtype=torch.bool)
            for _action_slice, _gripper_idx, controller_attr in DELTA_POSE_ARM_SPECS
        }

    def reset_envs(self, env_ids: Sequence[int]) -> None:
        if not env_ids:
            return
        indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for valid in self.anchor_valid.values():
            valid[indices] = False

    def _current_pose(self, env, controller_attr: str) -> torch.Tensor:
        controller = getattr(env.env.unwrapped, controller_attr, None)
        raw_pose = getattr(getattr(controller, "ee_pose_at_base", None), "raw_pose", None)
        if raw_pose is None:
            raise RuntimeError(f"Delta hold filter cannot read `{controller_attr}.ee_pose_at_base`.")
        pose = raw_pose.detach().to(device=self.device, dtype=torch.float32).reshape(-1, 7)
        if pose.shape != (self.num_envs, 7):
            raise ValueError(
                f"Unexpected `{controller_attr}` pose shape {tuple(pose.shape)}; "
                f"expected {(self.num_envs, 7)}."
            )
        return _normalize_pose_wxyz(pose)

    @torch.inference_mode()
    def apply(self, actions: torch.Tensor, env) -> torch.Tensor:
        actions = actions.detach().to(device=self.device, dtype=torch.float32).clone()
        if actions.shape != (self.num_envs, 14):
            raise ValueError(
                f"Delta hold filter expected {(self.num_envs, 14)}, got {tuple(actions.shape)}."
            )

        for action_slice, gripper_idx, controller_attr in DELTA_POSE_ARM_SPECS:
            delta = actions[:, action_slice]
            is_open = actions[:, gripper_idx] > self.gripper_threshold
            is_small = (
                torch.linalg.norm(delta[:, 0:3], dim=-1) <= self.pos_threshold
            ) & (torch.linalg.norm(delta[:, 3:6], dim=-1) <= self.rot_threshold)

            if self.mode == "deadband":
                actions[is_open & is_small, action_slice] = 0.0
                continue

            current_pose = self._current_pose(env, controller_attr)
            anchors = self.anchors[controller_attr]
            valid = self.anchor_valid[controller_attr]
            is_closed = ~is_open
            anchors[is_closed] = current_pose[is_closed]
            valid[is_closed] = True

            initialize = is_open & (~valid)
            anchors[initialize] = current_pose[initialize]
            valid[initialize] = True

            held = is_open & is_small
            held_delta = _delta_from_target_pose(current_pose, anchors)
            actions[:, action_slice] = torch.where(
                held[:, None],
                held_delta,
                actions[:, action_slice],
            )

            committed = is_open & (~is_small)
            committed_target = _target_pose_from_delta(current_pose, delta)
            anchors[:] = torch.where(
                committed[:, None],
                committed_target,
                anchors,
            )
        return actions


def _make_hold_filter(args, env, runtime):
    if args.delta_pose_hold_mode == "off":
        return None
    if DELTA_POSE_CONTROL_MODE not in (
        str(runtime.get("action_type") or ""),
        str(runtime.get("control_mode") or ""),
    ):
        raise ValueError("Delta-pose hold filtering requires the native delta-pose control mode.")
    return _BatchedDeltaPoseHoldFilter(
        mode=args.delta_pose_hold_mode,
        num_envs=env.num_envs,
        device=env.device,
        gripper_threshold=args.delta_pose_hold_gripper_threshold,
        pos_threshold=args.delta_pose_hold_pos_threshold,
        rot_threshold=args.delta_pose_hold_rot_threshold,
    )


def _init_obs_history(obs: Mapping[str, torch.Tensor], n_steps: int):
    return {
        key: value.unsqueeze(1).expand(-1, n_steps, *value.shape[1:]).clone()
        for key, value in obs.items()
    }


def _append_obs_history(obs_history, obs):
    return {
        key: torch.cat([value[:, 1:], obs[key].unsqueeze(1)], dim=1)
        for key, value in obs_history.items()
    }


def _reset_obs_history_env(obs_history, reset_obs, env_id: int) -> None:
    for key, value in obs_history.items():
        reset_value = reset_obs[key][env_id]
        value[env_id] = reset_value.unsqueeze(0).expand(value.shape[1], *reset_value.shape)


def _info_bool_tensor(info, key: str, *, num_envs: int, device: torch.device):
    if not isinstance(info, Mapping) or key not in info:
        return torch.zeros(num_envs, device=device, dtype=torch.bool)
    return torch.as_tensor(info[key], device=device, dtype=torch.bool).reshape(num_envs)


def _canonical_device(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    if device.type != "cuda":
        return device.type, device.index
    index = torch.cuda.current_device() if device.index is None else device.index
    return device.type, int(index)


def _new_episode_reservoir(seed: int) -> dict:
    return {
        "rng": np.random.default_rng(int(seed)),
        "candidate_count": 0,
        "features": [],
        "labels": [],
        "stages": [],
        "steps": [],
    }


def _reservoir_add(
    reservoir: dict,
    *,
    feature: np.ndarray,
    label: np.ndarray,
    stage: int,
    step: int,
    capacity: int,
) -> None:
    reservoir["candidate_count"] += 1
    if len(reservoir["features"]) < capacity:
        index = len(reservoir["features"])
        reservoir["features"].append(feature)
        reservoir["labels"].append(label)
        reservoir["stages"].append(int(stage))
        reservoir["steps"].append(int(step))
        return

    index = int(reservoir["rng"].integers(0, reservoir["candidate_count"]))
    if index >= capacity:
        return
    reservoir["features"][index] = feature
    reservoir["labels"][index] = label
    reservoir["stages"][index] = int(stage)
    reservoir["steps"][index] = int(step)


@torch.inference_mode()
def _collect_rollout_features(args, ckpt_cfg, policy):
    obs_shapes = _checkpoint_obs_shapes(ckpt_cfg)
    if not all(key in obs_shapes for key in VIEW_KEYS):
        raise ValueError(f"Checkpoint is missing required multiview keys {VIEW_KEYS}.")
    if int(ckpt_cfg.n_action_steps) != 1:
        raise ValueError("The vector geometry probe requires checkpoint n_action_steps=1.")

    env, runtime = _create_vector_environment(args, ckpt_cfg)
    if _canonical_device(env.device) != _canonical_device(policy.device):
        env.close()
        raise RuntimeError(
            f"Policy and vector environment must share one device: "
            f"policy={policy.device}, env={env.device}."
        )
    hold_filter = _make_hold_filter(args, env, runtime)
    obs, _ = env.reset()
    obs_history = _init_obs_history(obs, int(ckpt_cfg.n_obs_steps))
    episode_steps = np.zeros(env.num_envs, dtype=np.int64)
    episode_generations = np.zeros(env.num_envs, dtype=np.int64)
    episode_seeds = np.asarray(
        [args.seed + env_id for env_id in range(env.num_envs)],
        dtype=np.int64,
    )
    reservoirs = [
        _new_episode_reservoir(seed + 104729)
        for seed in episode_seeds.tolist()
    ]

    features = []
    labels = []
    stages = []
    episode_ids = []
    steps = []
    episode_summary = []
    valid_episodes = 0
    completed_attempts = 0
    vector_steps = 0
    start_time = time.perf_counter()

    try:
        while (
            valid_episodes < args.episodes
            and completed_attempts < args.max_rollout_episodes
        ):
            current_labels, current_stages = _geometry_labels_and_stages(env)
            distances = torch.linalg.norm(current_labels, dim=-1)
            eligible = distances < args.near_target_distance
            if vector_steps % args.sample_every == 0 and bool(eligible.any()):
                visual_features = _extract_visual_features(policy, obs)
                eligible_ids = eligible.nonzero(as_tuple=False).flatten()
                feature_values = visual_features[eligible_ids].detach().cpu().numpy()
                label_values = current_labels[eligible_ids].detach().cpu().numpy()
                stage_values = current_stages[eligible_ids].detach().cpu().numpy()
                for offset, env_id_value in enumerate(eligible_ids.detach().cpu().tolist()):
                    _reservoir_add(
                        reservoirs[env_id_value],
                        feature=feature_values[offset].astype(np.float32, copy=True),
                        label=label_values[offset].astype(np.float32, copy=True),
                        stage=int(stage_values[offset]),
                        step=int(episode_steps[env_id_value]),
                        capacity=args.max_samples_per_episode,
                    )

            prediction = policy.predict_action(obs_history)
            action_chunk = prediction["action"]
            if action_chunk.shape[:2] != (env.num_envs, 1):
                raise ValueError(
                    f"Expected policy action shape [B,1,Da], got {tuple(action_chunk.shape)}."
                )
            actions = action_chunk[:, 0].detach()
            if hold_filter is not None:
                actions = hold_filter.apply(actions, env)
            actions = env.clip_actions(actions)

            next_obs, _reward, terminated, truncated, info = env.step(actions)
            episode_steps += 1
            vector_steps += 1
            success = _info_bool_tensor(
                info,
                "success",
                num_envs=env.num_envs,
                device=env.device,
            )
            reached_limit = torch.as_tensor(
                episode_steps >= args.max_episode_steps,
                device=env.device,
                dtype=torch.bool,
            )
            finished = terminated | truncated | success | reached_limit
            next_obs_history = _append_obs_history(obs_history, next_obs)

            finished_ids = finished.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            stop_collection = False
            for env_id in finished_ids:
                if completed_attempts >= args.max_rollout_episodes:
                    stop_collection = True
                    break
                completed_attempts += 1
                reservoir = reservoirs[env_id]
                sample_count = len(reservoir["features"])
                is_valid = sample_count >= args.min_samples_per_episode
                assigned_episode_id = None
                if is_valid and valid_episodes < args.episodes:
                    assigned_episode_id = valid_episodes
                    features.extend(reservoir["features"])
                    labels.extend(reservoir["labels"])
                    stages.extend(reservoir["stages"])
                    steps.extend(reservoir["steps"])
                    episode_ids.extend([valid_episodes] * sample_count)
                    valid_episodes += 1

                episode_summary.append(
                    {
                        "attempt": completed_attempts - 1,
                        "seed": int(episode_seeds[env_id]),
                        "length": int(episode_steps[env_id]),
                        "success": bool(success[env_id].item()),
                        "candidate_count": int(reservoir["candidate_count"]),
                        "sample_count": int(sample_count),
                        "valid": bool(is_valid),
                        "valid_episode_id": assigned_episode_id,
                    }
                )
                print(
                    "collect "
                    f"attempts={completed_attempts}/{args.max_rollout_episodes} "
                    f"valid={valid_episodes}/{args.episodes} "
                    f"seed={episode_seeds[env_id]} length={episode_steps[env_id]} "
                    f"near_candidates={reservoir['candidate_count']} samples={sample_count}",
                    flush=True,
                )

                if valid_episodes >= args.episodes:
                    stop_collection = True
                    break

                episode_generations[env_id] += 1
                episode_seeds[env_id] = (
                    args.seed + episode_generations[env_id] * env.num_envs + env_id
                )
                reset_obs, _ = env.reset_at([env_id])
                for key in next_obs:
                    next_obs[key][env_id] = reset_obs[key][env_id]
                _reset_obs_history_env(next_obs_history, reset_obs, env_id)
                if hold_filter is not None:
                    hold_filter.reset_envs([env_id])
                episode_steps[env_id] = 0
                reservoirs[env_id] = _new_episode_reservoir(
                    int(episode_seeds[env_id]) + 104729
                )

            obs = next_obs
            obs_history = next_obs_history
            if stop_collection:
                break
    finally:
        env.close()

    elapsed_seconds = time.perf_counter() - start_time
    if valid_episodes < args.episodes:
        raise RuntimeError(
            f"Collected only {valid_episodes}/{args.episodes} valid episodes after "
            f"{completed_attempts} attempts. Increase --max-rollout-episodes or inspect "
            "the near-target policy behavior."
        )
    return {
        "features": np.stack(features).astype(np.float32),
        "labels": np.stack(labels).astype(np.float32),
        "stages": np.asarray(stages, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "steps": np.asarray(steps, dtype=np.int64),
        "episode_summary": episode_summary,
        "completed_attempts": int(completed_attempts),
        "vector_steps": int(vector_steps),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _split_episode_ids(args):
    ids = np.arange(args.episodes, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    train_end = args.train_episodes
    val_end = train_end + args.val_episodes
    return (
        np.sort(ids[:train_end]),
        np.sort(ids[train_end:val_end]),
        np.sort(ids[val_end:]),
    )


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    absolute_error = np.abs(error)
    euclidean_error = np.linalg.norm(error, axis=1)
    return {
        "count": int(target.shape[0]),
        "axis_mae_cm": (absolute_error.mean(axis=0) * 100.0).tolist(),
        "axis_rmse_cm": (np.sqrt(np.mean(np.square(error), axis=0)) * 100.0).tolist(),
        "euclidean_mean_cm": float(euclidean_error.mean() * 100.0),
        "euclidean_median_cm": float(np.median(euclidean_error) * 100.0),
        "euclidean_p90_cm": float(np.quantile(euclidean_error, 0.9) * 100.0),
        "task_tolerance_frac": float(
            np.mean(
                (absolute_error[:, 0] < 0.03)
                & (absolute_error[:, 1] < 0.03)
                & (absolute_error[:, 2] < 0.01)
            )
        ),
    }


def _predict_probe(model, features, feature_mean, feature_std, label_mean, label_std, device):
    x = torch.as_tensor(
        (features - feature_mean) / feature_std,
        device=device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        prediction = model(x).cpu().numpy()
    return prediction * label_std + label_mean


def _fit_probe(
    name,
    probe,
    train_features,
    train_labels,
    val_features,
    val_labels,
    test_features,
    test_labels,
    args,
    device,
):
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6)
    label_mean = train_labels.mean(axis=0, keepdims=True)
    label_std = np.maximum(train_labels.std(axis=0, keepdims=True), 1e-4)
    train_x = torch.as_tensor(
        (train_features - feature_mean) / feature_std,
        device=device,
        dtype=torch.float32,
    )
    train_y = torch.as_tensor(
        (train_labels - label_mean) / label_std,
        device=device,
        dtype=torch.float32,
    )
    val_x = torch.as_tensor(
        (val_features - feature_mean) / feature_std,
        device=device,
        dtype=torch.float32,
    )
    val_y = torch.as_tensor(
        (val_labels - label_mean) / label_std,
        device=device,
        dtype=torch.float32,
    )

    probe = probe.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    stale_epochs = 0

    for epoch in range(args.probe_epochs):
        probe.train()
        permutation = torch.randperm(train_x.shape[0], generator=generator)
        for start in range(0, train_x.shape[0], args.probe_batch_size):
            indices = permutation[start : start + args.probe_batch_size].to(device=device)
            prediction = probe(train_x[indices])
            loss = torch.mean(torch.square(prediction - train_y[indices]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.inference_mode():
            val_loss = float(torch.mean(torch.square(probe(val_x) - val_y)).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        else:
            stale_epochs += 1
        if (
            epoch + 1 >= args.probe_min_epochs
            and stale_epochs >= args.probe_patience
        ):
            break

    if best_state is None:
        raise RuntimeError(f"Probe `{name}` did not produce a finite validation checkpoint.")
    probe.load_state_dict(best_state)
    probe.eval()
    split_predictions = {
        "train": _predict_probe(
            probe,
            train_features,
            feature_mean,
            feature_std,
            label_mean,
            label_std,
            device,
        ),
        "validation": _predict_probe(
            probe,
            val_features,
            feature_mean,
            feature_std,
            label_mean,
            label_std,
            device,
        ),
        "test": _predict_probe(
            probe,
            test_features,
            feature_mean,
            feature_std,
            label_mean,
            label_std,
            device,
        ),
    }
    return {
        "name": name,
        "parameter_count": int(sum(parameter.numel() for parameter in probe.parameters())),
        "best_epoch": int(best_epoch + 1),
        "epochs_ran": int(epoch + 1),
        "best_validation_normalized_mse": float(best_val_loss),
        "train": _regression_metrics(train_labels, split_predictions["train"]),
        "validation": _regression_metrics(val_labels, split_predictions["validation"]),
        "test": _regression_metrics(test_labels, split_predictions["test"]),
    }


def _fit_probes(collected, train_ids, val_ids, test_ids, args, device):
    features = collected["features"]
    labels = collected["labels"]
    episode_ids = collected["episode_ids"]
    masks = {
        "train": np.isin(episode_ids, train_ids),
        "validation": np.isin(episode_ids, val_ids),
        "test": np.isin(episode_ids, test_ids),
    }
    split_data = {
        split: (features[mask], labels[mask])
        for split, mask in masks.items()
    }
    if any(values[0].shape[0] == 0 for values in split_data.values()):
        raise RuntimeError("Train/validation/test split contains an empty frame set.")

    train_features, train_labels = split_data["train"]
    val_features, val_labels = split_data["validation"]
    test_features, test_labels = split_data["test"]
    baseline_mean = train_labels.mean(axis=0, keepdims=True)
    baseline = {
        "name": "train_mean",
        "validation": _regression_metrics(
            val_labels,
            np.broadcast_to(baseline_mean, val_labels.shape),
        ),
        "test": _regression_metrics(
            test_labels,
            np.broadcast_to(baseline_mean, test_labels.shape),
        ),
    }
    feature_dim = int(features.shape[1])

    torch.manual_seed(args.seed)
    linear = _fit_probe(
        "linear_512_to_3",
        nn.Linear(feature_dim, 3),
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        args,
        device,
    )
    torch.manual_seed(args.seed)
    mlp = _fit_probe(
        "mlp_512_to_128_to_3",
        nn.Sequential(
            nn.Linear(feature_dim, MLP_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(MLP_HIDDEN_DIM, 3),
        ),
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        args,
        device,
    )
    return {
        "frame_counts": {
            split: int(values[0].shape[0])
            for split, values in split_data.items()
        },
        "baseline": baseline,
        "linear": linear,
        "mlp": mlp,
    }


def _print_probe_result(result: dict) -> None:
    for key in ("baseline", "linear", "mlp"):
        probe = result[key]
        test = probe["test"]
        axis = test["axis_mae_cm"]
        extra = ""
        if "best_epoch" in probe:
            extra = f" best_epoch={probe['best_epoch']} epochs_ran={probe['epochs_ran']}"
        print(
            f"probe model={probe['name']} split=test count={test['count']} "
            f"axis_mae_cm=[{axis[0]:.3f},{axis[1]:.3f},{axis[2]:.3f}] "
            f"euclidean_mean_cm={test['euclidean_mean_cm']:.3f} "
            f"p90_cm={test['euclidean_p90_cm']:.3f} "
            f"task_tolerance_frac={test['task_tolerance_frac']:.3f}{extra}",
            flush=True,
        )


def _validate_args(args) -> None:
    split_total = args.train_episodes + args.val_episodes + args.test_episodes
    if split_total != args.episodes:
        raise ValueError(
            "--train-episodes + --val-episodes + --test-episodes must equal --episodes; "
            f"got {split_total} != {args.episodes}."
        )
    positive_names = (
        "episodes",
        "train_episodes",
        "val_episodes",
        "test_episodes",
        "num_envs",
        "max_rollout_episodes",
        "max_episode_steps",
        "sample_every",
        "max_samples_per_episode",
        "min_samples_per_episode",
        "probe_epochs",
        "probe_min_epochs",
        "probe_patience",
        "probe_batch_size",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.min_samples_per_episode > args.max_samples_per_episode:
        raise ValueError("--min-samples-per-episode cannot exceed --max-samples-per-episode.")
    if args.max_rollout_episodes < args.episodes:
        raise ValueError("--max-rollout-episodes cannot be smaller than --episodes.")
    if args.num_envs > args.max_rollout_episodes:
        raise ValueError("--num-envs cannot exceed --max-rollout-episodes.")
    if args.near_target_distance <= 0.0:
        raise ValueError("--near-target-distance must be positive meters.")


def run(args) -> None:
    _validate_args(args)
    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    ckpt_cfg = payload["cfg"]
    adapter = importlib.import_module(ADAPTER_MODULES["flowmatching"])
    model = adapter.get_model({"ckpt_path": str(ckpt_path)})
    policy = model.policy
    policy.eval()
    policy.requires_grad_(False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    collected = _collect_rollout_features(args, ckpt_cfg, policy)
    train_ids, val_ids, test_ids = _split_episode_ids(args)
    probe_start = time.perf_counter()
    probes = _fit_probes(
        collected,
        train_ids,
        val_ids,
        test_ids,
        args,
        policy.device,
    )
    probe_elapsed = time.perf_counter() - probe_start

    label_distances = np.linalg.norm(collected["labels"], axis=1)
    summary = {
        "checkpoint": str(ckpt_path),
        "task_name": args.task_name,
        "task_profile": args.task_profile,
        "label_semantics": "target_top_center_minus_box_bottom_center_world_xyz_m",
        "feature_semantics": "frozen_normalized_multiview_visual_feature_before_agent_pos",
        "feature_dim": int(collected["features"].shape[1]),
        "sample_count": int(collected["features"].shape[0]),
        "valid_episode_count": int(args.episodes),
        "completed_rollout_attempts": collected["completed_attempts"],
        "vector_steps": collected["vector_steps"],
        "collection_elapsed_seconds": collected["elapsed_seconds"],
        "probe_elapsed_seconds": float(probe_elapsed),
        "near_target_distance_cm": float(args.near_target_distance * 100.0),
        "sample_distance_cm": {
            "mean": float(label_distances.mean() * 100.0),
            "max": float(label_distances.max() * 100.0),
        },
        "train_episode_ids": train_ids.tolist(),
        "validation_episode_ids": val_ids.tolist(),
        "test_episode_ids": test_ids.tolist(),
        "episode_summary": collected["episode_summary"],
        "probes": probes,
        "settings": {
            "seed": args.seed,
            "num_envs": args.num_envs,
            "max_episode_steps": args.max_episode_steps,
            "sample_every": args.sample_every,
            "max_samples_per_episode": args.max_samples_per_episode,
            "min_samples_per_episode": args.min_samples_per_episode,
            "probe_epochs": args.probe_epochs,
            "probe_min_epochs": args.probe_min_epochs,
            "probe_patience": args.probe_patience,
            "probe_batch_size": args.probe_batch_size,
            "probe_lr": args.probe_lr,
            "probe_weight_decay": args.probe_weight_decay,
            "mlp_hidden_dim": MLP_HIDDEN_DIM,
            "delta_pose_hold_mode": args.delta_pose_hold_mode,
        },
    }
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(
        f"dataset valid_episodes={args.episodes} samples={collected['features'].shape[0]} "
        f"attempts={collected['completed_attempts']} feature_dim={collected['features'].shape[1]} "
        f"collection_minutes={collected['elapsed_seconds'] / 60.0:.2f} "
        f"probe_seconds={probe_elapsed:.2f}",
        flush=True,
    )
    _print_probe_result(probes)
    print(f"summary_saved path={output_path}", flush=True)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
