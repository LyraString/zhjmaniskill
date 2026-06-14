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


ADAPTER_MODULES = {
    "flowmatching": "policy.FlowMatching.deploy_maniskill",
    "fpo": "policy.FPO.deploy_maniskill",
}

TASK_CACHE_ATTRS = ("reward_stage", "left_gripper_val", "right_gripper_val")
DEFAULT_TARGET_STAGES = (2, 3, 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether HandoverBlockUR5-v2 supports state restore for "
            "counterfactual coupling labels."
        )
    )
    parser.add_argument("policy_kind", choices=sorted(ADAPTER_MODULES.keys()))
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("task_name", type=str)
    parser.add_argument("--task-profile", default="handover_block_ur5_v2_smoke")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--probe-points-per-episode", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--identity-threshold", type=float, default=0.002)
    parser.add_argument("--coupled-threshold", type=float, default=0.02)
    parser.add_argument("--max-episode-steps", type=int, default=900)
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
    infos = []
    terminated = False
    truncated = False
    for action in action_seq:
        result = env.step(ActionSpec.from_value(action, action_type=action_type))
        positions.append(_box_pos(env.env.unwrapped))
        infos.append(_compact_info(result.info))
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)
        if terminated or truncated:
            break
    return {
        "box_positions": np.asarray(positions, dtype=np.float64),
        "infos": infos,
        "steps": int(len(positions)),
        "terminated": terminated,
        "truncated": truncated,
    }


def _trajectory_max_dev(reference: np.ndarray, candidate: np.ndarray) -> float:
    steps = min(reference.shape[0], candidate.shape[0])
    if steps <= 0:
        return float("nan")
    return float(np.linalg.norm(reference[:steps] - candidate[:steps], axis=1).max())


def _trajectory_final_dev(reference: np.ndarray, candidate: np.ndarray) -> float:
    steps = min(reference.shape[0], candidate.shape[0])
    if steps <= 0:
        return float("nan")
    return float(np.linalg.norm(reference[steps - 1] - candidate[steps - 1]))


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _frac(values: list[bool]) -> float:
    if not values:
        return float("nan")
    return float(np.mean([float(value) for value in values]))


def _stage_histogram(probes: list[dict[str, Any]]) -> dict[str, int]:
    histogram = {str(idx): 0 for idx in range(5)}
    histogram["unknown"] = 0
    for probe in probes:
        stage = int(probe.get("stage", -1))
        if 0 <= stage <= 4:
            histogram[str(stage)] += 1
        else:
            histogram["unknown"] += 1
    return histogram


def _probe_devs(probes: list[dict[str, Any]], key: str) -> list[float]:
    return [float(probe[key]) for probe in probes]


def _coupled_frac(probes: list[dict[str, Any]], key: str, threshold: float) -> float:
    devs = [float(probe[key]) for probe in probes if np.isfinite(float(probe[key]))]
    return _frac([dev > float(threshold) for dev in devs])


def _probe_from_snapshot(
    *,
    env,
    snapshot: dict[str, Any],
    action_seq: np.ndarray,
    action_type: str | None,
    noise_std: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    unwrapped = env.env.unwrapped

    _restore_snapshot(unwrapped, snapshot)
    baseline = _rollout_action_sequence(env, action_seq, action_type)

    _restore_snapshot(unwrapped, snapshot)
    identity = _rollout_action_sequence(env, action_seq, action_type)

    branches = {"baseline": baseline, "identity": identity}
    for branch in ("left_noise", "right_noise", "both_noise"):
        noisy_actions = _add_noise(
            action_seq,
            branch=branch,
            noise_std=noise_std,
            rng=rng,
            action_space=env.action_space,
        )
        _restore_snapshot(unwrapped, snapshot)
        branches[branch] = _rollout_action_sequence(env, noisy_actions, action_type)

    baseline_pos = branches["baseline"]["box_positions"]
    identity_pos = branches["identity"]["box_positions"]
    result = {
        "baseline_steps": branches["baseline"]["steps"],
        "identity_steps": branches["identity"]["steps"],
        "restore_identity_max_dev": _trajectory_max_dev(baseline_pos, identity_pos),
        "restore_identity_final_dev": _trajectory_final_dev(baseline_pos, identity_pos),
    }
    for branch in ("left_noise", "right_noise", "both_noise"):
        branch_pos = branches[branch]["box_positions"]
        result[f"{branch}_steps"] = branches[branch]["steps"]
        result[f"{branch}_max_dev"] = _trajectory_max_dev(baseline_pos, branch_pos)
        result[f"{branch}_final_dev"] = _trajectory_final_dev(baseline_pos, branch_pos)
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
        # Branch rollouts also advance Gym/ManiSkill step counters. Use a
        # larger backend limit and enforce the main rollout limit ourselves.
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
    except Exception as exc:  # noqa: BLE001 - report probe feasibility, do not crash.
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def run_probe(args) -> dict[str, Any]:
    adapter, model, ckpt_cfg = _load_policy(args)
    env, resolved_runtime, runtime_task_name = _build_env(args)
    rng = np.random.default_rng(int(args.seed))
    target_stages = set(DEFAULT_TARGET_STAGES)
    probes = []

    try:
        obs = env.reset(seed=int(args.seed))
        supported, reason = _state_restore_support(env.env.unwrapped)
        if not supported:
            return {
                "state_restore_supported": False,
                "state_restore_error": reason,
                "recommended_next_step": "stop_counterfactual_capo",
            }

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
                    action_seq = _make_action_sequence(action_chunk, int(args.horizon))
                    probe_result = _probe_from_snapshot(
                        env=env,
                        snapshot=snapshot,
                        action_seq=action_seq,
                        action_type=resolved_runtime.get("action_type"),
                        noise_std=float(args.noise_std),
                        rng=rng,
                    )
                    _restore_snapshot(env.env.unwrapped, snapshot)
                    identity_dev = float(probe_result["restore_identity_max_dev"])
                    probe_result.update(
                        {
                            "episode": int(episode_idx),
                            "step": int(episode_length),
                            "stage": int(stage),
                            "identity_pass": bool(
                                np.isfinite(identity_dev)
                                and identity_dev < float(args.identity_threshold)
                            ),
                            "box_pos": _box_pos(env.env.unwrapped).round(6).tolist(),
                            "info": _compact_info(last_info),
                        }
                    )
                    probes.append(probe_result)
                    probe_count += 1
                    next_probe_at += probe_interval

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

    identity_devs = _probe_devs(probes, "restore_identity_max_dev")
    identity_passes = [
        float(dev < float(args.identity_threshold))
        for dev in identity_devs
        if np.isfinite(dev)
    ]
    valid_probes = [probe for probe in probes if bool(probe.get("identity_pass", False))]
    invalid_identity_count = int(len(probes) - len(valid_probes))
    left_devs = _probe_devs(probes, "left_noise_max_dev")
    right_devs = _probe_devs(probes, "right_noise_max_dev")
    both_devs = _probe_devs(probes, "both_noise_max_dev")
    valid_left_devs = _probe_devs(valid_probes, "left_noise_max_dev")
    valid_right_devs = _probe_devs(valid_probes, "right_noise_max_dev")
    valid_both_devs = _probe_devs(valid_probes, "both_noise_max_dev")
    identity_pass_rate = float(np.mean(identity_passes)) if identity_passes else 0.0
    valid_probe_frac = float(len(valid_probes) / len(probes)) if probes else 0.0
    valid_has_noise_signal = (
        max(_mean(valid_left_devs), _mean(valid_right_devs), _mean(valid_both_devs))
        > float(args.coupled_threshold)
    )
    min_valid_probe_count = max(3, min(6, int(args.episodes) * 2))
    if probes and valid_probe_frac < 0.75:
        recommended_next_step = "debug_state_restore_before_labeling"
    elif len(valid_probes) < min_valid_probe_count:
        recommended_next_step = "rerun_with_more_episodes_or_probe_points"
    elif valid_has_noise_signal:
        recommended_next_step = "build_counterfactual_label_generator"
    else:
        recommended_next_step = "adjust_noise_or_horizon_and_rerun_probe"

    return {
        "state_restore_supported": True,
        "task_name": runtime_task_name,
        "task_profile": args.task_profile,
        "policy_kind": args.policy_kind,
        "ckpt_path": str(args.ckpt_path),
        "episodes": int(args.episodes),
        "probe_count": int(len(probes)),
        "valid_probe_count": int(len(valid_probes)),
        "invalid_identity_count": invalid_identity_count,
        "valid_probe_frac": valid_probe_frac,
        "horizon": int(args.horizon),
        "noise_std": float(args.noise_std),
        "target_stages": sorted(target_stages),
        "identity_threshold": float(args.identity_threshold),
        "coupled_threshold": float(args.coupled_threshold),
        "restore_identity_pass_rate": identity_pass_rate,
        "restore_identity_max_dev_mean": _mean(identity_devs),
        "left_noise_max_dev_mean": _mean(left_devs),
        "right_noise_max_dev_mean": _mean(right_devs),
        "both_noise_max_dev_mean": _mean(both_devs),
        "valid_left_noise_max_dev_mean": _mean(valid_left_devs),
        "valid_right_noise_max_dev_mean": _mean(valid_right_devs),
        "valid_both_noise_max_dev_mean": _mean(valid_both_devs),
        "valid_left_coupled_frac": _coupled_frac(
            valid_probes, "left_noise_max_dev", float(args.coupled_threshold)
        ),
        "valid_right_coupled_frac": _coupled_frac(
            valid_probes, "right_noise_max_dev", float(args.coupled_threshold)
        ),
        "valid_both_coupled_frac": _coupled_frac(
            valid_probes, "both_noise_max_dev", float(args.coupled_threshold)
        ),
        "stage_histogram": _stage_histogram(probes),
        "valid_stage_histogram": _stage_histogram(valid_probes),
        "recommended_next_step": recommended_next_step,
        "probes": probes,
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_probe(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
