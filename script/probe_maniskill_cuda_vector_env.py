#!/usr/bin/env python3
"""Probe native ManiSkill vector-env tensor I/O for the FPO CUDA fast path."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mani_skill.envs  # noqa: F401
import sim.maniskill  # noqa: F401
from sim.maniskill.task_profiles import (
    get_task_profile,
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
    resolve_task_profile,
)


def _shape_of(value: Any) -> tuple[int, ...] | str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "-"
    return tuple(int(dim) for dim in shape)


def _device_of(value: Any) -> str:
    device = getattr(value, "device", None)
    return "-" if device is None else str(device)


def _dtype_of(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return "-" if dtype is None else str(dtype)


def describe_tree(name: str, value: Any, *, max_depth: int = 6, depth: int = 0) -> None:
    indent = "  " * depth
    prefix = f"{indent}{name}"
    if isinstance(value, Mapping):
        print(f"{prefix}: dict keys={list(value.keys())}")
        if depth >= max_depth:
            return
        for key, child in value.items():
            describe_tree(str(key), child, max_depth=max_depth, depth=depth + 1)
        return

    if isinstance(value, torch.Tensor):
        print(
            f"{prefix}: torch shape={_shape_of(value)} dtype={_dtype_of(value)} "
            f"device={_device_of(value)} requires_grad={value.requires_grad}"
        )
        return

    if isinstance(value, np.ndarray):
        print(f"{prefix}: ndarray shape={_shape_of(value)} dtype={_dtype_of(value)}")
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        print(f"{prefix}: {type(value).__name__} len={len(value)}")
        if depth >= max_depth:
            return
        for idx, child in enumerate(value[:5]):
            describe_tree(f"[{idx}]", child, max_depth=max_depth, depth=depth + 1)
        return

    print(f"{prefix}: {type(value).__name__} value={value!r}")


def infer_action_shape(env: Any, num_envs: int, fallback_action_dim: int | None) -> tuple[int, ...]:
    action_space = getattr(env, "action_space", None)
    single_action_space = getattr(env, "single_action_space", None)

    for space in (action_space, single_action_space):
        shape = getattr(space, "shape", None)
        if shape is None:
            continue
        shape = tuple(int(dim) for dim in shape)
        if len(shape) >= 2 and shape[0] == num_envs:
            return shape
        if len(shape) >= 1:
            return (num_envs, *shape)

    if fallback_action_dim is None:
        raise RuntimeError(
            "Unable to infer action shape from env.action_space; pass --action-dim."
        )
    return (num_envs, int(fallback_action_dim))


def action_device_for(sim_backend: str | None) -> torch.device:
    if sim_backend is not None and "cuda" in str(sim_backend).lower() and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def make_env(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    env_id, resolved_profile_name = resolve_runtime_task_target(
        task_name=args.env_id,
        task_profile=args.task_profile,
        env_id=args.env_id,
    )
    profile = get_task_profile(resolved_profile_name) if resolved_profile_name else None
    if profile is None:
        profile = resolve_task_profile(env_id)

    action_type = args.action_type
    obs_mode = args.obs_mode
    control_mode = args.control_mode
    render_mode = args.render_mode
    reward_mode = args.reward_mode
    robot_uids = args.robot_uids
    sim_freq = args.sim_freq
    control_freq = args.control_freq
    fallback_action_dim = args.action_dim

    if profile is not None:
        action_type = action_type or profile.get("action_type")
        obs_mode = obs_mode or profile.get("obs_mode")
        control_mode = control_mode if control_mode is not None else profile.get("control_mode")
        render_mode = render_mode if render_mode is not None else profile.get("render_mode")
        reward_mode = reward_mode if reward_mode is not None else profile.get("reward_mode")
        robot_uids = robot_uids if robot_uids is not None else profile.get("robot_uids")
        sim_freq = sim_freq if sim_freq is not None else profile.get("sim_freq")
        control_freq = control_freq if control_freq is not None else profile.get("control_freq")
        shape_meta = profile.get("shape_meta") or {}
        action_shape = shape_meta.get("action")
        if fallback_action_dim is None and action_shape:
            fallback_action_dim = int(action_shape[0])

    obs_mode = obs_mode or "pointcloud+state_dict"
    action_type, control_mode = resolve_runtime_control_mode(
        action_type=action_type,
        control_mode=control_mode,
    )

    make_kwargs: dict[str, Any] = {
        "num_envs": int(args.num_envs),
        "obs_mode": obs_mode,
    }
    if control_mode is not None:
        make_kwargs["control_mode"] = control_mode
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    if args.sim_backend is not None:
        make_kwargs["sim_backend"] = args.sim_backend
    if args.render_backend is not None:
        make_kwargs["render_backend"] = args.render_backend
    if reward_mode is not None:
        make_kwargs["reward_mode"] = reward_mode
    if robot_uids not in (None, "none", "None"):
        make_kwargs["robot_uids"] = robot_uids

    sim_config: dict[str, Any] = {}
    if sim_freq is not None:
        sim_config["sim_freq"] = int(sim_freq)
    if control_freq is not None:
        sim_config["control_freq"] = int(control_freq)
    if sim_config:
        make_kwargs["sim_config"] = sim_config

    env = gym.make(env_id, **make_kwargs)
    metadata = {
        "env_id": env_id,
        "profile": resolved_profile_name,
        "action_type": action_type,
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "fallback_action_dim": fallback_action_dim,
        "make_kwargs": make_kwargs,
    }
    return env, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="HandoverBlock-v1")
    parser.add_argument("--task-profile", default="handover_block_native_smoke")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--action-type", default=None)
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--reward-mode", default=None)
    parser.add_argument("--robot-uids", default=None)
    parser.add_argument("--sim-freq", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--partial-env-idx", type=int, nargs="*", default=[0])
    parser.add_argument("--skip-partial-reset", action="store_true")
    args = parser.parse_args()

    env, metadata = make_env(args)
    print("env_metadata")
    describe_tree("metadata", metadata)
    print(f"env_type={type(env)}")
    print(f"unwrapped_type={type(env.unwrapped)}")
    print(f"env_action_space={getattr(env, 'action_space', None)}")
    print(f"env_single_action_space={getattr(env, 'single_action_space', None)}")

    try:
        seeds = [int(args.seed) + i for i in range(int(args.num_envs))]
        print("\nreset")
        obs, info = env.reset(seed=seeds)
        describe_tree("obs", obs)
        describe_tree("info", info, max_depth=3)

        action_shape = infer_action_shape(
            env,
            int(args.num_envs),
            metadata["fallback_action_dim"],
        )
        action_device = action_device_for(args.sim_backend)
        action = torch.zeros(action_shape, dtype=torch.float32, device=action_device)
        print(f"\naction_tensor shape={tuple(action.shape)} dtype={action.dtype} device={action.device}")

        print("\nstep_with_cuda_tensor_action")
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        describe_tree("next_obs", next_obs)
        describe_tree("reward", reward)
        describe_tree("terminated", terminated)
        describe_tree("truncated", truncated)
        describe_tree("step_info", step_info, max_depth=3)

        if not args.skip_partial_reset:
            env_idx = np.asarray(args.partial_env_idx, dtype=np.int64)
            print(f"\npartial_reset env_idx={env_idx.tolist()}")
            try:
                reset_obs, reset_info = env.reset(
                    seed=int(args.seed) + 1000,
                    options={"env_idx": env_idx},
                )
            except Exception as exc:
                print(f"partial_reset_failed type={type(exc).__name__} message={exc}")
            else:
                describe_tree("partial_obs", reset_obs)
                describe_tree("partial_info", reset_info, max_depth=3)
    finally:
        env.close()


if __name__ == "__main__":
    main()
