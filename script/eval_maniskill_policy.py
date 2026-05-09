import argparse
import importlib
import sys
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
}


def _to_bool_scalar(value) -> bool:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


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

    video_dir = Path(args.save_video_dir) if args.save_video_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    returns = []
    lengths = []
    successes = []

    try:
        for episode_idx in range(args.episodes):
            obs = env.reset(seed=args.seed + episode_idx)
            adapter.reset_model(model)
            encoded_obs = adapter.encode_obs(dict(obs.data))
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

            while episode_length < args.max_episode_steps:
                if len(model.obs_cache) == 0:
                    model.update_obs(encoded_obs)
                action_chunk = model.get_action()

                for action in action_chunk:
                    result = env.step(
                        ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                    )
                    episode_return += float(result.reward)
                    episode_length += 1
                    encoded_obs = adapter.encode_obs(dict(result.observation.data))
                    model.update_obs(encoded_obs)
                    if video_dir is not None:
                        _maybe_capture_frame(env, frames)
                    if _to_bool_scalar(result.info.get("success", False)):
                        success = True
                        break
                    if result.terminated or result.truncated or episode_length >= args.max_episode_steps:
                        break

                if success or result.terminated or result.truncated or episode_length >= args.max_episode_steps:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(float(success))
            if video_dir is not None and len(frames) > 0:
                episode_dir = video_dir
                episode_dir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(
                    episode_dir / f"episode{episode_idx}.mp4",
                    frames,
                    fps=int(args.video_fps),
                )
            print(
                f"episode={episode_idx} return={episode_return:.4f} "
                f"length={episode_length} success={int(success)}"
            )
    finally:
        env.close()

    print(
        "summary "
        f"mean_return={float(np.mean(returns)):.4f} "
        f"mean_length={float(np.mean(lengths)):.2f} "
        f"success_rate={float(np.mean(successes)):.4f}"
    )


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
