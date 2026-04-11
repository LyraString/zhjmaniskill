import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import (
    ManiSkillTaskFactory,
    ManiSkillTrajectoryRecorder,
    apply_task_profile,
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect ManiSkill episodes into `.npz` files.")
    parser.add_argument("task_name", type=str, help="ManiSkill env id or the env id associated with --task-profile")
    parser.add_argument("task_config", type=str, help="Dataset subdirectory name under data/<task_name>/")
    parser.add_argument("episode_num", type=int, help="Number of episodes to collect")
    parser.add_argument("--task-profile", type=str, default=None, help="Optional registered ManiSkill task profile")
    parser.add_argument("--seed", type=int, default=42, help="Base reset seed")
    parser.add_argument("--max-episode-steps", type=int, default=50)
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
    return parser


def run_collection(args) -> None:
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
    }
    _resolved_action_type, resolved_control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["control_mode"] = resolved_control_mode
    _, resolved_runtime, _profile_name = apply_task_profile(runtime_task_name, runtime_kwargs)
    save_dir = REPO_ROOT / "data" / runtime_task_name / args.task_config
    recorder = ManiSkillTrajectoryRecorder(save_dir)
    env = ManiSkillTaskFactory().create_env(
        task_name=runtime_task_name,
        **resolved_runtime,
    )

    try:
        for episode_idx in range(args.episode_num):
            obs = env.reset(seed=args.seed + episode_idx)
            recorder.start_episode(dict(obs.data))

            for _ in range(args.max_episode_steps):
                action = env.env.action_space.sample()
                result = env.step(
                    ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                )
                recorder.append_step(action, dict(result.observation.data))
                if result.terminated or result.truncated:
                    break

            out_path = recorder.save_episode(episode_idx)
            print(f"saved {out_path}")
    finally:
        env.close()


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_collection(args)


if __name__ == "__main__":
    main()
