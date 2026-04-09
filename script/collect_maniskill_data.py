import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory, ManiSkillTrajectoryRecorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect ManiSkill episodes into `.npz` files.")
    parser.add_argument("task_name", type=str, help="ManiSkill env id, e.g. PickCube-v1")
    parser.add_argument("task_config", type=str, help="Dataset subdirectory name under data/<task_name>/")
    parser.add_argument("episode_num", type=int, help="Number of episodes to collect")
    parser.add_argument("--seed", type=int, default=42, help="Base reset seed")
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--action-type", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--obs-mode", type=str, default="pointcloud+state_dict")
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--reward-mode", type=str, default="normalized_dense")
    parser.add_argument("--robot-uids", type=str, default="panda")
    parser.add_argument("--sim-freq", type=int, default=200)
    parser.add_argument("--control-freq", type=int, default=20)
    return parser


def run_collection(args) -> None:
    save_dir = REPO_ROOT / "data" / args.task_name / args.task_config
    recorder = ManiSkillTrajectoryRecorder(save_dir)
    env = ManiSkillTaskFactory().create_env(
        task_name=args.task_name,
        action_type=args.action_type,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        reward_mode=args.reward_mode,
        robot_uids=args.robot_uids,
        sim_freq=args.sim_freq,
        control_freq=args.control_freq,
    )

    try:
        for episode_idx in range(args.episode_num):
            obs = env.reset(seed=args.seed + episode_idx)
            recorder.start_episode(dict(obs.data))

            for _ in range(args.max_episode_steps):
                action = env.env.action_space.sample()
                result = env.step(
                    ActionSpec.from_value(action, action_type=args.action_type)
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
