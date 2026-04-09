import argparse
import importlib
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory, apply_task_profile


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
    parser.add_argument("task_name", type=str, help="ManiSkill env id, e.g. PickCube-v1")
    parser.add_argument("--task-profile", type=str, default=None, help="Optional registered ManiSkill task profile")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-type", type=str, default=None)
    parser.add_argument("--obs-mode", type=str, default=None)
    parser.add_argument("--control-mode", type=str, default=None)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--robot-uids", type=str, default=None)
    parser.add_argument("--sim-freq", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=None)
    return parser


def run_evaluation(args) -> None:
    adapter = importlib.import_module(ADAPTER_MODULES[args.policy_kind])
    model = adapter.get_model({"ckpt_path": args.ckpt_path})
    task_key = args.task_profile or args.task_name
    runtime_kwargs = {
        "action_type": args.action_type,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "render_mode": args.render_mode,
        "reward_mode": args.reward_mode,
        "robot_uids": args.robot_uids,
        "sim_freq": args.sim_freq,
        "control_freq": args.control_freq,
    }
    _, resolved_runtime, _profile_name = apply_task_profile(task_key, runtime_kwargs)

    env = ManiSkillTaskFactory().create_env(
        task_name=task_key,
        **resolved_runtime,
    )

    returns = []
    lengths = []
    successes = []

    try:
        for episode_idx in range(args.episodes):
            obs = env.reset(seed=args.seed + episode_idx)
            adapter.reset_model(model)
            encoded_obs = adapter.encode_obs(dict(obs.data))
            episode_return = 0.0
            episode_length = 0
            success = False

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
                    success = _to_bool_scalar(result.info.get("success", False))
                    if result.terminated or result.truncated or episode_length >= args.max_episode_steps:
                        break

                if result.terminated or result.truncated or episode_length >= args.max_episode_steps:
                    break

            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(float(success))
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
