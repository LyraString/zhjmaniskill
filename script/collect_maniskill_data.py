import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

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
    resolve_task_profile,
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
    parser.add_argument("--collector", choices=("random", "legacy_replay", "scripted_expert"), default="random")
    parser.add_argument("--teacher-zarr-path", type=str, default=None)
    parser.add_argument("--teacher-seed-path", type=str, default=None)
    parser.add_argument("--save-only-success", action="store_true")
    return parser


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _to_bool_scalar(value) -> bool:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


def _load_teacher_actions(zarr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open(str(zarr_path), mode="r")
    data_group = root["data"]
    if "action" in data_group:
        actions = np.asarray(data_group["action"], dtype=np.float32)
    elif "actions" in data_group:
        actions = np.asarray(data_group["actions"], dtype=np.float32)
    else:
        raise KeyError(f"No `action` or `actions` array found in teacher zarr: {zarr_path}")
    episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    return actions, episode_ends


def _load_teacher_seeds(seed_path: Path | None) -> list[int] | None:
    if seed_path is None or not seed_path.is_file():
        return None
    raw = seed_path.read_text(encoding="utf-8").split()
    if len(raw) == 0:
        return None
    return [int(value) for value in raw]


def _resolve_teacher_assets(args, resolved_profile_name: str | None) -> tuple[Path, Path | None]:
    teacher_zarr_path = _resolve_repo_path(args.teacher_zarr_path)
    teacher_seed_path = _resolve_repo_path(args.teacher_seed_path)
    if teacher_zarr_path is not None:
        return teacher_zarr_path, teacher_seed_path

    if resolved_profile_name is None:
        raise ValueError(
            "--collector legacy_replay requires either --teacher-zarr-path "
            "or a task profile with teacher_dataset defaults."
        )

    profile = resolve_task_profile(resolved_profile_name)
    teacher_cfg = (profile or {}).get("teacher_dataset", {})
    teacher_zarr_path = _resolve_repo_path(teacher_cfg.get("zarr_path"))
    teacher_seed_path = _resolve_repo_path(teacher_cfg.get("seed_path"))
    if teacher_zarr_path is None:
        raise ValueError(
            f"Task profile `{resolved_profile_name}` does not define a teacher_dataset.zarr_path "
            "and --teacher-zarr-path was not provided."
        )
    return teacher_zarr_path, teacher_seed_path


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

    def maybe_render() -> None:
        if getattr(env, "render_mode", None) == "human":
            env.render()

    try:
        if args.collector == "random":
            for episode_idx in range(args.episode_num):
                obs = env.reset(seed=args.seed + episode_idx)
                recorder.start_episode(dict(obs.data))
                maybe_render()

                for _ in range(args.max_episode_steps):
                    action = env.env.action_space.sample()
                    result = env.step(
                        ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                    )
                    recorder.append_step(action, dict(result.observation.data))
                    maybe_render()
                    if result.terminated or result.truncated:
                        break

                out_path = recorder.save_episode(episode_idx)
                print(f"saved {out_path}")
            return

        if args.collector == "scripted_expert":
            from sim.maniskill.expert.handover_block_solver import HandoverBlockSolver

            solver = HandoverBlockSolver(env)
            saved_episode_idx = 0
            attempted_episode_idx = 0
            while saved_episode_idx < args.episode_num:
                reset_seed = args.seed + attempted_episode_idx
                obs = env.reset(seed=reset_seed)
                recorder.start_episode(dict(obs.data))
                maybe_render()
                success, episode_length, status = solver.solve_episode(
                    recorder=recorder,
                    max_episode_steps=args.max_episode_steps,
                )

                if success or not args.save_only_success:
                    if episode_length > 0:
                        out_path = recorder.save_episode(saved_episode_idx)
                        print(
                            f"saved {out_path} "
                            f"(seed={reset_seed}, length={episode_length}, "
                            f"success={int(success)}, status={status})"
                        )
                        saved_episode_idx += 1
                    else:
                        print(
                            f"skipped scripted_expert attempt={attempted_episode_idx} "
                            f"(seed={reset_seed}, length=0, success={int(success)}, status={status})"
                        )
                    if (not success) and getattr(solver, "last_debug_info", None):
                        print("scripted_expert_debug=" + json.dumps(solver.last_debug_info, ensure_ascii=False))
                else:
                    print(
                        f"skipped scripted_expert attempt={attempted_episode_idx} "
                        f"(seed={reset_seed}, length={episode_length}, success=0, status={status})"
                    )
                    if getattr(solver, "last_debug_info", None):
                        print("scripted_expert_debug=" + json.dumps(solver.last_debug_info, ensure_ascii=False))
                attempted_episode_idx += 1
            return

        teacher_zarr_path, teacher_seed_path = _resolve_teacher_assets(args, resolved_profile_name)
        teacher_actions, teacher_episode_ends = _load_teacher_actions(teacher_zarr_path)
        teacher_seeds = _load_teacher_seeds(teacher_seed_path)

        saved_episode_idx = 0
        teacher_episode_idx = 0
        prev_end = 0
        num_teacher_episodes = len(teacher_episode_ends)

        while saved_episode_idx < args.episode_num and teacher_episode_idx < num_teacher_episodes:
            teacher_end = int(teacher_episode_ends[teacher_episode_idx])
            action_seq = teacher_actions[prev_end:teacher_end]
            prev_end = teacher_end

            reset_seed = (
                teacher_seeds[teacher_episode_idx]
                if teacher_seeds is not None and teacher_episode_idx < len(teacher_seeds)
                else args.seed + teacher_episode_idx
            )
            obs = env.reset(seed=int(reset_seed))
            recorder.start_episode(dict(obs.data))
            maybe_render()

            success = False
            episode_length = 0
            for action in action_seq[: args.max_episode_steps]:
                result = env.step(
                    ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                )
                recorder.append_step(action, dict(result.observation.data))
                maybe_render()
                episode_length += 1
                if _to_bool_scalar(result.info.get("success", False)):
                    success = True
                    break
                if result.terminated or result.truncated:
                    break

            if success or not args.save_only_success:
                out_path = recorder.save_episode(saved_episode_idx)
                print(
                    f"saved {out_path} "
                    f"(teacher_episode={teacher_episode_idx}, seed={reset_seed}, "
                    f"length={episode_length}, success={int(success)})"
                )
                saved_episode_idx += 1
            else:
                print(
                    f"skipped teacher_episode={teacher_episode_idx} "
                    f"(seed={reset_seed}, length={episode_length}, success=0)"
                )

            teacher_episode_idx += 1

        if saved_episode_idx < args.episode_num:
            print(
                f"warning: requested {args.episode_num} episodes but only saved {saved_episode_idx} "
                f"from {num_teacher_episodes} teacher episodes"
            )
    finally:
        env.close()


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_collection(args)


if __name__ == "__main__":
    main()
