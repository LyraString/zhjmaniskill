import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import (
    ManiSkillTaskFactory,
    apply_task_profile,
    resolve_runtime_control_mode,
    resolve_runtime_task_target,
    resolve_task_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quickly check whether a RoboTwin/ManiSkill zarr dataset is usable "
            "for training and/or legacy replay."
        )
    )
    parser.add_argument("zarr_path", type=str, help="Path to the source zarr dataset")
    parser.add_argument(
        "--task-profile",
        type=str,
        default="handover_block_native_smoke",
        help="Registered ManiSkill task profile used as the compatibility target",
    )
    parser.add_argument(
        "--teacher-seed-path",
        type=str,
        default=None,
        help="Optional seed.txt path used for replay smoke checks",
    )
    parser.add_argument(
        "--smoke-replay",
        action="store_true",
        help="Also run a short replay smoke test against the target ManiSkill env",
    )
    parser.add_argument(
        "--replay-episodes",
        type=int,
        default=1,
        help="How many episodes to replay during smoke test",
    )
    parser.add_argument(
        "--replay-steps",
        type=int,
        default=50,
        help="Maximum steps per episode during smoke replay",
    )
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


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_teacher_seeds(seed_path: Path | None) -> list[int] | None:
    if seed_path is None or not seed_path.is_file():
        return None
    raw = seed_path.read_text(encoding="utf-8").split()
    if not raw:
        return None
    return [int(value) for value in raw]


def _bool_icon(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _to_bool_scalar(value) -> bool:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = np.asarray(value).reshape(-1)[0]
    return bool(value)


def _shape_tuple(value) -> tuple[int, ...]:
    return tuple(int(x) for x in value)


def _get_action_array(data_group):
    if "action" in data_group:
        return data_group["action"], "action"
    if "actions" in data_group:
        return data_group["actions"], "actions"
    return None, None


def check_dataset_schema(zarr_path: Path, task_profile: str) -> dict:
    profile = resolve_task_profile(task_profile)
    if profile is None:
        raise KeyError(f"Unknown ManiSkill task profile: {task_profile}")

    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise KeyError("Dataset root must contain `data/` and `meta/` groups")

    data_group = root["data"]
    meta_group = root["meta"]
    action_arr, action_key = _get_action_array(data_group)
    state_arr = data_group["state"] if "state" in data_group else None
    point_cloud_arr = data_group["point_cloud"] if "point_cloud" in data_group else None
    head_camera_arr = data_group["head_camera"] if "head_camera" in data_group else None
    episode_ends = np.asarray(meta_group["episode_ends"], dtype=np.int64)

    checks = {}
    checks["has_episode_ends"] = len(episode_ends) > 0
    checks["episode_ends_monotonic"] = bool(
        len(episode_ends) > 0 and np.all(np.diff(episode_ends) > 0)
    )
    checks["has_action_for_replay"] = action_arr is not None

    n_steps = None
    if checks["has_episode_ends"]:
        n_steps = int(episode_ends[-1])

    if action_arr is not None and n_steps is not None:
        checks["action_length_matches_episode_ends"] = int(action_arr.shape[0]) == n_steps
    else:
        checks["action_length_matches_episode_ends"] = False

    if state_arr is not None and n_steps is not None:
        checks["state_length_matches_episode_ends"] = int(state_arr.shape[0]) == n_steps
    else:
        checks["state_length_matches_episode_ends"] = False

    if point_cloud_arr is not None and n_steps is not None:
        checks["point_cloud_length_matches_episode_ends"] = int(point_cloud_arr.shape[0]) == n_steps
    else:
        checks["point_cloud_length_matches_episode_ends"] = False

    expected_shape_meta = profile["shape_meta"]
    expected_action_shape = _shape_tuple(expected_shape_meta["action"])
    expected_agent_pos_shape = _shape_tuple(expected_shape_meta["agent_pos"])
    expected_point_cloud_shape = _shape_tuple(expected_shape_meta["point_cloud"])

    checks["action_rank_ok"] = bool(action_arr is not None and len(action_arr.shape) >= 2)
    checks["action_shape_matches_profile"] = bool(
        action_arr is not None and _shape_tuple(action_arr.shape[1:]) == expected_action_shape
    )
    checks["state_present"] = state_arr is not None
    checks["state_shape_matches_profile"] = bool(
        state_arr is not None and _shape_tuple(state_arr.shape[1:]) == expected_agent_pos_shape
    )
    checks["point_cloud_present"] = point_cloud_arr is not None
    checks["point_cloud_shape_matches_profile"] = bool(
        point_cloud_arr is not None
        and _shape_tuple(point_cloud_arr.shape[1:]) == expected_point_cloud_shape
    )

    checks["usable_for_legacy_replay"] = all(
        (
            checks["has_episode_ends"],
            checks["episode_ends_monotonic"],
            checks["has_action_for_replay"],
            checks["action_length_matches_episode_ends"],
            checks["action_shape_matches_profile"],
        )
    )
    checks["usable_for_training"] = all(
        (
            checks["usable_for_legacy_replay"],
            checks["state_present"],
            checks["state_length_matches_episode_ends"],
            checks["state_shape_matches_profile"],
            checks["point_cloud_present"],
            checks["point_cloud_length_matches_episode_ends"],
            checks["point_cloud_shape_matches_profile"],
        )
    )

    info = {
        "zarr_path": str(zarr_path),
        "task_profile": task_profile,
        "env_id": str(profile["env_id"]),
        "action_key": action_key,
        "action_shape": None if action_arr is None else tuple(int(x) for x in action_arr.shape),
        "state_shape": None if state_arr is None else tuple(int(x) for x in state_arr.shape),
        "point_cloud_shape": (
            None if point_cloud_arr is None else tuple(int(x) for x in point_cloud_arr.shape)
        ),
        "head_camera_shape": (
            None if head_camera_arr is None else tuple(int(x) for x in head_camera_arr.shape)
        ),
        "episode_count": int(len(episode_ends)),
        "n_steps": n_steps,
        "expected_action_shape": expected_action_shape,
        "expected_state_shape": expected_agent_pos_shape,
        "expected_point_cloud_shape": expected_point_cloud_shape,
        "checks": checks,
    }
    return info


def _to_numpy_vector(value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.ndim >= 2 and value.shape[0] == 1:
        value = value[0]
    return value.reshape(-1)


def _snapshot_runtime_state(env_backend) -> dict:
    unwrapped = env_backend.env.unwrapped
    state = {
        "left_qpos": None,
        "right_qpos": None,
        "box_pose_p": None,
    }
    if hasattr(unwrapped, "get_arm_joint_positions"):
        state["left_qpos"] = _to_numpy_vector(unwrapped.get_arm_joint_positions(0))
        state["right_qpos"] = _to_numpy_vector(unwrapped.get_arm_joint_positions(1))
    if hasattr(unwrapped, "box") and hasattr(unwrapped.box, "pose") and hasattr(unwrapped.box.pose, "p"):
        state["box_pose_p"] = _to_numpy_vector(unwrapped.box.pose.p)
    return state


def _accumulate_motion(motion_totals: dict, prev_state: dict, cur_state: dict) -> None:
    for key in ("left_qpos", "right_qpos", "box_pose_p"):
        prev_value = prev_state.get(key)
        cur_value = cur_state.get(key)
        if prev_value is None or cur_value is None:
            continue
        motion_totals[key] += float(np.linalg.norm(cur_value - prev_value))


def _maybe_render_human(env_backend) -> None:
    if getattr(env_backend, "render_mode", None) == "human":
        env_backend.env.render()


def _maybe_capture_frame(env_backend, frames: list[np.ndarray]) -> None:
    frame = env_backend.env.render()
    if frame is None:
        return
    if hasattr(frame, "detach") and hasattr(frame, "cpu"):
        frame = frame.detach().cpu()
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] >= 1:
        frame = frame[0]
    if frame.ndim != 3:
        return
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    frames.append(frame)


def run_replay_smoke(
    *,
    zarr_path: Path,
    task_profile: str,
    teacher_seed_path: Path | None,
    replay_episodes: int,
    replay_steps: int,
    runtime_overrides: dict,
    save_video_dir: Path | None,
    video_fps: int,
) -> dict:
    task_name, resolved_profile_name = resolve_runtime_task_target(
        task_name=task_profile,
        task_profile=task_profile,
        env_id=None,
    )
    runtime_kwargs = dict(runtime_overrides)
    if save_video_dir is not None and runtime_kwargs.get("render_mode") in (None, ""):
        runtime_kwargs["render_mode"] = "rgb_array"
    _resolved_action_type, resolved_control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["control_mode"] = resolved_control_mode
    _, resolved_runtime, _ = apply_task_profile(task_name, runtime_kwargs)

    root = zarr.open(str(zarr_path), mode="r")
    action_arr, _ = _get_action_array(root["data"])
    episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    teacher_seeds = _load_teacher_seeds(teacher_seed_path)

    env = ManiSkillTaskFactory().create_env(task_name=task_name, **resolved_runtime)
    video_dir = save_video_dir
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    prev_end = 0
    replayed = []
    try:
        for episode_idx, teacher_end in enumerate(episode_ends[:replay_episodes]):
            teacher_end = int(teacher_end)
            action_seq = np.asarray(action_arr[prev_end:teacher_end], dtype=np.float32)
            prev_end = teacher_end
            reset_seed = (
                teacher_seeds[episode_idx]
                if teacher_seeds is not None and episode_idx < len(teacher_seeds)
                else episode_idx
            )
            env.reset(seed=int(reset_seed))
            _maybe_render_human(env)
            start_state = _snapshot_runtime_state(env)
            prev_state = dict(start_state)
            motion_totals = {
                "left_qpos": 0.0,
                "right_qpos": 0.0,
                "box_pose_p": 0.0,
            }
            frames: list[np.ndarray] = []
            if video_dir is not None:
                _maybe_capture_frame(env, frames)
            step_count = 0
            terminated = False
            truncated = False
            success = False
            error = None
            for action in action_seq[:replay_steps]:
                try:
                    result = env.step(
                        ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                    )
                except Exception as exc:  # pragma: no cover - smoke diagnostic path
                    error = repr(exc)
                    break
                step_count += 1
                terminated = bool(result.terminated)
                truncated = bool(result.truncated)
                success = _to_bool_scalar(result.info.get("success", False))
                _maybe_render_human(env)
                cur_state = _snapshot_runtime_state(env)
                _accumulate_motion(motion_totals, prev_state, cur_state)
                prev_state = cur_state
                if video_dir is not None:
                    _maybe_capture_frame(env, frames)
                if success:
                    break
                if terminated or truncated:
                    break
            final_state = _snapshot_runtime_state(env)
            left_joint_displacement = None
            right_joint_displacement = None
            box_displacement = None
            if start_state["left_qpos"] is not None and final_state["left_qpos"] is not None:
                left_joint_displacement = float(
                    np.linalg.norm(final_state["left_qpos"] - start_state["left_qpos"])
                )
            if start_state["right_qpos"] is not None and final_state["right_qpos"] is not None:
                right_joint_displacement = float(
                    np.linalg.norm(final_state["right_qpos"] - start_state["right_qpos"])
                )
            if start_state["box_pose_p"] is not None and final_state["box_pose_p"] is not None:
                box_displacement = float(
                    np.linalg.norm(final_state["box_pose_p"] - start_state["box_pose_p"])
                )
            stalled_suspected = bool(
                step_count > 0
                and motion_totals["left_qpos"] < 1e-3
                and motion_totals["right_qpos"] < 1e-3
                and motion_totals["box_pose_p"] < 1e-5
            )
            video_path = None
            if video_dir is not None and len(frames) > 0:
                video_path = video_dir / f"replay_episode{episode_idx}.mp4"
                imageio.mimsave(video_path, frames, fps=int(video_fps))
            replayed.append(
                {
                    "episode_index": int(episode_idx),
                    "seed": int(reset_seed),
                    "steps_executed": int(step_count),
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": success,
                    "left_joint_motion": motion_totals["left_qpos"],
                    "right_joint_motion": motion_totals["right_qpos"],
                    "box_motion": motion_totals["box_pose_p"],
                    "left_joint_displacement": left_joint_displacement,
                    "right_joint_displacement": right_joint_displacement,
                    "box_displacement": box_displacement,
                    "stalled_suspected": stalled_suspected,
                    "video_path": None if video_path is None else str(video_path),
                    "error": error,
                }
            )
    finally:
        env.close()

    ok = all(item["error"] is None for item in replayed)
    return {
        "profile_name": resolved_profile_name,
        "episodes_checked": len(replayed),
        "ok": ok,
        "results": replayed,
    }


def print_report(schema_info: dict, replay_info: dict | None) -> None:
    checks = schema_info["checks"]
    print(f"Dataset: {schema_info['zarr_path']}")
    print(f"Target profile: {schema_info['task_profile']} (env_id={schema_info['env_id']})")
    print("")
    print("Observed arrays:")
    print(f"  action key: {schema_info['action_key']}")
    print(f"  action shape: {schema_info['action_shape']}")
    print(f"  state shape: {schema_info['state_shape']}")
    print(f"  point_cloud shape: {schema_info['point_cloud_shape']}")
    print(f"  head_camera shape: {schema_info['head_camera_shape']}")
    print(f"  episode_count: {schema_info['episode_count']}")
    print(f"  n_steps: {schema_info['n_steps']}")
    print("")
    print("Expected by target profile:")
    print(f"  action shape: {schema_info['expected_action_shape']}")
    print(f"  state shape: {schema_info['expected_state_shape']}")
    print(f"  point_cloud shape: {schema_info['expected_point_cloud_shape']}")
    print("")
    print("Checks:")
    for key in (
        "has_episode_ends",
        "episode_ends_monotonic",
        "has_action_for_replay",
        "action_length_matches_episode_ends",
        "action_shape_matches_profile",
        "state_present",
        "state_length_matches_episode_ends",
        "state_shape_matches_profile",
        "point_cloud_present",
        "point_cloud_length_matches_episode_ends",
        "point_cloud_shape_matches_profile",
    ):
        print(f"  {_bool_icon(checks[key])} {key}")
    print("")
    print(f"Verdict: usable_for_legacy_replay={checks['usable_for_legacy_replay']}")
    print(f"Verdict: usable_for_training={checks['usable_for_training']}")
    if replay_info is not None:
        print("")
        print(f"Replay smoke: ok={replay_info['ok']} episodes_checked={replay_info['episodes_checked']}")
        for item in replay_info["results"]:
            print(
                "  episode={episode_index} seed={seed} steps={steps_executed} "
                "terminated={terminated} truncated={truncated} success={success} "
                "left_motion={left_joint_motion:.6f} right_motion={right_joint_motion:.6f} "
                "box_motion={box_motion:.6f} left_disp={left_joint_displacement} "
                "right_disp={right_joint_displacement} box_disp={box_displacement} "
                "stalled_suspected={stalled_suspected} video_path={video_path} "
                "error={error}".format(**item)
            )


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    zarr_path = _resolve_repo_path(args.zarr_path)
    if zarr_path is None or not zarr_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.zarr_path}")
    teacher_seed_path = _resolve_repo_path(args.teacher_seed_path)
    save_video_dir = _resolve_repo_path(args.save_video_dir)

    schema_info = check_dataset_schema(zarr_path, args.task_profile)
    replay_info = None
    if args.smoke_replay:
        replay_info = run_replay_smoke(
            zarr_path=zarr_path,
            task_profile=args.task_profile,
            teacher_seed_path=teacher_seed_path,
            replay_episodes=args.replay_episodes,
            replay_steps=args.replay_steps,
            runtime_overrides={
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
            },
            save_video_dir=save_video_dir,
            video_fps=args.video_fps,
        )

    print_report(schema_info, replay_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
