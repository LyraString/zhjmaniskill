import argparse
import json
import sys
from pathlib import Path

import h5py
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
    resolve_task_profile,
)
from sim.maniskill.expert.primitives import FixedRateMotionPrimitives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare old RoboTwin demo states against current ManiSkill legacy replay around first grasp."
    )
    parser.add_argument("--task-profile", type=str, default="handover_block_native_legacy_rng_smoke")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--window-before", type=int, default=20)
    parser.add_argument("--window-after", type=int, default=20)
    parser.add_argument("--close-threshold", type=float, default=0.2)
    parser.add_argument("--ee-pos-threshold", type=float, default=0.02)
    parser.add_argument("--qpos-threshold", type=float, default=0.15)
    parser.add_argument("--box-motion-threshold", type=float, default=0.005)
    parser.add_argument("--teacher-zarr-path", type=str, default=None)
    parser.add_argument("--teacher-seed-path", type=str, default=None)
    parser.add_argument(
        "--legacy-demo-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "handover_block" / "demo_clean"),
    )
    parser.add_argument("--sim-backend", type=str, default=None)
    parser.add_argument("--render-backend", type=str, default=None)
    parser.add_argument("--render-mode", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _load_teacher_assets(profile_name: str, zarr_path_arg: str | None, seed_path_arg: str | None) -> tuple[Path, Path | None]:
    teacher_zarr_path = _resolve_repo_path(zarr_path_arg)
    teacher_seed_path = _resolve_repo_path(seed_path_arg)
    if teacher_zarr_path is not None:
        return teacher_zarr_path, teacher_seed_path

    profile = resolve_task_profile(profile_name)
    if profile is None:
        raise KeyError(f"Unknown task profile: {profile_name}")
    teacher_cfg = (profile or {}).get("teacher_dataset", {})
    teacher_zarr_path = _resolve_repo_path(teacher_cfg.get("zarr_path"))
    teacher_seed_path = _resolve_repo_path(teacher_cfg.get("seed_path"))
    if teacher_zarr_path is None:
        raise ValueError(f"Task profile `{profile_name}` does not define teacher_dataset.zarr_path")
    return teacher_zarr_path, teacher_seed_path


def _load_teacher_actions(zarr_path: Path, episode_index: int) -> np.ndarray:
    root = zarr.open(str(zarr_path), mode="r")
    actions = np.asarray(root["data"]["action"], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)
    if episode_index < 0 or episode_index >= len(episode_ends):
        raise IndexError(f"episode_index={episode_index} out of range for {len(episode_ends)} episodes")
    start = int(episode_ends[episode_index - 1]) if episode_index > 0 else 0
    end = int(episode_ends[episode_index])
    return actions[start:end]


def _load_teacher_seeds(seed_path: Path | None) -> list[int] | None:
    if seed_path is None or not seed_path.is_file():
        return None
    raw = seed_path.read_text(encoding="utf-8").split()
    return [int(value) for value in raw] if raw else None


def _to_numpy(value, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if shape is not None:
        return array.reshape(shape)
    return array


def _pose_to_list(value) -> list[float]:
    return _to_numpy(value).reshape(-1).astype(np.float64, copy=False).tolist()


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _load_legacy_demo(hdf5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as root:
        left_endpose = np.asarray(root["endpose/left_endpose"], dtype=np.float64)
        left_gripper = np.asarray(root["endpose/left_gripper"], dtype=np.float64)
        joint_vector = np.asarray(root["joint_action/vector"], dtype=np.float64)
    return {
        "left_endpose": left_endpose,
        "left_gripper": left_gripper,
        "joint_vector": joint_vector,
    }


def _capture_current_state(raw_env, ctrl: FixedRateMotionPrimitives) -> dict[str, object]:
    return {
        "left_endpose": np.asarray(ctrl.current_gripper_pose(0), dtype=np.float64),
        "left_qpos": raw_env.get_arm_joint_positions(0).astype(np.float64, copy=False),
        "left_gripper": float(ctrl.current_gripper_value(0)),
        "box_pose": _to_numpy(raw_env.box.pose.raw_pose, (-1, 7))[0].astype(np.float64, copy=False),
    }


def _align_record(old_demo: dict[str, np.ndarray], current_state: dict[str, object], old_index: int) -> dict[str, object]:
    old_ee = old_demo["left_endpose"][old_index]
    old_gripper = float(old_demo["left_gripper"][old_index])
    old_qpos = old_demo["joint_vector"][old_index, :6]
    current_ee = np.asarray(current_state["left_endpose"], dtype=np.float64)
    current_qpos = np.asarray(current_state["left_qpos"], dtype=np.float64)
    current_gripper = float(current_state["left_gripper"])
    box_pose = np.asarray(current_state["box_pose"], dtype=np.float64)
    return {
        "old_index": int(old_index),
        "old_left_ee_pose": old_ee.tolist(),
        "old_left_gripper": old_gripper,
        "old_left_qpos": old_qpos.tolist(),
        "current_left_endpose": current_ee.tolist(),
        "current_left_gripper": current_gripper,
        "current_left_qpos": current_qpos.tolist(),
        "current_box_pose": box_pose.tolist(),
        "ee_pos_err": float(np.linalg.norm(current_ee[:3] - old_ee[:3])),
        "ee_rot_err_deg": _quat_angle_deg(current_ee[3:], old_ee[3:]),
        "qpos_err": float(np.linalg.norm(current_qpos - old_qpos)),
    }


def _first_index(records: list[dict[str, object]], predicate) -> int | None:
    for idx, record in enumerate(records):
        if predicate(record):
            return idx
    return None


def main() -> None:
    args = build_parser().parse_args()

    teacher_zarr_path, teacher_seed_path = _load_teacher_assets(
        profile_name=args.task_profile,
        zarr_path_arg=args.teacher_zarr_path,
        seed_path_arg=args.teacher_seed_path,
    )
    teacher_actions = _load_teacher_actions(teacher_zarr_path, args.episode_index)
    teacher_seeds = _load_teacher_seeds(teacher_seed_path)
    if teacher_seeds is None:
        raise ValueError("Seed file is required for aligned replay comparison.")
    reset_seed = teacher_seeds[args.episode_index]

    legacy_demo_dir = _resolve_repo_path(args.legacy_demo_dir)
    if legacy_demo_dir is None:
        raise ValueError("--legacy-demo-dir resolved to None")
    hdf5_path = legacy_demo_dir / "data" / f"episode{args.episode_index}.hdf5"
    old_demo = _load_legacy_demo(hdf5_path)

    runtime_kwargs = {
        "render_mode": args.render_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
    }
    _, resolved_runtime, _ = apply_task_profile(args.task_profile, runtime_kwargs)
    _resolved_action_type, resolved_control_mode = resolve_runtime_control_mode(
        action_type=resolved_runtime["action_type"],
        control_mode=resolved_runtime["control_mode"],
    )
    resolved_runtime["control_mode"] = resolved_control_mode

    env = ManiSkillTaskFactory().create_env(task_name=args.task_profile, **resolved_runtime)
    records: list[dict[str, object]] = []
    try:
        env.reset(seed=reset_seed)
        raw_env = env.env.unwrapped
        ctrl = FixedRateMotionPrimitives(env)
        initial_box_pose = _to_numpy(raw_env.box.pose.raw_pose, (-1, 7))[0]
        records.append(_align_record(old_demo, _capture_current_state(raw_env, ctrl), old_index=0))

        max_steps = min(int(args.max_steps), len(teacher_actions), len(old_demo["left_endpose"]) - 1)
        for step_idx in range(max_steps):
            env.step(ActionSpec.from_value(teacher_actions[step_idx], action_type=resolved_runtime["action_type"]))
            records.append(_align_record(old_demo, _capture_current_state(raw_env, ctrl), old_index=step_idx + 1))
    finally:
        env.close()

    for record in records:
        box_pose = np.asarray(record["current_box_pose"], dtype=np.float64)
        record["box_disp"] = float(np.linalg.norm(box_pose[:3] - initial_box_pose[:3]))

    first_old_close_idx = _first_index(records, lambda r: float(r["old_left_gripper"]) < args.close_threshold)
    first_replay_close_rel_idx = _first_index(records[1:], lambda r: float(r["current_left_gripper"]) < args.close_threshold)
    first_replay_close_idx = None if first_replay_close_rel_idx is None else first_replay_close_rel_idx + 1
    first_box_motion_idx = _first_index(records, lambda r: float(r["box_disp"]) > args.box_motion_threshold)
    first_ee_diverge_idx = _first_index(records, lambda r: float(r["ee_pos_err"]) > args.ee_pos_threshold)
    first_qpos_diverge_idx = _first_index(records, lambda r: float(r["qpos_err"]) > args.qpos_threshold)

    if args.start_step is not None or args.end_step is not None:
        window_start = max(0, int(args.start_step or 0))
        window_end = min(len(records) - 1, int(args.end_step if args.end_step is not None else len(records) - 1))
    else:
        center_idx = first_old_close_idx if first_old_close_idx is not None else min(len(records) - 1, max(0, args.max_steps // 2))
        window_start = max(0, center_idx - int(args.window_before))
        window_end = min(len(records) - 1, center_idx + int(args.window_after))

    summary = {
        "task_profile": args.task_profile,
        "episode_index": int(args.episode_index),
        "seed": int(reset_seed),
        "teacher_zarr_path": str(teacher_zarr_path),
        "legacy_hdf5_path": str(hdf5_path),
        "record_count": len(records),
        "first_old_close_idx": first_old_close_idx,
        "first_replay_close_idx": first_replay_close_idx,
        "first_box_motion_idx": first_box_motion_idx,
        "first_ee_diverge_idx": first_ee_diverge_idx,
        "first_qpos_diverge_idx": first_qpos_diverge_idx,
        "window_start": window_start,
        "window_end": window_end,
    }

    print(
        f"episode={summary['episode_index']} seed={summary['seed']} "
        f"records={summary['record_count']} teacher_actions={len(teacher_actions)}"
    )
    print(
        "events "
        f"old_close={first_old_close_idx} "
        f"replay_close={first_replay_close_idx} "
        f"box_motion={first_box_motion_idx} "
        f"ee_diverge(>{args.ee_pos_threshold})={first_ee_diverge_idx} "
        f"qpos_diverge(>{args.qpos_threshold})={first_qpos_diverge_idx}"
    )
    print(
        "window "
        f"[{window_start}, {window_end}] "
        "(idx old_g cur_g ee_pos_err ee_rot_deg qpos_err box_disp old_xyz -> cur_xyz)"
    )
    for idx in range(window_start, window_end + 1):
        record = records[idx]
        old_xyz = np.asarray(record["old_left_ee_pose"], dtype=np.float64)[:3]
        cur_xyz = np.asarray(record["current_left_endpose"], dtype=np.float64)[:3]
        print(
            f"{idx:03d} "
            f"{float(record['old_left_gripper']):.2f} "
            f"{float(record['current_left_gripper']):.2f} "
            f"{float(record['ee_pos_err']):.4f} "
            f"{float(record['ee_rot_err_deg']):6.2f} "
            f"{float(record['qpos_err']):.4f} "
            f"{float(record['box_disp']):.4f} "
            f"{np.round(old_xyz, 4).tolist()} -> {np.round(cur_xyz, 4).tolist()}"
        )

    if args.output_json:
        output_path = _resolve_repo_path(args.output_json)
        if output_path is None:
            raise ValueError("--output-json resolved to None")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "records": records,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json={output_path}")


if __name__ == "__main__":
    main()
