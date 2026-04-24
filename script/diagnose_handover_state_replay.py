import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill import ManiSkillTaskFactory, apply_task_profile, resolve_task_profile
from sim.maniskill.expert.primitives import FixedRateMotionPrimitives


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose whether first-grasp replay failure comes from drive tracking or contact differences."
    )
    parser.add_argument("--task-profile", type=str, default="handover_block_native_legacy_rng_smoke")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--close-threshold", type=float, default=0.2)
    parser.add_argument("--qpos-threshold", type=float, default=0.05)
    parser.add_argument("--box-motion-threshold", type=float, default=0.005)
    parser.add_argument(
        "--legacy-demo-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "handover_block" / "demo_clean"),
    )
    parser.add_argument("--teacher-seed-path", type=str, default=None)
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


def _load_seed_path(profile_name: str, seed_path_arg: str | None) -> Path:
    seed_path = _resolve_repo_path(seed_path_arg)
    if seed_path is not None:
        return seed_path
    profile = resolve_task_profile(profile_name)
    if profile is None:
        raise KeyError(f"Unknown task profile: {profile_name}")
    teacher_cfg = (profile or {}).get("teacher_dataset", {})
    seed_path = _resolve_repo_path(teacher_cfg.get("seed_path"))
    if seed_path is None:
        raise ValueError(f"Task profile `{profile_name}` does not define teacher_dataset.seed_path")
    return seed_path


def _load_teacher_seeds(seed_path: Path) -> list[int]:
    raw = seed_path.read_text(encoding="utf-8").split()
    if not raw:
        raise ValueError(f"No seeds found in {seed_path}")
    return [int(value) for value in raw]


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


def _to_numpy(value, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if shape is not None:
        array = array.reshape(shape)
    return array


def _joint_name(joint) -> str:
    if hasattr(joint, "get_name"):
        return joint.get_name()
    return str(joint.name)


def _validate_joint_order(raw_env) -> None:
    for arm_index, robot, arm_joints, gripper_joints in (
        (0, raw_env.left_robot, raw_env.left_arm_joints, raw_env.left_gripper),
        (1, raw_env.right_robot, raw_env.right_arm_joints, raw_env.right_gripper),
    ):
        active_names = [_joint_name(joint) for joint in robot.get_active_joints()]
        expected_names = [_joint_name(joint) for joint in arm_joints] + [
            _joint_name(joint) for joint, _, _ in gripper_joints
        ]
        if active_names[: len(expected_names)] != expected_names:
            raise RuntimeError(
                f"Unexpected active joint order for arm {arm_index}: "
                f"expected prefix {expected_names}, got {active_names}"
            )


def _apply_teacher_robot_state(raw_env, arm_index: int, arm_qpos: np.ndarray, gripper_value: float) -> None:
    robot = raw_env.left_robot if arm_index == 0 else raw_env.right_robot
    arm_joints = raw_env.left_arm_joints if arm_index == 0 else raw_env.right_arm_joints
    gripper_joints = raw_env.left_gripper if arm_index == 0 else raw_env.right_gripper

    qpos = robot.get_qpos().clone()
    qvel = robot.get_qvel().clone()
    qvel.zero_()

    active_joints = robot.get_active_joints()
    name_to_index = {_joint_name(joint): idx for idx, joint in enumerate(active_joints)}

    arm_qpos = np.asarray(arm_qpos, dtype=np.float64).reshape(len(arm_joints))
    for joint, target in zip(arm_joints, arm_qpos):
        qpos[:, name_to_index[_joint_name(joint)]] = float(target)

    gripper_scale = raw_env.embodiment_spec.gripper_scale(arm_index)
    real_gripper_val = float(gripper_scale[0] + gripper_value * (gripper_scale[1] - gripper_scale[0]))
    for joint, sign, offset in gripper_joints:
        qpos[:, name_to_index[_joint_name(joint)]] = real_gripper_val * float(sign) + float(offset)

    robot.set_qpos(qpos)
    robot.set_qvel(qvel)
    robot.set_joint_drive_targets(qpos[:, : len(arm_joints)], joints=arm_joints)
    robot.set_joint_drive_velocity_targets(
        torch.zeros((1, len(arm_joints)), device=raw_env.device, dtype=qvel.dtype),
        joints=arm_joints,
    )
    raw_env._set_gripper(torch.tensor([gripper_value], device=raw_env.device, dtype=torch.float32), arm_index=arm_index)


def _advance_one_control_step(raw_env) -> None:
    if raw_env.gpu_sim_enabled:
        raw_env.scene._gpu_apply_all()
    raw_env._before_control_step()
    for _ in range(raw_env._sim_steps_per_control):
        raw_env._before_simulation_step()
        raw_env.scene.step()
        raw_env._after_simulation_step()
    raw_env._after_control_step()
    if raw_env.gpu_sim_enabled:
        raw_env.scene._gpu_fetch_all()


def _capture_state(raw_env, ctrl: FixedRateMotionPrimitives) -> dict[str, object]:
    return {
        "left_endpose": np.asarray(ctrl.current_gripper_pose(0), dtype=np.float64),
        "left_qpos": raw_env.get_arm_joint_positions(0).astype(np.float64, copy=False),
        "left_gripper": float(ctrl.current_gripper_value(0)),
        "box_pose": _to_numpy(raw_env.box.pose.raw_pose, (-1, 7))[0].astype(np.float64, copy=False),
    }


def _first_index(records: list[dict[str, object]], predicate):
    for idx, record in enumerate(records):
        if predicate(record):
            return idx
    return None


def main() -> None:
    args = build_parser().parse_args()

    seed_path = _load_seed_path(args.task_profile, args.teacher_seed_path)
    teacher_seeds = _load_teacher_seeds(seed_path)
    reset_seed = teacher_seeds[args.episode_index]

    legacy_demo_dir = _resolve_repo_path(args.legacy_demo_dir)
    if legacy_demo_dir is None:
        raise ValueError("--legacy-demo-dir resolved to None")
    hdf5_path = legacy_demo_dir / "data" / f"episode{args.episode_index}.hdf5"
    old_demo = _load_legacy_demo(hdf5_path)

    runtime_kwargs = {
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "render_mode": args.render_mode,
    }
    _, resolved_runtime, _ = apply_task_profile(args.task_profile, runtime_kwargs)
    env = ManiSkillTaskFactory().create_env(task_name=args.task_profile, **resolved_runtime)

    records: list[dict[str, object]] = []
    try:
        env.reset(seed=reset_seed)
        raw_env = env.env.unwrapped
        _validate_joint_order(raw_env)
        ctrl = FixedRateMotionPrimitives(env)
        initial_box_pose = _to_numpy(raw_env.box.pose.raw_pose, (-1, 7))[0]

        max_steps = min(int(args.max_steps), len(old_demo["joint_vector"]))
        for step_idx in range(max_steps):
            teacher_vector = np.asarray(old_demo["joint_vector"][step_idx], dtype=np.float64)
            _apply_teacher_robot_state(raw_env, 0, teacher_vector[:6], float(teacher_vector[6]))
            _apply_teacher_robot_state(raw_env, 1, teacher_vector[7:13], float(teacher_vector[13]))
            _advance_one_control_step(raw_env)
            if getattr(env, "render_mode", None) == "human":
                env.render()

            current = _capture_state(raw_env, ctrl)
            old_endpose = np.asarray(old_demo["left_endpose"][step_idx], dtype=np.float64)
            old_qpos = teacher_vector[:6]
            current_endpose = np.asarray(current["left_endpose"], dtype=np.float64)
            current_qpos = np.asarray(current["left_qpos"], dtype=np.float64)
            box_pose = np.asarray(current["box_pose"], dtype=np.float64)

            records.append(
                {
                    "idx": step_idx,
                    "teacher_left_gripper": float(old_demo["left_gripper"][step_idx]),
                    "current_left_gripper": float(current["left_gripper"]),
                    "teacher_left_qpos": old_qpos.tolist(),
                    "current_left_qpos": current_qpos.tolist(),
                    "teacher_left_endpose": old_endpose.tolist(),
                    "current_left_endpose": current_endpose.tolist(),
                    "current_box_pose": box_pose.tolist(),
                    "qpos_err": float(np.linalg.norm(current_qpos - old_qpos)),
                    "ee_pos_err": float(np.linalg.norm(current_endpose[:3] - old_endpose[:3])),
                    "box_disp": float(np.linalg.norm(box_pose[:3] - initial_box_pose[:3])),
                }
            )
    finally:
        env.close()

    first_teacher_close_idx = _first_index(records, lambda r: float(r["teacher_left_gripper"]) < args.close_threshold)
    first_current_close_idx = _first_index(records, lambda r: float(r["current_left_gripper"]) < args.close_threshold)
    first_box_motion_idx = _first_index(records, lambda r: float(r["box_disp"]) > args.box_motion_threshold)
    first_qpos_drift_idx = _first_index(records, lambda r: float(r["qpos_err"]) > args.qpos_threshold)

    if first_box_motion_idx is None:
        diagnosis = "no_box_motion"
    elif first_teacher_close_idx is None:
        diagnosis = "teacher_never_closed"
    elif first_box_motion_idx < first_teacher_close_idx:
        diagnosis = "contact_or_geometry_diff_persists_under_state_replay"
    else:
        diagnosis = "drive_tracking_is_likely_primary_issue"

    center_idx = first_teacher_close_idx if first_teacher_close_idx is not None else min(len(records) - 1, max_steps // 2)
    window_start = max(0, center_idx - 20)
    window_end = min(len(records) - 1, center_idx + 20)

    summary = {
        "task_profile": args.task_profile,
        "episode_index": int(args.episode_index),
        "seed": int(reset_seed),
        "legacy_hdf5_path": str(hdf5_path),
        "record_count": len(records),
        "first_teacher_close_idx": first_teacher_close_idx,
        "first_current_close_idx": first_current_close_idx,
        "first_box_motion_idx": first_box_motion_idx,
        "first_qpos_drift_idx": first_qpos_drift_idx,
        "diagnosis": diagnosis,
        "window_start": window_start,
        "window_end": window_end,
    }

    print(f"episode={summary['episode_index']} seed={summary['seed']} records={summary['record_count']}")
    print(
        "events "
        f"teacher_close={first_teacher_close_idx} "
        f"current_close={first_current_close_idx} "
        f"box_motion={first_box_motion_idx} "
        f"qpos_drift(>{args.qpos_threshold})={first_qpos_drift_idx}"
    )
    print(f"diagnosis={diagnosis}")
    print(
        "window "
        f"[{window_start}, {window_end}] "
        "(idx teacher_g current_g qpos_err ee_pos_err box_disp old_xyz -> current_xyz)"
    )
    for idx in range(window_start, window_end + 1):
        record = records[idx]
        old_xyz = np.asarray(record["teacher_left_endpose"], dtype=np.float64)[:3]
        current_xyz = np.asarray(record["current_left_endpose"], dtype=np.float64)[:3]
        print(
            f"{idx:03d} "
            f"{float(record['teacher_left_gripper']):.2f} "
            f"{float(record['current_left_gripper']):.2f} "
            f"{float(record['qpos_err']):.4f} "
            f"{float(record['ee_pos_err']):.4f} "
            f"{float(record['box_disp']):.4f} "
            f"{np.round(old_xyz, 4).tolist()} -> {np.round(current_xyz, 4).tolist()}"
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
