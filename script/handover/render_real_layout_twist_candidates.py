#!/usr/bin/env python3
"""Render UR5 real-layout home-pose candidates with different base twists.

This is a standalone calibration helper. It does not affect training, eval, or
data collection. The base center, base normal tilt, and home qpos are fixed from
the measured real setup; only the remaining twist around each base normal is
scanned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill import ManiSkillTaskFactory
from sim.maniskill.task_profiles import (
    UR5_REAL_LAYOUT_BASE_Y,
    UR5_REAL_LAYOUT_BASE_Z,
    UR5_REAL_LAYOUT_HOMESTATE,
    UR5_REAL_LAYOUT_ARMS_DISTANCE,
)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _quat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> list[float]:
    axis_arr = np.asarray(axis, dtype=np.float64)
    axis_arr = axis_arr / np.linalg.norm(axis_arr)
    half = float(angle) * 0.5
    xyz = axis_arr * np.sin(half)
    return [float(np.cos(half)), float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _normalise_quat(q: list[float]) -> list[float]:
    q_arr = np.asarray(q, dtype=np.float64)
    q_arr = q_arr / np.linalg.norm(q_arr)
    return [float(item) for item in q_arr]


def _real_layout_robot_pose(
    *,
    arms_distance: float,
    base_y: float,
    base_z: float,
    tilt_deg: float,
    twist_deg: float,
) -> list[list[float]]:
    tilt = np.deg2rad(float(tilt_deg))
    twist = np.deg2rad(float(twist_deg))
    twist_quat = _quat_from_axis_angle((0.0, 0.0, 1.0), twist)

    robot_pose = []
    for sign in (-1.0, 1.0):
        tilt_quat = _quat_from_axis_angle((0.0, 1.0, 0.0), sign * tilt)
        quat = _normalise_quat(_quat_mul(tilt_quat, twist_quat))
        robot_pose.append([0.0, float(base_y), float(base_z), *quat])
    return robot_pose


def _render_frame(env_backend) -> np.ndarray | None:
    frame = env_backend.env.render()
    if frame is None:
        return None
    if hasattr(frame, "detach") and hasattr(frame, "cpu"):
        frame = frame.detach().cpu()
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] >= 1:
        frame = frame[0]
    if frame.ndim != 3:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _render_candidate(args: argparse.Namespace, twist_deg: float, output_dir: Path) -> dict[str, Any]:
    robot_pose = _real_layout_robot_pose(
        arms_distance=float(args.arms_distance),
        base_y=float(args.base_y),
        base_z=float(args.base_z),
        tilt_deg=float(args.tilt_deg),
        twist_deg=float(twist_deg),
    )
    env = ManiSkillTaskFactory().create_env(
        "handover_block_ur5_v2_real_layout_probe",
        obs_mode="state_dict",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=5,
        arms_distance=float(args.arms_distance),
        robot_pose_override=robot_pose,
        homestate_override=UR5_REAL_LAYOUT_HOMESTATE,
    )
    try:
        env.reset(seed=int(args.seed))
        frame = None
        for _ in range(max(1, int(args.warmup_renders))):
            frame = _render_frame(env)
        if frame is None:
            raise RuntimeError("render() returned no RGB frame.")
        image_path = output_dir / f"twist_{twist_deg:+07.2f}deg.png"
        imageio.imwrite(image_path, frame)
        base_env = env.env.unwrapped
        return {
            "twist_deg": float(twist_deg),
            "image_path": str(image_path),
            "robot_pose_override": robot_pose,
            "left_root_pose": np.round(base_env.get_arm_root_pose_array(0), 8).tolist(),
            "right_root_pose": np.round(base_env.get_arm_root_pose_array(1), 8).tolist(),
            "left_home_qpos": np.round(base_env.get_arm_home_joint_positions(0), 8).tolist(),
            "right_home_qpos": np.round(base_env.get_arm_home_joint_positions(1), 8).tolist(),
        }
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render real-layout base-twist home-pose candidates.")
    parser.add_argument("--output-dir", default="/home/idiot/zhj/RoboTwin/tmp/handover/real_layout_twist_candidates")
    parser.add_argument("--twists-deg", default="0,90,180,270")
    parser.add_argument("--arms-distance", type=float, default=UR5_REAL_LAYOUT_ARMS_DISTANCE)
    parser.add_argument("--base-y", type=float, default=UR5_REAL_LAYOUT_BASE_Y)
    parser.add_argument("--base-z", type=float, default=UR5_REAL_LAYOUT_BASE_Z)
    parser.add_argument("--tilt-deg", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--warmup-renders", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for twist_deg in _parse_float_list(args.twists_deg):
        print(f"rendering twist={twist_deg:+.2f} deg")
        summaries.append(_render_candidate(args, twist_deg, output_dir))

    contact_frames = []
    for item in summaries:
        frame = imageio.imread(item["image_path"])
        contact_frames.append(np.asarray(frame, dtype=np.uint8))
    if contact_frames:
        min_height = min(frame.shape[0] for frame in contact_frames)
        contact_sheet = np.concatenate([frame[:min_height] for frame in contact_frames], axis=1)
        contact_path = output_dir / "twist_contact_sheet.png"
        imageio.imwrite(contact_path, contact_sheet)
    else:
        contact_path = None

    summary = {
        "output_dir": str(output_dir),
        "contact_sheet": str(contact_path) if contact_path is not None else None,
        "candidates": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
