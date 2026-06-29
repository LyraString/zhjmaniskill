#!/usr/bin/env python3
"""Render real-layout home qpos mapping candidates.

This calibration helper keeps the measured real-layout base pose fixed and
scans simple mappings for shoulder_pan / shoulder_lift. The goal is to find a
simulation home qpos that visually matches the real robot, because UR teach
pendant joint values may not share the same base-frame convention as this URDF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from mani_skill.utils import sapien_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill import ManiSkillTaskFactory
from sim.maniskill.task_profiles import (
    UR5_REAL_LAYOUT_HOMESTATE,
    UR5_REAL_LAYOUT_ROBOT_POSE,
    UR5_REAL_LAYOUT_ARMS_DISTANCE,
)


PAN_OFFSETS = {
    "pan": 0.0,
    "pan_plus_pi": np.pi,
    "pan_minus_pi": -np.pi,
}
LIFT_MODES = {
    "lift": lambda q: q,
    "neg_lift": lambda q: -q,
    "lift_plus_half_pi": lambda q: q + np.pi / 2.0,
    "lift_minus_half_pi": lambda q: q - np.pi / 2.0,
    "neg_lift_plus_half_pi": lambda q: -q + np.pi / 2.0,
    "neg_lift_minus_half_pi": lambda q: -q - np.pi / 2.0,
}


def _parse_candidates(value: str | None, default: list[str]) -> list[str]:
    if value in (None, ""):
        return default
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_offsets_deg(value: str | None, default: list[str] | None = None) -> list[str]:
    return _parse_candidates(value, default or ["0"])


def _offset_deg_to_rad(value: str) -> float:
    return float(np.deg2rad(float(value)))


def _candidate_homestate(candidate: dict[str, str]) -> list[list[float]]:
    homestate = [list(values) for values in UR5_REAL_LAYOUT_HOMESTATE]
    for arm_index, prefix in ((0, "left"), (1, "right")):
        pan_mode = candidate[f"{prefix}_pan"]
        homestate[arm_index][0] = float(homestate[arm_index][0] + PAN_OFFSETS[pan_mode])
        lift_offset_key = f"{prefix}_lift_offset_deg"
        if lift_offset_key in candidate:
            homestate[arm_index][1] = float(
                homestate[arm_index][1] + np.deg2rad(float(candidate[lift_offset_key]))
            )
        else:
            lift_mode = candidate[f"{prefix}_lift"]
            homestate[arm_index][1] = float(LIFT_MODES[lift_mode](homestate[arm_index][1]))
        for joint_index, joint_name in ((3, "wrist1"), (4, "wrist2"), (5, "wrist3")):
            offset_key = f"{prefix}_{joint_name}_offset_deg"
            if offset_key in candidate:
                homestate[arm_index][joint_index] = float(
                    homestate[arm_index][joint_index] + _offset_deg_to_rad(candidate[offset_key])
                )
    return homestate


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


def _front_render_camera_configs() -> dict[str, dict[str, Any]]:
    # View from the lab/front side toward the robot bases.
    pose = sapien_utils.look_at(eye=[0.0, 1.15, 1.62], target=[0.0, -0.65, 1.15])
    p = np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3].tolist()
    q = np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4].tolist()
    return {
        "render_camera": {
            "pose": [*p, *q],
            "width": 768,
            "height": 768,
            "fov": 0.9,
        }
    }


def _render_candidate(args: argparse.Namespace, candidate: dict[str, str], output_dir: Path, index: int) -> dict[str, Any]:
    homestate = _candidate_homestate(candidate)
    env_kwargs = {}
    if args.view == "front":
        env_kwargs["human_render_camera_configs"] = _front_render_camera_configs()
    env = ManiSkillTaskFactory().create_env(
        "handover_block_ur5_v2_real_layout_probe",
        obs_mode="state_dict",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=5,
        arms_distance=UR5_REAL_LAYOUT_ARMS_DISTANCE,
        robot_pose_override=UR5_REAL_LAYOUT_ROBOT_POSE,
        homestate_override=homestate,
        **env_kwargs,
    )
    try:
        env.reset(seed=int(args.seed))
        frame = None
        for _ in range(max(1, int(args.warmup_renders))):
            frame = _render_frame(env)
        if frame is None:
            raise RuntimeError("render() returned no RGB frame.")
        wrist_label = ""
        wrist_keys = [
            "left_wrist1_offset_deg",
            "left_wrist2_offset_deg",
            "left_wrist3_offset_deg",
            "right_wrist1_offset_deg",
            "right_wrist2_offset_deg",
            "right_wrist3_offset_deg",
        ]
        if any(float(candidate.get(key, "0")) != 0.0 for key in wrist_keys):
            wrist_label = (
                f"_LW1{candidate.get('left_wrist1_offset_deg', '0')}"
                f"_LW2{candidate.get('left_wrist2_offset_deg', '0')}"
                f"_LW3{candidate.get('left_wrist3_offset_deg', '0')}"
                f"_RW1{candidate.get('right_wrist1_offset_deg', '0')}"
                f"_RW2{candidate.get('right_wrist2_offset_deg', '0')}"
                f"_RW3{candidate.get('right_wrist3_offset_deg', '0')}"
            )
        label = (
            f"{index:03d}_"
            f"L-{candidate['left_pan']}-{candidate['left_lift']}_"
            f"R-{candidate['right_pan']}-{candidate['right_lift']}"
            f"{wrist_label}"
        )
        image_path = output_dir / f"{label}.png"
        imageio.imwrite(image_path, frame)
        base_env = env.env.unwrapped
        return {
            "index": int(index),
            "candidate": candidate,
            "image_path": str(image_path),
            "homestate_override": homestate,
            "left_root_pose": np.round(base_env.get_arm_root_pose_array(0), 8).tolist(),
            "right_root_pose": np.round(base_env.get_arm_root_pose_array(1), 8).tolist(),
        }
    finally:
        env.close()


def _candidate_grid(args: argparse.Namespace) -> list[dict[str, str]]:
    left_pan_modes = _parse_candidates(args.left_pan_modes, ["pan", "pan_plus_pi", "pan_minus_pi"])
    right_pan_modes = _parse_candidates(args.right_pan_modes, ["pan", "pan_plus_pi", "pan_minus_pi"])
    left_lift_offsets = _parse_candidates(args.left_lift_offsets_deg, [])
    right_lift_offsets = _parse_candidates(args.right_lift_offsets_deg, [])
    offset_sweep = bool(left_lift_offsets or right_lift_offsets)
    if offset_sweep:
        left_lift_modes = [f"offset_{value}deg" for value in left_lift_offsets]
        right_lift_modes = [f"offset_{value}deg" for value in right_lift_offsets]
    else:
        left_lift_modes = _parse_candidates(args.left_lift_modes, list(LIFT_MODES.keys()))
        right_lift_modes = _parse_candidates(args.right_lift_modes, list(LIFT_MODES.keys()))

    for mode in [*left_pan_modes, *right_pan_modes]:
        if mode not in PAN_OFFSETS:
            raise ValueError(f"Unknown pan mode: {mode}")
    if not offset_sweep:
        for mode in [*left_lift_modes, *right_lift_modes]:
            if mode not in LIFT_MODES:
                raise ValueError(f"Unknown lift mode: {mode}")

    left_wrist1_offsets = _parse_offsets_deg(args.left_wrist1_offsets_deg)
    left_wrist2_offsets = _parse_offsets_deg(args.left_wrist2_offsets_deg)
    left_wrist3_offsets = _parse_offsets_deg(args.left_wrist3_offsets_deg)
    right_wrist1_offsets = _parse_offsets_deg(args.right_wrist1_offsets_deg)
    right_wrist2_offsets = _parse_offsets_deg(args.right_wrist2_offsets_deg)
    right_wrist3_offsets = _parse_offsets_deg(args.right_wrist3_offsets_deg)
    if args.paired_wrist_offsets:
        wrist_offset_tuples = []
        max_len = max(
            len(left_wrist1_offsets),
            len(left_wrist2_offsets),
            len(left_wrist3_offsets),
            len(right_wrist1_offsets),
            len(right_wrist2_offsets),
            len(right_wrist3_offsets),
        )
        for values in (
            left_wrist1_offsets,
            left_wrist2_offsets,
            left_wrist3_offsets,
            right_wrist1_offsets,
            right_wrist2_offsets,
            right_wrist3_offsets,
        ):
            if len(values) not in (1, max_len):
                raise ValueError("--paired-wrist-offsets requires each wrist offset list length to be 1 or max length.")
        for idx in range(max_len):
            wrist_offset_tuples.append(
                (
                    left_wrist1_offsets[idx if len(left_wrist1_offsets) > 1 else 0],
                    left_wrist2_offsets[idx if len(left_wrist2_offsets) > 1 else 0],
                    left_wrist3_offsets[idx if len(left_wrist3_offsets) > 1 else 0],
                    right_wrist1_offsets[idx if len(right_wrist1_offsets) > 1 else 0],
                    right_wrist2_offsets[idx if len(right_wrist2_offsets) > 1 else 0],
                    right_wrist3_offsets[idx if len(right_wrist3_offsets) > 1 else 0],
                )
            )
    else:
        wrist_offset_tuples = [
            (lw1, lw2, lw3, rw1, rw2, rw3)
            for lw1 in left_wrist1_offsets
            for lw2 in left_wrist2_offsets
            for lw3 in left_wrist3_offsets
            for rw1 in right_wrist1_offsets
            for rw2 in right_wrist2_offsets
            for rw3 in right_wrist3_offsets
        ]

    candidates = []
    for left_pan in left_pan_modes:
        for right_pan in right_pan_modes:
            for left_idx, left_lift in enumerate(left_lift_modes):
                for right_idx, right_lift in enumerate(right_lift_modes):
                    for lw1, lw2, lw3, rw1, rw2, rw3 in wrist_offset_tuples:
                        candidate = {
                            "left_pan": left_pan,
                            "right_pan": right_pan,
                            "left_lift": left_lift,
                            "right_lift": right_lift,
                            "left_wrist1_offset_deg": str(lw1),
                            "left_wrist2_offset_deg": str(lw2),
                            "left_wrist3_offset_deg": str(lw3),
                            "right_wrist1_offset_deg": str(rw1),
                            "right_wrist2_offset_deg": str(rw2),
                            "right_wrist3_offset_deg": str(rw3),
                        }
                        if offset_sweep:
                            candidate["left_lift_offset_deg"] = str(left_lift_offsets[left_idx])
                            candidate["right_lift_offset_deg"] = str(right_lift_offsets[right_idx])
                        candidates.append(candidate)
                        if len(candidates) >= int(args.max_candidates):
                            return candidates
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render real-layout home-qpos mapping candidates.")
    parser.add_argument("--output-dir", default="/home/idiot/zhj/RoboTwin/tmp/handover/real_layout_home_candidates")
    parser.add_argument("--left-pan-modes", default="pan,pan_plus_pi,pan_minus_pi")
    parser.add_argument("--right-pan-modes", default="pan,pan_plus_pi,pan_minus_pi")
    parser.add_argument("--left-lift-modes", default="lift,neg_lift,lift_plus_half_pi,lift_minus_half_pi")
    parser.add_argument("--right-lift-modes", default="lift,neg_lift,lift_plus_half_pi,lift_minus_half_pi")
    parser.add_argument("--left-lift-offsets-deg", default="")
    parser.add_argument("--right-lift-offsets-deg", default="")
    parser.add_argument("--left-wrist1-offsets-deg", default="0")
    parser.add_argument("--left-wrist2-offsets-deg", default="0")
    parser.add_argument("--left-wrist3-offsets-deg", default="0")
    parser.add_argument("--right-wrist1-offsets-deg", default="0")
    parser.add_argument("--right-wrist2-offsets-deg", default="0")
    parser.add_argument("--right-wrist3-offsets-deg", default="0")
    parser.add_argument(
        "--paired-wrist-offsets",
        action="store_true",
        help="Zip wrist offset lists instead of taking the Cartesian product.",
    )
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--warmup-renders", type=int, default=2)
    parser.add_argument("--view", choices=("default", "front"), default="front")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for index, candidate in enumerate(_candidate_grid(args)):
        print(f"rendering candidate={index:03d} {candidate}")
        summaries.append(_render_candidate(args, candidate, output_dir, index))

    frames = [np.asarray(imageio.imread(item["image_path"]), dtype=np.uint8) for item in summaries]
    contact_sheet_path = None
    if frames:
        min_height = min(frame.shape[0] for frame in frames)
        rows = []
        cols = int(np.ceil(np.sqrt(len(frames))))
        for row_start in range(0, len(frames), cols):
            row_frames = frames[row_start : row_start + cols]
            row_frames = [frame[:min_height] for frame in row_frames]
            if len(row_frames) < cols:
                pad = np.zeros_like(row_frames[0])
                row_frames.extend([pad] * (cols - len(row_frames)))
            rows.append(np.concatenate(row_frames, axis=1))
        contact_sheet = np.concatenate(rows, axis=0)
        contact_sheet_path = output_dir / "home_candidates_contact_sheet.png"
        imageio.imwrite(contact_sheet_path, contact_sheet)

    summary = {
        "output_dir": str(output_dir),
        "contact_sheet": str(contact_sheet_path) if contact_sheet_path is not None else None,
        "candidates": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
