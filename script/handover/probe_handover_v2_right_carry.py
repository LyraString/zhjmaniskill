from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill.expert.handover_block_solver import HandoverBlockSolver, _NullRecorder
from sim.maniskill.task_factory import ManiSkillTaskFactory


def movement_segments(name: str, scale: float) -> tuple[tuple[float, float, float], ...]:
    base_segments: dict[str, tuple[tuple[float, float, float], ...]] = {
        "none": (),
        "small": ((0.015, 0.0, 0.0), (0.0, 0.015, 0.0), (0.0, 0.0, 0.015)),
        "xy": ((0.02, 0.0, 0.0), (0.0, 0.02, 0.0)),
        "lift": ((0.0, 0.0, 0.03),),
    }
    segments = base_segments[str(name)]
    return tuple(
        (float(x) * float(scale), float(y) * float(scale), float(z) * float(scale))
        for x, y, z in segments
    )


def compact_probe(probe: dict[str, object]) -> dict[str, object]:
    alignment = probe.get("right_handover_alignment")
    compact = {
        "stage": probe.get("stage"),
        "passed": probe.get("passed"),
        "failed_stage": probe.get("failed_stage"),
        "selected_contact_id": probe.get("selected_contact_id"),
        "selection_reason": probe.get("selection_reason"),
        "steps_taken": probe.get("steps_taken"),
        "metrics": probe.get("metrics"),
        "thresholds": probe.get("thresholds"),
        "handover_box_pose": probe.get("handover_box_pose"),
        "post_release_box_pose": probe.get("post_release_box_pose"),
        "box_pose": probe.get("box_pose"),
        "right_gripper_pose": probe.get("right_gripper_pose"),
        "right_ee_pose": probe.get("right_ee_pose"),
        "left_gripper_pose": probe.get("left_gripper_pose"),
    }
    if isinstance(alignment, dict):
        compact["right_handover_alignment"] = {
            "box_gripper_distance": alignment.get("box_gripper_distance"),
            "box_planned_distance": alignment.get("box_planned_distance"),
            "box_to_gripper_xyz": alignment.get("box_to_gripper_xyz"),
            "box_to_planned_xyz": alignment.get("box_to_planned_xyz"),
        }
    carry_trace = probe.get("carry_trace")
    if isinstance(carry_trace, list) and carry_trace:
        compact["carry_trace_tail"] = carry_trace[-3:]
    release_trace = probe.get("left_release_clear_trace")
    if isinstance(release_trace, list) and release_trace:
        compact["left_release_clear_trace_tail"] = release_trace[-2:]
    return {key: value for key, value in compact.items() if value is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe whether the right arm can carry the box after handover release."
    )
    parser.add_argument("--task-profile", default="handover_block_ur5_v2_smoke")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--no-preposition-right", action="store_true")
    parser.add_argument("--static-receiver", dest="static_receiver", action="store_true", default=True)
    parser.add_argument("--dynamic-receiver", dest="static_receiver", action="store_false")
    parser.add_argument("--right-grasp-distance", type=float, default=0.0)
    parser.add_argument("--movement", choices=("none", "small", "xy", "lift"), default="small")
    parser.add_argument("--movement-scale", type=float, default=1.0)
    parser.add_argument("--move-mode", choices=("ik", "planner"), default="ik")
    parser.add_argument("--full-debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = ManiSkillTaskFactory().create_env(
        args.task_profile,
        obs_mode="state_dict",
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    try:
        episodes = []
        segments = movement_segments(args.movement, args.movement_scale)
        for offset in range(int(args.num_seeds)):
            seed = int(args.seed) + offset
            backend.reset(seed=seed)
            solver = HandoverBlockSolver(backend)
            solver.PREPOSITION_RIGHT_FOR_HANDOVER = not bool(args.no_preposition_right)
            solver.RIGHT_HANDOVER_STATIC_RECEIVER = bool(args.static_receiver)
            solver.RIGHT_HANDOVER_GRASP_DISTANCE = float(args.right_grasp_distance)
            solver.RIGHT_CARRY_PROBE_MOVE_MODE = str(args.move_mode)
            success, episode_length, status = solver.solve_episode(
                recorder=_NullRecorder(),
                max_episode_steps=args.max_episode_steps,
                stop_after_right_carry_probe=True,
                right_carry_probe_segments=segments,
            )
            episode = {
                "seed": seed,
                "success": bool(success),
                "episode_length": int(episode_length),
                "status": str(status),
                "probe": solver.last_debug_info if args.full_debug else compact_probe(solver.last_debug_info),
            }
            if args.full_debug:
                episode["first_pick_debug"] = solver.first_pick_debug
            episodes.append(episode)

        success_count = sum(1 for episode in episodes if episode["success"])
        payload = {
            "task_profile": args.task_profile,
            "seed_start": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "preposition_right": not bool(args.no_preposition_right),
            "static_receiver": bool(args.static_receiver),
            "right_grasp_distance": float(args.right_grasp_distance),
            "movement": str(args.movement),
            "movement_scale": float(args.movement_scale),
            "move_mode": str(args.move_mode),
            "segments": segments,
            "success_count": int(success_count),
            "success_rate": float(success_count / max(len(episodes), 1)),
            "episodes": episodes,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
