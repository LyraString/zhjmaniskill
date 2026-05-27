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


def compact_probe(probe: dict[str, object]) -> dict[str, object]:
    alignment = probe.get("right_handover_alignment")
    compact = {
        "stage": probe.get("stage"),
        "passed": probe.get("passed"),
        "selected_contact_id": probe.get("selected_contact_id"),
        "selection_reason": probe.get("selection_reason"),
        "steps_taken": probe.get("steps_taken"),
        "metrics": probe.get("metrics"),
        "thresholds": probe.get("thresholds"),
        "handover_box_pose": probe.get("handover_box_pose"),
        "box_pose": probe.get("box_pose"),
        "right_gripper_pose": probe.get("right_gripper_pose"),
        "right_ee_pose": probe.get("right_ee_pose"),
        "right_qpos": probe.get("right_qpos"),
    }
    if isinstance(alignment, dict):
        compact["right_handover_alignment"] = {
            "box_gripper_distance": alignment.get("box_gripper_distance"),
            "box_planned_distance": alignment.get("box_planned_distance"),
            "box_to_gripper_xyz": alignment.get("box_to_gripper_xyz"),
            "box_to_planned_xyz": alignment.get("box_to_planned_xyz"),
        }
    close_trace = probe.get("right_handover_close_trace")
    if isinstance(close_trace, list) and close_trace:
        compact["right_handover_close_trace_tail"] = close_trace[-3:]
    return {key: value for key, value in compact.items() if value is not None}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the right-arm handover close stage for HandoverBlockUR5-v2."
    )
    parser.add_argument("--task-profile", default="handover_block_ur5_v2_smoke")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--no-preposition-right", action="store_true")
    parser.add_argument("--static-receiver", action="store_true")
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
        for offset in range(int(args.num_seeds)):
            seed = int(args.seed) + offset
            backend.reset(seed=seed)
            solver = HandoverBlockSolver(backend)
            solver.PREPOSITION_RIGHT_FOR_HANDOVER = not bool(args.no_preposition_right)
            solver.RIGHT_HANDOVER_STATIC_RECEIVER = bool(args.static_receiver)
            success, episode_length, status = solver.solve_episode(
                recorder=_NullRecorder(),
                max_episode_steps=args.max_episode_steps,
                stop_after_right_handover=True,
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
            "success_count": int(success_count),
            "success_rate": float(success_count / max(len(episodes), 1)),
            "episodes": episodes,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
