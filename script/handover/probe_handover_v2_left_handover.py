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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe the left-arm move-to-handover stage for HandoverBlockUR5-v2."
    )
    parser.add_argument("--task-profile", default="handover_block_ur5_v2_smoke")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--no-preposition-right", action="store_true")
    parser.add_argument("--hold-steps", type=int, default=0)
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
            success, episode_length, status = solver.solve_episode(
                recorder=_NullRecorder(),
                max_episode_steps=args.max_episode_steps,
                stop_after_left_handover=True,
                stop_after_left_handover_hold_steps=int(args.hold_steps),
            )
            episode = {
                "seed": seed,
                "success": bool(success),
                "episode_length": int(episode_length),
                "status": str(status),
                "probe": solver.last_debug_info,
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
            "hold_steps": int(args.hold_steps),
            "success_count": int(success_count),
            "success_rate": float(success_count / max(len(episodes), 1)),
            "episodes": episodes,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
