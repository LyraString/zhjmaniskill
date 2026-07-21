import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill.collection_scheduler import SharedCollectionScheduler


COLLECT_SCRIPT = REPO_ROOT / "script" / "collect_maniskill_data.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamically schedule seeds across persistent collection workers. "
            "This only parallelizes at the process level; it does not change the collector, "
            "controller, action semantics, or saved episode format."
        )
    )
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("episode_num", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated visible GPU ids. Workers are assigned round-robin. "
            "Use `--gpus 1 --num-workers 2` to run two workers on the same GPU."
        ),
    )
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=100000,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-attempts-per-worker",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for worker stdout/stderr logs. Defaults to tmp/collect_parallel/<task_config>.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=2.0,
        help="Poll saved episodes at this interval; print only when the count increases.",
    )
    parser.add_argument("--dry-run", action="store_true")

    # Arguments forwarded to collect_maniskill_data.py.
    parser.add_argument("--task-profile", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--episode-index-offset",
        type=int,
        default=0,
        help="Starting episode index in the shared dataset directory.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
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
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument(
        "--collector",
        choices=(
            "random",
            "legacy_replay",
            "scripted_expert",
            "scripted_delta_pose_expert",
            "policy_intervention",
            "policy_manual_intervention",
        ),
        default=None,
    )
    parser.add_argument("--teacher-zarr-path", type=str, default=None)
    parser.add_argument("--teacher-seed-path", type=str, default=None)
    parser.add_argument("--policy-kind", choices=("flowmatching", "fpo"), default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--save-only-success", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--debug-scripted-expert", action="store_true")
    parser.add_argument("--debug-policy-intervention", action="store_true")
    parser.add_argument("--debug-policy-manual-intervention", action="store_true")
    return parser


def maybe_add_option(cmd: list[str], name: str, value) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def build_worker_command(
    args,
    worker_idx: int,
    scheduler_path: Path,
    unknown_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(COLLECT_SCRIPT),
        args.task_name,
        args.task_config,
        str(args.episode_num),
        "--seed",
        str(args.seed),
        "--shared-collection-state",
        str(scheduler_path),
    ]
    maybe_add_option(cmd, "--task-profile", args.task_profile)
    maybe_add_option(cmd, "--max-episode-steps", args.max_episode_steps)
    maybe_add_option(cmd, "--action-type", args.action_type)
    maybe_add_option(cmd, "--obs-mode", args.obs_mode)
    maybe_add_option(cmd, "--control-mode", args.control_mode)
    maybe_add_option(cmd, "--render-mode", args.render_mode)
    maybe_add_option(cmd, "--sim-backend", args.sim_backend)
    maybe_add_option(cmd, "--render-backend", args.render_backend)
    maybe_add_option(cmd, "--reward-mode", args.reward_mode)
    maybe_add_option(cmd, "--robot-uids", args.robot_uids)
    maybe_add_option(cmd, "--sim-freq", args.sim_freq)
    maybe_add_option(cmd, "--control-freq", args.control_freq)
    maybe_add_option(cmd, "--save-video-dir", args.save_video_dir)
    maybe_add_option(cmd, "--video-fps", args.video_fps)
    maybe_add_option(cmd, "--collector", args.collector)
    maybe_add_option(cmd, "--teacher-zarr-path", args.teacher_zarr_path)
    maybe_add_option(cmd, "--teacher-seed-path", args.teacher_seed_path)
    maybe_add_option(cmd, "--policy-kind", args.policy_kind)
    maybe_add_option(cmd, "--ckpt-path", args.ckpt_path)
    if args.save_only_success:
        cmd.append("--save-only-success")
    if args.debug_scripted_expert:
        cmd.append("--debug-scripted-expert")
    if args.debug_policy_intervention:
        cmd.append("--debug-policy-intervention")
    if args.debug_policy_manual_intervention:
        cmd.append("--debug-policy-manual-intervention")
    cmd.extend(unknown_args)
    return cmd


def worker_env(args, worker_idx: int) -> dict[str, str]:
    env = dict(os.environ)
    if args.gpus is None or args.gpus.strip() == "":
        env["PYTHONUNBUFFERED"] = "1"
        return env
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip() != ""]
    if len(gpus) == 0:
        env["PYTHONUNBUFFERED"] = "1"
        return env
    env["CUDA_VISIBLE_DEVICES"] = gpus[worker_idx % len(gpus)]
    env["PYTHONUNBUFFERED"] = "1"
    return env


def count_saved_episodes(
    task_name: str,
    task_config: str,
    episode_index_offset: int,
    episode_count: int,
) -> int | None:
    data_dir = REPO_ROOT / "data" / task_name / task_config / "data"
    if not data_dir.is_dir():
        return None
    return sum(
        (data_dir / f"episode{episode_idx}.npz").is_file()
        and (data_dir / f"episode{episode_idx}.json").is_file()
        for episode_idx in range(
            episode_index_offset,
            episode_index_offset + episode_count,
        )
    )


def main(argv=None) -> int:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    if args.episode_num <= 0:
        raise ValueError("episode_num must be positive.")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    if args.episode_index_offset < 0:
        raise ValueError("--episode-index-offset must be non-negative.")
    if args.collector not in {"scripted_expert", "scripted_delta_pose_expert"}:
        raise ValueError(
            "Dynamic seed scheduling requires --collector scripted_expert or "
            "--collector scripted_delta_pose_expert."
        )

    max_attempts = args.max_attempts
    if max_attempts is None and args.max_attempts_per_worker is not None:
        max_attempts = int(args.max_attempts_per_worker) * int(args.num_workers)

    data_dir = REPO_ROOT / "data" / args.task_name / args.task_config / "data"
    collisions = [
        data_dir / f"episode{episode_idx}{suffix}"
        for episode_idx in range(
            int(args.episode_index_offset),
            int(args.episode_index_offset) + int(args.episode_num),
        )
        for suffix in (".npz", ".json")
        if (data_dir / f"episode{episode_idx}{suffix}").exists()
    ]
    if collisions:
        preview = ", ".join(str(path) for path in collisions[:3])
        raise FileExistsError(
            "Refusing to overwrite existing episode files in the shared output range: "
            f"{preview}"
        )

    log_dir = Path(args.log_dir) if args.log_dir is not None else REPO_ROOT / "tmp" / "collect_parallel" / args.task_config
    log_dir.mkdir(parents=True, exist_ok=True)
    scheduler_path = log_dir / "scheduler_state.json"

    worker_specs = []
    active_worker_count = min(int(args.num_workers), int(args.episode_num))
    for worker_idx in range(active_worker_count):
        cmd = build_worker_command(
            args,
            worker_idx,
            scheduler_path,
            unknown_args,
        )
        env = worker_env(args, worker_idx)
        log_path = log_dir / f"worker{worker_idx}.log"
        worker_specs.append((worker_idx, cmd, env, log_path))

    print(
        f"dynamic collection workers={len(worker_specs)} "
        f"target_episodes={args.episode_num} start_seed={args.seed} "
        f"max_attempts={max_attempts}",
        flush=True,
    )
    for worker_idx, cmd, env, log_path in worker_specs:
        gpu = env.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "inherit"))
        print(
            f"worker{worker_idx}: task_config={args.task_config} "
            f"cuda_visible_devices={gpu} log={log_path}",
            flush=True,
        )
        print("  " + " ".join(cmd), flush=True)

    if args.dry_run:
        return 0

    scheduler = SharedCollectionScheduler.initialize(
        scheduler_path,
        seed=int(args.seed),
        episode_index_offset=int(args.episode_index_offset),
        episode_count=int(args.episode_num),
        max_attempts=None if max_attempts is None else int(max_attempts),
    )

    processes = []
    start_time = time.time()
    for worker_idx, cmd, env, log_path in worker_specs:
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_idx, process, log_file, log_path))

    exit_code = 0
    remaining = {
        worker_idx: (process, log_file, log_path)
        for worker_idx, process, log_file, log_path in processes
    }
    initial_saved_count = count_saved_episodes(
        args.task_name,
        args.task_config,
        int(args.episode_index_offset),
        int(args.episode_num),
    )
    last_reported_saved_count = initial_saved_count or 0
    progress_interval = max(0.5, float(args.progress_interval_sec))
    while remaining:
        saved_count = count_saved_episodes(
            args.task_name,
            args.task_config,
            int(args.episode_index_offset),
            int(args.episode_num),
        )
        if (
            saved_count is not None
            and saved_count > last_reported_saved_count
        ):
            print(f"saved {saved_count}/{args.episode_num}", flush=True)
            last_reported_saved_count = saved_count

        finished_workers = []
        for worker_idx, (process, log_file, log_path) in remaining.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            print(
                f"worker{worker_idx} finished returncode={return_code} log={log_path}",
                flush=True,
            )
            if return_code != 0:
                exit_code = return_code if exit_code == 0 else exit_code
            finished_workers.append(worker_idx)
        for worker_idx in finished_workers:
            remaining.pop(worker_idx, None)
        if remaining:
            time.sleep(progress_interval)

    elapsed = time.time() - start_time
    scheduler_state = scheduler.snapshot()
    saved_count = count_saved_episodes(
        args.task_name,
        args.task_config,
        int(args.episode_index_offset),
        int(args.episode_num),
    )
    if saved_count is not None and saved_count > last_reported_saved_count:
        print(f"saved {saved_count}/{args.episode_num}", flush=True)
    if saved_count != int(args.episode_num):
        exit_code = 1 if exit_code == 0 else exit_code
        print(
            f"warning: shared dataset saved {saved_count}/{args.episode_num} episodes "
            f"in data/{args.task_name}/{args.task_config}/data",
            flush=True,
        )
    print(
        f"dynamic collection attempts={scheduler_state['attempt_count']} "
        f"next_seed={scheduler_state['next_seed']} elapsed_sec={elapsed:.1f}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
