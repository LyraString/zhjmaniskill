import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECT_SCRIPT = REPO_ROOT / "script" / "collect_maniskill_data.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch multiple independent collect_maniskill_data.py workers. "
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
        help="Seed offset between worker processes.",
    )
    parser.add_argument(
        "--max-attempts-per-worker",
        type=int,
        default=None,
        help="If set, pass this max-attempts value to every worker.",
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
        default=30.0,
        help="Print saved episode counts from worker output directories at this interval.",
    )
    parser.add_argument("--dry-run", action="store_true")

    # Arguments forwarded to collect_maniskill_data.py.
    parser.add_argument("--task-profile", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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


def split_counts(total: int, parts: int) -> list[int]:
    if total <= 0:
        raise ValueError("episode_num must be positive.")
    if parts <= 0:
        raise ValueError("--num-workers must be positive.")
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def maybe_add_option(cmd: list[str], name: str, value) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def worker_max_attempts(args, episode_count: int) -> int | None:
    if args.max_attempts_per_worker is not None:
        return int(args.max_attempts_per_worker)
    if args.max_attempts is None:
        return None
    attempts_per_episode = float(args.max_attempts) / float(args.episode_num)
    return max(int(episode_count), int(math.ceil(attempts_per_episode * episode_count)))


def build_worker_command(args, worker_idx: int, episode_count: int, unknown_args: list[str]) -> list[str]:
    task_config = f"{args.task_config}_part{worker_idx}"
    cmd = [
        sys.executable,
        "-u",
        str(COLLECT_SCRIPT),
        args.task_name,
        task_config,
        str(episode_count),
        "--seed",
        str(int(args.seed) + worker_idx * int(args.seed_stride)),
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
    maybe_add_option(cmd, "--max-attempts", worker_max_attempts(args, episode_count))
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


def count_saved_npz(task_name: str, task_config: str) -> int | None:
    data_dir = REPO_ROOT / "data" / task_name / task_config / "data"
    if not data_dir.is_dir():
        return None
    return len(list(data_dir.glob("*.npz")))


def main(argv=None) -> int:
    parser = build_parser()
    args, unknown_args = parser.parse_known_args(argv)
    episode_counts = split_counts(int(args.episode_num), int(args.num_workers))
    active_workers = [
        (worker_idx, episode_count)
        for worker_idx, episode_count in enumerate(episode_counts)
        if episode_count > 0
    ]
    log_dir = Path(args.log_dir) if args.log_dir is not None else REPO_ROOT / "tmp" / "collect_parallel" / args.task_config
    log_dir.mkdir(parents=True, exist_ok=True)

    worker_specs = []
    for worker_idx, episode_count in active_workers:
        cmd = build_worker_command(args, worker_idx, episode_count, unknown_args)
        env = worker_env(args, worker_idx)
        task_config = f"{args.task_config}_part{worker_idx}"
        log_path = log_dir / f"worker{worker_idx}.log"
        worker_specs.append((worker_idx, episode_count, task_config, cmd, env, log_path))

    print(f"parallel collection workers={len(worker_specs)} total_episodes={args.episode_num}", flush=True)
    for worker_idx, episode_count, task_config, cmd, env, log_path in worker_specs:
        gpu = env.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "inherit"))
        print(
            f"worker{worker_idx}: episodes={episode_count} task_config={task_config} "
            f"cuda_visible_devices={gpu} log={log_path}"
            ,
            flush=True,
        )
        print("  " + " ".join(cmd), flush=True)

    if args.dry_run:
        return 0

    processes = []
    start_time = time.time()
    for worker_idx, episode_count, task_config, cmd, env, log_path in worker_specs:
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_idx, episode_count, task_config, process, log_file, log_path))

    exit_code = 0
    remaining = {
        worker_idx: (episode_count, task_config, process, log_file, log_path)
        for worker_idx, episode_count, task_config, process, log_file, log_path in processes
    }
    last_progress_time = 0.0
    progress_interval = max(1.0, float(args.progress_interval_sec))
    while remaining:
        now = time.time()
        if now - last_progress_time >= progress_interval:
            parts = []
            for worker_idx, (episode_count, task_config, _process, _log_file, _log_path) in sorted(remaining.items()):
                saved_count = count_saved_npz(args.task_name, task_config)
                saved_text = "?" if saved_count is None else str(saved_count)
                parts.append(f"worker{worker_idx}={saved_text}/{episode_count}")
            print("progress " + " ".join(parts), flush=True)
            last_progress_time = now

        finished_workers = []
        for worker_idx, (episode_count, task_config, process, log_file, log_path) in remaining.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            saved_count = count_saved_npz(args.task_name, task_config)
            saved_text = "unknown" if saved_count is None else str(saved_count)
            print(
                f"worker{worker_idx} finished returncode={return_code} "
                f"requested={episode_count} saved={saved_text} log={log_path}",
                flush=True,
            )
            if return_code != 0:
                exit_code = return_code if exit_code == 0 else exit_code
            if saved_count is not None and saved_count != episode_count:
                print(
                    f"warning: worker{worker_idx} saved {saved_count}/{episode_count} episodes "
                    f"in data/{args.task_name}/{task_config}/data",
                    flush=True,
                )
            finished_workers.append(worker_idx)
        for worker_idx in finished_workers:
            remaining.pop(worker_idx, None)
        if remaining:
            time.sleep(min(5.0, progress_interval))

    elapsed = time.time() - start_time
    print(f"parallel collection elapsed_sec={elapsed:.1f}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
