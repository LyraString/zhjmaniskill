"""Evaluate a SmolVLA policy server on held-out or fresh-reset UR5 handovers."""

from __future__ import annotations

import argparse
from contextlib import closing
from io import BytesIO
import json
import math
from pathlib import Path
import socket
import struct
import sys
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from mani_skill.utils import sapien_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from script.check_vr_teleop_dataset import _render_frame, _restore_initial_state, _to_numpy
from sim.maniskill import ManiSkillTaskFactory
from sim.maniskill.task_profiles import get_task_profile
from sim.maniskill.teleop import capture_canonical_observation


CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")
DEFAULT_DATASET = REPO_ROOT / "data/HandoverBlockUR5-v2/pico_handover_clean_v2"
DEFAULT_HELDOUT_OUTPUT = REPO_ROOT / "tmp/handover/smolvla_050000/results.json"
DEFAULT_RESET_OUTPUT = REPO_ROOT / "tmp/handover/smolvla_050000_test/results.json"
DEFAULT_HELDOUT_EPISODES = (6, 33, 49, 55, 57, 66, 76, 78)
DEFAULT_TASK_PROFILE = "handover_block_ur5_v2_vr_robotwin_clean"
EXPECTED_CONTROL_MODE = "bimanual_pd_ee_delta_pose"
PACKET_LENGTH = struct.Struct("!Q")
MAX_PACKET_BYTES = 64 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--task-profile", default=None)
    parser.add_argument("--task-instruction", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument(
        "--evaluation-mode",
        choices=("heldout", "reset"),
        default="heldout",
        help="Restore held-out recordings or evaluate fresh unseen reset seeds.",
    )
    parser.add_argument("--episodes", nargs="+", type=int, default=list(DEFAULT_HELDOUT_EPISODES))
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--policy-seed", type=int, default=999)
    parser.add_argument("--policy-label", default="smolvla-050000")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--max-translation-step", type=float, default=0.015)
    parser.add_argument("--max-rotation-step", type=float, default=0.10)
    parser.add_argument("--video-every", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("Policy server closed the connection.")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_packet(connection: socket.socket) -> dict[str, np.ndarray]:
    size = PACKET_LENGTH.unpack(_recv_exact(connection, PACKET_LENGTH.size))[0]
    if size <= 0 or size > MAX_PACKET_BYTES:
        raise ValueError(f"Invalid packet size from policy server: {size} bytes")
    with np.load(BytesIO(_recv_exact(connection, size)), allow_pickle=False) as packet:
        return {key: packet[key] for key in packet.files}


def send_packet(connection: socket.socket, **arrays: np.ndarray) -> None:
    stream = BytesIO()
    np.savez(stream, **arrays)
    payload = stream.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError(f"Packet exceeds {MAX_PACKET_BYTES} bytes")
    connection.sendall(PACKET_LENGTH.pack(len(payload)) + payload)


class PolicyClient:
    def __init__(self, host: str, port: int):
        self.connection = socket.create_connection((host, port), timeout=10.0)
        self.connection.settimeout(None)

    def close(self) -> None:
        self.connection.close()

    def _request(self, **arrays: np.ndarray) -> dict[str, np.ndarray]:
        send_packet(self.connection, **arrays)
        response = recv_packet(self.connection)
        status = str(np.asarray(response.get("status", "")).reshape(()).item())
        if status != "ok":
            message = str(np.asarray(response.get("error", "unknown server error")).reshape(()).item())
            raise RuntimeError(f"Policy server error: {message}")
        return response

    def reset(self) -> None:
        self._request(command=np.asarray("reset"))

    def predict(self, observation: dict[str, Any], task: str, seed: int) -> np.ndarray:
        packet = {
            "command": np.asarray("predict"),
            "state": np.asarray(observation["robot"]["agent_qpos"], dtype=np.float32),
            "task": np.asarray(task),
            "seed": np.asarray(seed, dtype=np.int64),
        }
        for name in CAMERA_NAMES:
            packet[name] = np.asarray(observation["cameras"][name]["rgb"], dtype=np.uint8)
        response = self._request(**packet)
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise RuntimeError(f"Expected server actions [K,14], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("Policy server returned NaN or Inf actions.")
        return actions


def _bounded_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm == 0.0:
        return vector
    return vector * (max_norm / norm)


def constrain_delta_action(
    action: np.ndarray,
    *,
    max_translation_step: float,
    max_rotation_step: float,
) -> tuple[np.ndarray, bool]:
    raw = np.asarray(action, dtype=np.float32)
    if raw.shape != (14,) or not np.isfinite(raw).all():
        raise ValueError(f"Expected one finite 14D action, got shape={raw.shape}")
    constrained = raw.copy()
    for offset in (0, 7):
        constrained[offset : offset + 3] = _bounded_vector(
            constrained[offset : offset + 3], max_translation_step
        )
        constrained[offset + 3 : offset + 6] = _bounded_vector(
            constrained[offset + 3 : offset + 6], max_rotation_step
        )
        constrained[offset + 6] = np.clip(constrained[offset + 6], 0.0, 1.0)
    return constrained, not np.array_equal(raw, constrained)


def episode_dir(dataset: Path, episode: int) -> Path:
    path = dataset / "episodes" / f"episode_{episode:06d}"
    if not (path / "episode.json").is_file() or not (path / "data.hdf5").is_file():
        raise FileNotFoundError(f"Missing canonical episode {episode}: {path}")
    return path


def load_episode_metadata(dataset: Path, episode: int) -> dict[str, Any]:
    source = episode_dir(dataset, episode)
    return json.loads((source / "episode.json").read_text(encoding="utf-8"))


def recorded_dataset_seeds(dataset: Path) -> set[int]:
    seeds = set()
    for path in sorted((dataset / "episodes").glob("episode_*")):
        metadata_path = path / "episode.json"
        if metadata_path.is_file():
            seeds.add(int(json.loads(metadata_path.read_text(encoding="utf-8"))["seed"]))
    if not seeds:
        raise FileNotFoundError(f"No episode metadata found under {dataset}")
    return seeds


def evaluation_cases(args: argparse.Namespace) -> list[dict[str, int | str | None]]:
    if args.evaluation_mode == "heldout":
        return [
            {
                "case_id": f"episode_{episode:06d}",
                "episode": int(episode),
                "seed": int(load_episode_metadata(args.dataset, episode)["seed"]),
            }
            for episode in args.episodes
        ]

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_episodes)))
    if args.dataset is not None:
        overlap = sorted(set(seeds) & recorded_dataset_seeds(args.dataset))
        if overlap:
            raise ValueError(f"Reset evaluation seeds overlap the recorded dataset: {overlap}")
    return [
        {"case_id": f"seed_{seed:06d}", "episode": None, "seed": seed}
        for seed in seeds
    ]


def reset_task_profile(dataset: Path | None, task_profile: str | None) -> str:
    if task_profile is None and dataset is not None:
        task_profile = str(load_episode_metadata(dataset, 0)["task_profile"])
    profile_name = task_profile or DEFAULT_TASK_PROFILE
    profile = get_task_profile(profile_name)
    if profile["env_id"] != "HandoverBlockUR5-v2":
        raise ValueError(f"Unsupported evaluation task profile: {profile_name}")
    if profile["control_mode"] != EXPECTED_CONTROL_MODE:
        raise ValueError(f"Task profile has incompatible control mode: {profile_name}")
    return profile_name


def reset_case_metadata(task_profile: str) -> dict[str, Any]:
    profile = get_task_profile(task_profile)
    return {
        "task_profile": task_profile,
        "control_mode": profile["control_mode"],
        "sim_freq_hz": profile["sim_freq"],
        "control_freq_hz": profile["control_freq"],
        "action_semantics": "actual_relative_delta_pose_euler_xyz",
    }


def case_task_instruction(
    args: argparse.Namespace, metadata: dict[str, Any], env: Any
) -> str:
    if args.task_instruction is not None:
        return str(args.task_instruction)
    if args.evaluation_mode == "heldout":
        return str(metadata["task_instruction"])
    return str(env.env.unwrapped.task_description)


def create_env(metadata: dict[str, Any], args: argparse.Namespace, *, record_video: bool):
    viewer_pose = sapien_utils.look_at(
        eye=np.asarray([0.0, -0.45, 1.90], dtype=np.float32),
        target=np.asarray([0.0, -0.05, 0.75], dtype=np.float32),
    )
    return ManiSkillTaskFactory().create_env(
        task_name=str(metadata["task_profile"]),
        obs_mode="pointcloud+state_dict",
        control_mode=str(metadata["control_mode"]),
        render_mode="rgb_array" if record_video else None,
        sim_backend=args.sim_backend or str(metadata.get("sim_backend", "physx_cpu")),
        render_backend=args.render_backend,
        sim_freq=int(metadata["sim_freq_hz"]),
        control_freq=int(metadata["control_freq_hz"]),
        max_episode_steps=int(args.max_steps),
        human_render_camera_configs={"render_camera": {"pose": viewer_pose}},
    )


def _validate_absolute_action(action: np.ndarray, action_space: Any) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(14)
    if not np.isfinite(action).all():
        raise ValueError("Absolute action contains NaN or Inf.")
    if action.shape != action_space.shape:
        raise ValueError(f"Action shape {action.shape} does not match {action_space.shape}")
    if np.any(action < action_space.low) or np.any(action > action_space.high):
        raise ValueError("Absolute action lies outside the ManiSkill action space.")
    return action


def _scalar(value: Any) -> Any:
    return _to_numpy(value).reshape(-1)[0].item()


def run_case(
    args: argparse.Namespace,
    client: PolicyClient,
    case: dict[str, int | str | None],
    video_dir: Path,
) -> dict[str, Any]:
    episode = case["episode"]
    seed = int(case["seed"])
    case_id = str(case["case_id"])
    source = episode_dir(args.dataset, int(episode)) if episode is not None else None
    metadata = (
        load_episode_metadata(args.dataset, int(episode))
        if episode is not None
        else reset_case_metadata(args.task_profile)
    )
    if metadata.get("action_semantics") != "actual_relative_delta_pose_euler_xyz":
        raise ValueError(f"Case {case_id} has incompatible action semantics.")
    if metadata.get("control_mode") != EXPECTED_CONTROL_MODE:
        raise ValueError(f"Case {case_id} has incompatible control mode.")

    video_path = video_dir / f"{case_id}.mp4"
    env = create_env(metadata, args, record_video=True)
    writer = None
    h5_file = None
    result: dict[str, Any] = {
        "case_id": case_id,
        "evaluation_mode": args.evaluation_mode,
        "episode": int(episode) if episode is not None else None,
        "seed": seed,
        "task_instruction": None,
        "success": False,
        "max_stage": 0,
        "steps": 0,
        "replans": 0,
        "constrained_action_steps": 0,
        "initial_qpos_error": None,
        "initial_box_position_error": None,
        "video": str(video_path),
        "error": None,
    }
    try:
        env.reset(seed=seed)
        raw_env = env.env.unwrapped
        task_instruction = case_task_instruction(args, metadata, env)
        result["task_instruction"] = task_instruction
        if args.evaluation_mode == "heldout":
            h5_file = h5py.File(source / "data.hdf5", "r")
            _restore_initial_state(raw_env, h5_file)
            raw_env._get_reward_stage()[:] = 0

            actual_qpos = _to_numpy(raw_env._get_agent_qpos()).reshape(-1, 14)[0]
            recorded_qpos = np.asarray(h5_file["observations/robot/agent_qpos"][0])
            actual_box = _to_numpy(raw_env.box.pose.raw_pose).reshape(-1, 7)[0]
            recorded_box = np.asarray(h5_file["observations/task/box_pose"][0])
            result["initial_qpos_error"] = float(np.linalg.norm(actual_qpos - recorded_qpos))
            result["initial_box_position_error"] = float(
                np.linalg.norm(actual_box[:3] - recorded_box[:3])
            )

        raw_obs = raw_env.get_obs()
        observation = capture_canonical_observation(raw_obs, raw_env)
        client.reset()
        action_queue: list[np.ndarray] = []

        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            video_path,
            fps=int(args.video_fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        frame = _render_frame(env)
        if frame is not None:
            writer.append_data(frame)

        policy_case_seed = int(episode) if episode is not None else seed
        for step in range(int(args.max_steps)):
            if not action_queue:
                policy_seed = int(args.policy_seed) + 100_000 * policy_case_seed + step
                chunk = client.predict(
                    observation,
                    task_instruction,
                    policy_seed,
                )
                if args.execution_horizon > len(chunk):
                    raise ValueError(
                        f"Execution horizon {args.execution_horizon} exceeds chunk {len(chunk)}"
                    )
                action_queue = list(chunk[: args.execution_horizon])
                result["replans"] += 1

            delta, was_constrained = constrain_delta_action(
                action_queue.pop(0),
                max_translation_step=float(args.max_translation_step),
                max_rotation_step=float(args.max_rotation_step),
            )
            result["constrained_action_steps"] += int(was_constrained)
            absolute = np.asarray(env.prepare_action_chunk(delta[None, :]), dtype=np.float32)
            absolute = _validate_absolute_action(absolute.reshape(14), env.action_space)
            raw_obs, _, terminated, truncated, info = env.step_raw(
                ActionSpec.from_value(absolute, action_type=EXPECTED_CONTROL_MODE)
            )
            observation = capture_canonical_observation(raw_obs, raw_env)

            result["steps"] = step + 1
            result["max_stage"] = max(
                int(result["max_stage"]), int(_scalar(info.get("reward_stage", 0)))
            )
            success = bool(_scalar(info.get("success", False)))
            result["success"] = bool(result["success"] or success)
            if (step + 1) % int(args.video_every) == 0:
                frame = _render_frame(env)
                if frame is not None:
                    writer.append_data(frame)
            if result["success"] or terminated or truncated:
                break
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if h5_file is not None:
            h5_file.close()
        if writer is not None:
            writer.close()
        env.close()

    print("episode_result " + json.dumps(result), flush=True)
    return result


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1.0 + z**2 / total
    center = (probability + z**2 / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / total + z**2 / (4.0 * total**2)
    ) / denominator
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["success"]) for row in rows)
    total_steps = sum(int(row["steps"]) for row in rows)
    constrained_steps = sum(int(row["constrained_action_steps"]) for row in rows)
    return {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "success_rate_wilson_95": _wilson_interval(successes, len(rows)),
        "mean_max_stage": float(np.mean([row["max_stage"] for row in rows])) if rows else 0.0,
        "stage_reach_rate": {
            str(stage): sum(int(row["max_stage"]) >= stage for row in rows) / len(rows)
            if rows
            else 0.0
            for stage in range(1, 5)
        },
        "errors": sum(row["error"] is not None for row in rows),
        "constrained_action_steps": constrained_steps,
        "constrained_action_rate": constrained_steps / total_steps if total_steps else 0.0,
    }


def save_results(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    cases: list[dict[str, int | str | None]],
) -> None:
    payload = {
        "policy_label": args.policy_label,
        "evaluation_mode": args.evaluation_mode,
        "dataset": str(args.dataset) if args.dataset is not None else None,
        "task_profile": args.task_profile,
        "task_instruction": args.task_instruction,
        "case_ids": [str(case["case_id"]) for case in cases],
        "episodes": [int(case["episode"]) for case in cases if case["episode"] is not None],
        "seeds": [int(case["seed"]) for case in cases],
        "execution_horizon": int(args.execution_horizon),
        "max_steps": int(args.max_steps),
        "policy_seed": int(args.policy_seed),
        "server": f"{args.server_host}:{args.server_port}",
        "max_translation_step": float(args.max_translation_step),
        "max_rotation_step": float(args.max_rotation_step),
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)


def validate_args(args: argparse.Namespace) -> None:
    if args.execution_horizon <= 0 or args.max_steps <= 0:
        raise ValueError("Execution horizon and max steps must be positive.")
    if args.video_every <= 0 or args.video_fps <= 0:
        raise ValueError("Video cadence and FPS must be positive.")
    if args.max_translation_step <= 0 or args.max_rotation_step <= 0:
        raise ValueError("Action limits must be positive.")
    if args.seed_start < 0 or args.num_episodes <= 0:
        raise ValueError("Seed start must be non-negative and episode count must be positive.")
    if args.evaluation_mode == "heldout" and len(set(args.episodes)) != len(args.episodes):
        raise ValueError("Episode list contains duplicates.")
    if args.evaluation_mode == "heldout" and args.task_profile is not None:
        raise ValueError("Held-out task profiles come from the recorded episodes.")
    if args.task_instruction is not None and not args.task_instruction.strip():
        raise ValueError("Task instruction must not be empty.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.evaluation_mode == "heldout":
        args.dataset = (args.dataset or DEFAULT_DATASET).expanduser().resolve()
    elif args.dataset is not None:
        args.dataset = args.dataset.expanduser().resolve()
    if args.evaluation_mode == "reset":
        args.task_profile = reset_task_profile(args.dataset, args.task_profile)
    if args.output is None:
        args.output = (
            DEFAULT_RESET_OUTPUT if args.evaluation_mode == "reset" else DEFAULT_HELDOUT_OUTPUT
        )
    args.output = args.output.expanduser().resolve()
    cases = evaluation_cases(args)
    case_ids = [str(case["case_id"]) for case in cases]
    video_dir = (
        args.video_dir.expanduser().resolve()
        if args.video_dir is not None
        else args.output.parent / f"{args.output.stem}_videos"
    )

    rows_by_case: dict[str, dict[str, Any]] = {}
    if args.output.is_file() and not args.force:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        expected = {
            "policy_label": args.policy_label,
            "evaluation_mode": args.evaluation_mode,
            "dataset": str(args.dataset) if args.dataset is not None else None,
            "task_profile": args.task_profile,
            "task_instruction": args.task_instruction,
            "execution_horizon": int(args.execution_horizon),
            "policy_seed": int(args.policy_seed),
        }
        existing.setdefault("evaluation_mode", "heldout")
        existing.setdefault("task_profile", None)
        existing.setdefault("task_instruction", None)
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing result configuration differs: {mismatches}")
        existing_case_ids = existing.get(
            "case_ids",
            [f"episode_{int(episode):06d}" for episode in existing.get("episodes", [])],
        )
        if list(existing_case_ids) != case_ids:
            raise ValueError("Existing result cases differ from the requested evaluation cases.")
        rows_by_case = {
            (
                str(row["case_id"])
                if row.get("case_id") is not None
                else f"episode_{int(row['episode']):06d}"
            ): row
            for row in existing.get("rows", [])
        }

    pending = [case for case in cases if args.force or str(case["case_id"]) not in rows_by_case]
    if pending:
        with closing(PolicyClient(args.server_host, args.server_port)) as client:
            for case in pending:
                case_id = str(case["case_id"])
                rows_by_case[case_id] = run_case(args, client, case, video_dir)
                rows = [rows_by_case[key] for key in case_ids if key in rows_by_case]
                save_results(args, rows, cases)
    else:
        print("No pending episodes; existing result file is complete.", flush=True)

    rows = [rows_by_case[key] for key in case_ids if key in rows_by_case]
    save_results(args, rows, cases)
    summary = aggregate(rows)
    print("aggregate " + json.dumps(summary), flush=True)
    print(f"results={args.output}", flush=True)
    if summary["errors"]:
        raise RuntimeError(f"{summary['errors']} episode(s) failed; see the saved result rows.")


if __name__ == "__main__":
    main()
