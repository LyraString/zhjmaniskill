"""Compare one-step and four-step execution using an existing VR checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import dill
import h5py
import numpy as np
import torch
import zarr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FM_ROOT = REPO_ROOT / "policy" / "fm_diffusion"
if str(FM_ROOT) not in sys.path:
    sys.path.insert(0, str(FM_ROOT))

from core.schemas.action import ActionSpec
from diffusion_policy.common.sampler import get_val_mask
from policy.FlowMatching import deploy_maniskill as adapter
from script.check_vr_teleop_dataset import _restore_initial_state, _to_numpy
from script.eval_maniskill_policy import (
    _actual_eef_pose_root_wxyz,
    _validate_action_in_space,
)
from sim.maniskill import ManiSkillTaskFactory, convert_maniskill_obs


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "policy/FlowMatching/checkpoints/260913-145757-vr-ep500-seed114514/500.ckpt"
)
DEFAULT_ZARR = (
    REPO_ROOT / "policy/FlowMatching/data/HandoverBlockUR5-pico-clean-v2-80.zarr"
)
DEFAULT_DATASET = REPO_ROOT / "data/HandoverBlockUR5-v2/pico_handover_clean_v2"
DEFAULT_OUTPUT = REPO_ROOT / "tmp/handover/vr_execution_horizon/results.json"
PREDICTED_ACTIONS = 4


class RecordedFrame:
    """Expose one HDF5 frame to the existing state-restoration helper."""

    def __init__(self, h5_file: h5py.File, index: int):
        self.h5_file = h5_file
        self.index = int(index)

    def __getitem__(self, key: str):
        return self.h5_file[key][self.index : self.index + 1]


def episode_paths(dataset: Path, episode: int) -> tuple[Path, Path]:
    folder = dataset / "episodes" / f"episode_{episode:06d}"
    return folder / "episode.json", folder / "data.hdf5"


def load_checkpoint_cfg(checkpoint: Path):
    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu")
    return payload["cfg"]


def heldout_episodes(cfg, episode_count: int) -> list[int]:
    dataset_cfg = cfg.task.dataset
    mask = get_val_mask(
        n_episodes=episode_count,
        val_ratio=float(dataset_cfg.val_ratio),
        seed=int(dataset_cfg.seed),
    )
    return np.flatnonzero(mask).astype(int).tolist()


def episode_bounds(zarr_data, episode: int) -> tuple[int, int, int]:
    end = int(zarr_data["meta/episode_ends"][episode])
    begin = int(zarr_data["meta/episode_ends"][episode - 1]) if episode else 0
    trim = int(zarr_data["meta/episode_trim_starts"][episode])
    return begin, end, trim


def encode_live_observation(env):
    raw_obs = env.env.unwrapped.get_obs()
    observation = convert_maniskill_obs(
        raw_obs,
        metadata={
            "env_id": env.env_id,
            "point_cloud_views": env.point_cloud_views,
            "point_cloud_sampling": env.point_cloud_sampling,
        },
    )
    return adapter.encode_obs(dict(observation.data))


def quaternion_error_deg(actual_wxyz: np.ndarray, expert_wxyz: np.ndarray) -> np.ndarray:
    actual = actual_wxyz / np.linalg.norm(actual_wxyz, axis=-1, keepdims=True)
    expert = expert_wxyz / np.linalg.norm(expert_wxyz, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(actual * expert, axis=-1)), 0.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def pose_errors(actual: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    position_mm = np.linalg.norm(actual[:, :3] - expert[:, :3], axis=-1) * 1000.0
    rotation_deg = quaternion_error_deg(actual[:, 3:], expert[:, 3:])
    return {
        "left_eef_position_error_mm": float(position_mm[0]),
        "right_eef_position_error_mm": float(position_mm[1]),
        "left_eef_rotation_error_deg": float(rotation_deg[0]),
        "right_eef_rotation_error_deg": float(rotation_deg[1]),
    }


def create_env(metadata: dict, args):
    return ManiSkillTaskFactory().create_env(
        task_name=metadata["task_profile"],
        obs_mode="pointcloud+state_dict",
        control_mode=metadata["control_mode"],
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        sim_freq=int(metadata["sim_freq_hz"]),
        control_freq=int(metadata["control_freq_hz"]),
        max_episode_steps=args.max_steps,
    )


def run_episode(args, model, zarr_data, episode: int, execution_horizon: int, split: str):
    metadata_path, h5_path = episode_paths(args.dataset, episode)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    begin, end, trim = episode_bounds(zarr_data, episode)
    recorded_length = end - begin

    with h5py.File(h5_path, "r") as h5_file:
        stages = np.asarray(
            h5_file["transitions/info/reward_stage"][trim : trim + recorded_length],
            dtype=np.int64,
        )
        transfer_indices = np.flatnonzero(stages >= 3)
        if transfer_indices.size == 0:
            raise RuntimeError(f"Episode {episode} has no recorded handover completion stage.")
        transfer_frame = int(transfer_indices[0])
        measure_frame = max(0, transfer_frame - int(args.pre_handover_steps))
        expert_box_pose = np.asarray(
            h5_file["observations/task/box_pose"][trim + measure_frame],
            dtype=np.float32,
        )
        expert_eef_pose = np.asarray(
            h5_file["observations/robot/ee_pose_root_wxyz"][trim + measure_frame],
            dtype=np.float32,
        )

        env = create_env(metadata, args)
        result = {
            "episode": int(episode),
            "split": split,
            "seed": int(metadata["seed"]),
            "execution_horizon": int(execution_horizon),
            "predicted_actions": PREDICTED_ACTIONS,
            "measure_frame": measure_frame,
            "transfer_frame": transfer_frame,
            "success": False,
            "max_stage": 0,
            "steps": 0,
            "error": None,
            "pre_handover": None,
        }
        try:
            env.reset(seed=int(metadata["seed"]))
            raw_env = env.env.unwrapped
            _restore_initial_state(raw_env, RecordedFrame(h5_file, trim))
            raw_env._get_reward_stage()[:] = (
                int(h5_file["transitions/info/reward_stage"][trim - 1]) if trim else 0
            )

            adapter.reset_model(model)
            model.update_obs(encode_live_observation(env))
            action_queue: list[np.ndarray] = []

            for step in range(int(args.max_steps)):
                if step == measure_frame:
                    box_pose = _to_numpy(raw_env.box.pose.raw_pose).reshape(7)
                    eef_pose = _actual_eef_pose_root_wxyz(env)
                    result["pre_handover"] = {
                        "block_position_error_mm": float(
                            np.linalg.norm(box_pose[:3] - expert_box_pose[:3]) * 1000.0
                        ),
                        **pose_errors(eef_pose, expert_eef_pose),
                    }

                if not action_queue:
                    # Common replanning times use the same diffusion seed across H=1/H=4.
                    torch.manual_seed(int(args.policy_seed) + 100_000 * episode + step)
                    predicted = np.asarray(model.get_action(), dtype=np.float32)
                    if predicted.shape != (PREDICTED_ACTIONS, 14):
                        raise RuntimeError(
                            f"Expected model output {(PREDICTED_ACTIONS, 14)}, got {predicted.shape}."
                        )
                    absolute_chunk = np.asarray(env.prepare_action_chunk(predicted))
                    action_queue = list(absolute_chunk[:execution_horizon])

                target = _validate_action_in_space(action_queue.pop(0), env.action_space)
                transition = env.step(
                    ActionSpec.from_value(target, action_type=metadata["control_mode"])
                )
                model.update_obs(adapter.encode_obs(dict(transition.observation.data)))
                stage = int(_to_numpy(transition.info["reward_stage"]).reshape(-1)[0])
                success = bool(_to_numpy(transition.info["success"]).reshape(-1)[0])
                result["max_stage"] = max(int(result["max_stage"]), stage)
                result["success"] = bool(result["success"] or success)
                result["steps"] = step + 1
                if result["success"] or transition.terminated or transition.truncated:
                    break
        except Exception as error:
            result["error"] = repr(error)
        finally:
            env.close()

    print("episode_result " + json.dumps(result), flush=True)
    return result


def aggregate(rows: list[dict]) -> dict:
    output = {}
    scopes = {"all": rows}
    for split in ("heldout", "train"):
        selected = [row for row in rows if row["split"] == split]
        if selected:
            scopes[split] = selected

    for scope, scoped_rows in scopes.items():
        output[scope] = {}
        for horizon in (4, 1):
            selected = [
                row for row in scoped_rows if int(row["execution_horizon"]) == horizon
            ]
            if not selected:
                continue
            diagnostics = [
                row["pre_handover"]
                for row in selected
                if row["pre_handover"] is not None
            ]
            metric_names = tuple(diagnostics[0]) if diagnostics else ()
            output[scope][str(horizon)] = {
                "episodes": len(selected),
                "successes": int(sum(bool(row["success"]) for row in selected)),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "max_stage_mean": float(np.mean([row["max_stage"] for row in selected])),
                "errors": int(sum(row["error"] is not None for row in selected)),
                "pre_handover_mean": {
                    name: float(np.mean([item[name] for item in diagnostics]))
                    for name in metric_names
                },
                "pre_handover_median": {
                    name: float(np.median([item[name] for item in diagnostics]))
                    for name in metric_names
                },
            }
    return output


def save_results(args, rows: list[dict], heldout: list[int]) -> None:
    payload = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "zarr": str(args.zarr),
        "heldout_episodes": heldout,
        "measurement": (
            f"State error at {args.pre_handover_steps} control steps before the "
            "recorded expert transfer stage."
        ),
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", nargs="+", type=int, default=None)
    parser.add_argument("--execution-horizons", nargs="+", type=int, choices=(1, 4), default=(4, 1))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--pre-handover-steps", type=int, default=10)
    parser.add_argument("--policy-seed", type=int, default=999)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-split", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_checkpoint_cfg(args.checkpoint)
    zarr_data = zarr.open(str(args.zarr), mode="r")
    episode_count = len(zarr_data["meta/episode_ends"])
    heldout = heldout_episodes(cfg, episode_count)
    episodes = heldout if args.episodes is None else list(dict.fromkeys(args.episodes))
    if any(episode < 0 or episode >= episode_count for episode in episodes):
        raise ValueError(f"Episode ids must be in [0, {episode_count - 1}].")
    print(f"heldout_episodes={heldout}", flush=True)
    print(f"selected_episodes={episodes}", flush=True)
    if args.list_split:
        return

    existing_rows = []
    if args.output.is_file() and not args.force:
        existing_rows = json.loads(args.output.read_text(encoding="utf-8")).get("rows", [])
    rows_by_key = {
        (int(row["episode"]), int(row["execution_horizon"])): row
        for row in existing_rows
    }

    pending = [
        (episode, horizon)
        for horizon in args.execution_horizons
        for episode in episodes
        if args.force or (episode, horizon) not in rows_by_key
    ]
    if pending:
        model = adapter.get_model(
            {"ckpt_path": str(args.checkpoint), "n_action_steps": PREDICTED_ACTIONS}
        )
        for episode, horizon in pending:
            split = "heldout" if episode in heldout else "train"
            row = run_episode(args, model, zarr_data, episode, horizon, split)
            rows_by_key[(episode, horizon)] = row
            save_results(args, list(rows_by_key.values()), heldout)
    else:
        print("No pending episode/horizon pairs; existing output is complete.", flush=True)

    rows = list(rows_by_key.values())
    save_results(args, rows, heldout)
    print("aggregate " + json.dumps(aggregate(rows)), flush=True)
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
