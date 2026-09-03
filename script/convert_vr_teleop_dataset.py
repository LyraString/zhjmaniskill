"""Convert canonical VR episodes to the repo's multiview delta-pose NPZ format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill import (
    ManiSkillTrajectoryRecorder,
    convert_maniskill_obs,
    get_task_profile,
)
from sim.maniskill.teleop.canonical_dataset import DEFAULT_CAMERA_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert canonical HDF5/MP4 VR data into training NPZ episodes."
    )
    parser.add_argument("dataset_dir", help="Canonical dataset root")
    parser.add_argument("output_name", help="Output directory under data/<task-name>/")
    parser.add_argument("--task-name", default="HandoverBlockUR5-v2")
    parser.add_argument(
        "--task-profile",
        default="handover_block_ur5_v2_vr_robotwin_clean",
    )
    parser.add_argument("--point-cloud-sampling", choices=("fast", "fps"), default="fast")
    parser.add_argument(
        "--output-format",
        choices=("npz", "zarr"),
        default="npz",
        help="Write legacy per-episode NPZ files or one training-ready Zarr dataset.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit output path. Primarily useful with --output-format zarr.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Convert only the first N episodes. Default converts every completed episode.",
    )
    parser.add_argument("--max-translation-step-m", type=float, default=0.015)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.10)
    parser.add_argument(
        "--trim-idle-prefix",
        action="store_true",
        help=(
            "Drop the initial left-arm hold segment before its first meaningful "
            "translation or rotation command."
        ),
    )
    parser.add_argument(
        "--trim-translation-threshold-m",
        type=float,
        default=0.005,
        help="Left-arm translation threshold used by --trim-idle-prefix.",
    )
    parser.add_argument(
        "--trim-rotation-threshold-rad",
        type=float,
        default=0.02,
        help="Left-arm rotation threshold used by --trim-idle-prefix.",
    )
    return parser


def _completed_episode_dirs(
    dataset_dir: Path,
    *,
    num_episodes: int | None,
) -> list[Path]:
    episodes = sorted((dataset_dir / "episodes").glob("episode_*"))
    episodes = [path for path in episodes if path.is_dir()]
    if not episodes:
        raise FileNotFoundError(f"No canonical episodes found under {dataset_dir}")
    if num_episodes is not None:
        if int(num_episodes) <= 0:
            raise ValueError(f"num_episodes must be positive, got {num_episodes}.")
        if int(num_episodes) > len(episodes):
            raise ValueError(
                f"Requested {num_episodes} episodes, but only {len(episodes)} are available."
            )
        episodes = episodes[: int(num_episodes)]
    return episodes


def _read_actions_and_validate_alignment(
    h5_file: h5py.File,
    *,
    episode_name: str,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> np.ndarray:
    actions = np.asarray(
        h5_file["actions/control/delta_pose_actual"],
        dtype=np.float32,
    )
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(
            f"{episode_name}: expected action shape [T,14], got {actions.shape}."
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{episode_name}: actions contain NaN or Inf.")
    for offset in (0, 7):
        translation_norm = np.linalg.norm(actions[:, offset : offset + 3], axis=1)
        rotation_norm = np.linalg.norm(actions[:, offset + 3 : offset + 6], axis=1)
        if np.any(translation_norm > float(max_translation_step_m) + 1.0e-6) or np.any(
            rotation_norm > float(max_rotation_step_rad) + 1.0e-6
        ):
            raise ValueError(
                f"{episode_name}: delta action exceeds controller radial limits."
            )

    observation_count = int(
        h5_file.attrs.get(
            "observation_count",
            h5_file["observations/robot/agent_qpos"].shape[0],
        )
    )
    if observation_count != actions.shape[0] + 1:
        raise ValueError(
            f"{episode_name}: expected observations=actions+1, got "
            f"{observation_count} and {actions.shape[0]}."
        )
    return actions


def _idle_prefix_length(
    actions: np.ndarray,
    *,
    enabled: bool,
    translation_threshold_m: float,
    rotation_threshold_rad: float,
) -> int:
    if not enabled:
        return 0
    if translation_threshold_m <= 0.0 or rotation_threshold_rad <= 0.0:
        raise ValueError("Idle-prefix trim thresholds must be positive.")

    left_translation = np.linalg.norm(actions[:, :3], axis=1)
    left_rotation = np.linalg.norm(actions[:, 3:6], axis=1)
    active = np.logical_or(
        left_translation > float(translation_threshold_m),
        left_rotation > float(rotation_threshold_rad),
    )
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        raise ValueError(
            "Cannot trim idle prefix because the left arm never exceeds the "
            "configured motion thresholds."
        )
    return int(active_indices[0])


def _raw_obs_from_h5(
    h5_file: h5py.File,
    observation_index: int,
) -> dict[str, Any]:
    sensor_data: dict[str, dict[str, np.ndarray]] = {}
    sensor_param: dict[str, dict[str, np.ndarray]] = {}
    for camera_name in DEFAULT_CAMERA_NAMES:
        prefix = f"observations/cameras/{camera_name}"
        position = np.asarray(h5_file[f"{prefix}/position_mm"][observation_index])
        segmentation = np.asarray(
            h5_file[f"{prefix}/segmentation_id"][observation_index]
        )
        sensor_data[camera_name] = {
            "position": position[None, ...],
            "segmentation": segmentation[None, ..., None],
        }
        sensor_param[camera_name] = {
            key: np.asarray(h5_file[f"{prefix}/{key}"][observation_index])[None, ...]
            for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")
        }
    agent_qpos = np.asarray(
        h5_file["observations/robot/agent_qpos"][observation_index],
        dtype=np.float32,
    )
    return {
        "sensor_data": sensor_data,
        "sensor_param": sensor_param,
        "agent": {"qpos": agent_qpos[None, ...]},
    }


def _learner_observation(
    h5_file: h5py.File,
    observation_index: int,
    *,
    task_name: str,
    point_cloud_views: Any,
    point_cloud_sampling: str,
) -> dict[str, np.ndarray]:
    observation = convert_maniskill_obs(
        _raw_obs_from_h5(h5_file, observation_index),
        metadata={
            "env_id": task_name,
            "point_cloud_views": point_cloud_views,
            "point_cloud_sampling": point_cloud_sampling,
        },
    )
    learner_data = dict(observation.data)
    for view in point_cloud_views:
        key = str(view["key"])
        expected_shape = (int(view["points"]), 3)
        if key not in learner_data:
            learner_data[key] = np.zeros(expected_shape, dtype=np.float32)
        elif np.asarray(learner_data[key]).shape != expected_shape:
            raise ValueError(
                f"Observation {observation_index} view `{key}` has shape "
                f"{np.asarray(learner_data[key]).shape}, expected {expected_shape}."
            )
    if np.asarray(learner_data.get("agent_pos")).shape != (14,):
        raise ValueError(
            f"Observation {observation_index} agent_pos has shape "
            f"{np.asarray(learner_data.get('agent_pos')).shape}, expected (14,)."
        )
    return learner_data


def convert_dataset(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    episodes = _completed_episode_dirs(
        dataset_dir,
        num_episodes=args.num_episodes,
    )

    profile = get_task_profile(args.task_profile)
    point_cloud_views = profile.get("point_cloud_views")
    if not point_cloud_views:
        raise ValueError(f"Task profile `{args.task_profile}` is not multiview.")

    output_dir = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else REPO_ROOT / "data" / args.task_name / args.output_name
    )
    existing_npz = list((output_dir / "data").glob("episode*.npz"))
    if existing_npz:
        raise FileExistsError(f"Output already contains episodes: {output_dir}")
    recorder = ManiSkillTrajectoryRecorder(output_dir)

    for output_index, episode_dir in enumerate(episodes):
        h5_path = episode_dir / "data.hdf5"
        with h5py.File(h5_path, "r") as h5_file:
            actions = _read_actions_and_validate_alignment(
                h5_file,
                episode_name=episode_dir.name,
                max_translation_step_m=args.max_translation_step_m,
                max_rotation_step_rad=args.max_rotation_step_rad,
            )
            trim_start = _idle_prefix_length(
                actions,
                enabled=args.trim_idle_prefix,
                translation_threshold_m=args.trim_translation_threshold_m,
                rotation_threshold_rad=args.trim_rotation_threshold_rad,
            )
            actions = actions[trim_start:]
            initial_obs = _learner_observation(
                h5_file,
                trim_start,
                task_name=args.task_name,
                point_cloud_views=point_cloud_views,
                point_cloud_sampling=args.point_cloud_sampling,
            )
            recorder.start_episode(initial_obs)
            for action_index, action in enumerate(actions):
                next_obs = _learner_observation(
                    h5_file,
                    trim_start + action_index + 1,
                    task_name=args.task_name,
                    point_cloud_views=point_cloud_views,
                    point_cloud_sampling=args.point_cloud_sampling,
                )
                recorder.append_step(action, next_obs)
            output_path = recorder.save_episode(output_index)

        source_metadata_path = episode_dir / "episode.json"
        source_metadata = (
            json.loads(source_metadata_path.read_text(encoding="utf-8"))
            if source_metadata_path.exists()
            else {}
        )
        metadata = {
            **source_metadata,
            "source_canonical_episode": str(episode_dir),
            "converted_task_profile": args.task_profile,
            "point_cloud_sampling": args.point_cloud_sampling,
            "action_semantics": "delta_pose_actual",
            "idle_prefix_trimmed_steps": trim_start,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"converted {episode_dir.name} -> {output_path}")

    return output_dir


def _default_zarr_output(args: argparse.Namespace, episode_count: int) -> Path:
    if args.output_path is not None:
        return Path(args.output_path).expanduser().resolve()
    output_name = f"{args.task_name}-{args.output_name}-{episode_count}.zarr"
    return REPO_ROOT / "policy" / "FlowMatching" / "data" / output_name


def convert_dataset_to_zarr(args: argparse.Namespace) -> Path:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    episodes = _completed_episode_dirs(
        dataset_dir,
        num_episodes=args.num_episodes,
    )
    profile = get_task_profile(args.task_profile)
    point_cloud_views = profile.get("point_cloud_views")
    if not point_cloud_views:
        raise ValueError(f"Task profile `{args.task_profile}` is not multiview.")

    action_counts: list[int] = []
    trim_starts: list[int] = []
    episode_seeds: list[int] = []
    source_control_freq_hz: int | None = None
    source_sim_freq_hz: int | None = None
    for episode_dir in episodes:
        with h5py.File(episode_dir / "data.hdf5", "r") as h5_file:
            actions = _read_actions_and_validate_alignment(
                h5_file,
                episode_name=episode_dir.name,
                max_translation_step_m=args.max_translation_step_m,
                max_rotation_step_rad=args.max_rotation_step_rad,
            )
            trim_start = _idle_prefix_length(
                actions,
                enabled=args.trim_idle_prefix,
                translation_threshold_m=args.trim_translation_threshold_m,
                rotation_threshold_rad=args.trim_rotation_threshold_rad,
            )
            trim_starts.append(trim_start)
            action_counts.append(int(actions.shape[0] - trim_start))
        metadata_path = episode_dir / "episode.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        episode_seeds.append(int(metadata["seed"]))
        recorded_profile = str(metadata.get("task_profile", ""))
        if recorded_profile != str(args.task_profile):
            raise ValueError(
                f"{episode_dir.name}: recorded task_profile={recorded_profile!r}, "
                f"expected {args.task_profile!r}."
            )
        action_semantics = str(metadata.get("action_semantics", ""))
        if action_semantics != "actual_relative_delta_pose_euler_xyz":
            raise ValueError(
                f"{episode_dir.name}: unsupported action_semantics={action_semantics!r}."
            )
        control_freq_hz = int(metadata["control_freq_hz"])
        sim_freq_hz = int(metadata["sim_freq_hz"])
        if source_control_freq_hz is None:
            source_control_freq_hz = control_freq_hz
            source_sim_freq_hz = sim_freq_hz
        elif (
            control_freq_hz != source_control_freq_hz
            or sim_freq_hz != source_sim_freq_hz
        ):
            raise ValueError(
                f"{episode_dir.name}: frequency mismatch; got sim/control="
                f"{sim_freq_hz}/{control_freq_hz}, expected "
                f"{source_sim_freq_hz}/{source_control_freq_hz}."
            )

    profile_control_freq = int(profile.get("control_freq", source_control_freq_hz))
    profile_sim_freq = int(profile.get("sim_freq", source_sim_freq_hz))
    if (
        source_control_freq_hz != profile_control_freq
        or source_sim_freq_hz != profile_sim_freq
    ):
        raise ValueError(
            "Canonical data frequency does not match the target task profile: "
            f"data={source_sim_freq_hz}/{source_control_freq_hz} Hz, "
            f"profile={profile_sim_freq}/{profile_control_freq} Hz."
        )

    output_path = _default_zarr_output(args, len(episodes))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Zarr dataset: {output_path}")
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if partial_path.exists():
        raise FileExistsError(
            f"A partial conversion already exists: {partial_path}. "
            "Remove it only after confirming no conversion is still running."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = int(sum(action_counts))
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    root = zarr.group(str(partial_path))
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")
    point_cloud_shapes = {
        str(view["key"]): (int(view["points"]), 3) for view in point_cloud_views
    }
    zarr_arrays = {
        key: data_group.create_dataset(
            key,
            shape=(total_steps, *shape),
            chunks=(100, *shape),
            dtype="float32",
            compressor=compressor,
        )
        for key, shape in point_cloud_shapes.items()
    }
    zarr_arrays["state"] = data_group.create_dataset(
        "state",
        shape=(total_steps, 14),
        chunks=(100, 14),
        dtype="float32",
        compressor=compressor,
    )
    zarr_arrays["action"] = data_group.create_dataset(
        "action",
        shape=(total_steps, 14),
        chunks=(100, 14),
        dtype="float32",
        compressor=compressor,
    )
    episode_ends = np.cumsum(action_counts, dtype=np.int64)
    meta_group.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype="int64",
        compressor=compressor,
    )
    meta_group.create_dataset(
        "episode_seeds",
        data=np.asarray(episode_seeds, dtype=np.int64),
        dtype="int64",
        compressor=compressor,
    )
    meta_group.create_dataset(
        "episode_trim_starts",
        data=np.asarray(trim_starts, dtype=np.int64),
        dtype="int64",
        compressor=compressor,
    )
    root.attrs.update(
        {
            "source_dataset": str(dataset_dir),
            "task_name": str(args.task_name),
            "task_profile": str(args.task_profile),
            "episode_count": len(episodes),
            "sim_frequency_hz": source_sim_freq_hz,
            "control_frequency_hz": source_control_freq_hz,
            "point_cloud_sampling": str(args.point_cloud_sampling),
            "action_semantics": "delta_pose_actual",
            "max_translation_step_m": float(args.max_translation_step_m),
            "max_rotation_step_rad": float(args.max_rotation_step_rad),
            "translation_units": "m",
            "rotation_units": "rad",
            "rotation_representation": "euler_xyz",
            "idle_prefix_trimmed": bool(args.trim_idle_prefix),
            "trim_translation_threshold_m": float(
                args.trim_translation_threshold_m
            ),
            "trim_rotation_threshold_rad": float(args.trim_rotation_threshold_rad),
        }
    )

    write_start = 0
    for episode_index, (episode_dir, action_count, trim_start) in enumerate(
        zip(episodes, action_counts, trim_starts, strict=True)
    ):
        with h5py.File(episode_dir / "data.hdf5", "r") as h5_file:
            actions = _read_actions_and_validate_alignment(
                h5_file,
                episode_name=episode_dir.name,
                max_translation_step_m=args.max_translation_step_m,
                max_rotation_step_rad=args.max_rotation_step_rad,
            )
            actions = actions[trim_start:]
            episode_arrays = {
                key: np.empty((action_count, *shape), dtype=np.float32)
                for key, shape in point_cloud_shapes.items()
            }
            episode_arrays["state"] = np.empty((action_count, 14), dtype=np.float32)
            for action_index in range(action_count):
                observation = _learner_observation(
                    h5_file,
                    trim_start + action_index,
                    task_name=args.task_name,
                    point_cloud_views=point_cloud_views,
                    point_cloud_sampling=args.point_cloud_sampling,
                )
                for key in point_cloud_shapes:
                    episode_arrays[key][action_index] = observation[key]
                episode_arrays["state"][action_index] = observation["agent_pos"]

        write_end = write_start + action_count
        for key, value in episode_arrays.items():
            zarr_arrays[key][write_start:write_end] = value
        zarr_arrays["action"][write_start:write_end] = actions
        write_start = write_end
        print(
            f"converted {episode_index + 1}/{len(episodes)} "
            f"{episode_dir.name} trim={trim_start} steps={action_count} "
            f"total={write_start}/{total_steps}"
        )

    if write_start != total_steps:
        raise RuntimeError(f"Expected {total_steps} written steps, got {write_start}.")
    partial_path.replace(output_path)
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = (
        convert_dataset_to_zarr(args)
        if args.output_format == "zarr"
        else convert_dataset(args)
    )
    print(f"converted dataset: {output_path}")


if __name__ == "__main__":
    main()
