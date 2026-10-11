"""Re-render the original VR demonstrations with a shorter wrist camera near plane."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import sys

import av
import h5py
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.maniskill import ManiSkillTaskFactory
from sim.maniskill.task_profiles import get_task_profile


SOURCE_PROFILE = "handover_block_ur5_v2_vr_robotwin_clean"
TARGET_PROFILE = "handover_block_ur5_v2_vr_robotwin_clean_near001"
WRIST_CAMERA_NEAR_M = 0.01
WRIST_CAMERAS = ("left_wrist_camera", "right_wrist_camera")
CAMERA_FIELDS = ("position_mm", "segmentation_id", "intrinsic_cv", "extrinsic_cv", "cam2world_gl")
DEFAULT_SOURCE = REPO_ROOT / "data/HandoverBlockUR5-v2/pico_handover_clean_v2"
DEFAULT_OUTPUT = REPO_ROOT / "data/HandoverBlockUR5-v2/pico_handover_clean_v2_near001"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-episodes", type=int, default=None)
    return parser


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _set_actor_pose(actor: object, pose: np.ndarray, device: torch.device) -> None:
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    actor.set_pose(
        Pose.create_from_pq(
            p=torch.as_tensor(pose[:3], device=device).reshape(1, 3),
            q=torch.as_tensor(pose[3:], device=device).reshape(1, 4),
        )
    )


def _restore_frame(raw_env: object, source: h5py.File, frame_index: int) -> None:
    for arm_index, side in ((0, "left"), (1, "right")):
        robot = raw_env.get_arm_robot(arm_index)
        qpos = torch.as_tensor(
            source[f"observations/robot/{side}_full_qpos"][frame_index],
            device=raw_env.device,
            dtype=robot.get_qpos().dtype,
        ).reshape(1, -1)
        qvel = torch.as_tensor(
            source[f"observations/robot/{side}_full_qvel"][frame_index],
            device=raw_env.device,
            dtype=robot.get_qvel().dtype,
        ).reshape(1, -1)
        robot.set_qpos(qpos)
        robot.set_qvel(qvel)

    for actor, key in ((raw_env.box, "box_pose"), (raw_env.target_box, "target_box_pose")):
        _set_actor_pose(actor, source[f"observations/task/{key}"][frame_index], raw_env.device)
    if raw_env.gpu_sim_enabled:
        raw_env.scene._gpu_apply_all()


def _open_video(
    path: Path, fps: int, width: int, height: int
) -> tuple[av.container.OutputContainer, av.video.stream.VideoStream]:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "veryfast"}
    return container, stream


def _render_episode(source_dir: Path, output_dir: Path) -> dict[str, object]:
    metadata = json.loads((source_dir / "episode.json").read_text(encoding="utf-8"))
    if metadata["task_profile"] != SOURCE_PROFILE:
        raise ValueError(f"Unexpected source profile in {source_dir}: {metadata['task_profile']}")

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_dir / "data.hdf5", output_dir / "data.hdf5")
    shutil.copy2(source_dir / "head_camera.mp4", output_dir / "head_camera.mp4")

    env = ManiSkillTaskFactory().create_env(
        task_name=TARGET_PROFILE,
        obs_mode="pointcloud+state_dict",
        control_mode=str(metadata["control_mode"]),
        sim_backend=str(metadata["sim_backend"]),
        render_backend=str(metadata["render_backend"]),
        sim_freq=int(metadata["sim_freq_hz"]),
        control_freq=int(metadata["control_freq_hz"]),
    )
    videos: dict[str, tuple[av.container.OutputContainer, av.video.stream.VideoStream]] = {}
    max_camera_pose_error = 0.0
    try:
        env.reset(seed=int(metadata["seed"]))
        raw_env = env.env.unwrapped
        with h5py.File(source_dir / "data.hdf5", "r") as source, h5py.File(
            output_dir / "data.hdf5", "r+"
        ) as output:
            observation_count = int(source["observations/robot/agent_qpos"].shape[0])
            action_count = int(source["actions/control/delta_pose_actual"].shape[0])
            if observation_count != action_count + 1:
                raise ValueError(f"Observation/action alignment is invalid in {source_dir}")

            for frame_index in range(observation_count):
                _restore_frame(raw_env, source, frame_index)
                observation = raw_env.get_obs()
                for camera_name in WRIST_CAMERAS:
                    camera_data = observation["sensor_data"][camera_name]
                    camera_params = observation["sensor_param"][camera_name]
                    rgb = _numpy(camera_data["rgb"])[0].astype(np.uint8, copy=False)
                    if camera_name not in videos:
                        videos[camera_name] = _open_video(
                            output_dir / f"{camera_name}.mp4",
                            int(metadata["control_freq_hz"]),
                            int(rgb.shape[1]),
                            int(rgb.shape[0]),
                        )
                    container, stream = videos[camera_name]
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    frame.pts = frame_index
                    frame.time_base = Fraction(1, int(metadata["control_freq_hz"]))
                    for packet in stream.encode(frame):
                        container.mux(packet)

                    prefix = f"observations/cameras/{camera_name}"
                    values = {
                        "position_mm": _numpy(camera_data["position"])[0],
                        "segmentation_id": _numpy(camera_data["segmentation"])[0, ..., 0],
                        **{key: _numpy(camera_params[key])[0] for key in CAMERA_FIELDS[2:]},
                    }
                    pose_error = float(
                        np.max(
                            np.abs(
                                values["cam2world_gl"]
                                - source[f"{prefix}/cam2world_gl"][frame_index]
                            )
                        )
                    )
                    max_camera_pose_error = max(max_camera_pose_error, pose_error)
                    if pose_error > 1.0e-4:
                        raise ValueError(
                            f"Camera pose differs from the recording at {source_dir.name} "
                            f"frame {frame_index}: {pose_error:.6g}"
                        )
                    for key, value in values.items():
                        target = output[f"{prefix}/{key}"]
                        if value.shape != target.shape[1:] or value.dtype != target.dtype:
                            raise ValueError(f"Unexpected {prefix}/{key} shape or dtype at frame {frame_index}")
                        target[frame_index] = value
            output.attrs["wrist_camera_near_m"] = WRIST_CAMERA_NEAR_M
    finally:
        for container, stream in videos.values():
            try:
                for packet in stream.encode(None):
                    container.mux(packet)
            finally:
                container.close()
        env.close()

    metadata["source_task_profile"] = metadata["task_profile"]
    metadata["task_profile"] = TARGET_PROFILE
    metadata["wrist_camera_near_m"] = WRIST_CAMERA_NEAR_M
    metadata["rerender_source"] = str(source_dir)
    (output_dir / "episode.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "episode": source_dir.name,
        "observations": observation_count,
        "actions": action_count,
        "max_camera_pose_error": max_camera_pose_error,
    }


def main() -> None:
    args = build_parser().parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Source and output must be separate dataset directories.")
    if args.num_episodes is not None and args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")
    episode_dirs = sorted((source_root / "episodes").glob("episode_*"))
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes under {source_root}")
    if args.num_episodes is not None:
        episode_dirs = episode_dirs[: args.num_episodes]
    target_near = get_task_profile(TARGET_PROFILE)["env_kwargs"]["wrist_camera_near"]
    if target_near != WRIST_CAMERA_NEAR_M:
        raise ValueError("Target profile has a different wrist camera near plane.")

    output_root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_dataset": str(source_root),
        "source_task_profile": SOURCE_PROFILE,
        "task_profile": TARGET_PROFILE,
        "wrist_camera_near_m": WRIST_CAMERA_NEAR_M,
        "rerendered_cameras": list(WRIST_CAMERAS),
    }
    manifest_path = output_root / "rerender_info.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != provenance:
            raise ValueError(f"Output directory has different provenance: {manifest_path}")
    else:
        if any(output_root.iterdir()):
            raise ValueError(f"Output directory already contains unrelated data: {output_root}")
        manifest_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(source_root / "dataset_info.json", output_root / "dataset_info.json")

    output_episodes = output_root / "episodes"
    output_episodes.mkdir(exist_ok=True)
    for index, source_dir in enumerate(episode_dirs, start=1):
        destination = output_episodes / source_dir.name
        if destination.exists():
            metadata = json.loads((destination / "episode.json").read_text(encoding="utf-8"))
            if metadata.get("task_profile") != TARGET_PROFILE:
                raise ValueError(f"Existing episode has another profile: {destination}")
            print(f"[{index}/{len(episode_dirs)}] skip {source_dir.name}", flush=True)
            continue
        partial = output_episodes / f".{source_dir.name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        report = _render_episode(source_dir, partial)
        partial.rename(destination)
        print(f"[{index}/{len(episode_dirs)}] {report}", flush=True)
    print(f"Raw dataset ready: {output_root}", flush=True)


if __name__ == "__main__":
    main()
