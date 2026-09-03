"""Validate and replay canonical VR teleoperation episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import av
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory
from sim.maniskill.teleop.canonical_dataset import DEFAULT_CAMERA_NAMES


EXPECTED_CONTROL_MODE = "bimanual_pd_ee_delta_pose"
FINAL_DIAGNOSTIC_KEYS = (
    "reward_stage",
    "reward_target_xy_dist",
    "reward_target_z_dist",
    "spatial_success",
    "right_gripper_open",
    "right_ee_clear",
    "box_static",
    "success",
)
REQUIRED_DATASETS = (
    "actions/control/absolute_ee_target",
    "actions/control/delta_pose_actual",
    "actions/host_monotonic_ns",
    "observations/robot/agent_qpos",
    "observations/robot/left_full_qpos",
    "observations/robot/left_full_qvel",
    "observations/robot/right_full_qpos",
    "observations/robot/right_full_qvel",
    "observations/task/box_pose",
    "observations/task/target_box_pose",
    "transitions/success",
    "transitions/terminated",
    "transitions/truncated",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate canonical VR episodes and optionally replay actions."
    )
    parser.add_argument("dataset_dir", help="Canonical VR dataset root")
    parser.add_argument(
        "--replay-episodes",
        type=int,
        nargs="*",
        default=(),
        help="Episode indices to replay after validation",
    )
    parser.add_argument("--save-video-dir", default=None)
    parser.add_argument("--video-every", type=int, default=2)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Optional final open-gripper hold steps after the recorded horizon",
    )
    parser.add_argument("--sim-backend", default=None)
    parser.add_argument("--render-backend", default="sapien_cuda")
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_bool(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _scalar_value(value: Any) -> bool | int | float:
    scalar = _to_numpy(value).reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def _episode_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in (dataset_dir / "episodes").glob("episode_*")
        if path.is_dir() and (path / "episode.json").is_file()
    )


def _video_frame_count(path: Path) -> tuple[int, int, int, float]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return (
            int(stream.frames),
            int(stream.width),
            int(stream.height),
            float(stream.average_rate),
        )


def _finite(dataset: h5py.Dataset) -> bool:
    return bool(np.isfinite(np.asarray(dataset)).all())


def validate_episode(episode_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    h5_path = episode_dir / "data.hdf5"
    if not h5_path.is_file():
        return {"episode": episode_dir.name, "issues": ["missing data.hdf5"]}

    with h5py.File(h5_path, "r") as h5_file:
        missing = [name for name in REQUIRED_DATASETS if name not in h5_file]
        if missing:
            return {
                "episode": episode_dir.name,
                "seed": metadata.get("seed"),
                "issues": [f"missing datasets: {missing}"],
            }

        actions = np.asarray(
            h5_file["actions/control/delta_pose_actual"], dtype=np.float32
        )
        absolute_targets = np.asarray(
            h5_file["actions/control/absolute_ee_target"], dtype=np.float32
        )
        action_count = int(h5_file.attrs.get("action_count", actions.shape[0]))
        observation_count = int(
            h5_file.attrs.get(
                "observation_count",
                h5_file["observations/robot/agent_qpos"].shape[0],
            )
        )
        if actions.shape != (action_count, 14):
            issues.append(f"delta action shape={actions.shape}")
        if absolute_targets.shape != (action_count, 14):
            issues.append(f"absolute action shape={absolute_targets.shape}")
        if observation_count != action_count + 1:
            issues.append(
                f"observation/action mismatch={observation_count}/{action_count}"
            )
        if not np.isfinite(actions).all() or not np.isfinite(absolute_targets).all():
            issues.append("action contains NaN/Inf")

        translation_max = 0.0
        rotation_max = 0.0
        for offset in (0, 7):
            translation_max = max(
                translation_max,
                float(np.linalg.norm(actions[:, offset : offset + 3], axis=1).max()),
            )
            rotation_max = max(
                rotation_max,
                float(
                    np.linalg.norm(actions[:, offset + 3 : offset + 6], axis=1).max()
                ),
            )
        if translation_max > 0.015001:
            issues.append(f"translation delta exceeds limit: {translation_max:.6f} m")
        if rotation_max > 0.100001:
            issues.append(f"rotation delta exceeds limit: {rotation_max:.6f} rad")
        grippers = actions[:, (6, 13)]
        if np.any(grippers < 0.0) or np.any(grippers > 1.0):
            issues.append("gripper command outside [0,1]")

        for dataset_name in (
            "observations/robot/agent_qpos",
            "observations/robot/ee_pose_root_wxyz",
            "observations/task/box_pose",
            "observations/task/target_box_pose",
        ):
            if dataset_name not in h5_file or not _finite(h5_file[dataset_name]):
                issues.append(f"non-finite or missing {dataset_name}")

        action_timestamps = np.asarray(
            h5_file["actions/host_monotonic_ns"], dtype=np.int64
        )
        timestamp_deltas = np.diff(action_timestamps).astype(np.float64) * 1e-9
        if timestamp_deltas.size and np.any(timestamp_deltas <= 0.0):
            issues.append("action host timestamps are not strictly increasing")

        h5_success = np.asarray(h5_file["transitions/success"], dtype=np.bool_)
        terminated = np.asarray(h5_file["transitions/terminated"], dtype=np.bool_)
        truncated = np.asarray(h5_file["transitions/truncated"], dtype=np.bool_)
        if not bool(metadata.get("success", False)):
            issues.append("episode metadata is not successful")
        if h5_success.size == 0 or not bool(h5_success[-1]):
            issues.append("final transition is not successful")
        if terminated.size == 0 or not bool(terminated[-1]) or bool(truncated[-1]):
            issues.append("final transition is not clean termination")

        video_counts = {}
        for camera_name in DEFAULT_CAMERA_NAMES:
            prefix = f"observations/cameras/{camera_name}"
            position_name = f"{prefix}/position_mm"
            segmentation_name = f"{prefix}/segmentation_id"
            if position_name not in h5_file or segmentation_name not in h5_file:
                issues.append(f"missing geometry for {camera_name}")
                continue
            position = h5_file[position_name]
            segmentation = h5_file[segmentation_name]
            if position.shape != (observation_count, 240, 320, 3):
                issues.append(f"unexpected {camera_name} position shape={position.shape}")
            if segmentation.shape != (observation_count, 240, 320):
                issues.append(
                    f"unexpected {camera_name} segmentation shape={segmentation.shape}"
                )
            sample_indices = sorted({0, observation_count // 2, observation_count - 1})
            if not any(np.any(np.asarray(position[index])) for index in sample_indices):
                issues.append(f"sampled {camera_name} geometry is all zero")

            video_path = episode_dir / f"{camera_name}.mp4"
            if not video_path.is_file():
                issues.append(f"missing {camera_name}.mp4")
                continue
            frame_count, width, height, fps = _video_frame_count(video_path)
            video_counts[camera_name] = frame_count
            if frame_count != observation_count:
                issues.append(
                    f"{camera_name} frames={frame_count}, observations={observation_count}"
                )
            if (width, height) != (320, 240) or not np.isclose(fps, 20.0):
                issues.append(
                    f"{camera_name} video={width}x{height}@{fps:g} instead of 320x240@20"
                )

    return {
        "episode": episode_dir.name,
        "seed": int(metadata["seed"]),
        "actions": action_count,
        "duration_s": action_count / float(metadata.get("control_freq_hz", 20)),
        "translation_max_m": translation_max,
        "rotation_max_rad": rotation_max,
        "median_dt_s": (
            float(np.median(timestamp_deltas)) if timestamp_deltas.size else None
        ),
        "p95_dt_s": (
            float(np.percentile(timestamp_deltas, 95)) if timestamp_deltas.size else None
        ),
        "video_counts": video_counts,
        "issues": issues,
    }


def _set_actor_pose(actor: Any, pose_wxyz: np.ndarray, device: Any) -> None:
    pose_wxyz = np.asarray(pose_wxyz, dtype=np.float32).reshape(7)
    actor.set_pose(
        Pose.create_from_pq(
            p=torch.as_tensor(pose_wxyz[:3], device=device).reshape(1, 3),
            q=torch.as_tensor(pose_wxyz[3:], device=device).reshape(1, 4),
        )
    )


def _restore_initial_state(raw_env: Any, h5_file: h5py.File) -> None:
    for arm_index, side in ((0, "left"), (1, "right")):
        robot = raw_env.get_arm_robot(arm_index)
        qpos = torch.as_tensor(
            np.asarray(h5_file[f"observations/robot/{side}_full_qpos"][0]),
            device=raw_env.device,
            dtype=robot.get_qpos().dtype,
        ).reshape(1, -1)
        qvel = torch.as_tensor(
            np.asarray(h5_file[f"observations/robot/{side}_full_qvel"][0]),
            device=raw_env.device,
            dtype=robot.get_qvel().dtype,
        ).reshape(1, -1)
        robot.set_qpos(qpos)
        robot.set_qvel(qvel)
        arm_joints = raw_env.left_arm_joints if arm_index == 0 else raw_env.right_arm_joints
        arm_targets = qpos[:, : len(arm_joints)]
        robot.set_joint_drive_targets(arm_targets, joints=arm_joints)
        robot.set_joint_drive_velocity_targets(
            torch.zeros_like(arm_targets), joints=arm_joints
        )
        gripper_value = float(
            np.asarray(h5_file[f"observations/task/{side}_gripper_val"][0]).reshape(-1)[0]
        )
        raw_env._set_gripper(
            torch.tensor([gripper_value], device=raw_env.device), arm_index=arm_index
        )

    _set_actor_pose(
        raw_env.box,
        np.asarray(h5_file["observations/task/box_pose"][0]),
        raw_env.device,
    )
    _set_actor_pose(
        raw_env.target_box,
        np.asarray(h5_file["observations/task/target_box_pose"][0]),
        raw_env.device,
    )
    zero_velocity = torch.zeros((1, 3), device=raw_env.device)
    raw_env.box.set_linear_velocity(zero_velocity)
    raw_env.box.set_angular_velocity(zero_velocity)


def _render_frame(env: Any) -> np.ndarray | None:
    frame = _to_numpy(env.env.render())
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def replay_episode(
    episode_dir: Path,
    *,
    video_path: Path | None,
    video_every: int,
    video_fps: int,
    settle_steps: int,
    sim_backend_override: str | None,
    render_backend: str,
) -> dict[str, Any]:
    metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
    profile = str(metadata["task_profile"])
    seed = int(metadata["seed"])
    sim_backend = sim_backend_override or str(metadata.get("sim_backend", "physx_cpu"))
    render_mode = "rgb_array" if video_path is not None else None
    viewer_pose = sapien_utils.look_at(
        eye=np.asarray([0.0, -0.45, 1.90], dtype=np.float32),
        target=np.asarray([0.0, -0.05, 0.75], dtype=np.float32),
    )

    with h5py.File(episode_dir / "data.hdf5", "r") as h5_file:
        actions = np.asarray(
            h5_file["actions/control/absolute_ee_target"], dtype=np.float32
        )
        recorded_qpos = np.asarray(
            h5_file["observations/robot/agent_qpos"], dtype=np.float32
        )
        recorded_box_pose = np.asarray(
            h5_file["observations/task/box_pose"], dtype=np.float32
        )
        recorded_final_info = {
            key: _scalar_value(h5_file[f"transitions/info/{key}"][-1])
            for key in FINAL_DIAGNOSTIC_KEYS
            if f"transitions/info/{key}" in h5_file
        }

        env = ManiSkillTaskFactory().create_env(
            task_name=profile,
            obs_mode="state_dict",
            control_mode=EXPECTED_CONTROL_MODE,
            render_mode=render_mode,
            sim_backend=sim_backend,
            render_backend=render_backend,
            max_episode_steps=len(actions) + int(settle_steps) + 10,
            human_render_camera_configs={"render_camera": {"pose": viewer_pose}},
        )
        writer = None
        qpos_errors: list[float] = []
        box_position_errors: list[float] = []
        error = None
        success = False
        terminated = False
        truncated = False
        steps = 0
        settle_steps_executed = 0
        success_at_recorded_horizon = False
        initial_qpos_error = None
        initial_box_position_error = None
        last_info: dict[str, Any] = {}
        try:
            env.reset(seed=seed)
            raw_env = env.env.unwrapped
            _restore_initial_state(raw_env, h5_file)
            initial_qpos = _to_numpy(raw_env._get_agent_qpos()).reshape(-1, 14)[0]
            initial_box_pose = _to_numpy(raw_env.box.pose.raw_pose).reshape(-1, 7)[0]
            initial_qpos_error = float(np.linalg.norm(initial_qpos - recorded_qpos[0]))
            initial_box_position_error = float(
                np.linalg.norm(initial_box_pose[:3] - recorded_box_pose[0, :3])
            )
            if video_path is not None:
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(
                    video_path,
                    fps=int(video_fps),
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=None,
                )
                frame = _render_frame(env)
                if frame is not None:
                    writer.append_data(frame)

            for action_index, action in enumerate(actions):
                try:
                    raw_obs, _, terminated, truncated, info = env.step_raw(
                        ActionSpec.from_value(
                            action,
                            action_type=EXPECTED_CONTROL_MODE,
                        )
                    )
                except Exception as exc:
                    error = repr(exc)
                    break
                steps += 1
                last_info = dict(info)
                actual_qpos = _to_numpy(raw_env._get_agent_qpos()).reshape(-1, 14)[0]
                actual_box_pose = _to_numpy(raw_env.box.pose.raw_pose).reshape(-1, 7)[0]
                qpos_errors.append(
                    float(np.linalg.norm(actual_qpos - recorded_qpos[action_index + 1]))
                )
                box_position_errors.append(
                    float(
                        np.linalg.norm(
                            actual_box_pose[:3] - recorded_box_pose[action_index + 1, :3]
                        )
                    )
                )
                success = _scalar_bool(info.get("success", False))
                if writer is not None and (
                    steps % int(video_every) == 0 or terminated or truncated
                ):
                    frame = _render_frame(env)
                    if frame is not None:
                        writer.append_data(frame)
                if terminated or truncated:
                    break
            success_at_recorded_horizon = success
            if error is None and not terminated and len(actions) > 0:
                for _ in range(int(settle_steps)):
                    try:
                        _, _, terminated, truncated, info = env.step_raw(
                            ActionSpec.from_value(
                                actions[-1],
                                action_type=EXPECTED_CONTROL_MODE,
                            )
                        )
                    except Exception as exc:
                        error = repr(exc)
                        break
                    settle_steps_executed += 1
                    last_info = dict(info)
                    success = _scalar_bool(info.get("success", False))
                    if writer is not None and (
                        settle_steps_executed % int(video_every) == 0
                        or terminated
                        or truncated
                    ):
                        frame = _render_frame(env)
                        if frame is not None:
                            writer.append_data(frame)
                    if terminated or truncated:
                        break
        finally:
            if writer is not None:
                writer.close()
            env.close()

    return {
        "episode": episode_dir.name,
        "seed": seed,
        "recorded_steps": len(actions),
        "replayed_steps": steps,
        "settle_steps_executed": settle_steps_executed,
        "success_at_recorded_horizon": success_at_recorded_horizon,
        "success": success,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "initial_qpos_error": initial_qpos_error,
        "initial_box_position_error_m": initial_box_position_error,
        "qpos_error_mean": float(np.mean(qpos_errors)) if qpos_errors else None,
        "qpos_error_max": float(np.max(qpos_errors)) if qpos_errors else None,
        "box_position_error_mean_m": (
            float(np.mean(box_position_errors)) if box_position_errors else None
        ),
        "box_position_error_max_m": (
            float(np.max(box_position_errors)) if box_position_errors else None
        ),
        "video": None if video_path is None else str(video_path),
        "recorded_final_info": recorded_final_info,
        "replay_final_info": {
            key: _scalar_value(last_info[key])
            for key in FINAL_DIAGNOSTIC_KEYS
            if key in last_info
        },
        "error": error,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    episodes = _episode_dirs(dataset_dir)
    if not episodes:
        raise FileNotFoundError(f"No completed episodes found under {dataset_dir}")
    if args.video_every <= 0 or args.video_fps <= 0:
        raise ValueError("Video interval and FPS must be positive.")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")

    reports = [validate_episode(episode_dir) for episode_dir in episodes]
    for report in reports:
        status = "PASS" if not report["issues"] else "FAIL"
        print(
            f"{status} {report['episode']} seed={report.get('seed')} "
            f"actions={report.get('actions')} duration={report.get('duration_s')} "
            f"median_dt={report.get('median_dt_s')} issues={report['issues']}"
        )
    lengths = np.asarray(
        [report["actions"] for report in reports if "actions" in report], dtype=np.int64
    )
    seeds = [report["seed"] for report in reports if "seed" in report]
    failed = [report for report in reports if report["issues"]]
    print(
        "summary "
        f"episodes={len(reports)} valid={len(reports) - len(failed)} "
        f"total_actions={int(lengths.sum())} "
        f"length_min/median/mean/max={int(lengths.min())}/"
        f"{float(np.median(lengths)):.1f}/{float(np.mean(lengths)):.1f}/"
        f"{int(lengths.max())} duplicate_seeds={len(seeds) - len(set(seeds))}"
    )

    by_index = {int(path.name.split("_")[-1]): path for path in episodes}
    video_dir = (
        None
        if args.save_video_dir is None
        else Path(args.save_video_dir).expanduser().resolve()
    )
    for episode_index in args.replay_episodes:
        if episode_index not in by_index:
            raise KeyError(f"Episode {episode_index} does not exist.")
        video_path = (
            None
            if video_dir is None
            else video_dir / f"replay_episode_{episode_index:06d}.mp4"
        )
        replay = replay_episode(
            by_index[episode_index],
            video_path=video_path,
            video_every=args.video_every,
            video_fps=args.video_fps,
            settle_steps=args.settle_steps,
            sim_backend_override=args.sim_backend,
            render_backend=args.render_backend,
        )
        print("replay " + json.dumps(replay, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
