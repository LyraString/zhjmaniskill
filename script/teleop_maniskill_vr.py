"""Teleoperate the bimanual ManiSkill task with PICO controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from mani_skill.utils import sapien_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory, apply_task_profile
from sim.maniskill.teleop import (
    BimanualXRMapper,
    CanonicalTeleopDatasetRecorder,
    XRClient,
    capture_canonical_observation,
)
from sim.maniskill.teleop.pose_mapper import DEFAULT_XR_TO_WORLD_ROTATION
from sim.maniskill.teleop.xr_client import BUTTON_NAMES, XRState


EXPECTED_CONTROL_MODE = "bimanual_pd_ee_delta_pose"
CAMERA_PREVIEW_NAMES = (
    ("head_camera", "HEAD"),
    ("left_wrist_camera", "LEFT WRIST"),
    ("right_wrist_camera", "RIGHT WRIST"),
)


def _is_recoverable_pdee_ik_failure(error: RuntimeError) -> bool:
    return str(error).startswith("PDEE inverse kinematics")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PICO/XRoboToolkit teleoperation for HandoverBlockUR5-v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_name", help="Dataset directory name under data/<env-id>/")
    parser.add_argument(
        "--task-instruction",
        default=(
            "Pick up the block with the left arm, hand it to the right arm, "
            "and place it on the target."
        ),
        help="Language task annotation stored with every episode",
    )
    parser.add_argument(
        "--task-profile",
        default="handover_block_ur5_v2_vr_robotwin_clean",
    )
    parser.add_argument("--save-dir", default=None, help="Override the canonical dataset root")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Explicit starting environment seed. When omitted, continue from one "
            "greater than the largest seed in the saved dataset."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=72_000,
        help=(
            "Hard environment TimeLimit. The default is one hour at 20 Hz so "
            "idle teleoperation setup does not reset the scene unexpectedly."
        ),
    )
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--render-mode", choices=("human", "rgb_array", "none"), default="human")
    parser.add_argument(
        "--viewer-eye",
        type=float,
        nargs=3,
        default=(0.0, -0.45, 1.90),
        metavar=("X", "Y", "Z"),
        help="Initial operator-view camera position in the ManiSkill world frame",
    )
    parser.add_argument(
        "--viewer-target",
        type=float,
        nargs=3,
        default=(0.0, -0.05, 0.75),
        metavar=("X", "Y", "Z"),
        help="Initial operator-view look-at target in the ManiSkill world frame",
    )
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--grip-threshold", type=float, default=0.9)
    parser.add_argument("--max-xr-stale-seconds", type=float, default=0.25)
    parser.add_argument("--rgb-crf", type=int, default=18)
    parser.add_argument("--rgb-preset", default="veryfast")
    parser.add_argument("--writer-queue-size", type=int, default=8)
    parser.add_argument("--save-only-success", action="store_true")
    parser.add_argument(
        "--camera-preview",
        choices=("off", "multiview"),
        default="multiview",
        help="Show a separate live preview of the three policy cameras",
    )
    parser.add_argument(
        "--camera-preview-every",
        type=int,
        default=2,
        help="Refresh the camera preview every N control steps",
    )
    parser.add_argument(
        "--camera-preview-scale",
        type=float,
        default=1.5,
        help="Display-only scale applied to the multiview preview",
    )
    parser.add_argument(
        "--xr-to-world-rotation",
        type=float,
        nargs=9,
        default=DEFAULT_XR_TO_WORLD_ROTATION.reshape(-1).tolist(),
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    return parser


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar_bool(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _current_ee_pose_root(base_env: Any) -> np.ndarray:
    return np.stack(
        [
            _to_numpy(base_env.left_delta_pose_controller.ee_pose_at_base.raw_pose)
            .reshape(-1, 7)[0],
            _to_numpy(base_env.right_delta_pose_controller.ee_pose_at_base.raw_pose)
            .reshape(-1, 7)[0],
        ],
        axis=0,
    ).astype(np.float64, copy=False)


def _root_pose_world(base_env: Any) -> np.ndarray:
    return np.stack(
        [base_env.get_arm_root_pose_array(arm_index) for arm_index in (0, 1)],
        axis=0,
    ).astype(np.float64, copy=False)


def _numeric_scalars(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in values.items():
        try:
            array = _to_numpy(value)
        except (TypeError, ValueError):
            continue
        if array.size != 1 or array.dtype.kind not in "biuf":
            continue
        result[str(key)] = array.reshape(())
    return result


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _render_human_if_open(env: Any, base_env: Any) -> bool:
    """Render one GUI frame and report whether the viewer remains open."""

    viewer = getattr(base_env, "_viewer", None)
    if viewer is not None and bool(getattr(viewer, "closed", False)):
        return False
    env.render()
    viewer = getattr(base_env, "_viewer", None)
    return viewer is None or not bool(getattr(viewer, "closed", False))


class _MultiviewCameraPreview:
    WINDOW_NAME = "RoboTwin VR camera views"

    def __init__(self, *, scale: float, cv2_module: Any | None = None) -> None:
        if scale <= 0.0:
            raise ValueError("Camera preview scale must be positive.")
        if cv2_module is None:
            import cv2 as cv2_module

        self.cv2 = cv2_module
        self.scale = float(scale)
        self.cv2.namedWindow(self.WINDOW_NAME, self.cv2.WINDOW_NORMAL)

    @staticmethod
    def _camera_rgb(raw_obs: Mapping[str, Any], camera_name: str) -> np.ndarray | None:
        sensor_data = raw_obs.get("sensor_data")
        if not isinstance(sensor_data, Mapping):
            return None
        camera_data = sensor_data.get(camera_name)
        if not isinstance(camera_data, Mapping) or "rgb" not in camera_data:
            return None
        rgb = _to_numpy(camera_data["rgb"])
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            return None
        return rgb

    def show(self, raw_obs: Mapping[str, Any]) -> bool:
        panels = []
        for camera_name, label in CAMERA_PREVIEW_NAMES:
            rgb = self._camera_rgb(raw_obs, camera_name)
            if rgb is None:
                continue
            panel = np.ascontiguousarray(rgb[..., ::-1])
            self.cv2.putText(
                panel,
                label,
                (8, 22),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
            panels.append(panel)
        if not panels:
            return False
        min_height = min(panel.shape[0] for panel in panels)
        preview = np.concatenate([panel[:min_height] for panel in panels], axis=1)
        if not np.isclose(self.scale, 1.0):
            preview = self.cv2.resize(
                preview,
                dsize=None,
                fx=self.scale,
                fy=self.scale,
                interpolation=self.cv2.INTER_LINEAR,
            )
        self.cv2.imshow(self.WINDOW_NAME, preview)
        self.cv2.waitKey(1)
        return True

    def close(self) -> None:
        self.cv2.destroyWindow(self.WINDOW_NAME)
        self.cv2.waitKey(1)


class _ButtonEdges:
    def __init__(self) -> None:
        self._previous = np.zeros(len(BUTTON_NAMES), dtype=np.bool_)

    def update(self, state: XRState) -> dict[str, bool]:
        current = np.asarray(state.buttons, dtype=np.bool_)
        rising = np.logical_and(current, np.logical_not(self._previous))
        self._previous = current.copy()
        return {name: bool(rising[index]) for index, name in enumerate(BUTTON_NAMES)}


def _build_action_record(
    *,
    xr_state: XRState,
    mapper: BimanualXRMapper,
    desired_ee_pose_root: np.ndarray,
    delta_action: np.ndarray,
    absolute_action: np.ndarray,
) -> dict[str, Any]:
    return {
        "host_monotonic_ns": np.int64(time.monotonic_ns()),
        "xr_timestamp_ns": np.int64(xr_state.timestamp_ns),
        "operator": {
            "controller_pose_xyzw": xr_state.controller_pose_xyzw.astype(
                np.float32, copy=False
            ),
            "grip": xr_state.grip,
            "trigger": xr_state.trigger,
            "axis": xr_state.axis,
            "buttons": xr_state.buttons,
        },
        "control": {
            "clutch_active": mapper.clutch_active,
            "desired_ee_pose_root_wxyz": desired_ee_pose_root.astype(
                np.float32, copy=False
            ),
            "delta_pose_actual": np.asarray(delta_action, dtype=np.float32),
            "absolute_ee_target": np.asarray(absolute_action, dtype=np.float32),
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_xr_stale_seconds <= 0.0:
        raise ValueError("--max-xr-stale-seconds must be positive.")
    if not 0 <= args.rgb_crf <= 51:
        raise ValueError("--rgb-crf must be between 0 and 51.")
    if args.camera_preview_every <= 0:
        raise ValueError("--camera-preview-every must be positive.")
    if args.camera_preview_scale <= 0.0:
        raise ValueError("--camera-preview-scale must be positive.")

    render_mode = None if args.render_mode == "none" else args.render_mode
    env_id, resolved_runtime, profile_name = apply_task_profile(
        args.task_profile,
        {
            "render_mode": render_mode,
            "sim_backend": args.sim_backend,
            "render_backend": args.render_backend,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    if render_mode == "human":
        viewer_pose = sapien_utils.look_at(
            eye=np.asarray(args.viewer_eye, dtype=np.float32),
            target=np.asarray(args.viewer_target, dtype=np.float32),
        )
        resolved_runtime["human_render_camera_configs"] = {
            "render_camera": {"pose": viewer_pose}
        }
    if resolved_runtime.get("control_mode") != EXPECTED_CONTROL_MODE:
        raise ValueError(
            f"VR teleoperation requires control_mode={EXPECTED_CONTROL_MODE}; "
            f"profile resolved to {resolved_runtime.get('control_mode')}."
        )
    if args.save_dir is None:
        dataset_dir = REPO_ROOT / "data" / env_id / args.dataset_name
    else:
        dataset_dir = Path(args.save_dir).expanduser().resolve()

    env = ManiSkillTaskFactory().create_env(task_name=env_id, **resolved_runtime)
    step_raw = getattr(env, "step_raw", None)
    if not callable(step_raw):
        raise RuntimeError("VR teleoperation requires a ManiSkill backend with step_raw().")
    base_env = env.env.unwrapped
    if getattr(base_env, "num_envs", 1) != 1:
        raise ValueError("VR teleoperation requires a single ManiSkill environment.")
    if base_env.left_delta_pose_controller is None:
        raise RuntimeError("Absolute EEF controllers are not initialized.")

    control_freq = int(getattr(base_env, "control_freq", 20))
    control_period = 1.0 / float(control_freq)
    recorder = CanonicalTeleopDatasetRecorder(
        dataset_dir,
        fps=control_freq,
        rgb_crf=args.rgb_crf,
        rgb_preset=args.rgb_preset,
        queue_size=args.writer_queue_size,
    )
    camera_preview = (
        _MultiviewCameraPreview(scale=args.camera_preview_scale)
        if args.camera_preview == "multiview"
        else None
    )
    mapper = BimanualXRMapper(
        xr_to_world_rotation=np.asarray(args.xr_to_world_rotation).reshape(3, 3),
        translation_scale=args.translation_scale,
        grip_threshold=args.grip_threshold,
        max_translation_step=0.015,
        max_rotation_step=0.10,
    )

    if args.seed is None:
        current_seed = recorder.next_seed()
        seed_source = "continued from saved episodes"
    else:
        current_seed = int(args.seed)
        seed_source = "explicit --seed"
    last_info: dict[str, Any] = {}
    episode_steps = 0
    recording_start_step = 0
    last_fresh_trigger = np.zeros(2, dtype=np.float32)

    def reset_environment() -> None:
        nonlocal last_info, episode_steps, last_fresh_trigger
        env.reset(seed=current_seed)
        mapper.reset(_current_ee_pose_root(base_env))
        last_fresh_trigger = np.asarray(
            [
                1.0 - float(base_env.get_left_gripper_val()),
                1.0 - float(base_env.get_right_gripper_val()),
            ],
            dtype=np.float32,
        )
        last_info = {}
        episode_steps = 0
        if render_mode == "human" and not _render_human_if_open(env, base_env):
            raise RuntimeError("SAPIEN viewer closed during environment reset.")
        if camera_preview is not None:
            camera_preview.show(env.last_raw_obs)

    def current_success() -> bool:
        evaluator = getattr(base_env, "evaluate", None)
        if callable(evaluator):
            evaluated = evaluator()
            if "success" in evaluated:
                return _scalar_bool(evaluated["success"])
        return _scalar_bool(last_info.get("success", False))

    def start_recording(xr_state: XRState) -> None:
        nonlocal recording_start_step
        initial = capture_canonical_observation(env.last_raw_obs, base_env)
        recording_start_step = episode_steps
        episode_index = recorder.start_episode(
            initial,
            metadata={
                "task_profile": profile_name or args.task_profile,
                "task_instruction": str(args.task_instruction),
                "env_id": env_id,
                "seed": current_seed,
                "git_commit": _git_commit(),
                "sim_backend": args.sim_backend,
                "render_backend": args.render_backend,
                "sim_freq_hz": int(getattr(base_env, "sim_freq", 200)),
                "control_freq_hz": control_freq,
                "control_mode": EXPECTED_CONTROL_MODE,
                "translation_unit": "meter",
                "rotation_unit": "radian",
                "camera_position_unit": "millimeter",
                "action_semantics": "actual_relative_delta_pose_euler_xyz",
                "translation_frame": "per_arm_root_aligned",
                "gripper_semantics": "absolute_1_open_0_closed",
                "xr_to_world_rotation": np.asarray(args.xr_to_world_rotation)
                .reshape(3, 3)
                .tolist(),
                "translation_scale": float(args.translation_scale),
                "recording_start_xr_timestamp_ns": int(xr_state.timestamp_ns),
            },
        )
        print(f"recording episode_{episode_index:06d} seed={current_seed}")

    def finish_recording(reason: str, *, force_discard: bool = False) -> None:
        nonlocal current_seed
        if not recorder.recording:
            return
        success = current_success()
        recorded_steps = int(episode_steps - recording_start_step)
        if force_discard or recorded_steps <= 0 or (args.save_only_success and not success):
            recorder.discard_episode()
            print(
                f"discarded episode reason={reason} success={int(success)} "
                f"recorded_steps={recorded_steps}"
            )
        else:
            out_path = recorder.save_episode(
                final_metadata={
                    "success": bool(success),
                    "terminal_reason": str(reason),
                    "environment_steps": int(episode_steps),
                    "recorded_action_steps": recorded_steps,
                    "final_info": {
                        key: np.asarray(value).reshape(()).item()
                        for key, value in _numeric_scalars(last_info).items()
                    },
                }
            )
            print(f"saved {out_path} success={int(success)}")
        current_seed += 1
        reset_environment()

    reset_environment()
    print(
        "VR controls: Grip=clutch arm, Trigger=gripper, B=start/save, "
        "X=reset, Y=discard, Ctrl-C=exit"
    )
    print(f"canonical dataset: {dataset_dir}")
    print(f"starting environment seed={current_seed} ({seed_source})")

    button_edges = _ButtonEdges()
    last_xr_timestamp: int | None = None
    last_timestamp_change_local = time.monotonic()
    stale_reported = False
    next_deadline = time.monotonic()

    try:
        with XRClient() as xr_client:
            print("waiting for the first valid XR controller frame...")
            while True:
                xr_state = xr_client.read()
                if xr_state.timestamp_ns > 0:
                    break
                if render_mode == "human" and not _render_human_if_open(
                    env, base_env
                ):
                    print("SAPIEN viewer closed; stopping teleoperation")
                    return
                time.sleep(min(control_period, 0.05))
            print("XR stream ready; controller motion enabled")
            last_xr_timestamp = xr_state.timestamp_ns
            last_timestamp_change_local = time.monotonic()
            button_edges.update(xr_state)
            next_deadline = time.monotonic()

            while True:
                xr_state = xr_client.read()
                now = time.monotonic()
                timestamp_valid = xr_state.timestamp_ns > 0
                if timestamp_valid and (
                    last_xr_timestamp is None
                    or xr_state.timestamp_ns != last_xr_timestamp
                ):
                    last_xr_timestamp = xr_state.timestamp_ns
                    last_timestamp_change_local = now
                xr_fresh = timestamp_valid and (
                    now - last_timestamp_change_local <= args.max_xr_stale_seconds
                )
                if xr_fresh:
                    last_fresh_trigger = xr_state.trigger.copy()
                    if stale_reported:
                        print("XR stream recovered; controller motion enabled")
                    stale_reported = False
                elif not stale_reported:
                    print("XR stream stale; holding both EEF targets")
                    stale_reported = True

                edges = button_edges.update(xr_state) if xr_fresh else {}
                if edges.get("Y", False):
                    finish_recording("operator_discard", force_discard=True)
                    continue
                if edges.get("X", False):
                    print(f"operator requested reset at environment_step={episode_steps}")
                    if recorder.recording:
                        finish_recording("operator_reset", force_discard=True)
                    else:
                        current_seed += 1
                        reset_environment()
                    continue
                if edges.get("B", False):
                    if recorder.recording:
                        finish_recording("operator_stop")
                    else:
                        start_recording(xr_state)
                    continue

                current_ee_pose_root = _current_ee_pose_root(base_env)
                desired_ee_pose_root = mapper.update_targets(
                    xr_state,
                    root_pose_world=_root_pose_world(base_env),
                    current_ee_pose_root=current_ee_pose_root,
                    allow_controller_motion=xr_fresh,
                )
                delta_action = mapper.build_delta_action(
                    current_ee_pose_root=current_ee_pose_root,
                    trigger=last_fresh_trigger,
                )
                absolute_action = np.asarray(
                    env.prepare_action_chunk(delta_action[None, :]), dtype=np.float32
                ).reshape(14)
                action_record = _build_action_record(
                    xr_state=xr_state,
                    mapper=mapper,
                    desired_ee_pose_root=desired_ee_pose_root,
                    delta_action=delta_action,
                    absolute_action=absolute_action,
                )
                try:
                    (
                        raw_obs,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = step_raw(
                        ActionSpec.from_value(
                            absolute_action,
                            action_type=EXPECTED_CONTROL_MODE,
                        )
                    )
                except RuntimeError as error:
                    if not _is_recoverable_pdee_ik_failure(error):
                        raise
                    failed_arm = base_env.last_delta_control_debug.get(
                        "failed_arm", "unknown"
                    )
                    recovery = (
                        "discarding the active episode and resetting"
                        if recorder.recording
                        else "resetting"
                    )
                    print(
                        f"rejected unreachable {failed_arm}-arm IK target at "
                        f"environment_step={episode_steps}; {recovery} the environment"
                    )
                    if recorder.recording:
                        finish_recording("pdee_ik_failure", force_discard=True)
                    else:
                        current_seed += 1
                        reset_environment()
                    next_deadline = time.monotonic()
                    continue
                last_info = dict(info)
                episode_steps += 1
                if (
                    camera_preview is not None
                    and episode_steps % args.camera_preview_every == 0
                ):
                    camera_preview.show(raw_obs)
                if recorder.recording:
                    recorder.append_step(
                        action=action_record,
                        transition={
                            "reward": np.float32(reward),
                            "terminated": np.bool_(terminated),
                            "truncated": np.bool_(truncated),
                            "success": np.bool_(
                                _scalar_bool(info.get("success", False))
                            ),
                            "info": _numeric_scalars(info),
                        },
                        next_observation=capture_canonical_observation(
                            env.last_raw_obs, base_env
                        ),
                    )
                if render_mode == "human" and not _render_human_if_open(
                    env, base_env
                ):
                    print("SAPIEN viewer closed; stopping teleoperation")
                    break

                if terminated or truncated:
                    transition_reason = (
                        "environment_terminated"
                        if terminated
                        else "environment_truncated"
                    )
                    print(
                        f"environment requested reset reason={transition_reason} "
                        f"environment_step={episode_steps}"
                    )
                    finish_recording(
                        transition_reason,
                        force_discard=not current_success(),
                    )
                    if not recorder.recording and episode_steps > 0:
                        current_seed += 1
                        reset_environment()

                next_deadline += control_period
                sleep_seconds = next_deadline - time.monotonic()
                if sleep_seconds > 0.0:
                    time.sleep(sleep_seconds)
                elif sleep_seconds < -control_period:
                    next_deadline = time.monotonic()
    except KeyboardInterrupt:
        print("teleoperation interrupted")
    finally:
        if camera_preview is not None:
            camera_preview.close()
        recorder.close()
        env.close()


if __name__ == "__main__":
    main()
