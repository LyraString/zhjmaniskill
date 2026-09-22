"""Collect successful VR corrections after a SmolVLA policy reaches a bad state."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
from mani_skill.utils import sapien_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from script.handover.eval_smolvla import (
    EXPECTED_CONTROL_MODE,
    PolicyClient,
    _validate_absolute_action,
    constrain_delta_action,
)
from script.teleop_maniskill_vr import (
    _ButtonEdges,
    _MultiviewCameraPreview,
    _build_action_record,
    _current_ee_pose_root,
    _is_recoverable_pdee_ik_failure,
    _numeric_scalars,
    _render_human_if_open,
    _root_pose_world,
    _scalar_bool,
)
from sim.maniskill import ManiSkillTaskFactory, apply_task_profile
from sim.maniskill.teleop import (
    BimanualXRMapper,
    CanonicalTeleopDatasetRecorder,
    XRClient,
    capture_canonical_observation,
)
from sim.maniskill.teleop.pose_mapper import DEFAULT_XR_TO_WORLD_ROTATION
from sim.maniskill.teleop.xr_client import XRState


DEFAULT_TASK = (
    "Pick up the block with the left arm, hand it to the right arm, "
    "and place it on the target."
)


class CollectorMode(str, Enum):
    WAITING = "waiting"
    POLICY = "policy"
    TAKEOVER_READY = "takeover_ready"
    TELEOP = "teleop"
    FAILED = "failed"


class CollectorCommand(str, Enum):
    NONE = "none"
    START_POLICY = "start_policy"
    PREPARE_TAKEOVER = "prepare_takeover"
    START_TELEOP = "start_teleop"
    TRIGGER_NOT_ALIGNED = "trigger_not_aligned"
    RESET_CURRENT_SEED = "reset_current_seed"
    SKIP_CURRENT_SEED = "skip_current_seed"
    AUTONOMOUS_SUCCESS = "autonomous_success"
    SAVE_CORRECTION = "save_correction"


@dataclass
class CollectorState:
    mode: CollectorMode = CollectorMode.WAITING

    def reset(self) -> None:
        self.mode = CollectorMode.WAITING

    def fail(self) -> None:
        self.mode = CollectorMode.FAILED

    def handle_buttons(
        self,
        edges: Mapping[str, bool],
        *,
        trigger_aligned: bool = False,
    ) -> CollectorCommand:
        if edges.get("X", False):
            return CollectorCommand.RESET_CURRENT_SEED
        if edges.get("Y", False):
            return CollectorCommand.SKIP_CURRENT_SEED
        if self.mode == CollectorMode.WAITING and edges.get("B", False):
            self.mode = CollectorMode.POLICY
            return CollectorCommand.START_POLICY
        if self.mode == CollectorMode.POLICY and edges.get("A", False):
            self.mode = CollectorMode.TAKEOVER_READY
            return CollectorCommand.PREPARE_TAKEOVER
        if self.mode == CollectorMode.TAKEOVER_READY and edges.get("B", False):
            if not trigger_aligned:
                return CollectorCommand.TRIGGER_NOT_ALIGNED
            self.mode = CollectorMode.TELEOP
            return CollectorCommand.START_TELEOP
        return CollectorCommand.NONE

    def handle_success(self) -> CollectorCommand:
        if self.mode == CollectorMode.POLICY:
            return CollectorCommand.AUTONOMOUS_SUCCESS
        if self.mode == CollectorMode.TELEOP:
            return CollectorCommand.SAVE_CORRECTION
        return CollectorCommand.NONE


def triggers_match_grippers(
    trigger: np.ndarray,
    gripper_values: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    desired = 1.0 - np.asarray(trigger, dtype=np.float32).reshape(2)
    current = np.asarray(gripper_values, dtype=np.float32).reshape(2)
    return bool(np.all(np.abs(desired - current) <= float(tolerance)))


def initialize_takeover_mapper(
    mapper: BimanualXRMapper,
    xr_state: XRState,
    *,
    current_ee_pose_root: np.ndarray,
    root_pose_world: np.ndarray,
) -> np.ndarray:
    """Anchor the controllers at the current EEF poses without moving either arm."""

    mapper.reset(current_ee_pose_root)
    return mapper.update_targets(
        xr_state,
        root_pose_world=root_pose_world,
        current_ee_pose_root=current_ee_pose_root,
        allow_controller_motion=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_name", help="Correction dataset name under data/<env-id>/")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--task-instruction", default=DEFAULT_TASK)
    parser.add_argument(
        "--task-profile",
        default="handover_block_ur5_v2_vr_robotwin_clean",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--default-seed", type=int, default=2000)
    parser.add_argument("--policy-label", default="smolvla-050000")
    parser.add_argument("--policy-seed", type=int, default=999)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=72_000)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--render-mode", choices=("human", "none"), default="human")
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--grip-threshold", type=float, default=0.9)
    parser.add_argument("--trigger-alignment-tolerance", type=float, default=0.25)
    parser.add_argument("--max-xr-stale-seconds", type=float, default=0.25)
    parser.add_argument("--max-translation-step", type=float, default=0.015)
    parser.add_argument("--max-rotation-step", type=float, default=0.10)
    parser.add_argument("--rgb-crf", type=int, default=18)
    parser.add_argument("--rgb-preset", default="veryfast")
    parser.add_argument("--writer-queue-size", type=int, default=8)
    parser.add_argument(
        "--camera-preview",
        choices=("off", "multiview"),
        default="multiview",
    )
    parser.add_argument("--camera-preview-every", type=int, default=2)
    parser.add_argument("--camera-preview-scale", type=float, default=1.5)
    parser.add_argument(
        "--viewer-eye",
        type=float,
        nargs=3,
        default=(0.0, -0.45, 1.90),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--viewer-target",
        type=float,
        nargs=3,
        default=(0.0, -0.05, 0.75),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--xr-to-world-rotation",
        type=float,
        nargs=9,
        default=DEFAULT_XR_TO_WORLD_ROTATION.reshape(-1).tolist(),
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    )
    return parser


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _gripper_values(base_env: Any) -> np.ndarray:
    return np.asarray(
        [base_env.get_left_gripper_val(), base_env.get_right_gripper_val()],
        dtype=np.float32,
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.execution_horizon <= 0 or args.max_episode_steps <= 0:
        raise ValueError("Execution horizon and maximum episode steps must be positive.")
    if args.max_xr_stale_seconds <= 0.0:
        raise ValueError("--max-xr-stale-seconds must be positive.")
    if args.max_translation_step <= 0.0 or args.max_rotation_step <= 0.0:
        raise ValueError("Per-step motion limits must be positive.")
    if not 0.0 <= args.trigger_alignment_tolerance <= 1.0:
        raise ValueError("--trigger-alignment-tolerance must be between 0 and 1.")
    if not 0 <= args.rgb_crf <= 51:
        raise ValueError("--rgb-crf must be between 0 and 51.")
    if args.camera_preview_every <= 0 or args.camera_preview_scale <= 0.0:
        raise ValueError("Camera preview cadence and scale must be positive.")
    if args.seed is not None and args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if args.default_seed < 0:
        raise ValueError("--default-seed must be non-negative.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)
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
            f"Correction collection requires control_mode={EXPECTED_CONTROL_MODE}; "
            f"profile resolved to {resolved_runtime.get('control_mode')}."
        )

    dataset_dir = (
        args.save_dir.expanduser().resolve()
        if args.save_dir is not None
        else REPO_ROOT / "data" / env_id / args.dataset_name
    )
    env = ManiSkillTaskFactory().create_env(task_name=env_id, **resolved_runtime)
    base_env = env.env.unwrapped
    step_raw = getattr(env, "step_raw", None)
    if not callable(step_raw):
        raise RuntimeError("Correction collection requires a ManiSkill backend with step_raw().")
    if getattr(base_env, "num_envs", 1) != 1:
        raise ValueError("Correction collection requires a single environment.")
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
        max_translation_step=args.max_translation_step,
        max_rotation_step=args.max_rotation_step,
    )
    current_seed = (
        int(args.seed)
        if args.seed is not None
        else recorder.next_seed(default=int(args.default_seed))
    )
    state = CollectorState()
    episode_steps = 0
    policy_steps = 0
    correction_steps = 0
    takeover_step: int | None = None
    last_info: dict[str, Any] = {}
    observation: dict[str, Any]
    action_queue: list[np.ndarray] = []
    session_counts = {"autonomous_success": 0, "saved_corrections": 0, "skipped": 0}

    def reset_environment() -> None:
        nonlocal episode_steps, policy_steps, correction_steps, takeover_step
        nonlocal last_info, observation, action_queue
        if recorder.recording:
            recorder.discard_episode()
        env.reset(seed=current_seed)
        observation = capture_canonical_observation(env.last_raw_obs, base_env)
        mapper.reset(_current_ee_pose_root(base_env))
        episode_steps = 0
        policy_steps = 0
        correction_steps = 0
        takeover_step = None
        last_info = {}
        action_queue = []
        state.reset()
        if render_mode == "human" and not _render_human_if_open(env, base_env):
            raise RuntimeError("SAPIEN viewer closed during environment reset.")
        if camera_preview is not None:
            camera_preview.show(env.last_raw_obs)
        print(f"seed={current_seed} ready; press B to start policy", flush=True)

    def advance_seed(*, skipped: bool = False) -> None:
        nonlocal current_seed
        if skipped:
            session_counts["skipped"] += 1
        current_seed += 1
        reset_environment()

    def start_correction(xr_state: XRState) -> None:
        nonlocal correction_steps
        current_ee = _current_ee_pose_root(base_env)
        initialize_takeover_mapper(
            mapper,
            xr_state,
            current_ee_pose_root=current_ee,
            root_pose_world=_root_pose_world(base_env),
        )
        correction_steps = 0
        episode_index = recorder.start_episode(
            capture_canonical_observation(env.last_raw_obs, base_env),
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
                "collection_type": "smolvla_vr_correction",
                "collection_round": str(args.dataset_name),
                "source_policy": str(args.policy_label),
                "policy_server": f"{args.server_host}:{args.server_port}",
                "policy_steps_before_takeover": int(policy_steps),
                "takeover_environment_step": int(takeover_step or 0),
                "recording_start_xr_timestamp_ns": int(xr_state.timestamp_ns),
            },
        )
        print(
            f"recording correction episode_{episode_index:06d} seed={current_seed}",
            flush=True,
        )

    def save_correction() -> None:
        success = _scalar_bool(last_info.get("success", False))
        if not success:
            raise RuntimeError("Refusing to save a correction before task success.")
        out_path = recorder.save_episode(
            final_metadata={
                "success": True,
                "terminal_reason": "task_success",
                "environment_steps": int(episode_steps),
                "recorded_action_steps": int(correction_steps),
                "final_info": {
                    key: np.asarray(value).reshape(()).item()
                    for key, value in _numeric_scalars(last_info).items()
                },
            }
        )
        session_counts["saved_corrections"] += 1
        print(f"saved {out_path}", flush=True)

    reset_environment()
    print(
        "Controls: B=start policy; A=prepare takeover; B=confirm aligned takeover; "
        "Grip=clutch; Trigger=gripper; X=retry seed; Y=skip seed; Ctrl-C=exit",
        flush=True,
    )
    print(f"correction dataset: {dataset_dir}", flush=True)

    button_edges = _ButtonEdges()
    last_xr_timestamp: int | None = None
    last_timestamp_change_local = time.monotonic()
    stale_reported = False
    next_deadline = time.monotonic()

    try:
        with closing(PolicyClient(args.server_host, args.server_port)) as client, XRClient() as xr_client:
            print("waiting for the first valid XR controller frame...", flush=True)
            while True:
                xr_state = xr_client.read()
                if xr_state.timestamp_ns > 0:
                    break
                if render_mode == "human" and not _render_human_if_open(env, base_env):
                    return
                time.sleep(min(control_period, 0.05))
            button_edges.update(xr_state)
            last_xr_timestamp = xr_state.timestamp_ns
            last_timestamp_change_local = time.monotonic()
            print("XR stream ready", flush=True)

            while True:
                xr_state = xr_client.read()
                now = time.monotonic()
                if xr_state.timestamp_ns > 0 and xr_state.timestamp_ns != last_xr_timestamp:
                    last_xr_timestamp = xr_state.timestamp_ns
                    last_timestamp_change_local = now
                xr_fresh = xr_state.timestamp_ns > 0 and (
                    now - last_timestamp_change_local <= args.max_xr_stale_seconds
                )
                if xr_fresh:
                    if stale_reported:
                        print("XR stream recovered", flush=True)
                    stale_reported = False
                    edges = button_edges.update(xr_state)
                else:
                    edges = {}
                    if not stale_reported:
                        print("XR stream stale; simulation and recording paused", flush=True)
                        stale_reported = True

                aligned = xr_fresh and triggers_match_grippers(
                    xr_state.trigger,
                    _gripper_values(base_env),
                    tolerance=args.trigger_alignment_tolerance,
                )
                command = state.handle_buttons(edges, trigger_aligned=aligned)
                if command == CollectorCommand.RESET_CURRENT_SEED:
                    print(f"discarding and resetting seed={current_seed}", flush=True)
                    reset_environment()
                    next_deadline = time.monotonic()
                    continue
                if command == CollectorCommand.SKIP_CURRENT_SEED:
                    print(f"discarding and skipping seed={current_seed}", flush=True)
                    advance_seed(skipped=True)
                    next_deadline = time.monotonic()
                    continue
                if command == CollectorCommand.START_POLICY:
                    client.reset()
                    action_queue = []
                    print(f"policy started seed={current_seed}", flush=True)
                elif command == CollectorCommand.PREPARE_TAKEOVER:
                    action_queue = []
                    takeover_step = episode_steps
                    mapper.reset(_current_ee_pose_root(base_env))
                    print(
                        "takeover ready; align both triggers with the current grippers, "
                        "then press B",
                        flush=True,
                    )
                elif command == CollectorCommand.TRIGGER_NOT_ALIGNED:
                    desired = 1.0 - np.asarray(xr_state.trigger)
                    print(
                        "takeover rejected: trigger/gripper mismatch "
                        f"desired={desired.round(3).tolist()} "
                        f"current={_gripper_values(base_env).round(3).tolist()}",
                        flush=True,
                    )
                elif command == CollectorCommand.START_TELEOP:
                    start_correction(xr_state)

                stepped = False
                try:
                    if state.mode == CollectorMode.POLICY:
                        if not action_queue:
                            inference_seed = (
                                int(args.policy_seed) + 100_000 * int(current_seed) + episode_steps
                            )
                            chunk = client.predict(
                                observation,
                                str(args.task_instruction),
                                inference_seed,
                            )
                            if args.execution_horizon > len(chunk):
                                raise ValueError(
                                    f"Execution horizon {args.execution_horizon} exceeds "
                                    f"policy chunk {len(chunk)}."
                                )
                            action_queue = list(chunk[: args.execution_horizon])
                        delta, _ = constrain_delta_action(
                            action_queue.pop(0),
                            max_translation_step=args.max_translation_step,
                            max_rotation_step=args.max_rotation_step,
                        )
                        absolute = np.asarray(
                            env.prepare_action_chunk(delta[None, :]), dtype=np.float32
                        ).reshape(14)
                        absolute = _validate_absolute_action(absolute, env.action_space)
                        raw_obs, _, terminated, truncated, info = step_raw(
                            ActionSpec.from_value(absolute, action_type=EXPECTED_CONTROL_MODE)
                        )
                        policy_steps += 1
                        stepped = True
                    elif state.mode == CollectorMode.TELEOP and xr_fresh:
                        current_ee = _current_ee_pose_root(base_env)
                        desired_ee = mapper.update_targets(
                            xr_state,
                            root_pose_world=_root_pose_world(base_env),
                            current_ee_pose_root=current_ee,
                            allow_controller_motion=True,
                        )
                        delta = mapper.build_delta_action(
                            current_ee_pose_root=current_ee,
                            trigger=xr_state.trigger,
                        )
                        absolute = np.asarray(
                            env.prepare_action_chunk(delta[None, :]), dtype=np.float32
                        ).reshape(14)
                        absolute = _validate_absolute_action(absolute, env.action_space)
                        action_record = _build_action_record(
                            xr_state=xr_state,
                            mapper=mapper,
                            desired_ee_pose_root=desired_ee,
                            delta_action=delta,
                            absolute_action=absolute,
                        )
                        raw_obs, reward, terminated, truncated, info = step_raw(
                            ActionSpec.from_value(absolute, action_type=EXPECTED_CONTROL_MODE)
                        )
                        last_info = dict(info)
                        correction_steps += 1
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
                            next_observation=capture_canonical_observation(raw_obs, base_env),
                        )
                        stepped = True
                except RuntimeError as error:
                    if not _is_recoverable_pdee_ik_failure(error):
                        raise
                    state.fail()
                    failed_arm = base_env.last_delta_control_debug.get("failed_arm", "unknown")
                    print(
                        f"IK failure on {failed_arm} arm at step={episode_steps}; "
                        "press X to retry or Y to skip",
                        flush=True,
                    )

                if stepped:
                    last_info = dict(info)
                    episode_steps += 1
                    observation = capture_canonical_observation(raw_obs, base_env)
                    success_command = state.handle_success() if _scalar_bool(
                        info.get("success", False)
                    ) else CollectorCommand.NONE
                    if success_command == CollectorCommand.AUTONOMOUS_SUCCESS:
                        session_counts["autonomous_success"] += 1
                        print(
                            f"autonomous success seed={current_seed}; no data saved",
                            flush=True,
                        )
                        advance_seed()
                    elif success_command == CollectorCommand.SAVE_CORRECTION:
                        save_correction()
                        advance_seed()
                    elif terminated or truncated:
                        state.fail()
                        print(
                            f"environment ended without success at step={episode_steps}; "
                            "press X to retry or Y to skip",
                            flush=True,
                        )

                if camera_preview is not None and (
                    not stepped or episode_steps % args.camera_preview_every == 0
                ):
                    camera_preview.show(env.last_raw_obs)
                if render_mode == "human" and not _render_human_if_open(env, base_env):
                    break

                next_deadline += control_period
                sleep_seconds = next_deadline - time.monotonic()
                if sleep_seconds > 0.0:
                    time.sleep(sleep_seconds)
                elif sleep_seconds < -control_period:
                    next_deadline = time.monotonic()
    except KeyboardInterrupt:
        print("correction collection interrupted", flush=True)
    finally:
        print("session " + json.dumps(session_counts), flush=True)
        if camera_preview is not None:
            camera_preview.close()
        recorder.close()
        env.close()


if __name__ == "__main__":
    main()
