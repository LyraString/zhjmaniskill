"""Manual first-grasp intervention collector for ManiSkill policy rollouts."""

from __future__ import annotations

import importlib
import select
import sys
from typing import Any

import numpy as np

from core.schemas.action import ActionSpec
from sim.maniskill.expert.handover_block_solver import HandoverBlockSolver
from sim.maniskill.expert.primitives import EpisodeLimitReached, pose_to_matrix
from sim.maniskill.policy_intervention_collector import (
    ADAPTER_MODULES,
    _ObsTrackingRecorder,
    _to_bool_scalar,
)


class PolicyManualInterventionCollector:
    TRIGGER_MIN_STEP = 5
    TRIGGER_MAX_STEP = 180
    TRIGGER_GRIPPER_BOX_DISTANCE = 0.28
    BOX_ON_TABLE_MARGIN = 0.03
    INITIAL_OPEN_STEPS = 4
    MANUAL_TRANSLATION_STEP = 0.015
    MANUAL_LIFT_STEP = 0.02
    MANUAL_MOVE_STEPS = 8
    MANUAL_SETTLE_STEPS = 4
    MANUAL_INPUT_POLL_SEC = 0.05
    MANUAL_MAX_SHOULDER_PAN_DELTA = 0.45
    MANUAL_MAX_JOINT_MOTION = 1.2

    def __init__(
        self,
        env_backend,
        *,
        policy_kind: str,
        ckpt_path: str,
        action_type: str | None,
        debug: bool = False,
    ) -> None:
        if policy_kind not in ADAPTER_MODULES:
            raise ValueError(f"Unsupported policy kind `{policy_kind}`.")
        if getattr(env_backend, "env_id", None) != "HandoverBlock-v1":
            raise ValueError(
                "policy_manual_intervention collector currently only supports `HandoverBlock-v1`."
            )

        self.backend = env_backend
        self.base_env = env_backend.env.unwrapped
        self.policy_kind = policy_kind
        self.action_type = action_type
        self.debug = bool(debug)

        self.adapter = importlib.import_module(ADAPTER_MODULES[policy_kind])
        self.model = self.adapter.get_model({"ckpt_path": ckpt_path})
        self.solver = HandoverBlockSolver(env_backend)
        self.last_debug_info: dict[str, Any] = {}

    def _maybe_render(self) -> None:
        if getattr(self.backend, "render_mode", None) == "human":
            self.backend.render()

    def _box_is_still_on_table(self) -> bool:
        box_pose = self.solver._box_pose()
        table_top = float(self.base_env.TABLE_HEIGHT)
        box_half_z = float(self.solver.box_half_size[2])
        return float(box_pose[2]) <= table_top + box_half_z + self.BOX_ON_TABLE_MARGIN

    def _should_trigger_left_first_grasp(self, step_count: int) -> bool:
        if step_count < self.TRIGGER_MIN_STEP or step_count > self.TRIGGER_MAX_STEP:
            return False
        if not self._box_is_still_on_table():
            return False

        left_gripper_pose = self.solver.ctrl.current_gripper_pose(0)
        box_pose = self.solver._box_pose()
        distance = float(np.linalg.norm(left_gripper_pose[:3] - box_pose[:3]))
        return distance <= self.TRIGGER_GRIPPER_BOX_DISTANCE

    def _manual_state_snapshot(self) -> dict[str, Any]:
        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        left_gripper_pose = np.asarray(self.solver.ctrl.current_gripper_pose(0), dtype=np.float64)
        return {
            "box_pose": box_pose.round(4).tolist(),
            "left_gripper_pose": left_gripper_pose.round(4).tolist(),
            "left_qpos": np.asarray(self.solver.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
            "left_gripper_value": round(float(self.solver.ctrl.current_gripper_value(0)), 4),
            "gripper_box_distance": round(float(np.linalg.norm(left_gripper_pose[:3] - box_pose[:3])), 4),
        }

    def _print_manual_help(self) -> None:
        print(
            "[manual] commands: "
            "viewer: up/down=forward/back  left/right=left/right  u/j=up/down  "
            "o=open  c=close  p=status  h=help  n=resume  q=abort | "
            "terminal fallback: w/s=forward/back  a/d=left/right  r/f=up/down  "
            "o=open  c=close  u=lift  p=status  help=?  resume  abort"
        )

    def _viewer_window(self):
        if getattr(self.backend, "render_mode", None) != "human":
            return None
        self._maybe_render()
        viewer = getattr(self.base_env, "viewer", None)
        if viewer is None:
            return None
        return getattr(viewer, "window", None)

    def _read_manual_command_from_viewer(self) -> str | None:
        window = self._viewer_window()
        if window is None:
            return None
        if window.key_press("h"):
            return "help"
        if window.key_press("p"):
            return "status"
        if window.key_press("up"):
            return "w"
        if window.key_press("down"):
            return "s"
        if window.key_press("left"):
            return "a"
        if window.key_press("right"):
            return "d"
        if window.key_press("u"):
            return "r"
        if window.key_press("j"):
            return "f"
        if window.key_press("o"):
            return "o"
        if window.key_press("c"):
            return "c"
        if window.key_press("n"):
            return "resume"
        if window.key_press("q"):
            return "abort"
        return None

    def _read_manual_command_from_stdin(self) -> str:
        if not hasattr(sys.stdin, "fileno") or not sys.stdin.isatty():
            try:
                return input("manual(left-grasp)> ").strip().lower()
            except EOFError:
                return "resume"

        sys.stdout.write("manual(left-grasp)> ")
        sys.stdout.flush()
        while True:
            self._maybe_render()
            ready, _, _ = select.select([sys.stdin], [], [], self.MANUAL_INPUT_POLL_SEC)
            if not ready:
                continue
            line = sys.stdin.readline()
            if line == "":
                return "resume"
            return line.strip().lower()

    def _read_manual_command(self) -> str:
        while True:
            viewer_command = self._read_manual_command_from_viewer()
            if viewer_command is not None:
                return viewer_command
            if getattr(self.backend, "render_mode", None) != "human":
                return self._read_manual_command_from_stdin()
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                ready, _, _ = select.select([sys.stdin], [], [], self.MANUAL_INPUT_POLL_SEC)
                if ready:
                    line = sys.stdin.readline()
                    if line == "":
                        return "resume"
                    return line.strip().lower()

    def _handle_nonfatal_manual_failure(self, command: str) -> bool:
        if self.solver.ctrl.last_status in ("manual_move_failed",):
            failed = self.last_debug_info.setdefault("failed_commands", [])
            failed.append(
                {
                    "command": str(command),
                    "status": str(self.solver.ctrl.last_status),
                    "plan_debug": dict(self.solver.ctrl.last_plan_debug),
                }
            )
            print(f"[manual] command `{command}` failed: {self.solver.ctrl.last_status}")
            self.solver.ctrl.last_status = "running"
            self.solver.ctrl.last_plan_debug = {}
            return True
        return False

    def _run_delta_command(self, recorder, *, command: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> str | None:
        current_gripper_pose = np.asarray(self.solver.ctrl.current_gripper_pose(0), dtype=np.float64)
        current_ee_pose = np.asarray(self.solver.ctrl.current_ee_pose(0), dtype=np.float64)
        local_rotation = pose_to_matrix(current_gripper_pose)[:3, :3]
        local_forward = local_rotation[:, 0]
        local_left = local_rotation[:, 1]
        world_delta = (
            float(x) * local_left
            + float(y) * local_forward
            + np.asarray([0.0, 0.0, float(z)], dtype=np.float64)
        )
        target_pose = current_ee_pose.copy()
        target_pose[:3] += world_delta

        ik_qpos, ik_reason = self.solver.ctrl.ik_to_pose_debug(0, target_pose)
        if ik_qpos is None:
            self.solver.ctrl.last_plan_debug = {
                "command": str(command),
                "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
                "world_delta": np.asarray(world_delta, dtype=np.float64).round(4).tolist(),
                "ik_reason": str(ik_reason),
            }
            self.solver.ctrl.last_status = "manual_move_failed"
        else:
            target_arm_qpos = self.solver.ctrl._arm_qpos_from_plan(ik_qpos)
            current_arm_qpos = self.solver.ctrl.current_qpos(0)
            shoulder_pan_delta = float(np.abs(target_arm_qpos[0] - current_arm_qpos[0]))
            joint_motion = float(np.linalg.norm(target_arm_qpos - current_arm_qpos))
            self.solver.ctrl.last_plan_debug = {
                "command": str(command),
                "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
                "world_delta": np.asarray(world_delta, dtype=np.float64).round(4).tolist(),
                "target_arm_qpos": np.asarray(target_arm_qpos, dtype=np.float64).round(4).tolist(),
                "shoulder_pan_delta": round(shoulder_pan_delta, 4),
                "joint_motion": round(joint_motion, 4),
            }
            if (
                shoulder_pan_delta > self.MANUAL_MAX_SHOULDER_PAN_DELTA
                or joint_motion > self.MANUAL_MAX_JOINT_MOTION
            ):
                self.solver.ctrl.last_plan_debug["guard_reason"] = "large_rotation"
                self.solver.ctrl.last_status = "manual_move_rejected"
            else:
                self.solver.ctrl.move_to_qpos(
                    0,
                    target_arm_qpos,
                    recorder,
                    steps=self.MANUAL_MOVE_STEPS,
                    settle_steps=self.MANUAL_SETTLE_STEPS,
                )
        if self._handle_nonfatal_manual_failure(command):
            return None
        if self.solver.ctrl.last_status in ("manual_move_rejected",):
            failed = self.last_debug_info.setdefault("failed_commands", [])
            failed.append(
                {
                    "command": str(command),
                    "status": str(self.solver.ctrl.last_status),
                    "plan_debug": dict(self.solver.ctrl.last_plan_debug),
                }
            )
            print(f"[manual] command `{command}` rejected: large_rotation")
            self.solver.ctrl.last_status = "running"
            self.solver.ctrl.last_plan_debug = {}
            return None
        if self.solver.ctrl.last_status != "running":
            return self.solver.ctrl.last_status
        return None

    def _run_manual_command(self, command: str, recorder) -> str | None:
        if command == "w":
            return self._run_delta_command(recorder, command=command, y=self.MANUAL_TRANSLATION_STEP)
        if command == "s":
            return self._run_delta_command(recorder, command=command, y=-self.MANUAL_TRANSLATION_STEP)
        if command == "a":
            return self._run_delta_command(recorder, command=command, x=self.MANUAL_TRANSLATION_STEP)
        if command == "d":
            return self._run_delta_command(recorder, command=command, x=-self.MANUAL_TRANSLATION_STEP)
        if command == "r":
            return self._run_delta_command(recorder, command=command, z=self.MANUAL_TRANSLATION_STEP)
        if command == "f":
            return self._run_delta_command(recorder, command=command, z=-self.MANUAL_TRANSLATION_STEP)
        if command == "u":
            return self._run_delta_command(recorder, command=command, z=self.MANUAL_LIFT_STEP)
        if command == "o":
            self.solver.ctrl.commanded_gripper[0] = 1.0
            self.solver.ctrl.hold_pose(
                recorder,
                steps=self.solver.ctrl.RELEASE_SETTLE_STEPS,
                left_gripper=1.0,
                allow_success_early=False,
            )
            if self.solver.ctrl.last_status != "running":
                return self.solver.ctrl.last_status
            return None
        if command == "c":
            self.solver.ctrl.commanded_gripper[0] = 0.0
            self.solver.ctrl.hold_pose(
                recorder,
                steps=self.solver.ctrl.GRIPPER_SETTLE_STEPS,
                left_gripper=0.0,
                allow_success_early=False,
            )
            if self.solver.ctrl.last_status != "running":
                return self.solver.ctrl.last_status
            return None
        return None

    def _run_left_first_grasp_manual_intervention(
        self,
        recorder: _ObsTrackingRecorder,
        *,
        remaining_steps: int,
        trigger_step: int,
    ) -> str:
        self.solver.ctrl.reset_episode(max_episode_steps=max(int(remaining_steps), 1))
        previous_success_detection = self.solver.ctrl.success_detection_enabled
        self.solver.ctrl.success_detection_enabled = False

        snapshot = self._manual_state_snapshot()
        self.last_debug_info = {
            "stage": "policy_manual_intervention_triggered",
            "trigger_step": int(trigger_step),
            "trigger_gripper_box_distance": snapshot["gripper_box_distance"],
            "box_pose": snapshot["box_pose"],
            "left_gripper_pose": snapshot["left_gripper_pose"],
            "left_qpos": snapshot["left_qpos"],
            "manual_commands": [],
        }

        try:
            self.solver.ctrl.commanded_gripper[0] = 1.0
            self.solver.ctrl.hold_pose(
                recorder,
                steps=self.INITIAL_OPEN_STEPS,
                left_gripper=1.0,
                allow_success_early=False,
            )
            if self.solver.ctrl.last_status != "running":
                self.last_debug_info["status"] = self.solver.ctrl.last_status
                return self.solver.ctrl.last_status

            print("[manual] left first-grasp intervention triggered")
            self._print_manual_help()

            while True:
                snapshot = self._manual_state_snapshot()
                print(
                    "[manual] "
                    f"dist={snapshot['gripper_box_distance']:.4f} "
                    f"left_gripper={snapshot['left_gripper_pose'][:3]} "
                    f"box={snapshot['box_pose'][:3]} "
                    f"gripper={snapshot['left_gripper_value']:.2f}"
                )
                command = self._read_manual_command()

                if command == "":
                    continue
                if command in ("help", "?"):
                    self._print_manual_help()
                    continue
                if command in ("p", "status"):
                    continue

                self.last_debug_info["manual_commands"].append(command)

                if command == "resume":
                    self.last_debug_info["status"] = "manual_intervention_resumed"
                    self.last_debug_info.update(
                        {
                            "post_manual_box_pose": self._manual_state_snapshot()["box_pose"],
                            "post_manual_left_gripper_pose": self._manual_state_snapshot()["left_gripper_pose"],
                            "post_manual_left_gripper_value": self._manual_state_snapshot()["left_gripper_value"],
                        }
                    )
                    return "manual_intervention_resumed"

                if command == "abort":
                    self.last_debug_info["status"] = "manual_intervention_aborted"
                    self.last_debug_info.update(
                        {
                            "post_manual_box_pose": self._manual_state_snapshot()["box_pose"],
                            "post_manual_left_gripper_pose": self._manual_state_snapshot()["left_gripper_pose"],
                            "post_manual_left_gripper_value": self._manual_state_snapshot()["left_gripper_value"],
                        }
                    )
                    return "manual_intervention_aborted"

                terminal_status = self._run_manual_command(command, recorder)
                if terminal_status is not None:
                    self.last_debug_info["status"] = terminal_status
                    self.last_debug_info.update(
                        {
                            "post_manual_box_pose": self._manual_state_snapshot()["box_pose"],
                            "post_manual_left_gripper_pose": self._manual_state_snapshot()["left_gripper_pose"],
                            "post_manual_left_gripper_value": self._manual_state_snapshot()["left_gripper_value"],
                        }
                    )
                    return terminal_status

                if command not in ("w", "s", "a", "d", "r", "f", "u", "o", "c"):
                    print(f"[manual] unknown command: {command}")
        except EpisodeLimitReached:
            self.solver.ctrl.last_status = "episode_limit_reached"
            self.last_debug_info["status"] = "episode_limit_reached"
            return "episode_limit_reached"
        finally:
            self.solver.ctrl.success_detection_enabled = previous_success_detection

    def collect_episode(
        self,
        recorder,
        *,
        seed: int,
        max_episode_steps: int,
    ) -> tuple[bool, int, str]:
        tracking = _ObsTrackingRecorder(recorder)
        obs = self.backend.reset(seed=seed)
        tracking.start_episode(dict(obs.data))
        self._maybe_render()

        self.adapter.reset_model(self.model)
        encoded_obs = self.adapter.encode_obs(dict(obs.data))

        success = False
        status = "policy_rollout_finished"
        intervention_attempted = False

        while tracking.step_count < int(max_episode_steps):
            if len(self.model.obs_cache) == 0:
                self.model.update_obs(encoded_obs)
            action_chunk = self.model.get_action()
            resume_after_intervention = False

            for action in action_chunk:
                if (not intervention_attempted) and self._should_trigger_left_first_grasp(tracking.step_count):
                    intervention_attempted = True
                    status = self._run_left_first_grasp_manual_intervention(
                        tracking,
                        remaining_steps=int(max_episode_steps) - tracking.step_count,
                        trigger_step=tracking.step_count,
                    )
                    if tracking.last_obs is not None:
                        self.adapter.reset_model(self.model)
                        encoded_obs = self.adapter.encode_obs(tracking.last_obs)
                    if status == "manual_intervention_resumed":
                        resume_after_intervention = True
                    break

                result = self.backend.step(
                    ActionSpec.from_value(action, action_type=self.action_type)
                )
                tracking.append_step(action, dict(result.observation.data))
                self._maybe_render()
                encoded_obs = self.adapter.encode_obs(dict(result.observation.data))
                self.model.update_obs(encoded_obs)

                if _to_bool_scalar(result.info.get("success", False)):
                    success = True
                    status = "success"
                    break
                if result.terminated:
                    status = "terminated"
                    break
                if result.truncated:
                    status = "truncated"
                    break
                if tracking.step_count >= int(max_episode_steps):
                    status = "max_episode_steps_reached"
                    break

            if resume_after_intervention:
                if tracking.step_count >= int(max_episode_steps):
                    status = "max_episode_steps_reached"
                    break
                continue

            if success or status in (
                "terminated",
                "truncated",
                "max_episode_steps_reached",
                "manual_intervention_aborted",
                "episode_limit_reached",
            ):
                break

        if tracking.step_count >= int(max_episode_steps) and not success and status == "policy_rollout_finished":
            status = "max_episode_steps_reached"

        return success, tracking.step_count, status
