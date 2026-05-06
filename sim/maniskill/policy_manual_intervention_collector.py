"""Manual first-grasp intervention collector for ManiSkill policy rollouts."""

from __future__ import annotations

import importlib
import select
import sys
from typing import Any

import numpy as np

from core.schemas.action import ActionSpec
from sim.maniskill.expert.handover_block_solver import (
    HandoverBlockSolver,
    LEFT_GRASP_CONTACT_IDS,
    TOP_DOWN_PREFERRED_GRASP_QUATS,
)
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
    MANUAL_RIGHT_TRANSLATION_STEP = 0.03
    MANUAL_LIFT_STEP = 0.02
    MANUAL_RIGHT_LIFT_STEP = 0.03
    MANUAL_MOVE_STEPS = 8
    MANUAL_SETTLE_STEPS = 4
    MANUAL_INPUT_POLL_SEC = 0.05
    MANUAL_IK_THRESHOLD = 0.01
    MANUAL_RETRY_SCALES = (1.0, 0.5, 0.25)
    MANUAL_MAX_SHOULDER_PAN_DELTA = 0.45
    MANUAL_RIGHT_MAX_SHOULDER_PAN_DELTA = 0.7
    MANUAL_MAX_JOINT_MOTION = 1.2
    MANUAL_RIGHT_MAX_JOINT_MOTION = 1.8
    MANUAL_TRIGGER_KEY = "m"

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

    @staticmethod
    def _arm_name(arm_index: int) -> str:
        return "left" if int(arm_index) == 0 else "right"

    def _default_manual_arm_index(self) -> int:
        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        distances = [
            float(np.linalg.norm(np.asarray(self.solver.ctrl.current_gripper_pose(arm), dtype=np.float64)[:3] - box_pose[:3]))
            for arm in (0, 1)
        ]
        return int(np.argmin(distances))

    def _manual_state_snapshot(self, arm_index: int = 0) -> dict[str, Any]:
        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        gripper_pose = np.asarray(self.solver.ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        return {
            "arm_index": int(arm_index),
            "arm_name": self._arm_name(arm_index),
            "box_pose": box_pose.round(4).tolist(),
            "gripper_pose": gripper_pose.round(4).tolist(),
            "qpos": np.asarray(self.solver.ctrl.current_qpos(arm_index), dtype=np.float64).round(4).tolist(),
            "gripper_value": round(float(self.solver.ctrl.current_gripper_value(arm_index)), 4),
            "gripper_box_distance": round(float(np.linalg.norm(gripper_pose[:3] - box_pose[:3])), 4),
        }

    def _sync_manual_arm_targets(self) -> None:
        for arm_index in (0, 1):
            self.solver.ctrl.commanded_arm_qpos[arm_index] = self.solver.ctrl.current_qpos(arm_index).copy()

    def _manual_translation_step(self, arm_index: int) -> float:
        if int(arm_index) == 1:
            return self.MANUAL_RIGHT_TRANSLATION_STEP
        return self.MANUAL_TRANSLATION_STEP

    def _manual_lift_step(self, arm_index: int) -> float:
        if int(arm_index) == 1:
            return self.MANUAL_RIGHT_LIFT_STEP
        return self.MANUAL_LIFT_STEP

    def _manual_shoulder_pan_limit(self, arm_index: int) -> float:
        if int(arm_index) == 1:
            return self.MANUAL_RIGHT_MAX_SHOULDER_PAN_DELTA
        return self.MANUAL_MAX_SHOULDER_PAN_DELTA

    def _manual_joint_motion_limit(self, arm_index: int) -> float:
        if int(arm_index) == 1:
            return self.MANUAL_RIGHT_MAX_JOINT_MOTION
        return self.MANUAL_MAX_JOINT_MOTION

    def _manual_box_relative_delta(self, arm_index: int, command: str) -> np.ndarray:
        step = self._manual_translation_step(arm_index)
        lift_step = self._manual_lift_step(arm_index)
        if command in ("r", "u"):
            return np.asarray([0.0, 0.0, lift_step], dtype=np.float64)
        if command == "f":
            return np.asarray([0.0, 0.0, -lift_step], dtype=np.float64)

        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        gripper_pose = np.asarray(self.solver.ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        to_box_xy = box_pose[:2] - gripper_pose[:2]
        norm = float(np.linalg.norm(to_box_xy))
        if norm < 1e-6:
            forward_xy = np.asarray([1.0, 0.0], dtype=np.float64)
        else:
            forward_xy = to_box_xy / norm
        left_xy = np.asarray([-forward_xy[1], forward_xy[0]], dtype=np.float64)

        if command == "w":
            delta_xy = forward_xy * step
        elif command == "s":
            delta_xy = -forward_xy * step
        elif command == "a":
            delta_xy = left_xy * step
        elif command == "d":
            delta_xy = -left_xy * step
        else:
            delta_xy = np.asarray([0.0, 0.0], dtype=np.float64)
        return np.asarray([delta_xy[0], delta_xy[1], 0.0], dtype=np.float64)

    def _manual_grasp_candidate(self, arm_index: int):
        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        if int(arm_index) == 0:
            candidate = self.solver.ctrl.choose_grasp_candidate(
                desired_pose=box_pose,
                arm_index=0,
                contact_ids=LEFT_GRASP_CONTACT_IDS,
                pre_distance=self.solver.PRE_GRASP_DISTANCE,
                grasp_distance=self.solver.LEFT_FIRST_GRASP_DISTANCE,
                pose_builder=self.solver._grasp_pose_from_box_pose,
                preferred_quat=self.solver._preferred_grasp_quat(0),
                preferred_top_down_quat=TOP_DOWN_PREFERRED_GRASP_QUATS[0],
                pre_pose_candidates_builder=self.solver._left_first_grasp_pre_pose_candidates_from_box_pose,
                selection_mode="prefer_side",
                max_grasp_plan_length=self.solver.LEFT_FIRST_GRASP_MAX_GRASP_PLAN_LENGTH,
            )
            return candidate, {
                "arm_index": 0,
                "mode": "left_first_grasp_candidate",
                "box_pose": box_pose.round(4).tolist(),
                "attempts": list(self.solver.ctrl.last_grasp_attempts),
            }

        candidate, _place_candidate, selection_debug, selection_reason, selection_options = (
            self.solver._select_right_handover_candidate(box_pose)
        )
        return candidate, {
            "arm_index": 1,
            "mode": "right_handover_candidate",
            "box_pose": box_pose.round(4).tolist(),
            "selection_reason": str(selection_reason),
            "selection_debug": selection_debug,
            "selection_options": [
                {
                    "contact_id": int(item["contact_id"]),
                    "future_place_ok": bool(item.get("future_place_ok", False)),
                    "future_place_path_len": float(item.get("future_place_path_len", np.inf)),
                }
                for item in selection_options
            ],
        }

    def _execute_manual_candidate_plan(self, arm_index: int, recorder, candidate) -> str | None:
        self._sync_manual_arm_targets()
        if self.solver.ctrl.execute_plan(arm_index, candidate.pre_plan, recorder):
            return self.solver.ctrl.last_status
        if self.solver.ctrl.last_status != "running":
            return self.solver.ctrl.last_status
        self._sync_manual_arm_targets()
        if self.solver.ctrl.execute_plan(arm_index, candidate.grasp_plan, recorder):
            return self.solver.ctrl.last_status
        if self.solver.ctrl.last_status != "running":
            return self.solver.ctrl.last_status
        self._sync_manual_arm_targets()
        return None

    def _print_manual_help(self) -> None:
        print(
            "[manual] commands: "
            "viewer: up/down or w/s=toward/away box  left/right or a/d=box-relative left/right  u/j or r/f=up/down  "
            "g=auto-align  1=left  2=right  o=open  c=close  p=status  h=help  n=resume  q=abort | "
            "terminal fallback: w/s=toward/away box  a/d=box-relative left/right  r/f=up/down  "
            "g=auto-align  1/2=left/right  o=open  c=close  u=lift  p=status  h/help/?  n/resume  q/abort  "
            "(terminal commands require Enter)"
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
        if window.key_press("1"):
            return "left"
        if window.key_press("2"):
            return "right"
        if window.key_press("h"):
            return "help"
        if window.key_press("p"):
            return "status"
        if window.key_press("g"):
            return "g"
        if window.key_press("up") or window.key_press("w"):
            return "w"
        if window.key_press("down") or window.key_press("s"):
            return "s"
        if window.key_press("left") or window.key_press("a"):
            return "a"
        if window.key_press("right") or window.key_press("d"):
            return "d"
        if window.key_press("u") or window.key_press("r"):
            return "r"
        if window.key_press("j") or window.key_press("f"):
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

    def _manual_trigger_pressed(self) -> bool:
        window = self._viewer_window()
        if window is None:
            return False
        return bool(window.key_press(self.MANUAL_TRIGGER_KEY))

    def _read_manual_command_from_stdin(self) -> str:
        if not hasattr(sys.stdin, "fileno") or not sys.stdin.isatty():
            try:
                return input("manual> ").strip().lower()
            except EOFError:
                return "resume"

        sys.stdout.write("manual> ")
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

    def _run_delta_command(
        self,
        recorder,
        *,
        arm_index: int,
        command: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        world_delta: np.ndarray | None = None,
    ) -> str | None:
        current_gripper_pose = np.asarray(self.solver.ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        if world_delta is None:
            local_rotation = pose_to_matrix(current_gripper_pose)[:3, :3]
            local_forward = local_rotation[:, 0]
            local_left = local_rotation[:, 1]
            world_delta = (
                float(x) * local_left
                + float(y) * local_forward
                + np.asarray([0.0, 0.0, float(z)], dtype=np.float64)
            )
        else:
            world_delta = np.asarray(world_delta, dtype=np.float64).reshape(3)
        last_failure_debug: dict[str, Any] | None = None
        last_reject_debug: dict[str, Any] | None = None

        for scale in self.MANUAL_RETRY_SCALES:
            scaled_delta = world_delta * float(scale)
            target_gripper_pose = current_gripper_pose.copy()
            target_gripper_pose[:3] += scaled_delta
            target_pose = self.solver.ctrl.gripper_pose_to_planner_pose(arm_index, target_gripper_pose)

            ik_qpos, ik_reason = self.solver.ctrl.ik_to_pose_debug(
                arm_index,
                target_pose,
                threshold=self.MANUAL_IK_THRESHOLD,
            )
            if ik_qpos is None:
                last_failure_debug = {
                    "arm_index": int(arm_index),
                    "command": str(command),
                    "retry_scale": float(scale),
                    "target_gripper_pose": np.asarray(target_gripper_pose, dtype=np.float64).round(4).tolist(),
                    "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
                    "world_delta": np.asarray(scaled_delta, dtype=np.float64).round(4).tolist(),
                    "ik_reason": str(ik_reason),
                }
                continue

            target_arm_qpos = self.solver.ctrl._arm_qpos_from_plan(ik_qpos)
            current_arm_qpos = self.solver.ctrl.current_qpos(arm_index)
            shoulder_pan_delta = float(np.abs(target_arm_qpos[0] - current_arm_qpos[0]))
            joint_motion = float(np.linalg.norm(target_arm_qpos - current_arm_qpos))
            shoulder_pan_limit = self._manual_shoulder_pan_limit(arm_index)
            joint_motion_limit = self._manual_joint_motion_limit(arm_index)
            plan_debug = {
                "arm_index": int(arm_index),
                "command": str(command),
                "retry_scale": float(scale),
                "target_gripper_pose": np.asarray(target_gripper_pose, dtype=np.float64).round(4).tolist(),
                "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
                "world_delta": np.asarray(scaled_delta, dtype=np.float64).round(4).tolist(),
                "target_arm_qpos": np.asarray(target_arm_qpos, dtype=np.float64).round(4).tolist(),
                "shoulder_pan_delta": round(shoulder_pan_delta, 4),
                "joint_motion": round(joint_motion, 4),
                "shoulder_pan_limit": round(float(shoulder_pan_limit), 4),
                "joint_motion_limit": round(float(joint_motion_limit), 4),
            }
            if (
                shoulder_pan_delta > shoulder_pan_limit
                or joint_motion > joint_motion_limit
            ):
                plan_debug["guard_reason"] = "large_rotation"
                last_reject_debug = plan_debug
                continue

            self.solver.ctrl.last_plan_debug = plan_debug
            self.solver.ctrl.move_to_qpos(
                arm_index,
                target_arm_qpos,
                recorder,
                steps=self.MANUAL_MOVE_STEPS,
                settle_steps=self.MANUAL_SETTLE_STEPS,
            )
            break
        else:
            if last_reject_debug is not None:
                self.solver.ctrl.last_plan_debug = last_reject_debug
                self.solver.ctrl.last_status = "manual_move_rejected"
            else:
                self.solver.ctrl.last_plan_debug = last_failure_debug or {
                    "arm_index": int(arm_index),
                    "command": str(command),
                    "world_delta": np.asarray(world_delta, dtype=np.float64).round(4).tolist(),
                }
                self.solver.ctrl.last_status = "manual_move_failed"
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

    def _run_manual_command(self, command: str, recorder, *, arm_index: int) -> str | None:
        self._sync_manual_arm_targets()
        if command in ("w", "s", "a", "d", "r", "f", "u"):
            return self._run_delta_command(
                recorder,
                arm_index=arm_index,
                command=command,
                world_delta=self._manual_box_relative_delta(arm_index, command),
            )
        if command == "g":
            candidate, debug_info = self._manual_grasp_candidate(arm_index)
            debug_info["command"] = "g"
            if candidate is None:
                self.solver.ctrl.last_plan_debug = debug_info
                self.solver.ctrl.last_status = "manual_auto_align_failed"
                failed = self.last_debug_info.setdefault("failed_commands", [])
                failed.append(
                    {
                        "command": "g",
                        "status": str(self.solver.ctrl.last_status),
                        "plan_debug": dict(self.solver.ctrl.last_plan_debug),
                    }
                )
                print("[manual] command `g` failed: manual_auto_align_failed")
                self.solver.ctrl.last_status = "running"
                self.solver.ctrl.last_plan_debug = {}
                return None

            debug_info.update(
                {
                    "selected_contact_id": int(candidate.contact_id),
                    "selected_pre_pose": np.asarray(candidate.pre_pose, dtype=np.float64).round(4).tolist(),
                    "selected_grasp_pose": np.asarray(candidate.grasp_pose, dtype=np.float64).round(4).tolist(),
                    "selected_pre_plan_len": int(len(candidate.pre_plan["position"])),
                    "selected_grasp_plan_len": int(len(candidate.grasp_plan["position"])),
                    "selected_grasp_plan_source": str(candidate.grasp_plan_source),
                }
            )
            self.last_debug_info["last_auto_align"] = debug_info
            terminal_status = self._execute_manual_candidate_plan(arm_index, recorder, candidate)
            if terminal_status is not None:
                return terminal_status
            print(
                "[manual] command `g` auto-aligned "
                f"{self._arm_name(arm_index)} arm using contact {int(candidate.contact_id)}"
            )
            return None
        if command == "o":
            self.solver.ctrl.commanded_gripper[arm_index] = 1.0
            gripper_kwargs = {"left_gripper": 1.0} if arm_index == 0 else {"right_gripper": 1.0}
            self.solver.ctrl.hold_pose(
                recorder,
                steps=self.solver.ctrl.RELEASE_SETTLE_STEPS,
                allow_success_early=False,
                **gripper_kwargs,
            )
            if self.solver.ctrl.last_status != "running":
                return self.solver.ctrl.last_status
            return None
        if command == "c":
            self.solver.ctrl.commanded_gripper[arm_index] = 0.0
            gripper_kwargs = {"left_gripper": 0.0} if arm_index == 0 else {"right_gripper": 0.0}
            self.solver.ctrl.hold_pose(
                recorder,
                steps=self.solver.ctrl.GRIPPER_SETTLE_STEPS,
                allow_success_early=False,
                **gripper_kwargs,
            )
            if self.solver.ctrl.last_status != "running":
                return self.solver.ctrl.last_status
            return None
        return None

    def _run_manual_intervention(
        self,
        recorder: _ObsTrackingRecorder,
        *,
        remaining_steps: int,
        trigger_step: int,
        trigger_label: str,
        initial_arm_index: int,
        initial_open: bool = False,
    ) -> str:
        self.solver.ctrl.reset_episode(max_episode_steps=max(int(remaining_steps), 1))
        previous_success_detection = self.solver.ctrl.success_detection_enabled
        self.solver.ctrl.success_detection_enabled = False

        active_arm_index = int(initial_arm_index)
        snapshot = self._manual_state_snapshot(active_arm_index)
        self.last_debug_info = {
            "stage": "policy_manual_intervention_triggered",
            "trigger_label": str(trigger_label),
            "trigger_step": int(trigger_step),
            "trigger_gripper_box_distance": snapshot["gripper_box_distance"],
            "box_pose": snapshot["box_pose"],
            "active_arm_index": int(active_arm_index),
            "active_arm_name": snapshot["arm_name"],
            "active_gripper_pose": snapshot["gripper_pose"],
            "active_qpos": snapshot["qpos"],
            "manual_commands": [],
        }

        try:
            if initial_open:
                self.solver.ctrl.commanded_gripper[active_arm_index] = 1.0
                gripper_kwargs = (
                    {"left_gripper": 1.0}
                    if active_arm_index == 0
                    else {"right_gripper": 1.0}
                )
                self.solver.ctrl.hold_pose(
                    recorder,
                    steps=self.INITIAL_OPEN_STEPS,
                    allow_success_early=False,
                    **gripper_kwargs,
                )
                if self.solver.ctrl.last_status != "running":
                    self.last_debug_info["status"] = self.solver.ctrl.last_status
                    return self.solver.ctrl.last_status

            print(f"[manual] {trigger_label} intervention triggered")
            self._print_manual_help()

            while True:
                snapshot = self._manual_state_snapshot(active_arm_index)
                print(
                    "[manual] "
                    f"arm={snapshot['arm_name']} "
                    f"dist={snapshot['gripper_box_distance']:.4f} "
                    f"gripper={snapshot['gripper_pose'][:3]} "
                    f"box={snapshot['box_pose'][:3]} "
                    f"gripper_val={snapshot['gripper_value']:.2f}"
                )
                command = self._read_manual_command()

                if command == "":
                    continue
                if command in ("help", "h", "?"):
                    self._print_manual_help()
                    continue
                if command in ("p", "status"):
                    continue
                if command in ("1", "left", "l"):
                    active_arm_index = 0
                    self.last_debug_info["manual_commands"].append("switch:left")
                    print("[manual] active arm: left")
                    continue
                if command in ("2", "right"):
                    active_arm_index = 1
                    self.last_debug_info["manual_commands"].append("switch:right")
                    print("[manual] active arm: right")
                    continue

                self.last_debug_info["manual_commands"].append(
                    f"{self._arm_name(active_arm_index)}:{command}"
                )

                if command in ("resume", "n"):
                    self.last_debug_info["status"] = "manual_intervention_resumed"
                    post_snapshot = self._manual_state_snapshot(active_arm_index)
                    self.last_debug_info.update(
                        {
                            "post_manual_box_pose": post_snapshot["box_pose"],
                            "post_manual_active_arm_index": int(active_arm_index),
                            "post_manual_active_arm_name": post_snapshot["arm_name"],
                            "post_manual_gripper_pose": post_snapshot["gripper_pose"],
                            "post_manual_gripper_value": post_snapshot["gripper_value"],
                        }
                    )
                    return "manual_intervention_resumed"

                if command in ("abort", "q"):
                    self.last_debug_info["status"] = "manual_intervention_aborted"
                    post_snapshot = self._manual_state_snapshot(active_arm_index)
                    self.last_debug_info.update(
                        {
                            "post_manual_box_pose": post_snapshot["box_pose"],
                            "post_manual_active_arm_index": int(active_arm_index),
                            "post_manual_active_arm_name": post_snapshot["arm_name"],
                            "post_manual_gripper_pose": post_snapshot["gripper_pose"],
                            "post_manual_gripper_value": post_snapshot["gripper_value"],
                        }
                    )
                    return "manual_intervention_aborted"

                terminal_status = self._run_manual_command(command, recorder, arm_index=active_arm_index)
                if terminal_status is not None:
                    self.last_debug_info["status"] = terminal_status
                    post_snapshot = self._manual_state_snapshot(active_arm_index)
                    self.last_debug_info.update(
                        {
                            "post_manual_box_pose": post_snapshot["box_pose"],
                            "post_manual_active_arm_index": int(active_arm_index),
                            "post_manual_active_arm_name": post_snapshot["arm_name"],
                            "post_manual_gripper_pose": post_snapshot["gripper_pose"],
                            "post_manual_gripper_value": post_snapshot["gripper_value"],
                        }
                    )
                    return terminal_status

                if command not in ("w", "s", "a", "d", "r", "f", "u", "g", "o", "c"):
                    print(f"[manual] unknown command: {command}")
        except EpisodeLimitReached:
            self.solver.ctrl.last_status = "episode_limit_reached"
            self.last_debug_info["status"] = "episode_limit_reached"
            return "episode_limit_reached"
        finally:
            self.solver.ctrl.success_detection_enabled = previous_success_detection

    def _run_left_first_grasp_manual_intervention(
        self,
        recorder: _ObsTrackingRecorder,
        *,
        remaining_steps: int,
        trigger_step: int,
    ) -> str:
        return self._run_manual_intervention(
            recorder,
            remaining_steps=remaining_steps,
            trigger_step=trigger_step,
            trigger_label="left first-grasp",
            initial_arm_index=0,
            initial_open=True,
        )

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
                if self._manual_trigger_pressed():
                    status = self._run_manual_intervention(
                        tracking,
                        remaining_steps=int(max_episode_steps) - tracking.step_count,
                        trigger_step=tracking.step_count,
                        trigger_label="viewer manual",
                        initial_arm_index=self._default_manual_arm_index(),
                        initial_open=False,
                    )
                    if tracking.last_obs is not None:
                        self.adapter.reset_model(self.model)
                        encoded_obs = self.adapter.encode_obs(tracking.last_obs)
                    if status == "manual_intervention_resumed":
                        resume_after_intervention = True
                    break

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
