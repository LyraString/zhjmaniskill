"""Manual first-grasp intervention collector for ManiSkill policy rollouts."""

from __future__ import annotations

import importlib
import select
import sys
from typing import Any

import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

from core.schemas.action import ActionSpec
from sim.maniskill.expert.handover_block_solver import (
    HandoverBlockSolver,
    LEFT_GRASP_CONTACT_IDS,
    RIGHT_GRASP_CONTACT_IDS,
    RIGHT_GRASP_PREFERRED_X_AXIS_CONTACT_IDS,
    TOP_DOWN_PREFERRED_GRASP_QUATS,
)
from sim.maniskill.expert.primitives import EpisodeLimitReached, matrix_to_pose, pose_to_matrix, to_numpy
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
    MANUAL_TRANSLATION_STEP = 0.025
    MANUAL_RIGHT_TRANSLATION_STEP = 0.04
    MANUAL_LIFT_STEP = 0.015
    MANUAL_RIGHT_LIFT_STEP = 0.04
    MANUAL_MOVE_STEPS = 5
    MANUAL_SETTLE_STEPS = 1
    MANUAL_ENTRY_FREEZE_STEPS = 8
    MANUAL_GRIPPER_CLOSED_HOLD_THRESHOLD = 0.2
    MANUAL_GRIPPER_CLOSED_DRIVE_VALUE = -0.2
    POLICY_GRIP_GUARD_TAKEOVER_DISTANCE = 0.16
    POLICY_POSE_GUARD_APPROACH_DISTANCE = 0.28
    MANUAL_AUTO_ALIGN_STEP = 0.035
    MANUAL_RIGHT_AUTO_ALIGN_STEP = 0.055
    MANUAL_AUTO_ALIGN_GRASP_SWITCH_DISTANCE = 0.05
    MANUAL_AUTO_ALIGN_ORIENTATION_DISTANCE = 0.08
    MANUAL_AUTO_ALIGN_PLAN_STEPS = 24
    MANUAL_LEFT_AUTO_ALIGN_PLAN_STEPS = 16
    MANUAL_LEFT_AUTO_ALIGN_BOX_MOVE_REJECT = 0.008
    MANUAL_LEFT_AUTO_ALIGN_BOX_Z_DROP_REJECT = 0.01
    MANUAL_RIGHT_AUTO_ALIGN_BOX_MOVE_REJECT = 0.025
    MANUAL_RIGHT_AUTO_ALIGN_BOX_Z_DROP_REJECT = 0.02
    MANUAL_INPUT_POLL_SEC = 0.05
    MANUAL_IK_THRESHOLD = 0.01
    MANUAL_RETRY_SCALES = (1.0, 0.5, 0.25)
    MANUAL_VERTICAL_DIRECTION_MISMATCH_EPS = 0.003
    MANUAL_VERTICAL_MIN_PROGRESS = 0.006
    MANUAL_VERTICAL_PROGRESS_RATIO = 0.25
    MANUAL_HANDOVER_HOLDER_MIN_LIFT = 0.035
    MANUAL_MAX_SHOULDER_PAN_DELTA = 0.45
    MANUAL_RIGHT_MAX_SHOULDER_PAN_DELTA = 0.7
    MANUAL_MAX_JOINT_MOTION = 1.2
    MANUAL_RIGHT_MAX_JOINT_MOTION = 1.8
    MANUAL_JOINT_LIMIT_CLIP_TOL = 1e-3
    MANUAL_TRIGGER_KEY = "m"
    MANUAL_RIGHT_HORIZONTAL_GRASP = True
    MANUAL_RIGHT_HORIZONTAL_YAW_VARIANTS = (-0.16, -0.08, 0.0, 0.08, 0.16)
    MANUAL_HANDOVER_HOLDER_LIFT = 0.0
    MANUAL_HANDOVER_LEFT_CLEAR_LIFT = 0.10

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
        self._manual_hold_arm_qpos: dict[int, np.ndarray] = {}
        self._manual_hold_gripper_qpos: dict[int, np.ndarray] = {}
        self._policy_grip_guard_arm_index: int | None = None
        self._policy_grip_guard_pose_active = False
        self._manual_handover_holder_lift_done = False
        self._manual_force_guarded_box_hold = False
        self._manual_trigger_down_latched = False
        self._manual_guarded_box_relative_anchor: np.ndarray | None = None

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

    def _manual_trigger_arm_index(self) -> int:
        guarded_arm_index = self._policy_grip_guard_arm_index
        if guarded_arm_index is None:
            return self._default_manual_arm_index()
        # During handover, manual intervention should control the receiving /
        # placing right arm. The left arm is only a guarded holder before
        # takeover; after takeover the right arm itself is the guarded holder.
        return 1

    def _guarded_arm_manual_command_blocked(self, arm_index: int, command: str) -> bool:
        guarded_arm_index = self._policy_grip_guard_arm_index
        if guarded_arm_index is None or int(guarded_arm_index) != int(arm_index):
            return False
        guarded_arm_index = int(guarded_arm_index)
        if guarded_arm_index != 0:
            return False
        if self._opposite_gripper_has_taken_over(guarded_arm_index):
            return False
        return command in ("w", "s", "a", "d", "r", "f", "u", "g", "o", "c")

    def _manual_state_snapshot(self, arm_index: int = 0) -> dict[str, Any]:
        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        gripper_pose = np.asarray(self.solver.ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        other_arm_index = 1 - int(arm_index)
        other_gripper_pose = np.asarray(self.solver.ctrl.current_gripper_pose(other_arm_index), dtype=np.float64)
        return {
            "arm_index": int(arm_index),
            "arm_name": self._arm_name(arm_index),
            "box_pose": box_pose.round(4).tolist(),
            "gripper_pose": gripper_pose.round(4).tolist(),
            "other_arm_index": int(other_arm_index),
            "other_arm_name": self._arm_name(other_arm_index),
            "other_gripper_pose": other_gripper_pose.round(4).tolist(),
            "qpos": np.asarray(self.solver.ctrl.current_qpos(arm_index), dtype=np.float64).round(4).tolist(),
            "gripper_value": round(float(self.solver.ctrl.current_gripper_value(arm_index)), 4),
            "gripper_joint_qpos": np.asarray(
                self.solver.ctrl.current_gripper_joint_qpos(arm_index),
                dtype=np.float64,
            ).round(4).tolist(),
            "gripper_box_distance": round(float(np.linalg.norm(gripper_pose[:3] - box_pose[:3])), 4),
        }

    def _sync_manual_arm_targets(self) -> None:
        for arm_index in (0, 1):
            self.solver.ctrl.commanded_arm_qpos[arm_index] = self.solver.ctrl.current_qpos(arm_index).copy()

    def _reset_manual_hold_targets(self) -> None:
        self._manual_hold_arm_qpos = {
            arm_index: self.solver.ctrl.current_qpos(arm_index).copy()
            for arm_index in (0, 1)
        }
        self._manual_hold_gripper_qpos = {
            arm_index: self.solver.ctrl.current_gripper_joint_qpos(arm_index).copy()
            for arm_index in (0, 1)
        }

    def _remember_manual_hold_target(self, arm_index: int, qpos: np.ndarray | None = None) -> None:
        if qpos is None:
            qpos = self.solver.ctrl.current_qpos(int(arm_index))
        self._manual_hold_arm_qpos[int(arm_index)] = np.asarray(qpos, dtype=np.float64).copy()

    def _manual_hold_target(self, arm_index: int) -> np.ndarray:
        arm_index = int(arm_index)
        if arm_index not in self._manual_hold_arm_qpos:
            self._remember_manual_hold_target(arm_index)
        return self._manual_hold_arm_qpos[arm_index].copy()

    def _remember_manual_gripper_hold_target(self, arm_index: int, qpos: np.ndarray | None = None) -> None:
        if qpos is None:
            qpos = self.solver.ctrl.current_gripper_joint_qpos(int(arm_index))
        self._manual_hold_gripper_qpos[int(arm_index)] = np.asarray(qpos, dtype=np.float64).copy()

    def _manual_gripper_hold_target(self, arm_index: int) -> np.ndarray:
        arm_index = int(arm_index)
        if arm_index not in self._manual_hold_gripper_qpos:
            self._remember_manual_gripper_hold_target(arm_index)
        return self._manual_hold_gripper_qpos[arm_index].copy()

    def _manual_gripper_hold_value(self, arm_index: int) -> float:
        value = float(self.solver.ctrl.commanded_gripper[int(arm_index)])
        if value <= self.MANUAL_GRIPPER_CLOSED_HOLD_THRESHOLD:
            return 0.0
        return value

    def _set_manual_gripper_drive_override(self, arm_index: int, gripper_value: float) -> None:
        setter = getattr(self.base_env, "set_manual_gripper_drive_override", None)
        if setter is None:
            return
        if float(gripper_value) <= self.MANUAL_GRIPPER_CLOSED_HOLD_THRESHOLD:
            setter(int(arm_index), self.MANUAL_GRIPPER_CLOSED_DRIVE_VALUE)
        else:
            setter(int(arm_index), None)

    def _clear_manual_gripper_drive_overrides(self) -> None:
        clearer = getattr(self.base_env, "clear_manual_gripper_drive_overrides", None)
        if clearer is not None:
            clearer()

    def _disable_policy_grip_guard(self) -> None:
        if self._policy_grip_guard_arm_index is not None:
            self._set_manual_gripper_drive_override(self._policy_grip_guard_arm_index, 1.0)
        self._policy_grip_guard_arm_index = None
        self._policy_grip_guard_pose_active = False
        self._manual_force_guarded_box_hold = False

    def _enable_policy_grip_guard_if_closed(self, arm_index: int) -> None:
        arm_index = int(arm_index)
        if float(self.solver.ctrl.current_gripper_value(arm_index)) > self.MANUAL_GRIPPER_CLOSED_HOLD_THRESHOLD:
            return
        self._policy_grip_guard_arm_index = arm_index
        self._policy_grip_guard_pose_active = False
        self.solver.ctrl.commanded_gripper[arm_index] = 0.0
        self._remember_manual_gripper_hold_target(arm_index)
        self._set_manual_gripper_drive_override(arm_index, 0.0)
        if self.debug:
            print(f"[manual] policy grip guard enabled: {self._arm_name(arm_index)}")

    def _restore_policy_grip_guard_override(self) -> None:
        guarded_arm_index = self._policy_grip_guard_arm_index
        if guarded_arm_index is None:
            return
        guarded_arm_index = int(guarded_arm_index)
        self.solver.ctrl.commanded_gripper[guarded_arm_index] = 0.0
        self._set_manual_gripper_drive_override(guarded_arm_index, 0.0)

    def _opposite_gripper_has_taken_over(self, guarded_arm_index: int) -> bool:
        other_arm_index = 1 - int(guarded_arm_index)
        if float(self.solver.ctrl.current_gripper_value(other_arm_index)) > self.MANUAL_GRIPPER_CLOSED_HOLD_THRESHOLD:
            return False
        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        other_pose = np.asarray(self.solver.ctrl.current_gripper_pose(other_arm_index), dtype=np.float64)
        distance = float(np.linalg.norm(other_pose[:3] - box_pose[:3]))
        return distance <= self.POLICY_GRIP_GUARD_TAKEOVER_DISTANCE

    def _opposite_gripper_box_distance(self, guarded_arm_index: int) -> float:
        other_arm_index = 1 - int(guarded_arm_index)
        box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        other_pose = np.asarray(self.solver.ctrl.current_gripper_pose(other_arm_index), dtype=np.float64)
        return float(np.linalg.norm(other_pose[:3] - box_pose[:3]))

    def _ensure_policy_pose_guard_if_needed(self, guarded_arm_index: int) -> bool:
        guarded_arm_index = int(guarded_arm_index)
        if self._policy_grip_guard_pose_active:
            return True
        if self._opposite_gripper_box_distance(guarded_arm_index) > self.POLICY_POSE_GUARD_APPROACH_DISTANCE:
            return False

        # The left arm must remain free while it transports the box to the
        # handover pose. Once the opposite gripper is close enough to approach,
        # latch the current pose and hold it until takeover.
        self._remember_manual_hold_target(guarded_arm_index)
        self._remember_manual_gripper_hold_target(guarded_arm_index)
        self._policy_grip_guard_pose_active = True
        if self.debug:
            print(f"[manual] policy pose guard enabled: {self._arm_name(guarded_arm_index)}")
        return True

    def _apply_policy_grip_guard(self, action: np.ndarray) -> np.ndarray:
        guarded_arm_index = self._policy_grip_guard_arm_index
        if guarded_arm_index is None:
            return action
        if self._opposite_gripper_has_taken_over(guarded_arm_index):
            if self.debug:
                print(f"[manual] policy grip guard released: {self._arm_name(guarded_arm_index)}")
            self._disable_policy_grip_guard()
            return action

        guarded_action = np.asarray(action, dtype=np.float32).copy()
        guarded_arm_index = int(guarded_arm_index)
        if guarded_arm_index == 0:
            gripper_index = 6
        else:
            gripper_index = 13
        guarded_action[gripper_index] = 0.0
        self.solver.ctrl.commanded_gripper[guarded_arm_index] = 0.0
        self._set_manual_gripper_drive_override(guarded_arm_index, 0.0)

        if self._ensure_policy_pose_guard_if_needed(guarded_arm_index):
            guarded_qpos = self._manual_hold_target(guarded_arm_index)
            guarded_gripper_qpos = self._manual_gripper_hold_target(guarded_arm_index)
            if guarded_arm_index == 0:
                guarded_action[0:6] = guarded_qpos.astype(np.float32, copy=False)
            else:
                guarded_action[7:13] = guarded_qpos.astype(np.float32, copy=False)
            self.solver.ctrl.commanded_arm_qpos[guarded_arm_index] = guarded_qpos.copy()
            self._freeze_arm_qpos(guarded_arm_index, guarded_qpos)
            self._freeze_gripper_qpos(guarded_arm_index, guarded_gripper_qpos)
            self._zero_arm_velocity(guarded_arm_index)
        return guarded_action

    def _restore_policy_grip_guard_state(self) -> None:
        guarded_arm_index = self._policy_grip_guard_arm_index
        if guarded_arm_index is None:
            return
        guarded_arm_index = int(guarded_arm_index)
        if self._opposite_gripper_has_taken_over(guarded_arm_index):
            return
        self.solver.ctrl.commanded_gripper[guarded_arm_index] = 0.0
        self._set_manual_gripper_drive_override(guarded_arm_index, 0.0)
        if self._ensure_policy_pose_guard_if_needed(guarded_arm_index):
            guarded_qpos = self._manual_hold_target(guarded_arm_index)
            guarded_gripper_qpos = self._manual_gripper_hold_target(guarded_arm_index)
            self.solver.ctrl.commanded_arm_qpos[guarded_arm_index] = guarded_qpos.copy()
            self._freeze_arm_qpos(guarded_arm_index, guarded_qpos)
            self._freeze_gripper_qpos(guarded_arm_index, guarded_gripper_qpos)
            self._zero_arm_velocity(guarded_arm_index)

    @staticmethod
    def _manual_terminal_status(result, default_status: str) -> str | None:
        if result.truncated:
            return "episode_limit_reached"
        if result.terminated:
            return default_status
        return None

    def _zero_arm_velocity(self, arm_index: int) -> None:
        robot = self.base_env.get_arm_robot(int(arm_index))
        qvel = robot.get_qvel().clone()
        qvel[...] = 0.0
        robot.set_qvel(qvel)

    def _freeze_arm_qpos(self, arm_index: int, arm_qpos: np.ndarray) -> None:
        arm_index = int(arm_index)
        robot = self.base_env.get_arm_robot(arm_index)
        arm_joints = self.base_env.left_arm_joints if arm_index == 0 else self.base_env.right_arm_joints
        full_qpos = robot.get_qpos().clone()
        full_qvel = robot.get_qvel().clone()
        arm_qpos_tensor = torch.as_tensor(
            np.asarray(arm_qpos, dtype=np.float64),
            device=full_qpos.device,
            dtype=full_qpos.dtype,
        ).reshape(1, -1)
        if full_qpos.ndim == 1:
            full_qpos[: self.base_env.ARM_DOF] = arm_qpos_tensor.reshape(-1)[: self.base_env.ARM_DOF]
            full_qvel[: self.base_env.ARM_DOF] = 0.0
            drive_target = full_qpos[: self.base_env.ARM_DOF].reshape(1, -1)
        else:
            full_qpos[:, : self.base_env.ARM_DOF] = arm_qpos_tensor[:, : self.base_env.ARM_DOF].expand(
                full_qpos.shape[0],
                -1,
            )
            full_qvel[:, : self.base_env.ARM_DOF] = 0.0
            drive_target = full_qpos[:, : self.base_env.ARM_DOF]
        robot.set_qpos(full_qpos)
        robot.set_qvel(full_qvel)
        robot.set_joint_drive_targets(drive_target, joints=arm_joints)
        robot.set_joint_drive_velocity_targets(torch.zeros_like(drive_target), joints=arm_joints)

    def _freeze_gripper_qpos(self, arm_index: int, gripper_qpos: np.ndarray) -> None:
        arm_index = int(arm_index)
        robot = self.base_env.get_arm_robot(arm_index)
        gripper_joints = self.base_env.left_gripper if arm_index == 0 else self.base_env.right_gripper
        gripper_qpos = np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)
        if gripper_qpos.size == 0 or len(gripper_joints) == 0:
            return
        full_qpos = robot.get_qpos().clone()
        full_qvel = robot.get_qvel().clone()
        target_qpos = torch.as_tensor(
            gripper_qpos[: len(gripper_joints)],
            device=full_qpos.device,
            dtype=full_qpos.dtype,
        ).reshape(1, -1)
        start = int(self.base_env.ARM_DOF)
        end = start + int(target_qpos.shape[1])
        if full_qpos.ndim == 1:
            if full_qpos.shape[0] < end:
                return
            full_qpos[start:end] = target_qpos.reshape(-1)
            full_qvel[start:end] = 0.0
        else:
            if full_qpos.shape[1] < end:
                return
            full_qpos[:, start:end] = target_qpos.expand(full_qpos.shape[0], -1)
            full_qvel[:, start:end] = 0.0
        robot.set_qpos(full_qpos)
        robot.set_qvel(full_qvel)

    def _lock_guarded_arm_for_manual_step(self, active_arm_index: int) -> None:
        guarded_arm_index = self._policy_grip_guard_arm_index
        if guarded_arm_index is None or int(guarded_arm_index) == int(active_arm_index):
            return
        guarded_arm_index = int(guarded_arm_index)
        guarded_qpos = self._manual_hold_target(guarded_arm_index)
        guarded_gripper_qpos = self._manual_gripper_hold_target(guarded_arm_index)
        self.solver.ctrl.commanded_arm_qpos[guarded_arm_index] = guarded_qpos.copy()
        self.solver.ctrl.commanded_gripper[guarded_arm_index] = 0.0
        self._set_manual_gripper_drive_override(guarded_arm_index, 0.0)
        self._freeze_arm_qpos(guarded_arm_index, guarded_qpos)
        self._freeze_gripper_qpos(guarded_arm_index, guarded_gripper_qpos)
        self._zero_arm_velocity(guarded_arm_index)

    def _set_box_pose_and_zero_velocity(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        device = getattr(self.base_env, "device", None)
        p = torch.as_tensor(pose[:3], device=device, dtype=torch.float32).reshape(1, 3)
        q = torch.as_tensor(pose[3:7], device=device, dtype=torch.float32).reshape(1, 4)
        self.base_env.box.set_pose(Pose.create_from_pq(p=p, q=q))
        zero_vec = torch.zeros(3, device=device, dtype=torch.float32)
        try:
            self.base_env.box.set_linear_velocity(zero_vec)
            self.base_env.box.set_angular_velocity(zero_vec)
        except Exception:
            zero_batch = zero_vec.reshape(1, 3)
            self.base_env.box.set_linear_velocity(zero_batch)
            self.base_env.box.set_angular_velocity(zero_batch)

    def _sync_manual_state_to_sim(self) -> None:
        scene = getattr(self.base_env, "scene", None)
        if scene is None or not bool(getattr(scene, "gpu_sim_enabled", False)):
            return
        if bool(getattr(scene, "_needs_fetch", False)):
            scene._gpu_fetch_all()
        scene._gpu_apply_all()
        scene._gpu_fetch_all()

    def _apply_manual_guarded_box_hold(self, active_arm_index: int) -> None:
        guarded_arm_index = self._policy_grip_guard_arm_index
        if guarded_arm_index is None:
            self._manual_guarded_box_relative_anchor = None
            return
        guarded_arm_index = int(guarded_arm_index)
        if int(guarded_arm_index) == int(active_arm_index) and not self._manual_force_guarded_box_hold:
            self._manual_guarded_box_relative_anchor = None
            return
        if self._opposite_gripper_has_taken_over(guarded_arm_index):
            self._manual_guarded_box_relative_anchor = None
            return
        guarded_pose = np.asarray(self.solver.ctrl.current_gripper_pose(guarded_arm_index), dtype=np.float64)
        guarded_matrix = pose_to_matrix(guarded_pose)
        if self._manual_guarded_box_relative_anchor is None:
            box_matrix = pose_to_matrix(np.asarray(self.solver._box_pose(), dtype=np.float64))
            self._manual_guarded_box_relative_anchor = np.linalg.inv(guarded_matrix) @ box_matrix
            if self.debug:
                print(
                    "[manual] guarded box hold enabled: "
                    f"{self._arm_name(guarded_arm_index)} holds box while "
                    f"{self._arm_name(active_arm_index)} moves"
                )
        box_pose = matrix_to_pose(guarded_matrix @ self._manual_guarded_box_relative_anchor)
        self._set_box_pose_and_zero_velocity(box_pose)

    def _manual_step_and_record(
        self,
        action: np.ndarray,
        recorder,
        *,
        active_arm_index: int,
    ) -> tuple[Any, bool]:
        if self.solver.ctrl.steps_taken >= self.solver.ctrl.step_limit:
            raise EpisodeLimitReached
        action = np.asarray(action, dtype=np.float32).reshape(self.base_env.ACTION_DIM)
        self.solver.ctrl.commanded_arm_qpos[0] = action[0:6].astype(np.float64, copy=True)
        self.solver.ctrl.commanded_gripper[0] = float(action[6])
        self.solver.ctrl.commanded_arm_qpos[1] = action[7:13].astype(np.float64, copy=True)
        self.solver.ctrl.commanded_gripper[1] = float(action[13])

        self._lock_guarded_arm_for_manual_step(active_arm_index)
        self._apply_manual_guarded_box_hold(active_arm_index)
        self._sync_manual_state_to_sim()

        result = self.backend.step(ActionSpec.from_value(action))

        self._lock_guarded_arm_for_manual_step(active_arm_index)
        self._apply_manual_guarded_box_hold(active_arm_index)
        self._sync_manual_state_to_sim()

        recorder.append_step(action, dict(result.observation.data))
        self.solver.ctrl.steps_taken += 1
        self._maybe_render()
        success = self.solver.ctrl.success_detection_enabled and _to_bool_scalar(result.info.get("success", False))
        return result, success

    def _manual_action(
        self,
        active_arm_index: int,
        *,
        active_qpos: np.ndarray | None = None,
        inactive_qpos: np.ndarray | None = None,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> np.ndarray:
        active_arm_index = int(active_arm_index)
        inactive_arm_index = 1 - active_arm_index
        if active_qpos is None:
            active_qpos = self.solver.ctrl.current_qpos(active_arm_index)
        if inactive_qpos is None:
            inactive_qpos = self._manual_hold_target(inactive_arm_index)
        if left_gripper is None:
            left_gripper = self._manual_gripper_hold_value(0)
        if right_gripper is None:
            right_gripper = self._manual_gripper_hold_value(1)
        self._set_manual_gripper_drive_override(0, float(left_gripper))
        self._set_manual_gripper_drive_override(1, float(right_gripper))
        if active_arm_index == 0:
            return self.solver.ctrl.compose_action(
                left_qpos=active_qpos,
                right_qpos=inactive_qpos,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
            )
        return self.solver.ctrl.compose_action(
            left_qpos=inactive_qpos,
            right_qpos=active_qpos,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
        )

    def _manual_move_to_qpos(
        self,
        arm_index: int,
        target_qpos: np.ndarray,
        recorder,
        *,
        steps: int,
        settle_steps: int,
    ) -> bool:
        start_qpos = self.solver.ctrl.current_qpos(arm_index)
        inactive_qpos = self._manual_hold_target(1 - int(arm_index))
        inactive_start_pose = np.asarray(self.solver.ctrl.current_gripper_pose(1 - int(arm_index)), dtype=np.float64)
        target_qpos = np.asarray(target_qpos, dtype=np.float64).reshape(start_qpos.shape)
        trajectory = np.linspace(start_qpos, target_qpos, num=int(steps)).astype(np.float64, copy=False)
        for qpos in trajectory:
            self._freeze_arm_qpos(1 - int(arm_index), inactive_qpos)
            self._zero_arm_velocity(1 - int(arm_index))
            self._lock_guarded_arm_for_manual_step(arm_index)
            action = self._manual_action(arm_index, active_qpos=qpos, inactive_qpos=inactive_qpos)
            result, success = self._manual_step_and_record(action, recorder, active_arm_index=arm_index)
            self._lock_guarded_arm_for_manual_step(arm_index)
            if success or self.solver.ctrl.current_success():
                self.solver.ctrl.last_status = "success"
                return True
            terminal_status = self._manual_terminal_status(result, "terminated_during_qpos_move")
            if terminal_status is not None:
                self.solver.ctrl.last_status = terminal_status
                return False
        for _ in range(int(settle_steps)):
            self._freeze_arm_qpos(1 - int(arm_index), inactive_qpos)
            self._zero_arm_velocity(1 - int(arm_index))
            self._lock_guarded_arm_for_manual_step(arm_index)
            action = self._manual_action(arm_index, active_qpos=target_qpos, inactive_qpos=inactive_qpos)
            result, success = self._manual_step_and_record(action, recorder, active_arm_index=arm_index)
            self._lock_guarded_arm_for_manual_step(arm_index)
            if success or self.solver.ctrl.current_success():
                self.solver.ctrl.last_status = "success"
                return True
            terminal_status = self._manual_terminal_status(result, "terminated_during_qpos_move")
            if terminal_status is not None:
                self.solver.ctrl.last_status = terminal_status
                return False
        self._sync_manual_arm_targets()
        self._remember_manual_hold_target(arm_index, target_qpos)
        inactive_end_pose = np.asarray(self.solver.ctrl.current_gripper_pose(1 - int(arm_index)), dtype=np.float64)
        self.solver.ctrl.last_plan_debug["inactive_gripper_drift"] = round(
            float(np.linalg.norm(inactive_end_pose[:3] - inactive_start_pose[:3])),
            6,
        )
        self.solver.ctrl.last_plan_debug["inactive_start_gripper_pose"] = inactive_start_pose.round(4).tolist()
        self.solver.ctrl.last_plan_debug["inactive_end_gripper_pose"] = inactive_end_pose.round(4).tolist()
        return False

    def _manual_hold_pose(
        self,
        recorder,
        *,
        active_arm_index: int,
        steps: int,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> bool:
        active_qpos = self.solver.ctrl.current_qpos(active_arm_index)
        inactive_qpos = self._manual_hold_target(1 - int(active_arm_index))
        for _ in range(int(steps)):
            self._freeze_arm_qpos(active_arm_index, active_qpos)
            self._freeze_arm_qpos(1 - int(active_arm_index), inactive_qpos)
            self._zero_arm_velocity(active_arm_index)
            self._zero_arm_velocity(1 - int(active_arm_index))
            self._lock_guarded_arm_for_manual_step(active_arm_index)
            action = self._manual_action(
                active_arm_index,
                active_qpos=active_qpos,
                inactive_qpos=inactive_qpos,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
            )
            result, success = self._manual_step_and_record(action, recorder, active_arm_index=active_arm_index)
            self._lock_guarded_arm_for_manual_step(active_arm_index)
            if success or self.solver.ctrl.current_success():
                self.solver.ctrl.last_status = "success"
                return True
            terminal_status = self._manual_terminal_status(result, "terminated_during_hold")
            if terminal_status is not None:
                self.solver.ctrl.last_status = terminal_status
                return False
        self._sync_manual_arm_targets()
        self._remember_manual_hold_target(active_arm_index, active_qpos)
        return False

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

    def _manual_joint_limit_reject(
        self,
        arm_index: int,
        target_arm_qpos: np.ndarray,
        plan_debug: dict[str, Any],
    ) -> bool:
        robot = self.base_env.get_arm_robot(int(arm_index))
        qlimits = to_numpy(robot.get_qlimits())
        if qlimits.ndim == 3:
            qlimits = qlimits[0]
        target_arm_qpos = np.asarray(target_arm_qpos, dtype=np.float64)
        if qlimits.shape[0] < target_arm_qpos.shape[0]:
            return False
        lower = np.asarray(qlimits[: target_arm_qpos.shape[0], 0], dtype=np.float64)
        upper = np.asarray(qlimits[: target_arm_qpos.shape[0], 1], dtype=np.float64)
        clipped_qpos = np.minimum(np.maximum(target_arm_qpos, lower), upper)
        clip_norm = float(np.linalg.norm(clipped_qpos - target_arm_qpos))
        plan_debug["joint_limit_clip_norm"] = round(clip_norm, 6)
        if clip_norm <= self.MANUAL_JOINT_LIMIT_CLIP_TOL:
            return False
        plan_debug["clipped_target_arm_qpos"] = clipped_qpos.round(4).tolist()
        plan_debug["guard_reason"] = "joint_limit_clip"
        return True

    def _manual_auto_align_step(self, arm_index: int) -> float:
        if int(arm_index) == 1:
            return self.MANUAL_RIGHT_AUTO_ALIGN_STEP
        return self.MANUAL_AUTO_ALIGN_STEP

    def _manual_auto_align_plan_steps(self, arm_index: int) -> int:
        if int(arm_index) == 0:
            return int(self.MANUAL_LEFT_AUTO_ALIGN_PLAN_STEPS)
        return int(self.MANUAL_AUTO_ALIGN_PLAN_STEPS)

    def _manual_auto_align_box_move_reject(self, arm_index: int) -> float:
        if int(arm_index) == 0:
            return float(self.MANUAL_LEFT_AUTO_ALIGN_BOX_MOVE_REJECT)
        return float(self.MANUAL_RIGHT_AUTO_ALIGN_BOX_MOVE_REJECT)

    def _manual_auto_align_box_z_drop_reject(self, arm_index: int) -> float:
        if int(arm_index) == 0:
            return float(self.MANUAL_LEFT_AUTO_ALIGN_BOX_Z_DROP_REJECT)
        return float(self.MANUAL_RIGHT_AUTO_ALIGN_BOX_Z_DROP_REJECT)

    @staticmethod
    def _table_horizontal_pose(pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(7)
        matrix = pose_to_matrix(pose)
        x_axis = matrix[:3, 0].copy()
        x_axis[2] = 0.0
        norm = float(np.linalg.norm(x_axis))
        if norm < 1e-6:
            x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            x_axis /= norm
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        horizontal = np.eye(4, dtype=np.float64)
        horizontal[:3, :3] = rotation
        horizontal[:3, 3] = matrix[:3, 3]
        return matrix_to_pose(horizontal)

    def _right_horizontal_grasp_pose_from_box_pose(
        self,
        box_pose: np.ndarray,
        contact_id: int,
        pre_distance: float,
    ) -> np.ndarray:
        pose = self.solver._grasp_pose_from_box_pose(box_pose, contact_id, pre_distance)
        return self._table_horizontal_pose(pose)

    def _right_horizontal_pre_pose_candidates_from_box_pose(
        self,
        box_pose: np.ndarray,
        contact_id: int,
        pre_distance: float,
        arm_index: int,
    ) -> list[np.ndarray]:
        base_pose = self._right_horizontal_grasp_pose_from_box_pose(box_pose, contact_id, pre_distance)
        center_pose = self.solver._box_contact_world_pose(box_pose, contact_id)
        return [
            self.solver._rotate_pose_about_world_z(base_pose, center_pose, float(theta))
            for theta in self.MANUAL_RIGHT_HORIZONTAL_YAW_VARIANTS
        ]

    def _manual_box_relative_delta(self, arm_index: int, command: str) -> np.ndarray:
        step = self._manual_translation_step(arm_index)
        lift_step = self._manual_lift_step(arm_index)
        if command in ("r", "u"):
            return np.asarray([0.0, 0.0, lift_step], dtype=np.float64)
        if command == "f":
            return np.asarray([0.0, 0.0, -lift_step], dtype=np.float64)
        # Manual keyboard control should stay in the table frame.  Box-relative
        # directions become unintuitive once the gripper points down or crosses
        # the box, which is exactly the handover/repair case.
        if command == "w":
            delta_xy = np.asarray([0.0, step], dtype=np.float64)
        elif command == "s":
            delta_xy = np.asarray([0.0, -step], dtype=np.float64)
        elif command == "a":
            delta_xy = np.asarray([-step, 0.0], dtype=np.float64)
        elif command == "d":
            delta_xy = np.asarray([step, 0.0], dtype=np.float64)
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
                selection_mode="prefer_low_motion",
                max_grasp_plan_length=self.solver.LEFT_FIRST_GRASP_MAX_GRASP_PLAN_LENGTH,
            )
            return candidate, {
                "arm_index": 0,
                "mode": "left_first_grasp_candidate",
                "box_pose": box_pose.round(4).tolist(),
                "attempts": list(self.solver.ctrl.last_grasp_attempts),
            }

        if self.MANUAL_RIGHT_HORIZONTAL_GRASP:
            candidate = self.solver.ctrl.choose_grasp_candidate(
                desired_pose=box_pose,
                arm_index=1,
                contact_ids=tuple(RIGHT_GRASP_PREFERRED_X_AXIS_CONTACT_IDS) or tuple(RIGHT_GRASP_CONTACT_IDS),
                pre_distance=self.solver.HANDOVER_PRE_GRASP_DISTANCE,
                grasp_distance=self.solver.RIGHT_HANDOVER_GRASP_DISTANCE,
                pose_builder=self._right_horizontal_grasp_pose_from_box_pose,
                preferred_quat=self._table_horizontal_pose(self.solver.ctrl.current_gripper_pose(1))[3:],
                preferred_top_down_quat=None,
                pre_pose_candidates_builder=self._right_horizontal_pre_pose_candidates_from_box_pose,
                selection_mode="prefer_side",
            )
            selection_debug = list(self.solver.ctrl.last_grasp_attempts)
            selection_reason = "manual_right_table_horizontal"
            selection_options = [
                {
                    "contact_id": int(item.get("contact_id", -1)),
                    "variant_index": int(item.get("variant_index", -1)),
                    "pre_reason": item.get("pre_reason"),
                    "grasp_reason": item.get("grasp_reason"),
                    "pre_pose": item.get("pre_pose"),
                    "grasp_pose": item.get("grasp_pose"),
                }
                for item in selection_debug
            ]
        else:
            candidate, _place_candidate, selection_debug, selection_reason, selection_options = (
                self.solver._select_right_handover_candidate(box_pose)
            )
        return candidate, {
            "arm_index": 1,
            "mode": "right_handover_candidate",
            "right_horizontal_grasp": bool(self.MANUAL_RIGHT_HORIZONTAL_GRASP),
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

    def _servo_manual_candidate(self, arm_index: int, recorder, candidate) -> str | None:
        self._sync_manual_arm_targets()

        current_pose = np.asarray(self.solver.ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        pre_pose = np.asarray(candidate.pre_pose, dtype=np.float64)
        grasp_pose = np.asarray(candidate.grasp_pose, dtype=np.float64)
        pre_distance = float(np.linalg.norm(pre_pose[:3] - current_pose[:3]))
        target_phase = "grasp" if pre_distance <= self.MANUAL_AUTO_ALIGN_GRASP_SWITCH_DISTANCE else "pre"
        target_pose = grasp_pose if target_phase == "grasp" else pre_pose
        plan = candidate.grasp_plan if target_phase == "grasp" else candidate.pre_plan
        plan_positions = np.asarray(plan.get("position", []), dtype=np.float64)
        if plan_positions.ndim != 2 or len(plan_positions) == 0:
            self.solver.ctrl.last_plan_debug = {
                "arm_index": int(arm_index),
                "target_phase": target_phase,
                "current_gripper_pose": current_pose.round(4).tolist(),
                "target_gripper_pose": target_pose.round(4).tolist(),
                "guard_reason": "empty_plan",
            }
            self.solver.ctrl.last_status = "manual_auto_align_failed"
            return None

        shoulder_pan_limit = self._manual_shoulder_pan_limit(arm_index)
        joint_motion_limit = self._manual_joint_motion_limit(arm_index)

        inactive_qpos = self._manual_hold_target(1 - int(arm_index))
        current_arm_qpos = self.solver.ctrl.current_qpos(arm_index)
        initial_box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
        executed_steps = 0
        skipped_steps = 0
        max_steps = self._manual_auto_align_plan_steps(arm_index)
        box_move_reject = self._manual_auto_align_box_move_reject(arm_index)
        box_z_drop_reject = self._manual_auto_align_box_z_drop_reject(arm_index)

        for waypoint in plan_positions:
            target_arm_qpos = self.solver.ctrl._arm_qpos_from_plan(waypoint)
            joint_delta = target_arm_qpos - current_arm_qpos
            shoulder_pan_delta = float(np.abs(joint_delta[0]))
            joint_motion = float(np.linalg.norm(joint_delta))
            if joint_motion < 1e-4:
                skipped_steps += 1
                continue

            self.solver.ctrl.last_plan_debug = {
                "arm_index": int(arm_index),
                "target_phase": target_phase,
                "path_source": "planner_waypoints",
                "current_gripper_pose": current_pose.round(4).tolist(),
                "target_gripper_pose": target_pose.round(4).tolist(),
                "target_arm_qpos": np.asarray(target_arm_qpos, dtype=np.float64).round(4).tolist(),
                "shoulder_pan_delta": round(shoulder_pan_delta, 4),
                "joint_motion": round(joint_motion, 4),
                "shoulder_pan_limit": round(float(shoulder_pan_limit), 4),
                "joint_motion_limit": round(float(joint_motion_limit), 4),
                "plan_len": int(len(plan_positions)),
                "executed_steps": int(executed_steps),
                "skipped_steps": int(skipped_steps),
            }
            if shoulder_pan_delta > shoulder_pan_limit or joint_motion > joint_motion_limit:
                self.solver.ctrl.last_plan_debug["guard_reason"] = "large_rotation"
                self.solver.ctrl.last_status = "manual_move_rejected"
                return None
            if self._manual_joint_limit_reject(arm_index, target_arm_qpos, self.solver.ctrl.last_plan_debug):
                self.solver.ctrl.last_status = "manual_move_rejected"
                return None

            self._freeze_arm_qpos(1 - int(arm_index), inactive_qpos)
            self._zero_arm_velocity(1 - int(arm_index))
            self._lock_guarded_arm_for_manual_step(arm_index)
            action = self._manual_action(
                arm_index,
                active_qpos=target_arm_qpos,
                inactive_qpos=inactive_qpos,
            )
            result, success = self._manual_step_and_record(action, recorder, active_arm_index=arm_index)
            self._lock_guarded_arm_for_manual_step(arm_index)
            if success or self.solver.ctrl.current_success():
                self.solver.ctrl.last_status = "success"
                return self.solver.ctrl.last_status
            terminal_status = self._manual_terminal_status(result, "terminated_during_qpos_move")
            if terminal_status is not None:
                self.solver.ctrl.last_status = terminal_status
                return self.solver.ctrl.last_status

            box_pose_after = np.asarray(self.solver._box_pose(), dtype=np.float64)
            box_displacement = float(np.linalg.norm(box_pose_after[:3] - initial_box_pose[:3]))
            box_z_drop = float(initial_box_pose[2] - box_pose_after[2])
            self.solver.ctrl.last_plan_debug["box_pose_before"] = initial_box_pose.round(4).tolist()
            self.solver.ctrl.last_plan_debug["box_pose_after"] = box_pose_after.round(4).tolist()
            self.solver.ctrl.last_plan_debug["box_displacement"] = round(box_displacement, 4)
            self.solver.ctrl.last_plan_debug["box_z_drop"] = round(box_z_drop, 4)
            self.solver.ctrl.last_plan_debug["box_move_reject"] = round(float(box_move_reject), 4)
            self.solver.ctrl.last_plan_debug["box_z_drop_reject"] = round(float(box_z_drop_reject), 4)
            if (
                box_displacement > box_move_reject
                or box_z_drop > box_z_drop_reject
            ):
                self.solver.ctrl.last_plan_debug["guard_reason"] = "auto_align_box_collision"
                self.solver.ctrl.last_status = "manual_move_rejected"
                return None

            current_arm_qpos = target_arm_qpos
            executed_steps += 1
            if executed_steps >= max_steps:
                break

        if executed_steps == 0:
            self.solver.ctrl.last_plan_debug = {
                "arm_index": int(arm_index),
                "target_phase": target_phase,
                "path_source": "planner_waypoints",
                "current_gripper_pose": current_pose.round(4).tolist(),
                "target_gripper_pose": target_pose.round(4).tolist(),
                "plan_len": int(len(plan_positions)),
                "skipped_steps": int(skipped_steps),
                "guard_reason": "no_progress",
            }
            self.solver.ctrl.last_status = "manual_auto_align_failed"
            return None

        self.solver.ctrl.last_plan_debug["executed_steps"] = int(executed_steps)
        self.solver.ctrl.last_plan_debug["max_steps"] = int(max_steps)
        self._sync_manual_arm_targets()
        self._remember_manual_hold_target(arm_index, current_arm_qpos)
        return None

    def _print_manual_help(self) -> None:
        print(
            "[manual] commands: "
            "viewer: up/down or w/s=table +Y/-Y  left/right or a/d=table -X/+X  u/j or r/f=up/down  "
            "g=auto-align  1=left  2=right  o=open  c=close  p=status  h=help  n=resume  q=abort | "
            "terminal fallback: w/s=table +Y/-Y  a/d=table -X/+X  r/f=up/down  "
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
            self._manual_trigger_down_latched = False
            return False
        if bool(window.key_press(self.MANUAL_TRIGGER_KEY)):
            self._manual_trigger_down_latched = True
            return True

        key_down = getattr(window, "key_down", None)
        if key_down is None:
            self._manual_trigger_down_latched = False
            return False

        is_down = bool(key_down(self.MANUAL_TRIGGER_KEY))
        if not is_down:
            self._manual_trigger_down_latched = False
            return False
        if self._manual_trigger_down_latched:
            return False
        self._manual_trigger_down_latched = True
        return True

    def _manual_trigger_key_is_down(self) -> bool:
        if getattr(self.backend, "render_mode", None) != "human":
            return False
        viewer = getattr(self.base_env, "viewer", None)
        if viewer is None:
            return False
        window = getattr(viewer, "window", None)
        if window is None:
            return False
        key_down = getattr(window, "key_down", None)
        if key_down is None:
            return False
        return bool(key_down(self.MANUAL_TRIGGER_KEY))

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
        if self.solver.ctrl.last_status in ("manual_move_failed", "manual_auto_align_failed"):
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
        force_table_horizontal: bool = False,
        min_z_progress: float | None = None,
        return_nonfatal_status: bool = False,
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
            if force_table_horizontal or (int(arm_index) == 1 and self.MANUAL_RIGHT_HORIZONTAL_GRASP):
                target_gripper_pose = self._table_horizontal_pose(target_gripper_pose)
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
            start_arm_qpos = current_arm_qpos.copy()
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
            if self._manual_joint_limit_reject(arm_index, target_arm_qpos, plan_debug):
                last_reject_debug = plan_debug
                continue

            start_pose = current_gripper_pose.copy()
            self.solver.ctrl.last_plan_debug = plan_debug
            self._manual_move_to_qpos(
                arm_index,
                target_arm_qpos,
                recorder,
                steps=self.MANUAL_MOVE_STEPS,
                settle_steps=self.MANUAL_SETTLE_STEPS,
            )
            end_pose = np.asarray(self.solver.ctrl.current_gripper_pose(arm_index), dtype=np.float64)
            actual_delta = end_pose[:3] - start_pose[:3]
            self.solver.ctrl.last_plan_debug["actual_gripper_pose"] = end_pose.round(4).tolist()
            self.solver.ctrl.last_plan_debug["actual_delta"] = actual_delta.round(4).tolist()
            requested_z = float(scaled_delta[2])
            actual_z = float(actual_delta[2])
            vertical_guard_command = command in ("r", "f", "u", "handover_holder_lift")
            if self.debug and vertical_guard_command:
                print(
                    "[manual] command "
                    f"`{command}` requested_delta={np.asarray(scaled_delta, dtype=np.float64).round(4).tolist()} "
                    f"actual_delta={actual_delta.round(4).tolist()}"
                )
            min_required_z_progress = (
                float(min_z_progress)
                if min_z_progress is not None
                else max(
                    self.MANUAL_VERTICAL_MIN_PROGRESS,
                    abs(requested_z) * self.MANUAL_VERTICAL_PROGRESS_RATIO,
                )
            )
            if (
                vertical_guard_command
                and abs(requested_z) > 1e-6
                and actual_z * np.sign(requested_z)
                < min_required_z_progress
            ):
                restore_debug = dict(self.solver.ctrl.last_plan_debug)
                restore_debug["guard_reason"] = "vertical_direction_mismatch"
                restore_debug["requested_z"] = round(requested_z, 4)
                restore_debug["actual_z"] = round(actual_z, 4)
                restore_debug["min_required_z_progress"] = round(min_required_z_progress, 4)
                self._manual_move_to_qpos(
                    arm_index,
                    start_arm_qpos,
                    recorder,
                    steps=self.MANUAL_MOVE_STEPS,
                    settle_steps=0,
                )
                self.solver.ctrl.last_plan_debug = restore_debug
                self.solver.ctrl.last_plan_debug["guard_reason"] = "vertical_direction_mismatch"
                self.solver.ctrl.last_plan_debug["requested_z"] = round(requested_z, 4)
                self.solver.ctrl.last_plan_debug["actual_z"] = round(actual_z, 4)
                self.solver.ctrl.last_status = "manual_move_rejected"
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
        if return_nonfatal_status and self.solver.ctrl.last_status in (
            "manual_move_failed",
            "manual_move_rejected",
        ):
            return self.solver.ctrl.last_status
        if self._handle_nonfatal_manual_failure(command):
            return None
        if self.solver.ctrl.last_status in ("manual_move_rejected",):
            if return_nonfatal_status:
                return self.solver.ctrl.last_status
            failed = self.last_debug_info.setdefault("failed_commands", [])
            failed.append(
                {
                    "command": str(command),
                    "status": str(self.solver.ctrl.last_status),
                    "plan_debug": dict(self.solver.ctrl.last_plan_debug),
                }
            )
            reason = self.solver.ctrl.last_plan_debug.get("guard_reason", "manual_move_rejected")
            print(f"[manual] command `{command}` rejected: {reason}")
            self.solver.ctrl.last_status = "running"
            self.solver.ctrl.last_plan_debug = {}
            return None
        if self.solver.ctrl.last_status != "running":
            return self.solver.ctrl.last_status
        return None

    def _run_manual_command(self, command: str, recorder, *, arm_index: int) -> str | None:
        self._sync_manual_arm_targets()
        if self._guarded_arm_manual_command_blocked(arm_index, command):
            print(
                "[manual] guarded left arm is holding the box before right takeover; "
                "switch to right arm with `2` for handover control"
            )
            return None
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
            terminal_status = self._servo_manual_candidate(arm_index, recorder, candidate)
            if self._handle_nonfatal_manual_failure("g"):
                return None
            if self.solver.ctrl.last_status in ("manual_move_rejected",):
                failed = self.last_debug_info.setdefault("failed_commands", [])
                failed.append(
                    {
                        "command": "g",
                        "status": str(self.solver.ctrl.last_status),
                        "plan_debug": dict(self.solver.ctrl.last_plan_debug),
                    }
                )
                reason = self.solver.ctrl.last_plan_debug.get("guard_reason", "manual_move_rejected")
                print(f"[manual] command `g` rejected: {reason}")
                self.solver.ctrl.last_status = "running"
                self.solver.ctrl.last_plan_debug = {}
                return None
            if terminal_status is not None:
                return terminal_status
            step_debug = dict(self.solver.ctrl.last_plan_debug)
            print(
                "[manual] command `g` auto-align step "
                f"{self._arm_name(arm_index)} arm contact={int(candidate.contact_id)} "
                f"phase={step_debug.get('target_phase', 'unknown')} "
                f"steps={step_debug.get('executed_steps', '?')}/{step_debug.get('plan_len', '?')} "
                f"inactive_drift={step_debug.get('inactive_gripper_drift', 0.0)}"
            )
            return None
        if command == "o":
            self.solver.ctrl.commanded_gripper[arm_index] = 1.0
            gripper_kwargs = {"left_gripper": 1.0} if arm_index == 0 else {"right_gripper": 1.0}
            self._manual_hold_pose(
                recorder,
                active_arm_index=arm_index,
                steps=self.solver.ctrl.RELEASE_SETTLE_STEPS,
                **gripper_kwargs,
            )
            if self.solver.ctrl.last_status != "running":
                return self.solver.ctrl.last_status
            return None
        if command == "c":
            self.solver.ctrl.commanded_gripper[arm_index] = 0.0
            gripper_kwargs = {"left_gripper": 0.0} if arm_index == 0 else {"right_gripper": 0.0}
            self._manual_hold_pose(
                recorder,
                active_arm_index=arm_index,
                steps=self.solver.ctrl.GRIPPER_SETTLE_STEPS,
                **gripper_kwargs,
            )
            if self.solver.ctrl.last_status != "running":
                return self.solver.ctrl.last_status
            return None
        return None

    def _clear_left_before_policy_resume_after_right_takeover(self, recorder, active_arm_index: int) -> str | None:
        if int(active_arm_index) != 1:
            return None
        if self._policy_grip_guard_arm_index != 0:
            return None
        if not self._opposite_gripper_has_taken_over(0):
            return None

        self.last_debug_info["handover_clear_before_resume"] = {
            "attempted": True,
            "from_guarded_arm": "left",
            "to_guarded_arm": "right",
        }

        self._policy_grip_guard_arm_index = 1
        self.solver.ctrl.commanded_gripper[1] = 0.0
        self._remember_manual_hold_target(1)
        self._remember_manual_gripper_hold_target(1)
        self._set_manual_gripper_drive_override(1, 0.0)
        self._manual_guarded_box_relative_anchor = None

        self.solver.ctrl.commanded_gripper[0] = 1.0
        self._manual_hold_pose(
            recorder,
            active_arm_index=0,
            steps=self.solver.ctrl.RELEASE_SETTLE_STEPS,
            left_gripper=1.0,
            right_gripper=0.0,
        )
        if self.solver.ctrl.last_status != "running":
            return self.solver.ctrl.last_status

        terminal_status = self._run_delta_command(
            recorder,
            arm_index=0,
            command="left_clear_before_resume",
            world_delta=np.asarray([0.0, 0.0, self.MANUAL_HANDOVER_LEFT_CLEAR_LIFT], dtype=np.float64),
        )
        if terminal_status is not None:
            return terminal_status
        if self.solver.ctrl.last_status in ("manual_move_failed", "manual_move_rejected"):
            if self.debug:
                print(
                    "[manual] left clear before resume failed: "
                    f"{self.solver.ctrl.last_status}; stay in manual"
                )
            self.last_debug_info["handover_clear_before_resume"]["status"] = str(self.solver.ctrl.last_status)
            self.solver.ctrl.last_status = "running"
            self.solver.ctrl.last_plan_debug = {}
            return "manual_clear_blocked"

        self.last_debug_info["handover_clear_before_resume"]["status"] = "cleared"
        if self.debug:
            print("[manual] handover clear before resume: right guarded, left opened and lifted")
        return None

    def _raise_left_holder_before_right_manual(self, recorder, active_arm_index: int) -> str | None:
        if int(active_arm_index) != 1:
            return None
        if self._manual_handover_holder_lift_done:
            return None
        if self._policy_grip_guard_arm_index != 0:
            return None
        if self._opposite_gripper_has_taken_over(0):
            return None

        lift = float(self.MANUAL_HANDOVER_HOLDER_LIFT)
        if lift <= 0.0:
            return None
        attempts = (
            ("current_orientation", False, lift),
            ("current_orientation_small", False, max(lift * 0.67, self.MANUAL_HANDOVER_HOLDER_MIN_LIFT)),
        )
        for attempt_name, force_table_horizontal, attempt_lift in attempts:
            start_box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
            start_left_pose = np.asarray(self.solver.ctrl.current_gripper_pose(0), dtype=np.float64)
            start_left_qpos = np.asarray(self.solver.ctrl.current_qpos(0), dtype=np.float64).copy()
            start_left_gripper_qpos = np.asarray(
                self.solver.ctrl.current_gripper_joint_qpos(0),
                dtype=np.float64,
            ).copy()
            self._manual_force_guarded_box_hold = True
            try:
                self._manual_guarded_box_relative_anchor = None
                terminal_status = self._run_delta_command(
                    recorder,
                    arm_index=0,
                    command="handover_holder_lift",
                    world_delta=np.asarray([0.0, 0.0, float(attempt_lift)], dtype=np.float64),
                    force_table_horizontal=force_table_horizontal,
                    min_z_progress=min(self.MANUAL_HANDOVER_HOLDER_MIN_LIFT, float(attempt_lift) * 0.8),
                    return_nonfatal_status=True,
                )
                # Ensure the box pose is synchronized with the final holder pose,
                # not just the penultimate physics step.
                if terminal_status is None and self.solver.ctrl.last_status == "running":
                    self._apply_manual_guarded_box_hold(active_arm_index=0)
                    self._sync_manual_state_to_sim()
            finally:
                self._manual_force_guarded_box_hold = False
            if terminal_status in ("manual_move_failed", "manual_move_rejected"):
                self.solver.ctrl.commanded_arm_qpos[0] = start_left_qpos.copy()
                self.solver.ctrl.commanded_gripper[0] = 0.0
                self._freeze_arm_qpos(0, start_left_qpos)
                self._freeze_gripper_qpos(0, start_left_gripper_qpos)
                self._zero_arm_velocity(0)
                self._set_box_pose_and_zero_velocity(start_box_pose)
                self._manual_guarded_box_relative_anchor = None
                self._sync_manual_state_to_sim()
                self._remember_manual_hold_target(0, start_left_qpos)
                self._remember_manual_gripper_hold_target(0, start_left_gripper_qpos)
                if self.debug:
                    print(
                        "[manual] handover holder lift attempt failed: "
                        f"{attempt_name} {terminal_status}"
                    )
                self.solver.ctrl.last_status = "running"
                self.solver.ctrl.last_plan_debug = {}
                continue
            if terminal_status is not None:
                return terminal_status
            if self.solver.ctrl.last_status in ("manual_move_failed", "manual_move_rejected"):
                if self.debug:
                    print(
                        "[manual] handover holder lift attempt failed: "
                        f"{attempt_name} {self.solver.ctrl.last_status}"
                    )
                self.solver.ctrl.last_status = "running"
                self.solver.ctrl.last_plan_debug = {}
                continue

            end_box_pose = np.asarray(self.solver._box_pose(), dtype=np.float64)
            end_left_pose = np.asarray(self.solver.ctrl.current_gripper_pose(0), dtype=np.float64)
            actual_box_lift = float(end_box_pose[2] - start_box_pose[2])
            actual_left_lift = float(end_left_pose[2] - start_left_pose[2])
            if (
                actual_left_lift < self.MANUAL_HANDOVER_HOLDER_MIN_LIFT
                or actual_box_lift < self.MANUAL_HANDOVER_HOLDER_MIN_LIFT * 0.5
            ):
                if self.debug:
                    print(
                        "[manual] handover holder lift insufficient: "
                        f"mode={attempt_name} requested={float(attempt_lift):.3f}m "
                        f"actual_box={actual_box_lift:.3f}m actual_left={actual_left_lift:.3f}m"
                    )
                self.solver.ctrl.commanded_arm_qpos[0] = start_left_qpos.copy()
                self.solver.ctrl.commanded_gripper[0] = 0.0
                self._freeze_arm_qpos(0, start_left_qpos)
                self._freeze_gripper_qpos(0, start_left_gripper_qpos)
                self._zero_arm_velocity(0)
                self._set_box_pose_and_zero_velocity(start_box_pose)
                self._manual_guarded_box_relative_anchor = None
                self._sync_manual_state_to_sim()
                self._remember_manual_hold_target(0, start_left_qpos)
                self._remember_manual_gripper_hold_target(0, start_left_gripper_qpos)
                continue
            self._manual_handover_holder_lift_done = True
            if self.debug:
                print(
                    "[manual] handover holder lifted before right control: "
                    f"mode={attempt_name} requested={float(attempt_lift):.3f}m "
                    f"actual_box={actual_box_lift:.3f}m actual_left={actual_left_lift:.3f}m"
                )
            return None

        if self.debug:
            print(
                "[manual] handover holder lift failed: no upward IK move was accepted; "
                "continue without extra lift"
            )
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
        self._clear_manual_gripper_drive_overrides()
        self._restore_policy_grip_guard_override()
        self._manual_guarded_box_relative_anchor = None

        active_arm_index = int(initial_arm_index)
        self._reset_manual_hold_targets()
        self._manual_hold_pose(
            recorder,
            active_arm_index=active_arm_index,
            steps=self.MANUAL_ENTRY_FREEZE_STEPS,
        )
        lift_status = self._raise_left_holder_before_right_manual(recorder, active_arm_index)
        if lift_status is not None:
            self.last_debug_info = {"status": lift_status}
            return lift_status
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
                self._manual_hold_pose(
                    recorder,
                    active_arm_index=active_arm_index,
                    steps=self.INITIAL_OPEN_STEPS,
                    **gripper_kwargs,
                )
                if self.solver.ctrl.last_status != "running":
                    self.last_debug_info["status"] = self.solver.ctrl.last_status
                    return self.solver.ctrl.last_status

            print(f"[manual] {trigger_label} intervention triggered")
            self._print_manual_help()

            while True:
                snapshot = self._manual_state_snapshot(active_arm_index)
                steps_left = int(self.solver.ctrl.step_limit - self.solver.ctrl.steps_taken)
                print(
                    "[manual] "
                    f"arm={snapshot['arm_name']} "
                    f"dist={snapshot['gripper_box_distance']:.4f} "
                    f"gripper={snapshot['gripper_pose'][:3]} "
                    f"other_{snapshot['other_arm_name']}={snapshot['other_gripper_pose'][:3]} "
                    f"box={snapshot['box_pose'][:3]} "
                    f"gripper_val={snapshot['gripper_value']:.2f} "
                    f"gripper_qpos={snapshot['gripper_joint_qpos']} "
                    f"steps_left={steps_left}"
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
                    self._remember_manual_hold_target(active_arm_index)
                    active_arm_index = 0
                    self._remember_manual_hold_target(active_arm_index)
                    self.last_debug_info["manual_commands"].append("switch:left")
                    print("[manual] active arm: left")
                    continue
                if command in ("2", "right"):
                    self._remember_manual_hold_target(active_arm_index)
                    active_arm_index = 1
                    self._remember_manual_hold_target(active_arm_index)
                    self.last_debug_info["manual_commands"].append("switch:right")
                    print("[manual] active arm: right")
                    continue

                self.last_debug_info["manual_commands"].append(
                    f"{self._arm_name(active_arm_index)}:{command}"
                )

                if command in ("resume", "n"):
                    clear_status = self._clear_left_before_policy_resume_after_right_takeover(
                        recorder,
                        active_arm_index,
                    )
                    if clear_status == "manual_clear_blocked":
                        continue
                    if clear_status is not None:
                        self.last_debug_info["status"] = clear_status
                        return clear_status
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
            self._manual_trigger_down_latched = self._manual_trigger_key_is_down()
            self._manual_guarded_box_relative_anchor = None
            self._clear_manual_gripper_drive_overrides()
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
        self._disable_policy_grip_guard()
        self._clear_manual_gripper_drive_overrides()
        self._manual_handover_holder_lift_done = False
        self._manual_force_guarded_box_hold = False
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
                        initial_arm_index=self._manual_trigger_arm_index(),
                        initial_open=False,
                    )
                    if tracking.last_obs is not None:
                        self.adapter.reset_model(self.model)
                        encoded_obs = self.adapter.encode_obs(tracking.last_obs)
                    if status == "success":
                        success = True
                        break
                    if status == "manual_intervention_resumed":
                        guard_arm_index = int(
                            self.last_debug_info.get(
                                "post_manual_active_arm_index",
                                self._manual_trigger_arm_index(),
                            )
                        )
                        self._enable_policy_grip_guard_if_closed(guard_arm_index)
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
                    if status == "success":
                        success = True
                        break
                    if status == "manual_intervention_resumed":
                        guard_arm_index = int(self.last_debug_info.get("post_manual_active_arm_index", 0))
                        self._enable_policy_grip_guard_if_closed(guard_arm_index)
                        resume_after_intervention = True
                    break

                action = self._apply_policy_grip_guard(action)
                result = self.backend.step(
                    ActionSpec.from_value(action, action_type=self.action_type)
                )
                self._restore_policy_grip_guard_state()
                self._sync_manual_state_to_sim()
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
                if self._manual_trigger_pressed():
                    status = self._run_manual_intervention(
                        tracking,
                        remaining_steps=int(max_episode_steps) - tracking.step_count,
                        trigger_step=tracking.step_count,
                        trigger_label="viewer manual",
                        initial_arm_index=self._manual_trigger_arm_index(),
                        initial_open=False,
                    )
                    if tracking.last_obs is not None:
                        self.adapter.reset_model(self.model)
                        encoded_obs = self.adapter.encode_obs(tracking.last_obs)
                    if status == "success":
                        success = True
                        break
                    if status == "manual_intervention_resumed":
                        guard_arm_index = int(
                            self.last_debug_info.get(
                                "post_manual_active_arm_index",
                                self._manual_trigger_arm_index(),
                            )
                        )
                        self._enable_policy_grip_guard_if_closed(guard_arm_index)
                        resume_after_intervention = True
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

        self._disable_policy_grip_guard()
        self._clear_manual_gripper_drive_overrides()
        return success, tracking.step_count, status
