"""Hybrid ManiSkill collector that repairs the first left grasp during policy rollout."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from core.schemas.action import ActionSpec
from sim.maniskill.expert.handover_block_solver import (
    LEFT_GRASP_CONTACT_IDS,
    TOP_DOWN_PREFERRED_GRASP_QUATS,
    HandoverBlockSolver,
)
from sim.maniskill.expert.primitives import EpisodeLimitReached


ADAPTER_MODULES = {
    "flowmatching": "policy.FlowMatching.deploy_maniskill",
    "fpo": "policy.FPO.deploy_maniskill",
}


def _to_bool_scalar(value: Any) -> bool:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).reshape(-1)[0])
    return bool(value)


class _ObsTrackingRecorder:
    def __init__(self, recorder) -> None:
        self._recorder = recorder
        self.last_obs: dict[str, Any] | None = None
        self.step_count = 0

    def start_episode(self, initial_obs: dict[str, Any]) -> None:
        self.last_obs = dict(initial_obs)
        self.step_count = 0
        self._recorder.start_episode(initial_obs)

    def append_step(self, action: Any, next_obs: dict[str, Any]) -> None:
        self.last_obs = dict(next_obs)
        self.step_count += 1
        self._recorder.append_step(action, next_obs)

    def save_episode(self, episode_idx: int):
        return self._recorder.save_episode(episode_idx)


class PolicyInterventionCollector:
    TRIGGER_MIN_STEP = 5
    TRIGGER_MAX_STEP = 180
    TRIGGER_GRIPPER_BOX_DISTANCE = 0.28
    GRIPPER_OPEN_STEPS = 4
    RETREAT_WORLD_OFFSET = np.asarray([0.0, -0.10, 0.08], dtype=np.float64)
    BOX_ON_TABLE_MARGIN = 0.03

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
            raise ValueError("policy_intervention collector currently only supports `HandoverBlock-v1`.")

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

    def _left_retreat_gripper_pose(self) -> np.ndarray:
        retreat_pose = np.asarray(
            self.solver.ctrl.current_gripper_pose(0),
            dtype=np.float64,
        ).copy()
        retreat_pose[:3] += self.RETREAT_WORLD_OFFSET
        return retreat_pose

    def _choose_left_pick_candidate(self):
        box_pose = self.solver._box_pose()
        return self.solver.ctrl.choose_grasp_candidate(
            desired_pose=box_pose,
            arm_index=0,
            contact_ids=LEFT_GRASP_CONTACT_IDS,
            pre_distance=self.solver.PRE_GRASP_DISTANCE,
            grasp_distance=self.solver.GRASP_DISTANCE,
            pose_builder=self.solver._grasp_pose_from_box_pose,
            preferred_quat=self.solver._preferred_grasp_quat(0),
            preferred_top_down_quat=TOP_DOWN_PREFERRED_GRASP_QUATS[0],
            pre_pose_candidates_builder=self.solver._left_first_grasp_pre_pose_candidates_from_box_pose,
            selection_mode="prefer_side",
        )

    def _run_left_first_grasp_intervention(
        self,
        recorder: _ObsTrackingRecorder,
        *,
        remaining_steps: int,
        trigger_step: int,
    ) -> str:
        self.solver.ctrl.reset_episode(max_episode_steps=max(int(remaining_steps), 1))
        previous_success_detection = self.solver.ctrl.success_detection_enabled
        self.solver.ctrl.success_detection_enabled = False

        box_pose = self.solver._box_pose()
        left_gripper_pose = self.solver.ctrl.current_gripper_pose(0)
        trigger_distance = float(np.linalg.norm(left_gripper_pose[:3] - box_pose[:3]))
        self.last_debug_info = {
            "stage": "policy_intervention_triggered",
            "trigger_step": int(trigger_step),
            "trigger_gripper_box_distance": round(trigger_distance, 4),
            "box_pose": np.asarray(box_pose, dtype=np.float64).round(4).tolist(),
            "left_gripper_pose": np.asarray(left_gripper_pose, dtype=np.float64).round(4).tolist(),
            "left_qpos": np.asarray(self.solver.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
        }

        try:
            self.solver.ctrl.commanded_gripper[0] = 1.0
            self.solver.ctrl.hold_pose(
                recorder,
                steps=self.GRIPPER_OPEN_STEPS,
                left_gripper=1.0,
                allow_success_early=False,
            )
            if self.solver.ctrl.last_status != "running":
                self.last_debug_info["status"] = self.solver.ctrl.last_status
                return self.solver.ctrl.last_status

            retreat_gripper_pose = self._left_retreat_gripper_pose()
            retreat_target_pose = self.solver.ctrl.gripper_pose_to_planner_pose(0, retreat_gripper_pose)
            self.last_debug_info["retreat_target_pose"] = (
                np.asarray(retreat_target_pose, dtype=np.float64).round(4).tolist()
            )
            self.solver.ctrl.move_to_pose(
                0,
                retreat_target_pose,
                recorder,
                prefer_screw=True,
                fail_status="left_pick_retreat_failed",
            )
            if self.solver.ctrl.last_status != "running":
                self.last_debug_info["status"] = self.solver.ctrl.last_status
                return self.solver.ctrl.last_status

            self.last_debug_info.update(
                {
                    "post_retreat_box_pose": np.asarray(self.solver._box_pose(), dtype=np.float64).round(4).tolist(),
                    "post_retreat_left_qpos": np.asarray(
                        self.solver.ctrl.current_qpos(0),
                        dtype=np.float64,
                    ).round(4).tolist(),
                    "post_retreat_left_gripper_pose": np.asarray(
                        self.solver.ctrl.current_gripper_pose(0),
                        dtype=np.float64,
                    ).round(4).tolist(),
                    "post_retreat_left_gripper_value": round(
                        float(self.solver.ctrl.current_gripper_value(0)),
                        4,
                    ),
                }
            )

            pick_candidate = self._choose_left_pick_candidate()
            if pick_candidate is None:
                self.last_debug_info.update(
                    {
                        "status": "left_pick_plan_failed",
                        "attempts": list(self.solver.ctrl.last_grasp_attempts),
                    }
                )
                return "left_pick_plan_failed"

            self.last_debug_info.update(
                {
                    "selected_contact_id": int(pick_candidate.contact_id),
                    "selected_pre_pose": np.asarray(pick_candidate.pre_pose, dtype=np.float64).round(4).tolist(),
                    "selected_grasp_pose": np.asarray(pick_candidate.grasp_pose, dtype=np.float64).round(4).tolist(),
                    "selected_grasp_plan_source": str(pick_candidate.grasp_plan_source),
                }
            )

            left_approach_pose = self.solver.ctrl.gripper_pose_to_planner_pose(
                0,
                self.solver._get_left_first_grasp_approach_pose(pick_candidate.pre_pose),
            )
            self.solver.ctrl.move_to_pose(
                0,
                left_approach_pose,
                recorder,
                fail_status="left_pick_approach_failed",
            )
            if self.solver.ctrl.last_status != "running":
                self.last_debug_info["status"] = self.solver.ctrl.last_status
                return self.solver.ctrl.last_status

            left_pre_pose = self.solver.ctrl.gripper_pose_to_planner_pose(0, pick_candidate.pre_pose)
            self.solver.ctrl.move_to_pose(
                0,
                left_pre_pose,
                recorder,
                prefer_screw=True,
                fail_status="left_pick_pre_failed",
            )
            if self.solver.ctrl.last_status != "running":
                self.last_debug_info["status"] = self.solver.ctrl.last_status
                return self.solver.ctrl.last_status

            self.solver.ctrl.execute_plan(0, pick_candidate.grasp_plan, recorder)
            if self.solver.ctrl.last_status != "running":
                self.last_debug_info["status"] = self.solver.ctrl.last_status
                return self.solver.ctrl.last_status

            self.solver.ctrl.commanded_gripper[0] = 0.0
            self.solver.ctrl.hold_pose(
                recorder,
                steps=self.solver.LEFT_FIRST_GRASP_CLOSE_STEPS,
                left_gripper=0.0,
                allow_success_early=False,
            )
            if self.solver.ctrl.last_status != "running":
                self.last_debug_info["status"] = self.solver.ctrl.last_status
                return self.solver.ctrl.last_status

            self.last_debug_info.update(
                {
                    "status": "intervention_completed",
                    "post_close_box_pose": np.asarray(self.solver._box_pose(), dtype=np.float64).round(4).tolist(),
                    "post_close_left_gripper_pose": np.asarray(
                        self.solver.ctrl.current_gripper_pose(0),
                        dtype=np.float64,
                    ).round(4).tolist(),
                    "post_close_left_gripper_value": round(float(self.solver.ctrl.current_gripper_value(0)), 4),
                }
            )
            return "intervention_completed"
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
                    status = self._run_left_first_grasp_intervention(
                        tracking,
                        remaining_steps=int(max_episode_steps) - tracking.step_count,
                        trigger_step=tracking.step_count,
                    )
                    if tracking.last_obs is not None:
                        self.adapter.reset_model(self.model)
                        encoded_obs = self.adapter.encode_obs(tracking.last_obs)
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
                if status in ("episode_limit_reached",) or status.startswith("terminated"):
                    break
                continue

            if success or status in ("terminated", "truncated", "max_episode_steps_reached"):
                break

        if tracking.step_count >= int(max_episode_steps) and not success and status == "policy_rollout_finished":
            status = "max_episode_steps_reached"

        return success, tracking.step_count, status
