from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import xml.etree.ElementTree as ET

import mplib
import numpy as np
import scipy.spatial
import transforms3d.quaternions as tquat
import trimesh

from core.schemas.action import ActionSpec


class EpisodeLimitReached(RuntimeError):
    pass


@dataclass
class GraspCandidate:
    contact_id: int
    pre_pose: np.ndarray
    grasp_pose: np.ndarray
    pre_plan: dict[str, Any]
    grasp_plan: dict[str, Any]
    score: float
    path_length: int


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_bool(value: Any) -> bool:
    return bool(to_numpy(value).reshape(-1)[0])


def pose_to_matrix(pose_pq: np.ndarray) -> np.ndarray:
    pose_pq = np.asarray(pose_pq, dtype=np.float64).reshape(7)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = tquat.quat2mat(pose_pq[3:])
    matrix[:3, 3] = pose_pq[:3]
    return matrix


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    quat = tquat.mat2quat(matrix[:3, :3])
    return np.concatenate([matrix[:3, 3], quat], axis=0).astype(np.float64, copy=False)


def scaled_local_matrix(local_matrix: np.ndarray, scale: np.ndarray) -> np.ndarray:
    matrix = np.asarray(local_matrix, dtype=np.float64).copy()
    matrix[:3, 3] *= np.asarray(scale, dtype=np.float64).reshape(3)
    return matrix


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    return 1.0 - float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))


def offset_pose_along_local_x(pose_pq: np.ndarray, delta: float) -> np.ndarray:
    pose_pq = np.asarray(pose_pq, dtype=np.float64).reshape(7).copy()
    rotation = tquat.quat2mat(pose_pq[3:])
    pose_pq[:3] += np.asarray([delta, 0.0, 0.0], dtype=np.float64) @ np.linalg.inv(rotation)
    return pose_pq


def generate_convex_mesh(input_stl_path: Path, output_stl_path: Path) -> None:
    mesh = trimesh.load(str(input_stl_path), force="mesh")
    vertices = np.asarray(mesh.vertices)
    hull = scipy.spatial.ConvexHull(vertices)
    convex_vertices = vertices[hull.vertices]
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(hull.vertices)}
    convex_faces = np.asarray([[vertex_map[idx] for idx in face] for face in hull.simplices], dtype=np.int64)
    convex_mesh = trimesh.Trimesh(vertices=convex_vertices, faces=convex_faces)
    output_stl_path.parent.mkdir(parents=True, exist_ok=True)
    convex_mesh.export(str(output_stl_path))


def ensure_convex_collision_meshes(urdf_path: str) -> None:
    urdf_file = Path(urdf_path).resolve()
    root = ET.parse(urdf_file).getroot()
    for mesh in root.findall(".//collision/geometry/mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or not filename.endswith(".stl"):
            continue
        mesh_path = (urdf_file.parent / filename).resolve()
        convex_path = Path(str(mesh_path) + ".convex.stl")
        if convex_path.is_file() or not mesh_path.is_file():
            continue
        generate_convex_mesh(mesh_path, convex_path)


class FixedRateMotionPrimitives:
    GRIPPER_SETTLE_STEPS = 18
    RELEASE_SETTLE_STEPS = 12
    HOLD_SETTLE_STEPS = 6
    PLANNER_TIME = 1.0

    def __init__(self, env_backend):
        self.backend = env_backend
        self.base_env = env_backend.env.unwrapped
        if getattr(self.base_env, "num_envs", 1) != 1:
            raise ValueError("FixedRateMotionPrimitives only supports num_envs=1.")
        self.control_timestep = float(getattr(self.base_env, "control_timestep", 0.05))
        self.steps_taken = 0
        self.step_limit = 0
        self.last_status = "not_started"
        self.last_grasp_attempts: list[dict[str, Any]] = []
        self.planners = {
            0: self._build_planner(0),
            1: self._build_planner(1),
        }

    def maybe_render(self) -> None:
        if getattr(self.backend, "render_mode", None) == "human":
            self.backend.render()

    def reset_episode(self, max_episode_steps: int) -> None:
        self.steps_taken = 0
        self.step_limit = int(max_episode_steps)
        self.last_status = "running"
        self.last_grasp_attempts = []

    def _build_planner(self, arm_index: int) -> mplib.Planner:
        assets = self.base_env.get_arm_planner_assets(arm_index)
        ensure_convex_collision_meshes(assets["urdf_path"])
        planner = mplib.Planner(
            urdf=assets["urdf_path"],
            srdf=assets["srdf_path"],
            move_group=assets["move_group"],
        )
        planner.set_base_pose(self.base_env.get_arm_root_pose_array(arm_index))
        return planner

    def _arm_config(self, arm_index: int) -> dict[str, Any]:
        if arm_index == 0:
            return self.base_env.embodiment_spec.left_config
        return self.base_env.embodiment_spec.right_config

    def gripper_pose_to_planner_pose(self, arm_index: int, target_pose: np.ndarray) -> np.ndarray:
        config = self._arm_config(arm_index)
        gripper_bias = float(config.get("gripper_bias", 0.0))
        delta_matrix = np.asarray(config.get("delta_matrix", np.eye(3)), dtype=np.float64).reshape(3, 3)
        inv_delta_matrix = np.linalg.inv(delta_matrix)

        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(7)
        gripper_pose_pos = target_pose[:3].copy()
        gripper_pose_quat = target_pose[3:].copy()
        gripper_pose_mat = tquat.quat2mat(gripper_pose_quat)
        gripper_pose_pos += gripper_pose_mat @ np.array([0.12 - gripper_bias, 0.0, 0.0], dtype=np.float64)
        gripper_pose_mat = gripper_pose_mat @ inv_delta_matrix
        gripper_pose_quat = tquat.mat2quat(gripper_pose_mat)
        return np.concatenate([gripper_pose_pos, gripper_pose_quat], axis=0).astype(np.float64, copy=False)

    def current_qpos(self, arm_index: int) -> np.ndarray:
        return self.base_env.get_arm_joint_positions(arm_index)

    def current_full_qpos(self, arm_index: int) -> np.ndarray:
        return self.base_env.get_arm_full_joint_positions(arm_index)

    def current_ee_pose(self, arm_index: int) -> np.ndarray:
        return self.base_env.get_arm_end_effector_pose_array(arm_index)

    def current_gripper_value(self, arm_index: int) -> float:
        value = self.base_env.get_left_gripper_val() if arm_index == 0 else self.base_env.get_right_gripper_val()
        if isinstance(value, (float, int)):
            return float(value)
        return float(to_numpy(value).reshape(-1)[0])

    def current_success(self) -> bool:
        info = self.base_env.evaluate()
        return scalar_bool(info["success"])

    def _arm_qpos_from_plan(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        return qpos[: self.base_env.ARM_DOF]

    def compose_action(
        self,
        *,
        left_qpos: np.ndarray | None = None,
        right_qpos: np.ndarray | None = None,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
    ) -> np.ndarray:
        if left_qpos is None:
            left_qpos = self.current_qpos(0)
        if right_qpos is None:
            right_qpos = self.current_qpos(1)
        if left_gripper is None:
            left_gripper = self.current_gripper_value(0)
        if right_gripper is None:
            right_gripper = self.current_gripper_value(1)
        return np.concatenate(
            [
                np.asarray(left_qpos, dtype=np.float32).reshape(self.base_env.ARM_DOF),
                np.asarray([left_gripper], dtype=np.float32),
                np.asarray(right_qpos, dtype=np.float32).reshape(self.base_env.ARM_DOF),
                np.asarray([right_gripper], dtype=np.float32),
            ],
            axis=0,
        )

    def _step_and_record(self, action: np.ndarray, recorder) -> tuple[Any, bool]:
        if self.steps_taken >= self.step_limit:
            raise EpisodeLimitReached
        result = self.backend.step(ActionSpec.from_value(action))
        recorder.append_step(action, dict(result.observation.data))
        self.steps_taken += 1
        self.maybe_render()
        success = scalar_bool(result.info.get("success", False))
        return result, success

    def _execute_plan(self, arm_index: int, plan_result: dict[str, Any], recorder) -> bool:
        positions = np.asarray(plan_result["position"], dtype=np.float64)
        for qpos in positions:
            arm_qpos = self._arm_qpos_from_plan(qpos)
            if arm_index == 0:
                action = self.compose_action(left_qpos=arm_qpos)
            else:
                action = self.compose_action(right_qpos=arm_qpos)
            result, success = self._step_and_record(action, recorder)
            if success:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_plan"
                return False
        return False

    def hold_pose(self, recorder, *, steps: int, left_gripper: float | None = None, right_gripper: float | None = None) -> bool:
        for _ in range(steps):
            action = self.compose_action(left_gripper=left_gripper, right_gripper=right_gripper)
            result, success = self._step_and_record(action, recorder)
            if success:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_hold"
                return False
        return False

    def plan_to_pose(self, arm_index: int, target_pose: np.ndarray, *, start_qpos: np.ndarray | None = None) -> dict[str, Any] | None:
        result, _reason = self.plan_to_pose_debug(arm_index, target_pose, start_qpos=start_qpos)
        return result

    def plan_to_pose_debug(
        self,
        arm_index: int,
        target_pose: np.ndarray,
        *,
        start_qpos: np.ndarray | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        planner = self.planners[arm_index]
        current_qpos = self.current_full_qpos(arm_index) if start_qpos is None else np.asarray(start_qpos, dtype=np.float64)
        current_qpos = planner.pad_qpos(current_qpos.copy())
        try:
            result = planner.plan_qpos_to_pose(
                np.asarray(target_pose, dtype=np.float64),
                current_qpos,
                time_step=self.control_timestep,
                planning_time=self.PLANNER_TIME,
                wrt_world=True,
            )
        except Exception as exc:
            return None, f"exception:{type(exc).__name__}:{exc}"
        status = str(result.get("status"))
        if status != "Success":
            return None, f"status:{status}"
        return result, "success"

    def plan_screw_to_pose(self, arm_index: int, target_pose: np.ndarray, *, start_qpos: np.ndarray | None = None) -> dict[str, Any] | None:
        result, _reason = self.plan_screw_to_pose_debug(arm_index, target_pose, start_qpos=start_qpos)
        return result

    def plan_screw_to_pose_debug(
        self,
        arm_index: int,
        target_pose: np.ndarray,
        *,
        start_qpos: np.ndarray | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        planner = self.planners[arm_index]
        current_qpos = self.current_full_qpos(arm_index) if start_qpos is None else np.asarray(start_qpos, dtype=np.float64)
        current_qpos = planner.pad_qpos(current_qpos.copy())
        try:
            result = planner.plan_screw(
                np.asarray(target_pose, dtype=np.float64),
                current_qpos,
                time_step=self.control_timestep,
                wrt_world=True,
            )
        except Exception as exc:
            return None, f"exception:{type(exc).__name__}:{exc}"
        status = str(result.get("status"))
        if status != "Success":
            return None, f"status:{status}"
        return result, "success"

    def move_to_pose(
        self,
        arm_index: int,
        pose: np.ndarray,
        recorder,
        *,
        prefer_screw: bool = False,
        fail_status: str,
    ) -> bool:
        plan_result = self.plan_screw_to_pose(arm_index, pose) if prefer_screw else None
        if plan_result is None:
            plan_result = self.plan_to_pose(arm_index, pose)
        if plan_result is None:
            self.last_status = fail_status
            return False
        return self._execute_plan(arm_index, plan_result, recorder)

    def move_by_displacement(
        self,
        arm_index: int,
        recorder,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quat: np.ndarray | None = None,
        move_axis: str = "world",
        fail_status: str,
    ) -> bool:
        origin_pose = self.current_ee_pose(arm_index).copy()
        if move_axis == "world":
            origin_pose[:3] += np.asarray([x, y, z], dtype=np.float64)
        else:
            dir_vec = pose_to_matrix(origin_pose)[:3, 0]
            dir_vec /= np.linalg.norm(dir_vec)
            origin_pose[:3] += -z * dir_vec
        if quat is not None:
            origin_pose[3:] = np.asarray(quat, dtype=np.float64).reshape(4)
        return self.move_to_pose(arm_index, origin_pose, recorder, fail_status=fail_status)

    def back_to_origin(self, arm_index: int, recorder, *, fail_status: str) -> bool:
        target_qpos = self.base_env.get_arm_home_joint_positions(arm_index)
        steps = np.linspace(self.current_qpos(arm_index), target_qpos, num=25).astype(np.float64, copy=False)
        for qpos in steps:
            if arm_index == 0:
                action = self.compose_action(left_qpos=qpos)
            else:
                action = self.compose_action(right_qpos=qpos)
            result, success = self._step_and_record(action, recorder)
            if success:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_home"
                return False
        return False

    def open_gripper(self, arm_index: int, recorder, *, pos: float = 1.0, fail_status: str | None = None) -> bool:
        success = self.hold_pose(
            recorder,
            steps=self.RELEASE_SETTLE_STEPS,
            left_gripper=pos if arm_index == 0 else None,
            right_gripper=pos if arm_index == 1 else None,
        )
        if not success and fail_status is not None and self.last_status == "running":
            self.last_status = fail_status
        return success

    def close_gripper(self, arm_index: int, recorder, *, pos: float = 0.0, fail_status: str | None = None) -> bool:
        success = self.hold_pose(
            recorder,
            steps=self.GRIPPER_SETTLE_STEPS,
            left_gripper=pos if arm_index == 0 else None,
            right_gripper=pos if arm_index == 1 else None,
        )
        if not success and fail_status is not None and self.last_status == "running":
            self.last_status = fail_status
        return success

    def choose_grasp_candidate(
        self,
        *,
        desired_pose: np.ndarray,
        arm_index: int,
        contact_ids: tuple[int, ...],
        pre_distance: float,
        grasp_distance: float,
        pose_builder: Callable[[np.ndarray, int, float], np.ndarray],
        preferred_quat: np.ndarray,
        pre_pose_candidates_builder: Callable[[np.ndarray, int, float, int], list[np.ndarray]] | None = None,
    ) -> GraspCandidate | None:
        current_qpos = self.current_full_qpos(arm_index)
        best: GraspCandidate | None = None
        attempts: list[dict[str, Any]] = []
        for contact_id in contact_ids:
            if pre_pose_candidates_builder is None:
                pre_pose_candidates = [pose_builder(desired_pose, contact_id, pre_distance)]
            else:
                pre_pose_candidates = list(pre_pose_candidates_builder(desired_pose, contact_id, pre_distance, arm_index))
            best_for_contact: GraspCandidate | None = None
            for variant_index, pre_pose in enumerate(pre_pose_candidates):
                if pre_pose_candidates_builder is None:
                    grasp_pose = pose_builder(desired_pose, contact_id, grasp_distance)
                else:
                    grasp_pose = offset_pose_along_local_x(pre_pose, pre_distance - grasp_distance)
                planner_pre_pose = self.gripper_pose_to_planner_pose(arm_index, pre_pose)
                planner_grasp_pose = self.gripper_pose_to_planner_pose(arm_index, grasp_pose)
                pre_plan, pre_reason = self.plan_to_pose_debug(arm_index, planner_pre_pose, start_qpos=current_qpos)
                if pre_plan is None:
                    attempts.append(
                        {
                            "contact_id": int(contact_id),
                            "variant_index": int(variant_index),
                            "pre_reason": pre_reason,
                            "grasp_reason": None,
                        }
                    )
                    continue
                next_qpos = np.asarray(pre_plan["position"][-1], dtype=np.float64)
                grasp_plan, grasp_reason = self.plan_screw_to_pose_debug(
                    arm_index,
                    planner_grasp_pose,
                    start_qpos=next_qpos,
                )
                if grasp_plan is None:
                    grasp_plan, grasp_reason = self.plan_to_pose_debug(
                        arm_index,
                        planner_grasp_pose,
                        start_qpos=next_qpos,
                    )
                if grasp_plan is None:
                    attempts.append(
                        {
                            "contact_id": int(contact_id),
                            "variant_index": int(variant_index),
                            "pre_reason": pre_reason,
                            "grasp_reason": grasp_reason,
                        }
                    )
                    continue
                candidate = GraspCandidate(
                    contact_id=contact_id,
                    pre_pose=pre_pose,
                    grasp_pose=grasp_pose,
                    pre_plan=pre_plan,
                    grasp_plan=grasp_plan,
                    score=quat_distance(grasp_pose[3:], preferred_quat),
                    path_length=len(pre_plan["position"]) + len(grasp_plan["position"]),
                )
                if best_for_contact is None or candidate.path_length < best_for_contact.path_length:
                    best_for_contact = candidate
                attempts.append(
                    {
                        "contact_id": int(contact_id),
                        "variant_index": int(variant_index),
                        "pre_reason": pre_reason,
                        "grasp_reason": grasp_reason,
                    }
                )
            if best_for_contact is None:
                continue
            if (
                best is None
                or best_for_contact.score < best.score
                or (
                    np.isclose(best_for_contact.score, best.score)
                    and best_for_contact.path_length < best.path_length
                )
            ):
                best = best_for_contact
        self.last_grasp_attempts = attempts
        return best

    def grasp_actor(
        self,
        *,
        desired_pose: np.ndarray,
        arm_index: int,
        contact_ids: tuple[int, ...],
        pre_grasp_dis: float,
        grasp_dis: float,
        pose_builder: Callable[[np.ndarray, int, float], np.ndarray],
        preferred_quat: np.ndarray,
        recorder,
        fail_status: str,
        gripper_pos: float = 0.0,
        pre_pose_candidates_builder: Callable[[np.ndarray, int, float, int], list[np.ndarray]] | None = None,
    ) -> tuple[GraspCandidate | None, bool]:
        candidate = self.choose_grasp_candidate(
            desired_pose=desired_pose,
            arm_index=arm_index,
            contact_ids=contact_ids,
            pre_distance=pre_grasp_dis,
            grasp_distance=grasp_dis,
            pose_builder=pose_builder,
            preferred_quat=preferred_quat,
            pre_pose_candidates_builder=pre_pose_candidates_builder,
        )
        if candidate is None:
            self.last_status = fail_status
            return None, False
        success = self._execute_plan(arm_index, candidate.pre_plan, recorder)
        if success:
            return candidate, True
        success = self._execute_plan(arm_index, candidate.grasp_plan, recorder)
        if success:
            return candidate, True
        success = self.close_gripper(arm_index, recorder, pos=gripper_pos)
        return candidate, success

    def place_actor(
        self,
        *,
        arm_index: int,
        pre_pose: np.ndarray,
        place_pose: np.ndarray,
        recorder,
        fail_status: str,
        open_after: bool = True,
        prefer_screw_final: bool = False,
    ) -> bool:
        if not np.allclose(pre_pose, place_pose):
            success = self.move_to_pose(arm_index, pre_pose, recorder, fail_status=fail_status)
            if success:
                return True
            if self.last_status == fail_status:
                return False
        success = self.move_to_pose(arm_index, place_pose, recorder, prefer_screw=prefer_screw_final, fail_status=fail_status)
        if success:
            return True
        if self.last_status == fail_status:
            return False
        if open_after:
            success = self.open_gripper(arm_index, recorder)
            if success:
                return True
        return False
