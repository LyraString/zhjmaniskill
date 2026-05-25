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
    grasp_plan_source: str
    score: float
    top_down_score: float
    side_score: float
    path_length: int
    shoulder_pan_delta: float
    joint_motion: float


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
    INITIAL_GRIPPER_SETTLE_STEPS = 30
    GRIPPER_JOINT_SETTLE_EPS = 0.02
    PLANNER_TIME = 0.35

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
        self.last_plan_debug: dict[str, Any] = {}
        self.success_detection_enabled = True
        self.commanded_arm_qpos = {
            0: np.zeros(self.base_env.ARM_DOF, dtype=np.float64),
            1: np.zeros(self.base_env.ARM_DOF, dtype=np.float64),
        }
        self.commanded_gripper = {
            0: 0.0,
            1: 0.0,
        }
        self.planners = {
            0: self._build_planner(0),
            1: self._build_planner(1),
        }

    def maybe_render(self) -> None:
        if getattr(self.backend, "render_mode", None) == "human":
            self.backend.render()

    def sync_sim_state(self) -> None:
        scene = getattr(self.base_env, "scene", None)
        if scene is None:
            return
        if not getattr(self.base_env, "gpu_sim_enabled", False):
            return
        if hasattr(scene, "_gpu_sim_initialized") and not getattr(scene, "_gpu_sim_initialized"):
            return
        scene._gpu_fetch_all()

    def reset_episode(self, max_episode_steps: int) -> None:
        self.steps_taken = 0
        self.step_limit = int(max_episode_steps)
        self.last_status = "running"
        self.last_grasp_attempts = []
        self.last_plan_debug = {}
        self.sync_sim_state()
        self.commanded_arm_qpos[0] = self.current_qpos(0).copy()
        self.commanded_arm_qpos[1] = self.current_qpos(1).copy()
        self.commanded_gripper[0] = self.current_gripper_value(0)
        self.commanded_gripper[1] = self.current_gripper_value(1)

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

    def planner_pose_to_gripper_pose(self, arm_index: int, planner_pose: np.ndarray) -> np.ndarray:
        config = self._arm_config(arm_index)
        gripper_bias = float(config.get("gripper_bias", 0.0))
        delta_matrix = np.asarray(config.get("delta_matrix", np.eye(3)), dtype=np.float64).reshape(3, 3)

        planner_pose = np.asarray(planner_pose, dtype=np.float64).reshape(7)
        planner_pose_pos = planner_pose[:3].copy()
        planner_pose_quat = planner_pose[3:].copy()
        planner_pose_mat = tquat.quat2mat(planner_pose_quat)

        gripper_pose_mat = planner_pose_mat @ delta_matrix
        gripper_pose_pos = planner_pose_pos - gripper_pose_mat @ np.array([0.12 - gripper_bias, 0.0, 0.0], dtype=np.float64)
        gripper_pose_quat = tquat.mat2quat(gripper_pose_mat)
        return np.concatenate([gripper_pose_pos, gripper_pose_quat], axis=0).astype(np.float64, copy=False)

    def current_qpos(self, arm_index: int) -> np.ndarray:
        self.sync_sim_state()
        return self.base_env.get_arm_joint_positions(arm_index)

    def current_full_qpos(self, arm_index: int) -> np.ndarray:
        self.sync_sim_state()
        return self.base_env.get_arm_full_joint_positions(arm_index)

    def current_ee_pose(self, arm_index: int) -> np.ndarray:
        self.sync_sim_state()
        return self.base_env.get_arm_end_effector_pose_array(arm_index)

    def current_gripper_pose(self, arm_index: int) -> np.ndarray:
        return self.planner_pose_to_gripper_pose(arm_index, self.current_ee_pose(arm_index))

    def current_gripper_value(self, arm_index: int) -> float:
        self.sync_sim_state()
        value = self.base_env.get_left_gripper_val() if arm_index == 0 else self.base_env.get_right_gripper_val()
        if isinstance(value, (float, int)):
            return float(value)
        return float(to_numpy(value).reshape(-1)[0])

    def current_gripper_joint_qpos(self, arm_index: int) -> np.ndarray:
        joints = self.base_env.left_gripper if arm_index == 0 else self.base_env.right_gripper
        full_qpos = self.current_full_qpos(arm_index)
        start = self.base_env.ARM_DOF
        end = start + len(joints)
        return np.asarray(full_qpos[start:end], dtype=np.float64).copy()

    def target_gripper_joint_qpos(self, arm_index: int, gripper_value: float) -> np.ndarray:
        gripper_scale = self.base_env.embodiment_spec.gripper_scale(arm_index)
        real_gripper_val = float(gripper_scale[0]) + float(gripper_value) * float(gripper_scale[1] - gripper_scale[0])
        joints = self.base_env.left_gripper if arm_index == 0 else self.base_env.right_gripper
        return np.asarray(
            [real_gripper_val * float(sign) + float(offset) for _, sign, offset in joints],
            dtype=np.float64,
        )

    def current_success(self) -> bool:
        self.sync_sim_state()
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
            left_qpos = self.commanded_arm_qpos[0]
        if right_qpos is None:
            right_qpos = self.commanded_arm_qpos[1]
        if left_gripper is None:
            left_gripper = self.commanded_gripper[0]
        if right_gripper is None:
            right_gripper = self.commanded_gripper[1]
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
        action = np.asarray(action, dtype=np.float32).reshape(self.base_env.ACTION_DIM)
        self.commanded_arm_qpos[0] = action[0:6].astype(np.float64, copy=True)
        self.commanded_gripper[0] = float(action[6])
        self.commanded_arm_qpos[1] = action[7:13].astype(np.float64, copy=True)
        self.commanded_gripper[1] = float(action[13])
        result = self.backend.step(ActionSpec.from_value(action))
        recorder.append_step(action, dict(result.observation.data))
        self.steps_taken += 1
        self.maybe_render()
        success = self.success_detection_enabled and scalar_bool(result.info.get("success", False))
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

    def execute_plan(self, arm_index: int, plan_result: dict[str, Any], recorder) -> bool:
        return self._execute_plan(arm_index, plan_result, recorder)

    def execute_plan_with_gripper(
        self,
        arm_index: int,
        plan_result: dict[str, Any],
        recorder,
        *,
        gripper: float | None = None,
    ) -> bool:
        positions = np.asarray(plan_result["position"], dtype=np.float64)
        for qpos in positions:
            arm_qpos = self._arm_qpos_from_plan(qpos)
            if arm_index == 0:
                action = self.compose_action(left_qpos=arm_qpos, left_gripper=gripper)
            else:
                action = self.compose_action(right_qpos=arm_qpos, right_gripper=gripper)
            result, success = self._step_and_record(action, recorder)
            if success:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_plan"
                return False
        return False

    def hold_pose(
        self,
        recorder,
        *,
        steps: int,
        left_gripper: float | None = None,
        right_gripper: float | None = None,
        allow_success_early: bool = True,
    ) -> bool:
        for _ in range(steps):
            action = self.compose_action(left_gripper=left_gripper, right_gripper=right_gripper)
            result, success = self._step_and_record(action, recorder)
            if success and allow_success_early:
                self.last_status = "success"
                return True
            if success and result.terminated:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_hold"
                return False
        return False

    def settle_grippers_to_targets(
        self,
        recorder,
        *,
        left_target: float | None = None,
        right_target: float | None = None,
        max_steps: int | None = None,
        joint_eps: float | None = None,
        fail_status: str | None = None,
        allow_success_early: bool = False,
    ) -> bool:
        max_steps = self.INITIAL_GRIPPER_SETTLE_STEPS if max_steps is None else int(max_steps)
        joint_eps = self.GRIPPER_JOINT_SETTLE_EPS if joint_eps is None else float(joint_eps)
        left_joint_target = None if left_target is None else self.target_gripper_joint_qpos(0, left_target)
        right_joint_target = None if right_target is None else self.target_gripper_joint_qpos(1, right_target)

        for _ in range(max_steps):
            action = self.compose_action(left_gripper=left_target, right_gripper=right_target)
            result, success = self._step_and_record(action, recorder)
            if success and allow_success_early:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_gripper_settle"
                return False

            left_ready = True
            if left_joint_target is not None:
                left_ready = np.all(
                    np.abs(self.current_gripper_joint_qpos(0) - left_joint_target) <= joint_eps
                )
            right_ready = True
            if right_joint_target is not None:
                right_ready = np.all(
                    np.abs(self.current_gripper_joint_qpos(1) - right_joint_target) <= joint_eps
                )
            if left_ready and right_ready:
                return False

        if fail_status is not None:
            self.last_status = fail_status
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
        self.sync_sim_state()
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

    def ik_to_pose_debug(
        self,
        arm_index: int,
        target_pose: np.ndarray,
        *,
        start_qpos: np.ndarray | None = None,
        n_init_qpos: int = 20,
        threshold: float = 1e-3,
    ) -> tuple[np.ndarray | None, str]:
        planner = self.planners[arm_index]
        self.sync_sim_state()
        current_qpos = self.current_full_qpos(arm_index) if start_qpos is None else np.asarray(start_qpos, dtype=np.float64)
        current_qpos = planner.pad_qpos(current_qpos.copy())
        goal_pose = planner.transform_goal_to_wrt_base(np.asarray(target_pose, dtype=np.float64))
        try:
            status, results = planner.IK(
                goal_pose,
                current_qpos,
                n_init_qpos=int(n_init_qpos),
                threshold=float(threshold),
            )
        except Exception as exc:
            return None, f"exception:{type(exc).__name__}:{exc}"
        if status != "Success" or len(results) == 0:
            return None, f"status:{status}"
        move_group_indices = np.asarray(planner.move_group_joint_indices, dtype=np.int64)
        best_qpos = min(
            results,
            key=lambda qpos: float(
                np.linalg.norm(np.asarray(qpos, dtype=np.float64)[move_group_indices] - current_qpos[move_group_indices])
            ),
        )
        return np.asarray(best_qpos, dtype=np.float64), "success"

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
        self.sync_sim_state()
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
        allow_pose_fallback: bool = True,
        fail_status: str,
    ) -> bool:
        target_pose = np.asarray(pose, dtype=np.float64)
        screw_reason = None
        pose_reason = None
        plan_result = None
        if prefer_screw:
            plan_result, screw_reason = self.plan_screw_to_pose_debug(arm_index, target_pose)
        if plan_result is None and (allow_pose_fallback or not prefer_screw):
            plan_result, pose_reason = self.plan_to_pose_debug(arm_index, target_pose)
        if plan_result is None:
            self.last_plan_debug = {
                "arm_index": int(arm_index),
                "target_pose": target_pose.round(4).tolist(),
                "prefer_screw": bool(prefer_screw),
                "allow_pose_fallback": bool(allow_pose_fallback),
                "screw_reason": screw_reason,
                "plan_reason": pose_reason,
            }
            self.last_status = fail_status
            return False
        self.last_plan_debug = {}
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
        return self.move_to_pose(
            arm_index,
            origin_pose,
            recorder,
            prefer_screw=True,
            fail_status=fail_status,
        )

    def move_by_displacement_via_ik(
        self,
        arm_index: int,
        recorder,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quat: np.ndarray | None = None,
        move_axis: str = "world",
        steps: int = 10,
        settle_steps: int = 8,
        fail_status: str,
        ik_threshold: float | None = None,
    ) -> bool:
        target_pose = self.current_ee_pose(arm_index).copy()
        if move_axis == "world":
            target_pose[:3] += np.asarray([x, y, z], dtype=np.float64)
        else:
            dir_vec = pose_to_matrix(target_pose)[:3, 0]
            dir_vec /= np.linalg.norm(dir_vec)
            target_pose[:3] += -z * dir_vec
        if quat is not None:
            target_pose[3:] = np.asarray(quat, dtype=np.float64).reshape(4)

        if ik_threshold is None:
            ik_qpos, ik_reason = self.ik_to_pose_debug(arm_index, target_pose)
        else:
            ik_qpos, ik_reason = self.ik_to_pose_debug(arm_index, target_pose, threshold=float(ik_threshold))
        if ik_qpos is None:
            self.last_plan_debug = {
                "arm_index": int(arm_index),
                "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
                "move_axis": str(move_axis),
                "ik_reason": ik_reason,
                "steps": int(steps),
                "settle_steps": int(settle_steps),
            }
            self.last_status = fail_status
            return False

        self.last_plan_debug = {
            "arm_index": int(arm_index),
            "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
            "target_arm_qpos": self._arm_qpos_from_plan(ik_qpos).round(4).tolist(),
            "move_axis": str(move_axis),
            "ik_reason": ik_reason,
            "steps": int(steps),
            "settle_steps": int(settle_steps),
        }
        return self.move_to_qpos(
            arm_index,
            self._arm_qpos_from_plan(ik_qpos),
            recorder,
            steps=steps,
            settle_steps=settle_steps,
        )

    def move_gripper_by_displacement_via_ik(
        self,
        arm_index: int,
        recorder,
        *,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        quat: np.ndarray | None = None,
        move_axis: str = "world",
        steps: int = 10,
        settle_steps: int = 8,
        fail_status: str,
        ik_threshold: float | None = None,
    ) -> bool:
        target_gripper_pose = self.current_gripper_pose(arm_index).copy()
        if move_axis == "world":
            target_gripper_pose[:3] += np.asarray([x, y, z], dtype=np.float64)
        else:
            dir_vec = pose_to_matrix(target_gripper_pose)[:3, 0]
            dir_vec /= np.linalg.norm(dir_vec)
            target_gripper_pose[:3] += -z * dir_vec
        if quat is not None:
            target_gripper_pose[3:] = np.asarray(quat, dtype=np.float64).reshape(4)
        target_pose = self.gripper_pose_to_planner_pose(arm_index, target_gripper_pose)

        if ik_threshold is None:
            ik_qpos, ik_reason = self.ik_to_pose_debug(arm_index, target_pose)
        else:
            ik_qpos, ik_reason = self.ik_to_pose_debug(arm_index, target_pose, threshold=float(ik_threshold))
        if ik_qpos is None:
            self.last_plan_debug = {
                "arm_index": int(arm_index),
                "target_gripper_pose": np.asarray(target_gripper_pose, dtype=np.float64).round(4).tolist(),
                "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
                "move_axis": str(move_axis),
                "ik_reason": ik_reason,
                "steps": int(steps),
                "settle_steps": int(settle_steps),
            }
            self.last_status = fail_status
            return False

        self.last_plan_debug = {
            "arm_index": int(arm_index),
            "target_gripper_pose": np.asarray(target_gripper_pose, dtype=np.float64).round(4).tolist(),
            "target_pose": np.asarray(target_pose, dtype=np.float64).round(4).tolist(),
            "target_arm_qpos": self._arm_qpos_from_plan(ik_qpos).round(4).tolist(),
            "move_axis": str(move_axis),
            "ik_reason": ik_reason,
            "steps": int(steps),
            "settle_steps": int(settle_steps),
        }
        return self.move_to_qpos(
            arm_index,
            self._arm_qpos_from_plan(ik_qpos),
            recorder,
            steps=steps,
            settle_steps=settle_steps,
        )

    def move_to_qpos(
        self,
        arm_index: int,
        target_qpos: np.ndarray,
        recorder,
        *,
        steps: int = 25,
        settle_steps: int = 0,
    ) -> bool:
        start_qpos = self.current_qpos(arm_index)
        target_qpos = np.asarray(target_qpos, dtype=np.float64).reshape(start_qpos.shape)
        trajectory = np.linspace(start_qpos, target_qpos, num=int(steps)).astype(np.float64, copy=False)
        for qpos in trajectory:
            if arm_index == 0:
                action = self.compose_action(left_qpos=qpos)
            else:
                action = self.compose_action(right_qpos=qpos)
            result, success = self._step_and_record(action, recorder)
            if success:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_qpos_move"
                return False
        for _ in range(int(settle_steps)):
            if arm_index == 0:
                action = self.compose_action(left_qpos=target_qpos)
            else:
                action = self.compose_action(right_qpos=target_qpos)
            result, success = self._step_and_record(action, recorder)
            if success:
                self.last_status = "success"
                return True
            if result.terminated or result.truncated:
                self.last_status = "terminated_during_qpos_move"
                return False
        return False

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

    def open_gripper(
        self,
        arm_index: int,
        recorder,
        *,
        pos: float = 1.0,
        fail_status: str | None = None,
        allow_success_early: bool = True,
    ) -> bool:
        success = self.hold_pose(
            recorder,
            steps=self.RELEASE_SETTLE_STEPS,
            left_gripper=pos if arm_index == 0 else None,
            right_gripper=pos if arm_index == 1 else None,
            allow_success_early=allow_success_early,
        )
        if not success and fail_status is not None and self.last_status not in ("running", "success"):
            self.last_status = fail_status
        return success

    def close_gripper(
        self,
        arm_index: int,
        recorder,
        *,
        pos: float = 0.0,
        fail_status: str | None = None,
        allow_success_early: bool = True,
        smooth_steps: int | None = None,
        settle_steps: int | None = None,
        step_callback: Callable[[dict[str, Any]], None] | None = None,
        paired_gripper_arm_index: int | None = None,
        paired_gripper_pos: float | None = None,
        paired_gripper_release_after_smooth: bool = False,
    ) -> bool:
        if smooth_steps is None:
            success = self.hold_pose(
                recorder,
                steps=self.GRIPPER_SETTLE_STEPS,
                left_gripper=pos if arm_index == 0 else None,
                right_gripper=pos if arm_index == 1 else None,
                allow_success_early=allow_success_early,
            )
        else:
            success = False
            start_pos = float(self.commanded_gripper[arm_index])
            step_count = max(1, int(smooth_steps))
            target_positions = np.linspace(start_pos, float(pos), num=step_count, dtype=np.float64)
            paired_positions = None
            paired_start_pos = None
            if paired_gripper_arm_index is not None and paired_gripper_pos is not None:
                paired_start_pos = float(self.commanded_gripper[int(paired_gripper_arm_index)])
                if paired_gripper_release_after_smooth:
                    paired_positions = np.full(step_count, paired_start_pos, dtype=np.float64)
                else:
                    paired_positions = np.linspace(
                        paired_start_pos,
                        float(paired_gripper_pos),
                        num=step_count,
                        dtype=np.float64,
                    )
            for step_index, gripper_pos in enumerate(target_positions):
                left_gripper = float(gripper_pos) if arm_index == 0 else None
                right_gripper = float(gripper_pos) if arm_index == 1 else None
                paired_gripper = None if paired_positions is None else float(paired_positions[step_index])
                if paired_gripper is not None:
                    if int(paired_gripper_arm_index) == 0:
                        left_gripper = paired_gripper
                    else:
                        right_gripper = paired_gripper
                action = self.compose_action(
                    left_gripper=left_gripper,
                    right_gripper=right_gripper,
                )
                result, step_success = self._step_and_record(action, recorder)
                if step_callback is not None:
                    step_callback(
                        {
                            "phase": "smooth",
                            "step_index": int(step_index),
                            "commanded_gripper": float(gripper_pos),
                            "paired_commanded_gripper": paired_gripper,
                            "global_step": int(self.steps_taken),
                        }
                    )
                if step_success and allow_success_early:
                    self.last_status = "success"
                    success = True
                    break
                if result.terminated or result.truncated:
                    self.last_status = "terminated_during_gripper_smooth_close"
                    break
            if not success and self.last_status == "running":
                settle_count = self.GRIPPER_SETTLE_STEPS if settle_steps is None else max(0, int(settle_steps))
                paired_settle_positions = None
                if (
                    paired_gripper_release_after_smooth
                    and paired_gripper_arm_index is not None
                    and paired_gripper_pos is not None
                    and paired_start_pos is not None
                    and settle_count > 0
                ):
                    paired_settle_positions = np.linspace(
                        float(paired_start_pos),
                        float(paired_gripper_pos),
                        num=settle_count + 1,
                        dtype=np.float64,
                    )[1:]
                for step_index in range(settle_count):
                    left_gripper = pos if arm_index == 0 else None
                    right_gripper = pos if arm_index == 1 else None
                    if paired_gripper_arm_index is not None and paired_gripper_pos is not None:
                        paired_gripper = (
                            float(paired_settle_positions[step_index])
                            if paired_settle_positions is not None
                            else float(paired_gripper_pos)
                        )
                        if int(paired_gripper_arm_index) == 0:
                            left_gripper = paired_gripper
                        else:
                            right_gripper = paired_gripper
                    else:
                        paired_gripper = None
                    action = self.compose_action(
                        left_gripper=left_gripper,
                        right_gripper=right_gripper,
                    )
                    result, step_success = self._step_and_record(action, recorder)
                    if step_callback is not None:
                        step_callback(
                            {
                                "phase": "settle",
                                "step_index": int(step_index),
                                "commanded_gripper": float(pos),
                                "paired_commanded_gripper": paired_gripper,
                                "global_step": int(self.steps_taken),
                            }
                        )
                    if step_success and allow_success_early:
                        self.last_status = "success"
                        success = True
                        break
                    if result.terminated or result.truncated:
                        self.last_status = "terminated_during_gripper_smooth_close"
                        break
        if not success and fail_status is not None and self.last_status not in ("running", "success"):
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
        preferred_top_down_quat: np.ndarray | None = None,
        pre_pose_candidates_builder: Callable[[np.ndarray, int, float, int], list[np.ndarray]] | None = None,
        selection_mode: str = "auto",
        max_grasp_plan_length: int | None = None,
    ) -> GraspCandidate | None:
        self.sync_sim_state()

        def pose_list(value: np.ndarray) -> list[float]:
            return np.asarray(value, dtype=np.float64).round(4).tolist()

        def better_candidate(
            lhs: GraspCandidate | None,
            rhs: GraspCandidate | None,
            *,
            score_attr: str,
        ) -> bool:
            if lhs is None:
                return False
            if rhs is None:
                return True
            lhs_score = float(getattr(lhs, score_attr))
            rhs_score = float(getattr(rhs, score_attr))
            score_tol = 0.03
            if lhs_score < rhs_score - score_tol:
                return True
            if lhs_score > rhs_score + score_tol:
                return False
            shoulder_tol = 0.05
            if lhs.shoulder_pan_delta < rhs.shoulder_pan_delta - shoulder_tol:
                return True
            if lhs.shoulder_pan_delta > rhs.shoulder_pan_delta + shoulder_tol:
                return False
            motion_tol = 0.1
            if lhs.joint_motion < rhs.joint_motion - motion_tol:
                return True
            if lhs.joint_motion > rhs.joint_motion + motion_tol:
                return False
            return lhs.path_length < rhs.path_length

        def lower_motion_candidate(lhs: GraspCandidate | None, rhs: GraspCandidate | None) -> bool:
            if lhs is None:
                return False
            if rhs is None:
                return True
            shoulder_tol = 0.05
            if lhs.shoulder_pan_delta < rhs.shoulder_pan_delta - shoulder_tol:
                return True
            if lhs.shoulder_pan_delta > rhs.shoulder_pan_delta + shoulder_tol:
                return False
            motion_tol = 0.1
            if lhs.joint_motion < rhs.joint_motion - motion_tol:
                return True
            if lhs.joint_motion > rhs.joint_motion + motion_tol:
                return False
            if lhs.path_length < rhs.path_length:
                return True
            if lhs.path_length > rhs.path_length:
                return False
            return lhs.score < rhs.score

        current_qpos = self.current_full_qpos(arm_index)
        current_arm_qpos = self.current_qpos(arm_index)
        current_ee_pose = self.current_ee_pose(arm_index)
        current_gripper_pose = self.current_gripper_pose(arm_index)
        planner_base_pose = self.base_env.get_arm_root_pose_array(arm_index)
        best: GraspCandidate | None = None
        best_top_down: GraspCandidate | None = None
        best_side: GraspCandidate | None = None
        best_low_motion: GraspCandidate | None = None
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
                            "pre_pose": pose_list(pre_pose),
                            "planner_pre_pose": pose_list(planner_pre_pose),
                            "grasp_pose": pose_list(grasp_pose),
                            "planner_grasp_pose": pose_list(planner_grasp_pose),
                            "desired_pose": pose_list(desired_pose),
                            "current_ee_pose": pose_list(current_ee_pose),
                            "current_gripper_pose": pose_list(current_gripper_pose),
                            "planner_base_pose": pose_list(planner_base_pose),
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
                grasp_plan_source = "screw"
                if (
                    grasp_plan is not None
                    and max_grasp_plan_length is not None
                    and len(grasp_plan["position"]) > int(max_grasp_plan_length)
                ):
                    grasp_reason = (
                        f"rejected:screw_path_length "
                        f"{len(grasp_plan['position'])}>{int(max_grasp_plan_length)}"
                    )
                    grasp_plan = None
                if grasp_plan is None:
                    grasp_plan, grasp_reason = self.plan_to_pose_debug(
                        arm_index,
                        planner_grasp_pose,
                        start_qpos=next_qpos,
                    )
                    if grasp_plan is not None:
                        grasp_plan_source = "pose_fallback"
                        if (
                            max_grasp_plan_length is not None
                            and len(grasp_plan["position"]) > int(max_grasp_plan_length)
                        ):
                            grasp_reason = (
                                f"rejected:pose_path_length "
                                f"{len(grasp_plan['position'])}>{int(max_grasp_plan_length)}"
                            )
                            grasp_plan = None
                if grasp_plan is None:
                    attempts.append(
                        {
                            "contact_id": int(contact_id),
                            "variant_index": int(variant_index),
                            "pre_pose": pose_list(pre_pose),
                            "planner_pre_pose": pose_list(planner_pre_pose),
                            "grasp_pose": pose_list(grasp_pose),
                            "planner_grasp_pose": pose_list(planner_grasp_pose),
                            "desired_pose": pose_list(desired_pose),
                            "current_ee_pose": pose_list(current_ee_pose),
                            "current_gripper_pose": pose_list(current_gripper_pose),
                            "planner_base_pose": pose_list(planner_base_pose),
                            "pre_reason": pre_reason,
                            "grasp_reason": grasp_reason,
                            "grasp_plan_source": None,
                            "max_grasp_plan_length": (
                                None if max_grasp_plan_length is None else int(max_grasp_plan_length)
                            ),
                        }
                    )
                    continue
                candidate = GraspCandidate(
                    contact_id=contact_id,
                    pre_pose=pre_pose,
                    grasp_pose=grasp_pose,
                    pre_plan=pre_plan,
                    grasp_plan=grasp_plan,
                    grasp_plan_source=grasp_plan_source,
                    score=0.0,
                    top_down_score=(
                        quat_distance(grasp_pose[3:], preferred_top_down_quat)
                        if preferred_top_down_quat is not None
                        else np.inf
                    ),
                    side_score=quat_distance(grasp_pose[3:], preferred_quat),
                    path_length=len(pre_plan["position"]) + len(grasp_plan["position"]),
                    shoulder_pan_delta=float(
                        np.abs(self._arm_qpos_from_plan(pre_plan["position"][-1])[0] - current_arm_qpos[0])
                    ),
                    joint_motion=float(
                        np.linalg.norm(self._arm_qpos_from_plan(pre_plan["position"][-1]) - current_arm_qpos)
                        + np.linalg.norm(
                            self._arm_qpos_from_plan(grasp_plan["position"][-1])
                            - self._arm_qpos_from_plan(pre_plan["position"][-1])
                        )
                    ),
                )
                candidate.score = (
                    0.7 * candidate.top_down_score + 0.3 * candidate.side_score
                    if preferred_top_down_quat is not None
                    else candidate.side_score
                )
                if best_for_contact is None or better_candidate(candidate, best_for_contact, score_attr="score"):
                    best_for_contact = candidate
                if lower_motion_candidate(candidate, best_low_motion):
                    best_low_motion = candidate
                attempts.append(
                    {
                        "contact_id": int(contact_id),
                        "variant_index": int(variant_index),
                        "pre_pose": pose_list(pre_pose),
                        "planner_pre_pose": pose_list(planner_pre_pose),
                        "grasp_pose": pose_list(grasp_pose),
                        "planner_grasp_pose": pose_list(planner_grasp_pose),
                        "desired_pose": pose_list(desired_pose),
                        "current_ee_pose": pose_list(current_ee_pose),
                        "current_gripper_pose": pose_list(current_gripper_pose),
                        "planner_base_pose": pose_list(planner_base_pose),
                        "pre_reason": pre_reason,
                        "grasp_reason": grasp_reason,
                        "grasp_plan_source": grasp_plan_source,
                        "pre_plan_len": int(len(pre_plan["position"])),
                        "grasp_plan_len": int(len(grasp_plan["position"])),
                    }
                )
            if best_for_contact is None:
                continue
            if better_candidate(best_for_contact, best_top_down, score_attr="top_down_score"):
                best_top_down = best_for_contact
            if better_candidate(best_for_contact, best_side, score_attr="side_score"):
                best_side = best_for_contact
            if better_candidate(best_for_contact, best, score_attr="score"):
                best = best_for_contact
        self.last_grasp_attempts = attempts
        if selection_mode == "prefer_top_down":
            return best_top_down if best_top_down is not None else best
        if selection_mode == "prefer_side":
            return best_side if best_side is not None else best
        if selection_mode == "prefer_low_motion":
            return best_low_motion if best_low_motion is not None else best
        if best_top_down is not None and best_top_down.top_down_score < 0.15:
            return best_top_down
        if best_side is not None and best_side.side_score < 0.15:
            return best_side
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
        preferred_top_down_quat: np.ndarray | None,
        recorder,
        fail_status: str,
        gripper_pos: float = 0.0,
        pre_close_settle_steps: int = 0,
        pre_close_paired_gripper_arm_index: int | None = None,
        pre_close_paired_gripper_pos: float | None = None,
        pre_close_paired_gripper_steps: int = 0,
        close_smooth_steps: int | None = None,
        close_settle_steps: int | None = None,
        close_step_callback: Callable[[dict[str, Any]], None] | None = None,
        paired_close_gripper_arm_index: int | None = None,
        paired_close_gripper_pos: float | None = None,
        paired_close_gripper_release_after_smooth: bool = False,
        freeze_current_qpos_before_close: bool = False,
        pre_pose_candidates_builder: Callable[[np.ndarray, int, float, int], list[np.ndarray]] | None = None,
        selection_mode: str = "auto",
    ) -> tuple[GraspCandidate | None, bool]:
        candidate = self.choose_grasp_candidate(
            desired_pose=desired_pose,
            arm_index=arm_index,
            contact_ids=contact_ids,
            pre_distance=pre_grasp_dis,
            grasp_distance=grasp_dis,
            pose_builder=pose_builder,
            preferred_quat=preferred_quat,
            preferred_top_down_quat=preferred_top_down_quat,
            pre_pose_candidates_builder=pre_pose_candidates_builder,
            selection_mode=selection_mode,
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
        if freeze_current_qpos_before_close and pre_close_settle_steps > 0:
            success = self.hold_pose(
                recorder,
                steps=int(pre_close_settle_steps),
                allow_success_early=False,
            )
            if success:
                return candidate, True
            pre_close_settle_steps = 0
        if freeze_current_qpos_before_close:
            # Stop arm drift while the fingers close. If the hand is still too
            # far from contact, the task-level grasp validation should reject
            # the sample instead of letting the arm push the object around.
            self.commanded_arm_qpos[int(arm_index)] = self.current_qpos(int(arm_index)).copy()
            for paired_arm_index in (pre_close_paired_gripper_arm_index, paired_close_gripper_arm_index):
                if paired_arm_index is not None:
                    self.commanded_arm_qpos[int(paired_arm_index)] = self.current_qpos(int(paired_arm_index)).copy()
        if (
            pre_close_paired_gripper_arm_index is not None
            and pre_close_paired_gripper_pos is not None
            and pre_close_paired_gripper_steps > 0
        ):
            success = self.close_gripper(
                int(pre_close_paired_gripper_arm_index),
                recorder,
                pos=float(pre_close_paired_gripper_pos),
                allow_success_early=False,
                smooth_steps=int(pre_close_paired_gripper_steps),
                settle_steps=0,
            )
            if success:
                return candidate, True
        if pre_close_settle_steps > 0:
            success = self.hold_pose(
                recorder,
                steps=int(pre_close_settle_steps),
                allow_success_early=False,
            )
            if success:
                return candidate, True
        success = self.close_gripper(
            arm_index,
            recorder,
            pos=gripper_pos,
            smooth_steps=close_smooth_steps,
            settle_steps=close_settle_steps,
            step_callback=close_step_callback,
            paired_gripper_arm_index=paired_close_gripper_arm_index,
            paired_gripper_pos=paired_close_gripper_pos,
            paired_gripper_release_after_smooth=paired_close_gripper_release_after_smooth,
        )
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
        allow_success_on_release: bool = True,
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
            success = self.open_gripper(
                arm_index,
                recorder,
                allow_success_early=allow_success_on_release,
            )
            if success:
                return True
        return False
