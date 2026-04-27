from __future__ import annotations

import copy
import numpy as np

from sim.maniskill.expert.primitives import (
    EpisodeLimitReached,
    FixedRateMotionPrimitives,
    GraspCandidate,
    matrix_to_pose,
    offset_pose_along_local_x,
    pose_to_matrix,
    scaled_local_matrix,
    to_numpy,
)
from sim.maniskill.tasks.handover_block_env import (
    BLOCK_MIDDLE_POSE,
    BOX_HALF_SIZE,
    TARGET_HALF_SIZE,
)


CONTACT_POSE_TO_GRASP_FRAME = np.array(
    [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)
LONG_BOX_CONTACT_MATRICES = np.asarray(
    [
        [[0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -0.7], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, -0.7], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, -0.7], [0.0, 0.0, 0.0, 1.0]],
        [[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, -0.7], [0.0, 0.0, 0.0, 1.0]],
    ],
    dtype=np.float64,
)
BOX_FUNCTIONAL_MATRICES = np.asarray(
    [
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
    ],
    dtype=np.float64,
)
PREFERRED_GRASP_QUATS = {
    0: np.asarray([-0.853532, 0.146484, -0.353542, -0.3536], dtype=np.float64),
    1: np.asarray([-0.353518, 0.353564, -0.14642, -0.853568], dtype=np.float64),
}
GRASP_DIRECTION_DIC = {
    "left": np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float64),
    "front_left": np.asarray([-0.383, 0.0, 0.0, -0.924], dtype=np.float64),
    "front": np.asarray([-0.707, 0.0, 0.0, -0.707], dtype=np.float64),
    "front_right": np.asarray([-0.924, 0.0, 0.0, -0.383], dtype=np.float64),
    "right": np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "top_down": np.asarray([-0.5, 0.5, -0.5, -0.5], dtype=np.float64),
    "down_right": np.asarray([-0.707, 0.0, -0.707, 0.0], dtype=np.float64),
    "down_left": np.asarray([0.0, 0.707, 0.0, -0.707], dtype=np.float64),
    "top_down_little_left": np.asarray([-0.353523, 0.61239, -0.353524, -0.61239], dtype=np.float64),
    "top_down_little_right": np.asarray([-0.61239, 0.353523, -0.61239, -0.353524], dtype=np.float64),
    "left_arm_perf": np.asarray([-0.853532, 0.146484, -0.353542, -0.3536], dtype=np.float64),
    "right_arm_perf": np.asarray([-0.353518, 0.353564, -0.14642, -0.853568], dtype=np.float64),
}
TOP_DOWN_PREFERRED_GRASP_QUATS = {
    0: np.asarray([-0.61239, 0.353523, -0.61239, -0.353524], dtype=np.float64),
    1: np.asarray([-0.353523, 0.61239, -0.353524, -0.61239], dtype=np.float64),
}
LEFT_GRASP_CONTACT_IDS = (0,)
RIGHT_GRASP_CONTACT_IDS = (4, 5, 6, 7)
RIGHT_GRASP_PREFERRED_X_AXIS_CONTACT_IDS = (5, 6)


class _NullRecorder:
    def record_step(self, *args, **kwargs) -> None:
        return None

    def append_step(self, *args, **kwargs) -> None:
        return None


class HandoverBlockSolver:
    PRE_GRASP_DISTANCE = 0.12
    HANDOVER_PRE_GRASP_DISTANCE = 0.08
    GRASP_DISTANCE = 0.0
    LEFT_FIRST_GRASP_DISTANCE = -0.02
    RIGHT_HANDOVER_GRASP_DISTANCE = 0.0
    ROTATE_NUM = 10
    LEFT_ROTATE_LIM = (-0.6, 0.6)
    RIGHT_ROTATE_LIM = (-0.6, 0.6)
    LEFT_FIRST_GRASP_YAW_VARIANTS = (0.0,)
    LEFT_LIFT_DISTANCE = 0.16
    LEFT_HANDOVER_CLEARANCE_Z = 0.18
    LEFT_RELEASE_LIFT_DISTANCE = 0.0
    LEFT_FIRST_GRASP_CLOSE_STEPS = 18
    LEFT_FIRST_GRASP_CLAMP_LIFT_DISTANCE = 0.05
    LEFT_FIRST_GRASP_CLAMP_LIFT_QPOS_STEPS = 10
    LEFT_FIRST_GRASP_CLAMP_LIFT_SETTLE_STEPS = 8
    LEFT_FIRST_GRASP_APPROACH_OFFSET = np.asarray([0.0, -0.10, 0.0], dtype=np.float64)
    ALLOW_LEFT_FIRST_GRASP_APPROACH_FALLBACK = True
    LEFT_FIRST_GRASP_BOX_AXIS_SHIFT = 0.04
    RIGHT_END_GRASP_BOX_AXIS_SHIFT = -0.03
    ABOVE_TARGET_OFFSET = np.asarray([0.05, -0.12, 0.20], dtype=np.float64)
    HANDOVER_APPROACH_OFFSET = np.asarray([0.0, -0.07, 0.10], dtype=np.float64)
    RIGHT_PLACE_TARGET_OFFSET = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    RIGHT_PRE_PLACE_WORLD_OFFSET = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    RIGHT_PLACE_WORLD_OFFSET = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    USE_SELECTED_PLACE_CANDIDATE_AT_RUNTIME = False
    RETURN_LEFT_HOME_AFTER_HANDOVER = False
    RIGHT_CARRY_RETRACT_SEGMENTS = ((0.02, -0.02, 0.0), (0.02, -0.02, 0.0))
    RIGHT_CARRY_LIFT_SEGMENTS = (0.03, 0.03, 0.03, 0.03, 0.02)
    RIGHT_CARRY_RELAX_BLEND = 1.0
    RIGHT_CARRY_RELAX_STEPS = 4
    RIGHT_CARRY_RELAX_SETTLE_STEPS = 2
    ENABLE_FUTURE_PLACE_SELECTION = False
    PREFER_RIGHT_HANDOVER_X_AXIS_CONTACTS = True
    ENABLE_RIGHT_HANDOVER_BRANCH_TRIAL = False
    RIGHT_HANDOVER_PLACE_VARIANT_TRIAL_LIMIT = 1
    RIGHT_HANDOVER_TOTAL_TRIAL_BRANCH_LIMIT = 2
    RIGHT_HANDOVER_TRIAL_SWITCH_DISTANCE_MARGIN = 0.02
    RIGHT_CARRY_TRANSITION_WAYPOINTS = 4
    RIGHT_RELEASE_LIFT_DISTANCE = 0.12
    RIGHT_RELEASE_RETRACT_DISTANCE = 0.03
    RIGHT_PLACE_SETTLE_STEPS = 3
    RIGHT_PLACE_CORRECTION_ITERS = 3
    RIGHT_PLACE_CORRECTION_EPS = 0.01
    RIGHT_PLACE_CORRECTION_MAX_STEP = 0.06
    RIGHT_PRE_PLACE_FALLBACK_IK_THRESHOLD = 1e-2
    RIGHT_PRE_PLACE_FALLBACK_STEPS = 12
    RIGHT_PRE_PLACE_FALLBACK_SETTLE_STEPS = 4
    RIGHT_PLACE_FALLBACK_IK_THRESHOLD = 5e-2
    RIGHT_PLACE_FALLBACK_MAX_SHOULDER_DELTA = 0.8
    RIGHT_PLACE_FALLBACK_MAX_JOINT_MOTION = 1.5
    RIGHT_PLACE_FALLBACK_STEPS = 12
    RIGHT_PLACE_FALLBACK_SETTLE_STEPS = 4
    RIGHT_PLACE_RELATIVE_FALLBACK_MAX_STEP = 0.02
    RIGHT_PLACE_RELATIVE_FALLBACK_IK_THRESHOLD = 6e-2
    RIGHT_PLACE_RELATIVE_FALLBACK_STEPS = 6
    RIGHT_PLACE_RELATIVE_FALLBACK_SETTLE_STEPS = 1
    RIGHT_PLACE_ALLOW_RELEASE_FROM_PRE_PLACE = True
    RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_XY_ERROR = 0.06
    RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_Z_ABOVE = 0.12
    RIGHT_HANDOVER_DUAL_HOLD_STEPS = 0
    RIGHT_HANDOVER_PRE_CLOSE_SETTLE_STEPS = 8
    RIGHT_HANDOVER_CLOSE_SMOOTH_STEPS = 16
    RIGHT_HANDOVER_CLOSE_SETTLE_STEPS = 12
    RIGHT_HANDOVER_POST_CLOSE_HOLD_STEPS = 8
    RIGHT_HANDOVER_GRIPPER_HOLD_POS = 0.0
    LEFT_PRERELEASE_BEFORE_RIGHT_HANDOVER_CLOSE = 0.0
    LEFT_PRERELEASE_BEFORE_RIGHT_HANDOVER_CLOSE_STEPS = 0
    LEFT_RELEASE_DURING_RIGHT_HANDOVER_CLOSE = 1.0
    LEFT_RELEASE_AFTER_RIGHT_HANDOVER_CLOSE = True
    LEFT_RELEASE_OPEN_SMOOTH_STEPS = 16
    LEFT_RELEASE_OPEN_SETTLE_STEPS = 6
    LEFT_RELEASE_CLEAR_WORLD_DELTA = np.asarray([0.0, -0.05, 0.04], dtype=np.float64)
    LEFT_RELEASE_CLEAR_LIFT_DELTA = np.asarray([0.0, 0.0, 0.06], dtype=np.float64)
    LEFT_RELEASE_CLEAR_RETREAT_DELTA = np.asarray([0.0, -0.10, 0.02], dtype=np.float64)
    LEFT_RELEASE_CLEAR_STEPS = 16
    LEFT_RELEASE_CLEAR_SETTLE_STEPS = 4
    FINAL_RELEASE_SETTLE_STEPS = 4
    FINAL_SETTLE_STEPS = 12

    def __init__(self, env_backend):
        self.backend = env_backend
        self.base_env = env_backend.env.unwrapped
        self.ctrl = FixedRateMotionPrimitives(env_backend)
        self.box_half_size = to_numpy(BOX_HALF_SIZE).reshape(3)
        self.target_half_size = to_numpy(TARGET_HALF_SIZE).reshape(3)
        self.last_debug_info: dict[str, object] = {}
        self.first_pick_debug: dict[str, object] = {}

    def _set_success_gate_enabled(self, enabled: bool) -> bool | None:
        if not hasattr(self.base_env, "success_gate_enabled"):
            return None
        previous = bool(self.base_env.success_gate_enabled)
        self.base_env.success_gate_enabled = bool(enabled)
        return previous

    @property
    def last_status(self) -> str:
        return self.ctrl.last_status

    def _box_pose(self) -> np.ndarray:
        self.ctrl.sync_sim_state()
        return np.concatenate(
            [
                to_numpy(self.base_env.box.pose.p).reshape(-1, 3)[0],
                to_numpy(self.base_env.box.pose.q).reshape(-1, 4)[0],
            ],
            axis=0,
        ).astype(np.float64, copy=False)

    def _target_pose(self) -> np.ndarray:
        self.ctrl.sync_sim_state()
        return np.concatenate(
            [
                to_numpy(self.base_env.target_box.pose.p).reshape(-1, 3)[0],
                to_numpy(self.base_env.target_box.pose.q).reshape(-1, 4)[0],
            ],
            axis=0,
        ).astype(np.float64, copy=False)

    def _box_contact_world_pose(self, box_pose: np.ndarray, contact_id: int) -> np.ndarray:
        local_matrix = scaled_local_matrix(LONG_BOX_CONTACT_MATRICES[contact_id], self.box_half_size)
        world_matrix = pose_to_matrix(box_pose) @ local_matrix
        return matrix_to_pose(world_matrix)

    def _box_functional_world_pose(self, box_pose: np.ndarray, functional_id: int) -> np.ndarray:
        local_matrix = scaled_local_matrix(BOX_FUNCTIONAL_MATRICES[functional_id], self.box_half_size)
        world_matrix = pose_to_matrix(box_pose) @ local_matrix
        return matrix_to_pose(world_matrix)

    def _target_functional_world_pose(self, functional_id: int) -> np.ndarray:
        local_matrix = scaled_local_matrix(BOX_FUNCTIONAL_MATRICES[functional_id], self.target_half_size)
        world_matrix = pose_to_matrix(self._target_pose()) @ local_matrix
        return matrix_to_pose(world_matrix)

    def _solve_box_pose_from_functional_target(self, target_pose: np.ndarray, functional_id: int) -> np.ndarray:
        local_matrix = scaled_local_matrix(BOX_FUNCTIONAL_MATRICES[functional_id], self.box_half_size)
        box_matrix = pose_to_matrix(target_pose) @ np.linalg.inv(local_matrix)
        return matrix_to_pose(box_matrix)

    def _grasp_pose_from_box_pose(self, box_pose: np.ndarray, contact_id: int, pre_distance: float) -> np.ndarray:
        contact_matrix = pose_to_matrix(self._box_contact_world_pose(box_pose, contact_id))
        grasp_frame = contact_matrix @ CONTACT_POSE_TO_GRASP_FRAME
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = grasp_frame[:3, :3]
        pose[:3, 3] = grasp_frame[:3, 3] + grasp_frame[:3, :3] @ np.array([-0.12 - pre_distance, 0.0, 0.0])
        return matrix_to_pose(pose)

    def _rotate_pose_along_target_axis(
        self,
        target_pose: np.ndarray,
        center_pose: np.ndarray,
        *,
        axis: np.ndarray,
        theta: float,
        towards: np.ndarray | None = None,
    ) -> np.ndarray:
        if np.isclose(theta, 0.0):
            return np.asarray(target_pose, dtype=np.float64).copy()
        target_matrix = pose_to_matrix(target_pose)
        center_matrix = pose_to_matrix(center_pose)
        world_axis = target_matrix[:3, :3] @ np.asarray(axis, dtype=np.float64).reshape(3)
        world_axis = world_axis / np.linalg.norm(world_axis)
        rotvec = world_axis * float(theta)
        theta_norm = np.linalg.norm(rotvec)
        kx, ky, kz = rotvec / theta_norm
        K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=np.float64)
        rotate_mat = np.eye(3, dtype=np.float64) + np.sin(theta_norm) * K + (1.0 - np.cos(theta_norm)) * (K @ K)
        translated = rotate_mat @ (target_matrix[:3, 3] - center_matrix[:3, 3]) + center_matrix[:3, 3]
        if towards is not None and np.dot(translated - center_matrix[:3, 3], np.asarray(towards, dtype=np.float64)) < 0:
            return self._rotate_pose_along_target_axis(
                target_pose,
                center_pose,
                axis=axis,
                theta=-theta,
                towards=None,
            )
        rotated = np.eye(4, dtype=np.float64)
        rotated[:3, :3] = rotate_mat @ target_matrix[:3, :3]
        rotated[:3, 3] = translated
        return matrix_to_pose(rotated)

    def _get_rotate_limits(self, arm_index: int) -> tuple[float, float]:
        config = self.base_env.embodiment_spec.left_config if arm_index == 0 else self.base_env.embodiment_spec.right_config
        rotate_lim = config.get("rotate_lim")
        if isinstance(rotate_lim, (list, tuple)) and len(rotate_lim) >= 2 and not np.isclose(float(rotate_lim[0]), float(rotate_lim[1])):
            return float(rotate_lim[0]), float(rotate_lim[1])
        return self.LEFT_ROTATE_LIM if arm_index == 0 else self.RIGHT_ROTATE_LIM

    def _preferred_grasp_quat(self, arm_index: int) -> np.ndarray:
        config = self.base_env.embodiment_spec.left_config if arm_index == 0 else self.base_env.embodiment_spec.right_config
        perfect_direction = config.get("grasp_perfect_direction", ["front_right", "front_left"])
        if isinstance(perfect_direction, (list, tuple)):
            direction_name = str(perfect_direction[min(arm_index, len(perfect_direction) - 1)])
        else:
            direction_name = str(perfect_direction)
        return GRASP_DIRECTION_DIC.get(direction_name, PREFERRED_GRASP_QUATS[arm_index]).copy()

    def _rotate_pose_about_world_z(
        self,
        target_pose: np.ndarray,
        center_pose: np.ndarray,
        theta: float,
    ) -> np.ndarray:
        if np.isclose(theta, 0.0):
            return np.asarray(target_pose, dtype=np.float64).copy()
        target_matrix = pose_to_matrix(target_pose)
        center_matrix = pose_to_matrix(center_pose)
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        rotate_mat = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rotated = np.eye(4, dtype=np.float64)
        rotated[:3, :3] = rotate_mat @ target_matrix[:3, :3]
        rotated[:3, 3] = rotate_mat @ (target_matrix[:3, 3] - center_matrix[:3, 3]) + center_matrix[:3, 3]
        return matrix_to_pose(rotated)

    def _grasp_pre_pose_candidates_from_box_pose(
        self,
        box_pose: np.ndarray,
        contact_id: int,
        pre_distance: float,
        arm_index: int,
    ) -> list[np.ndarray]:
        base_pose = self._grasp_pose_from_box_pose(box_pose, contact_id, pre_distance)
        center_pose = self._box_contact_world_pose(box_pose, contact_id)
        rotate_min, rotate_max = self._get_rotate_limits(arm_index)
        angles = np.linspace(rotate_min, rotate_max, num=self.ROTATE_NUM, dtype=np.float64)
        candidates = [np.asarray(base_pose, dtype=np.float64)]
        for theta in angles:
            if np.isclose(theta, 0.0):
                continue
            candidates.append(
                self._rotate_pose_along_target_axis(
                    base_pose,
                    center_pose,
                    axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    theta=float(theta),
                    towards=np.array([0.0, -1.0, 0.0], dtype=np.float64),
                )
            )
        return candidates

    def _left_first_grasp_pre_pose_candidates_from_box_pose(
        self,
        box_pose: np.ndarray,
        contact_id: int,
        pre_distance: float,
        arm_index: int,
    ) -> list[np.ndarray]:
        box_axis = pose_to_matrix(box_pose)[:3, 2]
        box_axis = box_axis / np.linalg.norm(box_axis)
        shift_vec = box_axis * self.LEFT_FIRST_GRASP_BOX_AXIS_SHIFT
        base_pose = self._grasp_pose_from_box_pose(box_pose, contact_id, pre_distance)
        base_pose = np.asarray(base_pose, dtype=np.float64).copy()
        base_pose[:3] += shift_vec
        center_pose = self._box_contact_world_pose(box_pose, contact_id)
        center_pose = np.asarray(center_pose, dtype=np.float64).copy()
        center_pose[:3] += shift_vec
        candidates = []
        for theta in self.LEFT_FIRST_GRASP_YAW_VARIANTS:
            candidates.append(self._rotate_pose_about_world_z(base_pose, center_pose, float(theta)))
        return candidates

    def _right_end_grasp_pre_pose_candidates_from_box_pose(
        self,
        box_pose: np.ndarray,
        contact_id: int,
        pre_distance: float,
        arm_index: int,
    ) -> list[np.ndarray]:
        box_axis = pose_to_matrix(box_pose)[:3, 2]
        box_axis = box_axis / np.linalg.norm(box_axis)
        shift_vec = -box_axis * self.RIGHT_END_GRASP_BOX_AXIS_SHIFT
        base_pose = self._grasp_pose_from_box_pose(box_pose, contact_id, pre_distance)
        base_pose = np.asarray(base_pose, dtype=np.float64).copy()
        base_pose[:3] += shift_vec
        center_pose = self._box_contact_world_pose(box_pose, contact_id)
        center_pose = np.asarray(center_pose, dtype=np.float64).copy()
        center_pose[:3] += shift_vec
        rotate_min, rotate_max = self._get_rotate_limits(arm_index)
        angles = np.linspace(rotate_min, rotate_max, num=self.ROTATE_NUM, dtype=np.float64)
        candidates = [np.asarray(base_pose, dtype=np.float64)]
        for theta in angles:
            if np.isclose(theta, 0.0):
                continue
            candidates.append(
                self._rotate_pose_along_target_axis(
                    base_pose,
                    center_pose,
                    axis=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                    theta=float(theta),
                    towards=np.array([0.0, -1.0, 0.0], dtype=np.float64),
                )
            )
        return candidates

    def _get_handover_approach_pose(self, pre_grasp_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pre_grasp_pose, dtype=np.float64).copy()
        pose[:3] += self.HANDOVER_APPROACH_OFFSET
        return pose

    def _get_left_first_grasp_approach_pose(self, pre_grasp_pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pre_grasp_pose, dtype=np.float64).copy()
        pose[:3] += self.LEFT_FIRST_GRASP_APPROACH_OFFSET
        return pose

    def _place_pose_from_current_grasp(
        self,
        *,
        arm_index: int,
        target_pose: np.ndarray,
        functional_id: int | None,
        pre_dis: float,
        pre_dis_axis: str | np.ndarray = "grasp",
    ) -> np.ndarray:
        box_pose = self._box_pose()
        box_matrix = pose_to_matrix(box_pose)
        if functional_id is None:
            place_start_pose = np.asarray(box_pose, dtype=np.float64)
        else:
            place_start_pose = self._box_functional_world_pose(box_pose, functional_id)
        place_start_matrix = pose_to_matrix(place_start_pose)
        target_pose = np.asarray(target_pose, dtype=np.float64)
        target_matrix = pose_to_matrix(target_pose)

        start_to_target_rot = target_matrix[:3, :3] @ place_start_matrix[:3, :3].T
        target_point = start_to_target_rot @ (box_matrix[:3, 3] - place_start_pose[:3]) + target_pose[:3]

        ee_pose = self.ctrl.current_gripper_pose(arm_index)
        ee_matrix = pose_to_matrix(ee_pose)
        target_grasp_matrix = start_to_target_rot @ ee_matrix[:3, :3]

        ee_to_box = box_matrix[:3, 3] - ee_pose[:3]
        ee_to_box_local = np.linalg.inv(ee_matrix[:3, :3]) @ ee_to_box
        grasp_bias = target_grasp_matrix @ ee_to_box_local

        if isinstance(pre_dis_axis, str):
            if pre_dis_axis == "grasp":
                target_dis_vec = grasp_bias.copy()
            elif pre_dis_axis == "fp":
                target_dis_vec = target_matrix[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                raise ValueError(f"Unsupported pre_dis_axis: {pre_dis_axis}")
        else:
            axis = np.asarray(pre_dis_axis, dtype=np.float64).reshape(3)
            axis /= np.linalg.norm(axis)
            target_dis_vec = target_matrix[:3, :3] @ axis

        if np.linalg.norm(target_dis_vec) < 1e-8:
            target_dis_vec = target_grasp_matrix[:, 0]
        target_dis_vec = target_dis_vec / np.linalg.norm(target_dis_vec)

        result_matrix = np.eye(4, dtype=np.float64)
        result_matrix[:3, :3] = target_grasp_matrix
        result_matrix[:3, 3] = target_point - grasp_bias - float(pre_dis) * target_dis_vec
        return matrix_to_pose(result_matrix)

    def _place_pose_by_current_grasp_translation(
        self,
        *,
        arm_index: int,
        target_pose: np.ndarray,
        pre_dis: float,
    ) -> np.ndarray:
        box_pose = self._box_pose()
        target_top_center = np.asarray(target_pose, dtype=np.float64)[:3].copy()
        desired_box_center = target_top_center + np.array([0.0, 0.0, float(self.box_half_size[2])], dtype=np.float64)
        result_pose = np.asarray(self.ctrl.current_gripper_pose(arm_index), dtype=np.float64).copy()
        result_pose[:3] += desired_box_center - box_pose[:3]
        result_pose[2] += float(pre_dis)
        return result_pose

    def _offset_pose_world(self, pose: np.ndarray, *, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
        shifted = np.asarray(pose, dtype=np.float64).copy()
        shifted[:3] += np.asarray([x, y, z], dtype=np.float64)
        return shifted

    def _box_gripper_alignment_debug(
        self,
        *,
        arm_index: int,
        planned_box_pose: np.ndarray | None = None,
        planned_gripper_pose: np.ndarray | None = None,
    ) -> dict[str, object]:
        box_pose = np.asarray(self._box_pose(), dtype=np.float64)
        gripper_pose = np.asarray(self.ctrl.current_gripper_pose(arm_index), dtype=np.float64)
        box_to_gripper = box_pose[:3] - gripper_pose[:3]
        debug: dict[str, object] = {
            "arm_index": int(arm_index),
            "box_pose": box_pose.round(4).tolist(),
            "gripper_pose": gripper_pose.round(4).tolist(),
            "box_to_gripper_xyz": box_to_gripper.round(4).tolist(),
            "box_gripper_distance": round(float(np.linalg.norm(box_to_gripper)), 4),
        }
        if planned_box_pose is not None:
            planned_box_pose = np.asarray(planned_box_pose, dtype=np.float64)
            box_to_planned = box_pose[:3] - planned_box_pose[:3]
            debug.update(
                {
                    "planned_box_pose": planned_box_pose.round(4).tolist(),
                    "box_to_planned_xyz": box_to_planned.round(4).tolist(),
                    "box_planned_distance": round(float(np.linalg.norm(box_to_planned)), 4),
                }
            )
        if planned_gripper_pose is not None:
            planned_gripper_pose = np.asarray(planned_gripper_pose, dtype=np.float64)
            gripper_to_planned = gripper_pose[:3] - planned_gripper_pose[:3]
            debug.update(
                {
                    "planned_gripper_pose": planned_gripper_pose.round(4).tolist(),
                    "gripper_to_planned_xyz": gripper_to_planned.round(4).tolist(),
                    "gripper_planned_distance": round(float(np.linalg.norm(gripper_to_planned)), 4),
                }
            )
        return debug

    def _future_place_feasibility_for_contact(
        self,
        *,
        contact_id: int,
        handover_candidate: GraspCandidate,
    ) -> tuple[bool, float, dict[str, object] | None, list[dict[str, object]], dict[str, object]]:
        place_target_pose = self._target_functional_world_pose(1).copy()
        place_target_pose[:3] += self.RIGHT_PLACE_TARGET_OFFSET
        place_box_pose = self._solve_box_pose_from_functional_target(place_target_pose, functional_id=0)

        predicted_grasp_qpos = np.asarray(handover_candidate.grasp_plan["position"][-1], dtype=np.float64).reshape(-1)
        predicted_grasp_arm_qpos = predicted_grasp_qpos[: self.base_env.ARM_DOF]
        right_home_qpos = np.asarray(self.base_env.get_arm_home_joint_positions(1), dtype=np.float64).reshape(-1)
        relaxed_start_qpos = predicted_grasp_arm_qpos + self.RIGHT_CARRY_RELAX_BLEND * (
            right_home_qpos - predicted_grasp_arm_qpos
        )

        attempts: list[dict[str, object]] = []
        best_path_len = np.inf
        selected_pre_pose = None
        selected_grasp_pose = None
        best_candidate: dict[str, object] | None = None
        viable_candidate_count = 0
        viable_candidates: list[dict[str, object]] = []

        pre_pose_candidates = self._right_end_grasp_pre_pose_candidates_from_box_pose(
            place_box_pose,
            contact_id,
            0.08,
            1,
        )
        for variant_index, pre_pose in enumerate(pre_pose_candidates):
            grasp_pose = offset_pose_along_local_x(pre_pose, 0.08 - self.GRASP_DISTANCE)
            pre_pose = self._offset_pose_world(
                pre_pose,
                x=float(self.RIGHT_PRE_PLACE_WORLD_OFFSET[0]),
                y=float(self.RIGHT_PRE_PLACE_WORLD_OFFSET[1]),
                z=float(self.RIGHT_PRE_PLACE_WORLD_OFFSET[2]),
            )
            grasp_pose = self._offset_pose_world(
                grasp_pose,
                x=float(self.RIGHT_PLACE_WORLD_OFFSET[0]),
                y=float(self.RIGHT_PLACE_WORLD_OFFSET[1]),
                z=float(self.RIGHT_PLACE_WORLD_OFFSET[2]),
            )
            planner_pre_pose = self.ctrl.gripper_pose_to_planner_pose(1, pre_pose)
            planner_grasp_pose = self.ctrl.gripper_pose_to_planner_pose(1, grasp_pose)

            pre_plan, pre_reason = self.ctrl.plan_to_pose_debug(
                1,
                planner_pre_pose,
                start_qpos=relaxed_start_qpos,
            )
            if pre_plan is None:
                attempts.append(
                    {
                        "contact_id": int(contact_id),
                        "variant_index": int(variant_index),
                        "pre_pose": np.asarray(pre_pose, dtype=np.float64).round(4).tolist(),
                        "grasp_pose": np.asarray(grasp_pose, dtype=np.float64).round(4).tolist(),
                        "planner_pre_pose": np.asarray(planner_pre_pose, dtype=np.float64).round(4).tolist(),
                        "planner_grasp_pose": np.asarray(planner_grasp_pose, dtype=np.float64).round(4).tolist(),
                        "pre_reason": pre_reason,
                        "grasp_reason": None,
                        "grasp_plan_source": None,
                    }
                )
                continue

            next_qpos = np.asarray(pre_plan["position"][-1], dtype=np.float64)
            grasp_plan, grasp_reason = self.ctrl.plan_screw_to_pose_debug(
                1,
                planner_grasp_pose,
                start_qpos=next_qpos,
            )
            grasp_plan_source = "screw"
            if grasp_plan is None:
                grasp_plan, grasp_reason = self.ctrl.plan_to_pose_debug(
                    1,
                    planner_grasp_pose,
                    start_qpos=next_qpos,
                )
                if grasp_plan is not None:
                    grasp_plan_source = "pose_fallback"

            if grasp_plan is None:
                attempts.append(
                    {
                        "contact_id": int(contact_id),
                        "variant_index": int(variant_index),
                        "pre_pose": np.asarray(pre_pose, dtype=np.float64).round(4).tolist(),
                        "grasp_pose": np.asarray(grasp_pose, dtype=np.float64).round(4).tolist(),
                        "planner_pre_pose": np.asarray(planner_pre_pose, dtype=np.float64).round(4).tolist(),
                        "planner_grasp_pose": np.asarray(planner_grasp_pose, dtype=np.float64).round(4).tolist(),
                        "pre_reason": pre_reason,
                        "grasp_reason": grasp_reason,
                        "grasp_plan_source": None,
                    }
                )
                continue

            path_len = float(len(pre_plan["position"]) + len(grasp_plan["position"]))
            candidate_data = {
                "contact_id": int(contact_id),
                "variant_index": int(variant_index),
                "pre_pose": np.asarray(pre_pose, dtype=np.float64).copy(),
                "grasp_pose": np.asarray(grasp_pose, dtype=np.float64).copy(),
                "pre_plan": pre_plan,
                "grasp_plan": grasp_plan,
                "relaxed_start_qpos": relaxed_start_qpos.copy(),
                "grasp_plan_source": str(grasp_plan_source),
                "path_len": path_len,
            }
            viable_candidate_count += 1
            viable_candidates.append(candidate_data)
            if path_len < best_path_len:
                best_path_len = path_len
                selected_pre_pose = np.asarray(pre_pose, dtype=np.float64).round(4).tolist()
                selected_grasp_pose = np.asarray(grasp_pose, dtype=np.float64).round(4).tolist()
                best_candidate = candidate_data
            attempts.append(
                {
                    "contact_id": int(contact_id),
                    "variant_index": int(variant_index),
                    "pre_pose": np.asarray(pre_pose, dtype=np.float64).round(4).tolist(),
                    "grasp_pose": np.asarray(grasp_pose, dtype=np.float64).round(4).tolist(),
                    "planner_pre_pose": np.asarray(planner_pre_pose, dtype=np.float64).round(4).tolist(),
                    "planner_grasp_pose": np.asarray(planner_grasp_pose, dtype=np.float64).round(4).tolist(),
                    "pre_reason": pre_reason,
                    "grasp_reason": grasp_reason,
                    "grasp_plan_source": grasp_plan_source,
                    "path_len": path_len,
                }
            )

        success = best_candidate is not None
        viable_candidates.sort(
            key=lambda item: (
                float(item["path_len"]),
                int(item["variant_index"]),
            )
        )
        debug_info = {
            "contact_id": int(contact_id),
            "place_target_pose": np.asarray(place_target_pose, dtype=np.float64).round(4).tolist(),
            "place_box_pose": np.asarray(place_box_pose, dtype=np.float64).round(4).tolist(),
            "predicted_grasp_arm_qpos": predicted_grasp_arm_qpos.round(4).tolist(),
            "relaxed_start_qpos": relaxed_start_qpos.round(4).tolist(),
            "selected_pre_pose": selected_pre_pose,
            "selected_grasp_pose": selected_grasp_pose,
            "selected_variant_index": None if best_candidate is None else int(best_candidate["variant_index"]),
            "selected_grasp_plan_source": None if best_candidate is None else str(best_candidate["grasp_plan_source"]),
            "best_path_len": None if not success else float(best_path_len),
            "viable_variant_indices": sorted(
                {
                    int(attempt["variant_index"])
                    for attempt in attempts
                    if attempt.get("path_len") is not None
                }
            ),
            "viable_candidate_count": int(viable_candidate_count),
            "attempts": attempts,
        }
        return success, float(best_path_len), best_candidate, viable_candidates, debug_info

    def _select_right_handover_candidate(
        self,
        handover_box_pose: np.ndarray,
    ) -> tuple[
        GraspCandidate | None,
        dict[str, object] | None,
        list[dict[str, object]],
        str,
        list[dict[str, object]],
    ]:
        candidates: list[tuple[GraspCandidate, bool, float, dict[str, object] | None]] = []
        selection_debug: list[dict[str, object]] = []
        selection_options: list[dict[str, object]] = []
        preferred_quat = self._preferred_grasp_quat(1)
        preferred_top_down_quat = TOP_DOWN_PREFERRED_GRASP_QUATS[1]

        for contact_id in RIGHT_GRASP_CONTACT_IDS:
            candidate = self.ctrl.choose_grasp_candidate(
                desired_pose=handover_box_pose,
                arm_index=1,
                contact_ids=(int(contact_id),),
                pre_distance=self.HANDOVER_PRE_GRASP_DISTANCE,
                grasp_distance=self.RIGHT_HANDOVER_GRASP_DISTANCE,
                pose_builder=self._grasp_pose_from_box_pose,
                preferred_quat=preferred_quat,
                preferred_top_down_quat=preferred_top_down_quat,
                pre_pose_candidates_builder=self._right_end_grasp_pre_pose_candidates_from_box_pose,
                selection_mode="auto",
            )
            if candidate is None:
                selection_debug.append(
                    {
                        "contact_id": int(contact_id),
                        "handover_candidate_found": False,
                        "handover_attempts": list(self.ctrl.last_grasp_attempts),
                    }
                )
                continue

            if not self.ENABLE_FUTURE_PLACE_SELECTION:
                selection_debug.append(
                    {
                        "contact_id": int(contact_id),
                        "handover_candidate_found": True,
                        "handover_score": float(candidate.score),
                        "handover_path_length": int(candidate.path_length),
                        "handover_shoulder_pan_delta": float(candidate.shoulder_pan_delta),
                        "handover_joint_motion": float(candidate.joint_motion),
                        "future_place_ok": None,
                        "future_place_candidate_count": 0,
                        "future_place_path_len": None,
                        "future_place_trial_variant_indices": [],
                        "future_place_debug": None,
                    }
                )
                candidates.append((candidate, False, np.inf, None))
                selection_options.append(
                    {
                        "contact_id": int(contact_id),
                        "candidate": candidate,
                        "future_place_ok": False,
                        "future_place_path_len": float(np.inf),
                        "future_place_candidate": None,
                        "future_place_candidates": [],
                        "future_place_variant_index": None,
                    }
                )
                continue

            (
                future_place_ok,
                future_place_path_len,
                future_place_candidate,
                future_place_candidates,
                future_place_debug,
            ) = self._future_place_feasibility_for_contact(
                contact_id=int(contact_id),
                handover_candidate=candidate,
            )
            selection_debug.append(
                {
                    "contact_id": int(contact_id),
                    "handover_candidate_found": True,
                    "handover_score": float(candidate.score),
                    "handover_path_length": int(candidate.path_length),
                    "handover_shoulder_pan_delta": float(candidate.shoulder_pan_delta),
                    "handover_joint_motion": float(candidate.joint_motion),
                    "future_place_ok": bool(future_place_ok),
                    "future_place_candidate_count": int(future_place_debug["viable_candidate_count"]),
                    "future_place_path_len": None if not np.isfinite(future_place_path_len) else float(future_place_path_len),
                    "future_place_trial_variant_indices": [
                        int(item["variant_index"])
                        for item in future_place_candidates[: self.RIGHT_HANDOVER_PLACE_VARIANT_TRIAL_LIMIT]
                    ],
                    "future_place_debug": future_place_debug,
                }
            )
            candidates.append((candidate, future_place_ok, future_place_path_len, future_place_candidate))
            selection_options.append(
                {
                    "contact_id": int(contact_id),
                    "candidate": candidate,
                    "future_place_ok": bool(future_place_ok),
                    "future_place_path_len": float(future_place_path_len),
                    "future_place_candidate": future_place_candidate,
                    "future_place_candidates": list(future_place_candidates),
                    "future_place_variant_index": (
                        None if future_place_candidate is None else int(future_place_candidate["variant_index"])
                    ),
                }
            )

        if not candidates:
            return None, None, selection_debug, "no_handover_candidate", selection_options

        if not self.ENABLE_FUTURE_PLACE_SELECTION:
            preferred_options = []
            if self.PREFER_RIGHT_HANDOVER_X_AXIS_CONTACTS:
                preferred_options = [
                    item
                    for item in selection_options
                    if int(item["contact_id"]) in RIGHT_GRASP_PREFERRED_X_AXIS_CONTACT_IDS
                ]
            selectable_options = preferred_options if preferred_options else selection_options
            selected_option = min(
                selectable_options,
                key=lambda item: (
                    float(item["candidate"].score),
                    int(item["candidate"].path_length),
                    float(item["candidate"].shoulder_pan_delta),
                    float(item["candidate"].joint_motion),
                ),
            )
            return (
                selected_option["candidate"],
                None,
                selection_debug,
                "preferred_x_axis_handover" if preferred_options else "handover_score_only",
                selection_options,
            )

        place_viable_options = [item for item in selection_options if item["future_place_ok"]]
        if place_viable_options:
            preferred_place_viable_options = []
            if self.PREFER_RIGHT_HANDOVER_X_AXIS_CONTACTS:
                preferred_place_viable_options = [
                    item
                    for item in place_viable_options
                    if int(item["contact_id"]) in RIGHT_GRASP_PREFERRED_X_AXIS_CONTACT_IDS
                ]
            selectable_place_options = preferred_place_viable_options if preferred_place_viable_options else place_viable_options
            selected_option = min(
                selectable_place_options,
                key=lambda item: (
                    float(item["future_place_path_len"]),
                    float(item["candidate"].score),
                    int(item["candidate"].path_length),
                    float(item["candidate"].shoulder_pan_delta),
                    float(item["candidate"].joint_motion),
                    int(item.get("future_place_variant_index") or 0),
                ),
            )
            return (
                selected_option["candidate"],
                selected_option["future_place_candidate"],
                selection_debug,
                "preferred_x_axis_future_place_viable" if preferred_place_viable_options else "future_place_viable",
                selection_options,
            )

        preferred_candidates = []
        if self.PREFER_RIGHT_HANDOVER_X_AXIS_CONTACTS:
            preferred_candidates = [
                item
                for item in candidates
                if int(item[0].contact_id) in RIGHT_GRASP_PREFERRED_X_AXIS_CONTACT_IDS
            ]
        selectable_candidates = preferred_candidates if preferred_candidates else candidates
        selected_candidate, _, _, _ = min(
            selectable_candidates,
            key=lambda item: (
                float(item[0].score),
                int(item[0].path_length),
                float(item[0].shoulder_pan_delta),
                float(item[0].joint_motion),
            ),
        )
        selection_reason = "preferred_x_axis_handover_fallback" if preferred_candidates else "handover_fallback"
        return selected_candidate, None, selection_debug, selection_reason, selection_options

    def _capture_rollout_snapshot(self) -> dict[str, object]:
        return {
            "env_state": copy.deepcopy(self.base_env.get_state_dict()),
            "elapsed_steps": copy.deepcopy(self.base_env._elapsed_steps),
            "steps_taken": int(self.ctrl.steps_taken),
            "step_limit": int(self.ctrl.step_limit),
            "last_status": str(self.ctrl.last_status),
            "last_grasp_attempts": copy.deepcopy(self.ctrl.last_grasp_attempts),
            "last_plan_debug": copy.deepcopy(self.ctrl.last_plan_debug),
            "commanded_arm_qpos": {
                0: np.asarray(self.ctrl.commanded_arm_qpos[0], dtype=np.float64).copy(),
                1: np.asarray(self.ctrl.commanded_arm_qpos[1], dtype=np.float64).copy(),
            },
            "commanded_gripper": {
                0: float(self.ctrl.commanded_gripper[0]),
                1: float(self.ctrl.commanded_gripper[1]),
            },
            "last_debug_info": copy.deepcopy(self.last_debug_info),
            "first_pick_debug": copy.deepcopy(self.first_pick_debug),
        }

    def _restore_rollout_snapshot(self, snapshot: dict[str, object]) -> None:
        self.base_env.set_state_dict(copy.deepcopy(snapshot["env_state"]))
        self.base_env._elapsed_steps = copy.deepcopy(snapshot["elapsed_steps"])
        self.ctrl.sync_sim_state()
        self.ctrl.steps_taken = int(snapshot["steps_taken"])
        self.ctrl.step_limit = int(snapshot["step_limit"])
        self.ctrl.last_status = str(snapshot["last_status"])
        self.ctrl.last_grasp_attempts = copy.deepcopy(snapshot["last_grasp_attempts"])
        self.ctrl.last_plan_debug = copy.deepcopy(snapshot["last_plan_debug"])
        self.ctrl.commanded_arm_qpos[0] = np.asarray(snapshot["commanded_arm_qpos"][0], dtype=np.float64).copy()
        self.ctrl.commanded_arm_qpos[1] = np.asarray(snapshot["commanded_arm_qpos"][1], dtype=np.float64).copy()
        self.ctrl.commanded_gripper[0] = float(snapshot["commanded_gripper"][0])
        self.ctrl.commanded_gripper[1] = float(snapshot["commanded_gripper"][1])
        self.last_debug_info = copy.deepcopy(snapshot["last_debug_info"])
        self.first_pick_debug = copy.deepcopy(snapshot["first_pick_debug"])

    def _allow_right_handover_branch_trial(self) -> bool:
        return self.ENABLE_RIGHT_HANDOVER_BRANCH_TRIAL and getattr(self.base_env, "render_mode", None) != "human"

    def _trial_branch_score(
        self,
        *,
        success: bool,
        status: str,
        debug_info: dict[str, object],
        steps_taken: int,
    ) -> tuple[float, ...]:
        xy_aligned = bool(debug_info.get("xy_aligned", False))
        z_aligned = bool(debug_info.get("z_aligned", False))
        spatial_success = bool(debug_info.get("spatial_success", False))
        final_check = debug_info.get("stage") == "final_success_check"
        box_to_target_xy = debug_info.get("box_to_target_xy")
        xy_distance = np.inf
        if box_to_target_xy is not None:
            xy_distance = float(np.linalg.norm(np.asarray(box_to_target_xy, dtype=np.float64)))
        box_pose = debug_info.get("box_pose")
        target_fp_pose = debug_info.get("target_fp_pose")
        z_distance = np.inf
        if box_pose is not None and target_fp_pose is not None:
            box_center = np.asarray(box_pose, dtype=np.float64)
            target_top = np.asarray(target_fp_pose, dtype=np.float64)
            box_bottom_center_z = float(box_center[2] - self.box_half_size[2])
            z_distance = float(abs(box_bottom_center_z - target_top[2]))
        combined_distance = xy_distance + z_distance
        return (
            float(success or status == "success"),
            float(spatial_success),
            float(xy_aligned and z_aligned),
            float(final_check),
            -combined_distance,
            float(xy_aligned),
            float(z_aligned),
            -xy_distance,
            -z_distance,
            -float(steps_taken),
        )

    def _trial_branch_supports_switch(
        self,
        *,
        planner_score: tuple[float, ...] | None,
        candidate_score: tuple[float, ...],
    ) -> bool:
        if planner_score is None:
            return True
        if candidate_score[0] != planner_score[0]:
            return candidate_score[0] > planner_score[0]
        if candidate_score[1] != planner_score[1]:
            return candidate_score[1] > planner_score[1]
        if candidate_score[2] != planner_score[2]:
            return candidate_score[2] > planner_score[2]
        if candidate_score[3] != planner_score[3]:
            return candidate_score[3] > planner_score[3]
        if candidate_score[5] != planner_score[5]:
            return candidate_score[5] > planner_score[5]
        if candidate_score[6] != planner_score[6]:
            return candidate_score[6] > planner_score[6]
        return candidate_score[4] > planner_score[4] + self.RIGHT_HANDOVER_TRIAL_SWITCH_DISTANCE_MARGIN

    def _execute_right_branch(
        self,
        *,
        recorder,
        handover_box_pose: np.ndarray,
        handover_candidate: GraspCandidate,
        selected_place_candidate: dict[str, object] | None,
        right_handover_selection_debug: list[dict[str, object]],
        right_handover_selection_reason: str,
    ) -> tuple[bool, int, str]:
        approach_pose = self.ctrl.gripper_pose_to_planner_pose(
            1,
            self._get_handover_approach_pose(handover_candidate.pre_pose),
        )
        if self.ctrl.move_to_pose(1, approach_pose, recorder, fail_status="right_handover_approach_failed"):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            self.last_debug_info = {
                "stage": "right_handover_approach",
                "handover_box_pose": np.asarray(handover_box_pose, dtype=np.float64).round(4).tolist(),
                "selection_reason": right_handover_selection_reason,
                "selection_debug": right_handover_selection_debug,
                "selected_contact_id": int(handover_candidate.contact_id),
                "approach_pose": np.asarray(approach_pose, dtype=np.float64).round(4).tolist(),
                "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                "plan_debug": dict(self.ctrl.last_plan_debug),
            }
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        right_handover_close_trace: list[dict[str, object]] = []

        def record_right_handover_close_step(event: dict[str, object]) -> None:
            right_handover_close_trace.append(
                {
                    "phase": event.get("phase"),
                    "step_index": event.get("step_index"),
                    "global_step": event.get("global_step"),
                    "commanded_gripper": round(float(event.get("commanded_gripper", 0.0)), 4),
                    "paired_commanded_gripper": (
                        None
                        if event.get("paired_commanded_gripper") is None
                        else round(float(event["paired_commanded_gripper"]), 4)
                    ),
                    "actual_gripper": round(float(self.ctrl.current_gripper_value(1)), 4),
                    "actual_left_gripper": round(float(self.ctrl.current_gripper_value(0)), 4),
                    "commanded_right_qpos": np.asarray(
                        self.ctrl.commanded_arm_qpos[1],
                        dtype=np.float64,
                    ).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                }
            )

        handover_candidate, success = self.ctrl.grasp_actor(
            desired_pose=handover_box_pose,
            arm_index=1,
            contact_ids=(handover_candidate.contact_id,),
            pre_grasp_dis=self.HANDOVER_PRE_GRASP_DISTANCE,
            grasp_dis=self.RIGHT_HANDOVER_GRASP_DISTANCE,
            pose_builder=self._grasp_pose_from_box_pose,
            preferred_quat=self._preferred_grasp_quat(1),
            preferred_top_down_quat=TOP_DOWN_PREFERRED_GRASP_QUATS[1],
            pre_pose_candidates_builder=self._right_end_grasp_pre_pose_candidates_from_box_pose,
            recorder=recorder,
            fail_status="right_handover_plan_failed",
            selection_mode="auto",
            gripper_pos=self.RIGHT_HANDOVER_GRIPPER_HOLD_POS,
            pre_close_settle_steps=self.RIGHT_HANDOVER_PRE_CLOSE_SETTLE_STEPS,
            pre_close_paired_gripper_arm_index=0,
            pre_close_paired_gripper_pos=self.LEFT_PRERELEASE_BEFORE_RIGHT_HANDOVER_CLOSE,
            pre_close_paired_gripper_steps=self.LEFT_PRERELEASE_BEFORE_RIGHT_HANDOVER_CLOSE_STEPS,
            close_smooth_steps=self.RIGHT_HANDOVER_CLOSE_SMOOTH_STEPS,
            close_settle_steps=self.RIGHT_HANDOVER_CLOSE_SETTLE_STEPS,
            close_step_callback=record_right_handover_close_step,
            freeze_current_qpos_before_close=True,
        )
        if handover_candidate is None:
            self.last_debug_info = {
                "stage": "right_handover_grasp",
                "handover_box_pose": np.asarray(handover_box_pose, dtype=np.float64).round(4).tolist(),
                "selection_reason": right_handover_selection_reason,
                "selection_debug": right_handover_selection_debug,
            }
            return False, self.ctrl.steps_taken, self.ctrl.last_status
        right_handover_alignment = self._box_gripper_alignment_debug(
            arm_index=1,
            planned_box_pose=handover_box_pose,
            planned_gripper_pose=handover_candidate.grasp_pose,
        )
        if success:
            return True, self.ctrl.steps_taken, self.ctrl.last_status

        if self.RIGHT_HANDOVER_DUAL_HOLD_STEPS > 0:
            if self.ctrl.hold_pose(
                recorder,
                steps=self.RIGHT_HANDOVER_DUAL_HOLD_STEPS,
                left_gripper=self.LEFT_RELEASE_DURING_RIGHT_HANDOVER_CLOSE,
                right_gripper=self.RIGHT_HANDOVER_GRIPPER_HOLD_POS,
                allow_success_early=False,
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_handover_dual_hold",
                    "handover_box_pose": np.asarray(handover_box_pose, dtype=np.float64).round(4).tolist(),
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "selected_contact_id": int(handover_candidate.contact_id),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status
            right_handover_alignment = self._box_gripper_alignment_debug(
                arm_index=1,
                planned_box_pose=handover_box_pose,
                planned_gripper_pose=handover_candidate.grasp_pose,
            )

        if self.RIGHT_HANDOVER_POST_CLOSE_HOLD_STEPS > 0:
            if self.ctrl.hold_pose(
                recorder,
                steps=self.RIGHT_HANDOVER_POST_CLOSE_HOLD_STEPS,
                left_gripper=self.LEFT_PRERELEASE_BEFORE_RIGHT_HANDOVER_CLOSE,
                right_gripper=self.RIGHT_HANDOVER_GRIPPER_HOLD_POS,
                allow_success_early=False,
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_handover_post_close_hold",
                    "handover_box_pose": np.asarray(handover_box_pose, dtype=np.float64).round(4).tolist(),
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "selected_contact_id": int(handover_candidate.contact_id),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.ctrl.close_gripper(
            0,
            recorder,
            pos=self.LEFT_RELEASE_DURING_RIGHT_HANDOVER_CLOSE,
            smooth_steps=self.LEFT_RELEASE_OPEN_SMOOTH_STEPS,
            settle_steps=self.LEFT_RELEASE_OPEN_SETTLE_STEPS,
            paired_gripper_arm_index=1,
            paired_gripper_pos=self.RIGHT_HANDOVER_GRIPPER_HOLD_POS,
            allow_success_early=False,
            fail_status="left_open_failed",
        ):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        left_release_before_clear_pose = np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64)
        left_release_clear_trace: list[dict[str, object]] = []
        left_release_clear_stages = (
            ("left_release_clear_lift", np.asarray(self.LEFT_RELEASE_CLEAR_LIFT_DELTA, dtype=np.float64)),
            ("left_release_clear_retreat", np.asarray(self.LEFT_RELEASE_CLEAR_RETREAT_DELTA, dtype=np.float64)),
        )
        for left_release_clear_stage, left_release_clear_delta in left_release_clear_stages:
            if float(np.linalg.norm(left_release_clear_delta)) <= 1e-8:
                continue
            left_release_stage_start_pose = np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64)
            if self.ctrl.move_by_displacement_via_ik(
                0,
                recorder,
                x=float(left_release_clear_delta[0]),
                y=float(left_release_clear_delta[1]),
                z=float(left_release_clear_delta[2]),
                move_axis="world",
                steps=self.LEFT_RELEASE_CLEAR_STEPS,
                settle_steps=self.LEFT_RELEASE_CLEAR_SETTLE_STEPS,
                fail_status=f"{left_release_clear_stage}_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            left_release_stage_end_pose = np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64)
            left_release_clear_trace.append(
                {
                    "stage": left_release_clear_stage,
                    "command_delta": left_release_clear_delta.round(4).tolist(),
                    "start_pose": left_release_stage_start_pose.round(4).tolist(),
                    "end_pose": left_release_stage_end_pose.round(4).tolist(),
                    "actual_delta": (left_release_stage_end_pose[:3] - left_release_stage_start_pose[:3])
                    .round(4)
                    .tolist(),
                }
            )
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": left_release_clear_stage,
                    "left_release_before_clear_pose": left_release_before_clear_pose.round(4).tolist(),
                    "left_release_after_clear_pose": left_release_stage_end_pose.round(4).tolist(),
                    "left_release_clear_trace": left_release_clear_trace,
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status
        left_release_after_clear_pose = np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64)
        left_release_clear_debug = {
            "left_release_before_clear_pose": left_release_before_clear_pose.round(4).tolist(),
            "left_release_after_clear_pose": left_release_after_clear_pose.round(4).tolist(),
            "left_release_clear_trace": left_release_clear_trace,
        }

        if self.LEFT_RELEASE_LIFT_DISTANCE > 1e-8:
            if self.ctrl.move_by_displacement(
                0,
                recorder,
                z=self.LEFT_RELEASE_LIFT_DISTANCE,
                move_axis="world",
                fail_status="left_release_lift_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "left_release_lift",
                    **left_release_clear_debug,
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "right_handover_alignment": right_handover_alignment,
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.RETURN_LEFT_HOME_AFTER_HANDOVER:
            if self.ctrl.back_to_origin(0, recorder, fail_status="left_back_to_origin_failed"):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                return False, self.ctrl.steps_taken, self.ctrl.last_status

        current_right_qpos = self.ctrl.current_qpos(1)
        right_home_qpos = self.base_env.get_arm_home_joint_positions(1)
        if selected_place_candidate is not None and self.USE_SELECTED_PLACE_CANDIDATE_AT_RUNTIME:
            right_carry_relax_qpos = np.asarray(selected_place_candidate["relaxed_start_qpos"], dtype=np.float64).copy()
        else:
            right_carry_relax_qpos = current_right_qpos.copy()
        if self.ctrl.move_to_qpos(
            1,
            right_carry_relax_qpos,
            recorder,
            steps=self.RIGHT_CARRY_RELAX_STEPS,
            settle_steps=self.RIGHT_CARRY_RELAX_SETTLE_STEPS,
        ):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            self.last_debug_info = {
                "stage": "right_carry_relax",
                **left_release_clear_debug,
                "current_right_qpos": np.asarray(current_right_qpos, dtype=np.float64).round(4).tolist(),
                "right_home_qpos": np.asarray(right_home_qpos, dtype=np.float64).round(4).tolist(),
                "right_carry_relax_qpos": np.asarray(right_carry_relax_qpos, dtype=np.float64).round(4).tolist(),
                "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                "right_handover_alignment": right_handover_alignment,
                "right_handover_close_trace": right_handover_close_trace,
            }
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        target_fp_pose = self._target_functional_world_pose(1)
        place_target_pose = target_fp_pose.copy()
        place_target_pose[:3] += self.RIGHT_PLACE_TARGET_OFFSET
        place_pose_source = "runtime_translation_from_current_grasp"
        selected_place_variant_index = None
        selected_place_grasp_plan_source = None
        selected_pre_plan_len = None
        selected_place_plan_len = None
        use_selected_place_candidate = (
            selected_place_candidate is not None and self.USE_SELECTED_PLACE_CANDIDATE_AT_RUNTIME
        )
        right_pre_place_fallback_debug = None
        right_place_fallback_debug = None
        right_place_relative_fallback_debug = None
        right_place_release_from_pre_place_debug = None
        skip_right_place_correction = False
        if selected_place_candidate is not None:
            selected_place_variant_index = int(selected_place_candidate["variant_index"])
            selected_place_grasp_plan_source = str(selected_place_candidate["grasp_plan_source"])
            selected_pre_plan_len = int(len(selected_place_candidate["pre_plan"]["position"]))
            selected_place_plan_len = int(len(selected_place_candidate["grasp_plan"]["position"]))
        if use_selected_place_candidate:
            right_pre_place_pose = np.asarray(selected_place_candidate["pre_pose"], dtype=np.float64).copy()
            right_place_pose = np.asarray(selected_place_candidate["grasp_pose"], dtype=np.float64).copy()
            place_pose_source = "future_place_candidate_plan"
        else:
            right_pre_place_pose = self._place_pose_by_current_grasp_translation(
                arm_index=1,
                target_pose=place_target_pose,
                pre_dis=0.08,
            )
            right_place_pose = self._place_pose_by_current_grasp_translation(
                arm_index=1,
                target_pose=place_target_pose,
                pre_dis=self.GRASP_DISTANCE,
            )
            right_pre_place_pose = self._offset_pose_world(
                right_pre_place_pose,
                x=float(self.RIGHT_PRE_PLACE_WORLD_OFFSET[0]),
                y=float(self.RIGHT_PRE_PLACE_WORLD_OFFSET[1]),
                z=float(self.RIGHT_PRE_PLACE_WORLD_OFFSET[2]),
            )
            right_place_pose = self._offset_pose_world(
                right_place_pose,
                x=float(self.RIGHT_PLACE_WORLD_OFFSET[0]),
                y=float(self.RIGHT_PLACE_WORLD_OFFSET[1]),
                z=float(self.RIGHT_PLACE_WORLD_OFFSET[2]),
            )

        if use_selected_place_candidate:
            if self.ctrl.execute_plan(1, selected_place_candidate["pre_plan"], recorder):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_pre_place",
                    **left_release_clear_debug,
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "selected_handover_contact_id": int(handover_candidate.contact_id),
                    "place_pose_source": place_pose_source,
                    "selected_place_variant_index": selected_place_variant_index,
                    "selected_place_grasp_plan_source": selected_place_grasp_plan_source,
                    "selected_pre_plan_len": selected_pre_plan_len,
                    "selected_place_plan_len": selected_place_plan_len,
                    "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                    "place_target_pose": np.asarray(place_target_pose, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_world_offset": np.asarray(self.RIGHT_PRE_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_place_world_offset": np.asarray(self.RIGHT_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_pose": np.asarray(right_pre_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_place_pose": np.asarray(right_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                self.ctrl.last_status = "right_pre_place_failed"
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            right_place_pose = np.asarray(selected_place_candidate["grasp_pose"], dtype=np.float64).copy()
            current_qpos_before_place = np.asarray(self.ctrl.current_qpos(1), dtype=np.float64)
            current_full_qpos_before_place = np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64)
            candidate_place_plan_start = np.asarray(selected_place_candidate["grasp_plan"]["position"][0], dtype=np.float64)
            candidate_place_plan_start_arm = candidate_place_plan_start[: current_qpos_before_place.shape[0]]
            place_plan_start_delta = current_qpos_before_place - candidate_place_plan_start_arm
            if self.ctrl.execute_plan(
                1,
                selected_place_candidate["grasp_plan"],
                recorder,
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                release_desired_box_center = np.asarray(target_fp_pose, dtype=np.float64)[:3].copy()
                release_desired_box_center[2] += float(self.box_half_size[2])
                release_box_pose = np.asarray(self._box_pose(), dtype=np.float64)
                release_residual = release_desired_box_center - release_box_pose[:3]
                release_xy_error = float(np.linalg.norm(release_residual[:2]))
                release_z_above = float(release_box_pose[2] - release_desired_box_center[2])
                right_place_release_from_pre_place_debug = {
                    "used": False,
                    "reason": "right_place_failed",
                    "max_xy_error": float(self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_XY_ERROR),
                    "max_z_above": float(self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_Z_ABOVE),
                    "desired_box_center": release_desired_box_center.round(4).tolist(),
                    "box_pose": release_box_pose.round(4).tolist(),
                    "residual": release_residual.round(4).tolist(),
                    "xy_error": round(release_xy_error, 4),
                    "z_above": round(release_z_above, 4),
                    "failed_status": self.ctrl.last_status,
                }
                if (
                    self.RIGHT_PLACE_ALLOW_RELEASE_FROM_PRE_PLACE
                    and release_xy_error <= self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_XY_ERROR
                    and 0.0 <= release_z_above <= self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_Z_ABOVE
                ):
                    right_place_release_from_pre_place_debug["used"] = True
                    right_place_release_from_pre_place_debug["reason"] = "box_already_over_target"
                    skip_right_place_correction = True
                    self.ctrl.last_status = "running"
                    self.ctrl.last_plan_debug = {}

            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_place",
                    **left_release_clear_debug,
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "selected_handover_contact_id": int(handover_candidate.contact_id),
                    "place_pose_source": place_pose_source,
                    "selected_place_variant_index": selected_place_variant_index,
                    "selected_place_grasp_plan_source": selected_place_grasp_plan_source,
                    "selected_pre_plan_len": selected_pre_plan_len,
                    "selected_place_plan_len": selected_place_plan_len,
                    "candidate_place_plan_start_qpos": np.asarray(
                        candidate_place_plan_start_arm,
                        dtype=np.float64,
                    ).round(4).tolist(),
                    "current_qpos_before_place": current_qpos_before_place.round(4).tolist(),
                    "current_full_qpos_before_place": current_full_qpos_before_place.round(4).tolist(),
                    "place_plan_start_delta": place_plan_start_delta.round(4).tolist(),
                    "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                    "place_target_pose": np.asarray(place_target_pose, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_world_offset": np.asarray(self.RIGHT_PRE_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_place_world_offset": np.asarray(self.RIGHT_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_pose": np.asarray(right_pre_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_place_pose": np.asarray(right_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                self.ctrl.last_status = "right_place_failed"
                return False, self.ctrl.steps_taken, self.ctrl.last_status
        else:
            right_pre_place_planner_pose = self.ctrl.gripper_pose_to_planner_pose(1, right_pre_place_pose)
            if self.ctrl.move_to_pose(
                1,
                right_pre_place_planner_pose,
                recorder,
                fail_status="right_pre_place_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                right_pre_place_primary_plan_debug = dict(self.ctrl.last_plan_debug)
                fallback_qpos, fallback_reason = self.ctrl.ik_to_pose_debug(
                    1,
                    right_pre_place_planner_pose,
                    threshold=self.RIGHT_PRE_PLACE_FALLBACK_IK_THRESHOLD,
                )
                if fallback_qpos is not None:
                    right_pre_place_fallback_arm_qpos = self.ctrl._arm_qpos_from_plan(fallback_qpos)
                    right_pre_place_fallback_debug = {
                        "used": True,
                        "threshold": float(self.RIGHT_PRE_PLACE_FALLBACK_IK_THRESHOLD),
                        "ik_reason": fallback_reason,
                        "target_arm_qpos": right_pre_place_fallback_arm_qpos.round(4).tolist(),
                        "primary_plan_debug": right_pre_place_primary_plan_debug,
                    }
                    self.ctrl.last_status = "running"
                    if self.ctrl.move_to_qpos(
                        1,
                        right_pre_place_fallback_arm_qpos,
                        recorder,
                        steps=self.RIGHT_PRE_PLACE_FALLBACK_STEPS,
                        settle_steps=self.RIGHT_PRE_PLACE_FALLBACK_SETTLE_STEPS,
                    ):
                        return True, self.ctrl.steps_taken, self.ctrl.last_status
                    if self.ctrl.last_status == "running":
                        self.ctrl.last_plan_debug = {}
                    else:
                        right_pre_place_fallback_debug["move_debug"] = dict(self.ctrl.last_plan_debug)
                else:
                    right_pre_place_fallback_debug = {
                        "used": False,
                        "threshold": float(self.RIGHT_PRE_PLACE_FALLBACK_IK_THRESHOLD),
                        "ik_reason": fallback_reason,
                        "primary_plan_debug": right_pre_place_primary_plan_debug,
                    }

            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_pre_place",
                    **left_release_clear_debug,
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "selected_handover_contact_id": int(handover_candidate.contact_id),
                    "place_pose_source": place_pose_source,
                    "selected_place_variant_index": selected_place_variant_index,
                    "selected_place_grasp_plan_source": selected_place_grasp_plan_source,
                    "selected_pre_plan_len": selected_pre_plan_len,
                    "selected_place_plan_len": selected_place_plan_len,
                    "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                    "place_target_pose": np.asarray(place_target_pose, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_world_offset": np.asarray(self.RIGHT_PRE_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_place_world_offset": np.asarray(self.RIGHT_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_pose": np.asarray(right_pre_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_place_pose": np.asarray(right_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "right_pre_place_fallback_debug": right_pre_place_fallback_debug,
                    "plan_debug": right_pre_place_primary_plan_debug,
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            right_place_prefer_screw = selected_place_grasp_plan_source != "pose_fallback"
            right_place_planner_pose = self.ctrl.gripper_pose_to_planner_pose(1, right_place_pose)
            if self.ctrl.move_to_pose(
                1,
                right_place_planner_pose,
                recorder,
                prefer_screw=right_place_prefer_screw,
                fail_status="right_place_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                right_place_primary_plan_debug = dict(self.ctrl.last_plan_debug)
                right_place_failure_plan_debug = right_place_primary_plan_debug
                fallback_qpos, fallback_reason = self.ctrl.ik_to_pose_debug(
                    1,
                    right_place_planner_pose,
                    n_init_qpos=1,
                    threshold=self.RIGHT_PLACE_FALLBACK_IK_THRESHOLD,
                )
                if fallback_qpos is not None:
                    current_right_qpos = np.asarray(self.ctrl.current_qpos(1), dtype=np.float64)
                    right_place_fallback_arm_qpos = self.ctrl._arm_qpos_from_plan(fallback_qpos)
                    right_place_fallback_delta = right_place_fallback_arm_qpos - current_right_qpos
                    right_place_fallback_shoulder_delta = abs(float(right_place_fallback_delta[0]))
                    right_place_fallback_joint_motion = float(np.linalg.norm(right_place_fallback_delta))
                    right_place_fallback_debug = {
                        "used": False,
                        "threshold": float(self.RIGHT_PLACE_FALLBACK_IK_THRESHOLD),
                        "ik_reason": fallback_reason,
                        "target_arm_qpos": right_place_fallback_arm_qpos.round(4).tolist(),
                        "current_arm_qpos": current_right_qpos.round(4).tolist(),
                        "shoulder_pan_delta": round(right_place_fallback_shoulder_delta, 4),
                        "joint_motion": round(right_place_fallback_joint_motion, 4),
                        "primary_plan_debug": right_place_primary_plan_debug,
                    }
                    if (
                        right_place_fallback_shoulder_delta > self.RIGHT_PLACE_FALLBACK_MAX_SHOULDER_DELTA
                        or right_place_fallback_joint_motion > self.RIGHT_PLACE_FALLBACK_MAX_JOINT_MOTION
                    ):
                        fallback_qpos = None
                        fallback_reason = "rejected_large_joint_jump"
                        right_place_fallback_debug["reject_reason"] = fallback_reason
                    else:
                        right_place_fallback_debug["used"] = True
                if fallback_qpos is not None:
                    right_place_fallback_arm_qpos = self.ctrl._arm_qpos_from_plan(fallback_qpos)
                    self.ctrl.last_status = "running"
                    if self.ctrl.move_to_qpos(
                        1,
                        right_place_fallback_arm_qpos,
                        recorder,
                        steps=self.RIGHT_PLACE_FALLBACK_STEPS,
                        settle_steps=self.RIGHT_PLACE_FALLBACK_SETTLE_STEPS,
                    ):
                        return True, self.ctrl.steps_taken, self.ctrl.last_status
                    if self.ctrl.last_status == "running":
                        self.ctrl.last_plan_debug = {}
                    else:
                        right_place_failure_plan_debug = dict(self.ctrl.last_plan_debug)
                        right_place_fallback_debug["move_debug"] = right_place_failure_plan_debug
                else:
                    if right_place_fallback_debug is None:
                        right_place_fallback_debug = {
                            "used": False,
                            "threshold": float(self.RIGHT_PLACE_FALLBACK_IK_THRESHOLD),
                            "ik_reason": fallback_reason,
                            "primary_plan_debug": right_place_primary_plan_debug,
                        }
                    relative_trace: list[dict[str, object]] = []
                    relative_target_delta = (
                        np.asarray(right_place_pose, dtype=np.float64)[:3]
                        - np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64)[:3]
                    )
                    right_place_relative_fallback_debug = {
                        "used": False,
                        "target_delta": relative_target_delta.round(4).tolist(),
                        "max_step": float(self.RIGHT_PLACE_RELATIVE_FALLBACK_MAX_STEP),
                        "trace": relative_trace,
                    }
                    self.ctrl.last_status = "running"
                    max_iter = int(np.ceil(np.linalg.norm(relative_target_delta) / self.RIGHT_PLACE_RELATIVE_FALLBACK_MAX_STEP)) + 2
                    for step_index in range(max_iter):
                        current_gripper_pose = np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64)
                        remaining_delta = np.asarray(right_place_pose, dtype=np.float64)[:3] - current_gripper_pose[:3]
                        remaining_norm = float(np.linalg.norm(remaining_delta))
                        if remaining_norm <= 1e-3:
                            right_place_relative_fallback_debug["used"] = True
                            self.ctrl.last_plan_debug = {}
                            break
                        step_delta = remaining_delta.copy()
                        if remaining_norm > self.RIGHT_PLACE_RELATIVE_FALLBACK_MAX_STEP:
                            step_delta *= float(self.RIGHT_PLACE_RELATIVE_FALLBACK_MAX_STEP) / remaining_norm
                        step_start_pose = current_gripper_pose.copy()
                        step_target_gripper_pose = current_gripper_pose.copy()
                        step_target_gripper_pose[:3] += step_delta
                        step_target_planner_pose = self.ctrl.gripper_pose_to_planner_pose(1, step_target_gripper_pose)
                        step_ik_qpos, step_ik_reason = self.ctrl.ik_to_pose_debug(
                            1,
                            step_target_planner_pose,
                            n_init_qpos=1,
                            threshold=self.RIGHT_PLACE_RELATIVE_FALLBACK_IK_THRESHOLD,
                        )
                        if step_ik_qpos is None:
                            self.ctrl.last_status = "right_place_relative_fallback_failed"
                            self.ctrl.last_plan_debug = {
                                "arm_index": 1,
                                "target_gripper_pose": step_target_gripper_pose.round(4).tolist(),
                                "target_planner_pose": step_target_planner_pose.round(4).tolist(),
                                "ik_threshold": float(self.RIGHT_PLACE_RELATIVE_FALLBACK_IK_THRESHOLD),
                                "ik_reason": step_ik_reason,
                                "steps": int(self.RIGHT_PLACE_RELATIVE_FALLBACK_STEPS),
                                "settle_steps": int(self.RIGHT_PLACE_RELATIVE_FALLBACK_SETTLE_STEPS),
                            }
                        else:
                            current_right_qpos = np.asarray(self.ctrl.current_qpos(1), dtype=np.float64)
                            step_target_arm_qpos = self.ctrl._arm_qpos_from_plan(step_ik_qpos)
                            step_qpos_delta = step_target_arm_qpos - current_right_qpos
                            step_shoulder_delta = abs(float(step_qpos_delta[0]))
                            step_joint_motion = float(np.linalg.norm(step_qpos_delta))
                            self.ctrl.last_status = "running"
                            self.ctrl.last_plan_debug = {
                                "arm_index": 1,
                                "target_gripper_pose": step_target_gripper_pose.round(4).tolist(),
                                "target_planner_pose": step_target_planner_pose.round(4).tolist(),
                                "current_arm_qpos": current_right_qpos.round(4).tolist(),
                                "target_arm_qpos": step_target_arm_qpos.round(4).tolist(),
                                "shoulder_pan_delta": round(step_shoulder_delta, 4),
                                "joint_motion": round(step_joint_motion, 4),
                                "ik_threshold": float(self.RIGHT_PLACE_RELATIVE_FALLBACK_IK_THRESHOLD),
                                "ik_reason": step_ik_reason,
                                "steps": int(self.RIGHT_PLACE_RELATIVE_FALLBACK_STEPS),
                                "settle_steps": int(self.RIGHT_PLACE_RELATIVE_FALLBACK_SETTLE_STEPS),
                            }
                            if (
                                step_shoulder_delta > self.RIGHT_PLACE_FALLBACK_MAX_SHOULDER_DELTA
                                or step_joint_motion > self.RIGHT_PLACE_FALLBACK_MAX_JOINT_MOTION
                            ):
                                self.ctrl.last_status = "right_place_relative_fallback_rejected"
                                self.ctrl.last_plan_debug["reject_reason"] = "large_joint_jump"
                            elif self.ctrl.move_to_qpos(
                                1,
                                step_target_arm_qpos,
                                recorder,
                                steps=self.RIGHT_PLACE_RELATIVE_FALLBACK_STEPS,
                                settle_steps=self.RIGHT_PLACE_RELATIVE_FALLBACK_SETTLE_STEPS,
                            ):
                                return True, self.ctrl.steps_taken, self.ctrl.last_status
                        step_end_pose = np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64)
                        relative_trace.append(
                            {
                                "step_index": int(step_index),
                                "remaining_delta": remaining_delta.round(4).tolist(),
                                "command_delta": step_delta.round(4).tolist(),
                                "start_pose": step_start_pose.round(4).tolist(),
                                "target_pose": step_target_gripper_pose.round(4).tolist(),
                                "end_pose": step_end_pose.round(4).tolist(),
                                "actual_delta": (step_end_pose[:3] - step_start_pose[:3]).round(4).tolist(),
                                "status": self.ctrl.last_status,
                                "plan_debug": dict(self.ctrl.last_plan_debug),
                            }
                        )
                        if self.ctrl.last_status != "running":
                            right_place_failure_plan_debug = dict(self.ctrl.last_plan_debug)
                            break

            if self.ctrl.last_status != "running":
                release_desired_box_center = np.asarray(target_fp_pose, dtype=np.float64)[:3].copy()
                release_desired_box_center[2] += float(self.box_half_size[2])
                release_box_pose = np.asarray(self._box_pose(), dtype=np.float64)
                release_residual = release_desired_box_center - release_box_pose[:3]
                release_xy_error = float(np.linalg.norm(release_residual[:2]))
                release_z_above = float(release_box_pose[2] - release_desired_box_center[2])
                right_place_release_from_pre_place_debug = {
                    "used": False,
                    "reason": "right_place_failed",
                    "max_xy_error": float(self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_XY_ERROR),
                    "max_z_above": float(self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_Z_ABOVE),
                    "desired_box_center": release_desired_box_center.round(4).tolist(),
                    "box_pose": release_box_pose.round(4).tolist(),
                    "residual": release_residual.round(4).tolist(),
                    "xy_error": round(release_xy_error, 4),
                    "z_above": round(release_z_above, 4),
                    "failed_status": self.ctrl.last_status,
                }
                if (
                    self.RIGHT_PLACE_ALLOW_RELEASE_FROM_PRE_PLACE
                    and release_xy_error <= self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_XY_ERROR
                    and 0.0 <= release_z_above <= self.RIGHT_PLACE_RELEASE_FROM_PRE_PLACE_MAX_Z_ABOVE
                ):
                    right_place_release_from_pre_place_debug["used"] = True
                    right_place_release_from_pre_place_debug["reason"] = "box_already_over_target"
                    skip_right_place_correction = True
                    self.ctrl.last_status = "running"
                    self.ctrl.last_plan_debug = {}

            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_place",
                    **left_release_clear_debug,
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "selected_handover_contact_id": int(handover_candidate.contact_id),
                    "place_pose_source": place_pose_source,
                    "selected_place_variant_index": selected_place_variant_index,
                    "selected_place_grasp_plan_source": selected_place_grasp_plan_source,
                    "selected_pre_plan_len": selected_pre_plan_len,
                    "selected_place_plan_len": selected_place_plan_len,
                    "right_place_prefer_screw": bool(right_place_prefer_screw),
                    "right_pre_place_fallback_debug": right_pre_place_fallback_debug,
                    "right_place_fallback_debug": right_place_fallback_debug,
                    "right_place_relative_fallback_debug": right_place_relative_fallback_debug,
                    "right_place_release_from_pre_place_debug": right_place_release_from_pre_place_debug,
                    "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                    "place_target_pose": np.asarray(place_target_pose, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_world_offset": np.asarray(self.RIGHT_PRE_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_place_world_offset": np.asarray(self.RIGHT_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                    "right_pre_place_pose": np.asarray(right_pre_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_place_pose": np.asarray(right_place_pose, dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "plan_debug": right_place_failure_plan_debug,
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.ctrl.hold_pose(
            recorder,
            steps=self.RIGHT_PLACE_SETTLE_STEPS,
            right_gripper=self.RIGHT_HANDOVER_GRIPPER_HOLD_POS,
            allow_success_early=False,
        ):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            self.last_debug_info = {
                "stage": "right_place_settle",
                **left_release_clear_debug,
                "selection_reason": right_handover_selection_reason,
                "selection_debug": right_handover_selection_debug,
                "selected_handover_contact_id": int(handover_candidate.contact_id),
                "place_pose_source": place_pose_source,
                "selected_place_variant_index": selected_place_variant_index,
                "selected_place_grasp_plan_source": selected_place_grasp_plan_source,
                "selected_pre_plan_len": selected_pre_plan_len,
                "selected_place_plan_len": selected_place_plan_len,
                "right_pre_place_fallback_debug": right_pre_place_fallback_debug,
                "right_place_fallback_debug": right_place_fallback_debug,
                "right_place_relative_fallback_debug": right_place_relative_fallback_debug,
                "right_place_release_from_pre_place_debug": right_place_release_from_pre_place_debug,
                "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                "place_target_pose": np.asarray(place_target_pose, dtype=np.float64).round(4).tolist(),
                "right_pre_place_pose": np.asarray(right_pre_place_pose, dtype=np.float64).round(4).tolist(),
                "right_place_pose": np.asarray(right_place_pose, dtype=np.float64).round(4).tolist(),
                "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                "right_handover_alignment": right_handover_alignment,
                "right_handover_close_trace": right_handover_close_trace,
            }
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        right_place_correction_trace: list[dict[str, object]] = []
        desired_box_center = np.asarray(target_fp_pose, dtype=np.float64)[:3].copy()
        desired_box_center[2] += float(self.box_half_size[2])
        if skip_right_place_correction:
            right_place_correction_trace.append(
                {
                    "skipped": True,
                    "reason": "release_from_pre_place",
                    "desired_box_center": desired_box_center.round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                }
            )
        for correction_index in range(0 if skip_right_place_correction else int(self.RIGHT_PLACE_CORRECTION_ITERS)):
            current_box_pose = np.asarray(self._box_pose(), dtype=np.float64)
            residual = desired_box_center - current_box_pose[:3]
            if (
                np.linalg.norm(residual[:2]) < self.RIGHT_PLACE_CORRECTION_EPS
                and abs(float(residual[2])) < self.RIGHT_PLACE_CORRECTION_EPS
            ):
                break
            step_delta = residual.copy()
            step_norm = float(np.linalg.norm(step_delta))
            if step_norm > self.RIGHT_PLACE_CORRECTION_MAX_STEP:
                step_delta *= float(self.RIGHT_PLACE_CORRECTION_MAX_STEP) / step_norm
            right_place_correction_trace.append(
                {
                    "iter": int(correction_index),
                    "box_pose": current_box_pose.round(4).tolist(),
                    "desired_box_center": desired_box_center.round(4).tolist(),
                    "residual": residual.round(4).tolist(),
                    "step_delta": step_delta.round(4).tolist(),
                }
            )
            if self.ctrl.move_by_displacement_via_ik(
                1,
                recorder,
                x=float(step_delta[0]),
                y=float(step_delta[1]),
                z=float(step_delta[2]),
                steps=8,
                settle_steps=3,
                fail_status="right_place_correction_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_place_correction",
                    **left_release_clear_debug,
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "selected_handover_contact_id": int(handover_candidate.contact_id),
                    "place_pose_source": place_pose_source,
                    "right_pre_place_fallback_debug": right_pre_place_fallback_debug,
                    "right_place_fallback_debug": right_place_fallback_debug,
                    "right_place_relative_fallback_debug": right_place_relative_fallback_debug,
                    "right_place_release_from_pre_place_debug": right_place_release_from_pre_place_debug,
                    "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                    "desired_box_center": desired_box_center.round(4).tolist(),
                    "right_place_correction_trace": right_place_correction_trace,
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "right_handover_alignment": right_handover_alignment,
                    "right_handover_close_trace": right_handover_close_trace,
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.ctrl.open_gripper(
            1,
            recorder,
            allow_success_early=False,
            fail_status="right_open_failed",
        ):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            self.last_debug_info = {
                "stage": "right_open",
                **left_release_clear_debug,
                "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                "right_pre_place_world_offset": np.asarray(self.RIGHT_PRE_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                "right_place_world_offset": np.asarray(self.RIGHT_PLACE_WORLD_OFFSET, dtype=np.float64).round(4).tolist(),
                "right_place_pose": np.asarray(right_place_pose, dtype=np.float64).round(4).tolist(),
                "right_place_fallback_debug": right_place_fallback_debug,
                "right_place_relative_fallback_debug": right_place_relative_fallback_debug,
                "right_place_release_from_pre_place_debug": right_place_release_from_pre_place_debug,
                "right_place_correction_trace": right_place_correction_trace,
                "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                "right_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(1), dtype=np.float64).round(4).tolist(),
                "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
            }
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.ctrl.hold_pose(
            recorder,
            steps=self.FINAL_RELEASE_SETTLE_STEPS,
            right_gripper=1.0,
            allow_success_early=False,
        ):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.RIGHT_RELEASE_RETRACT_DISTANCE > 1e-8:
            if self.ctrl.move_by_displacement(
                1,
                recorder,
                z=self.RIGHT_RELEASE_RETRACT_DISTANCE,
                move_axis="grasp",
                fail_status="right_release_retract_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_release_retract",
                    **left_release_clear_debug,
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.ctrl.move_by_displacement(
            1,
            recorder,
            z=self.RIGHT_RELEASE_LIFT_DISTANCE,
            move_axis="world",
            fail_status="right_release_lift_failed",
        ):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            self.last_debug_info = {
                "stage": "right_release_lift",
                **left_release_clear_debug,
                "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                "plan_debug": dict(self.ctrl.last_plan_debug),
            }
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.ctrl.back_to_origin(1, recorder, fail_status="right_release_clear_failed"):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            self.last_debug_info = {
                "stage": "right_release_clear",
                **left_release_clear_debug,
                "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
            }
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        if self.ctrl.hold_pose(
            recorder,
            steps=self.FINAL_SETTLE_STEPS,
            right_gripper=1.0,
            allow_success_early=False,
        ):
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        if self.ctrl.last_status != "running":
            return False, self.ctrl.steps_taken, self.ctrl.last_status

        previous_success_gate_enabled = self._set_success_gate_enabled(True)
        try:
            final_info = self.base_env.evaluate()
        finally:
            if previous_success_gate_enabled is not None:
                self._set_success_gate_enabled(previous_success_gate_enabled)
        target_fp_pose = self._target_functional_world_pose(1)
        final_debug_info = {
            "stage": "final_success_check",
            **left_release_clear_debug,
            "selection_reason": right_handover_selection_reason,
            "selection_debug": right_handover_selection_debug,
            "selected_handover_contact_id": int(handover_candidate.contact_id),
            "right_pre_place_fallback_debug": right_pre_place_fallback_debug,
            "right_place_fallback_debug": right_place_fallback_debug,
            "right_place_relative_fallback_debug": right_place_relative_fallback_debug,
            "right_place_release_from_pre_place_debug": right_place_release_from_pre_place_debug,
            "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
            "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
            "box_to_target_xy": (
                np.asarray(self._box_pose(), dtype=np.float64)[:2] - np.asarray(target_fp_pose, dtype=np.float64)[:2]
            ).round(4).tolist(),
            "right_handover_alignment": right_handover_alignment,
            "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
            "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
            "xy_aligned": bool(to_numpy(final_info["xy_aligned"]).reshape(-1)[0]),
            "z_aligned": bool(to_numpy(final_info["z_aligned"]).reshape(-1)[0]),
            "spatial_success": bool(to_numpy(final_info["spatial_success"]).reshape(-1)[0]),
            "right_gripper_open": bool(to_numpy(final_info["right_gripper_open"]).reshape(-1)[0]),
            "right_ee_clear": bool(to_numpy(final_info["right_ee_clear"]).reshape(-1)[0]),
            "box_static": bool(to_numpy(final_info["box_static"]).reshape(-1)[0]),
            "right_handover_close_trace": right_handover_close_trace,
            "right_place_correction_trace": right_place_correction_trace,
        }
        self.last_debug_info = final_debug_info
        if bool(to_numpy(final_info["success"]).reshape(-1)[0]):
            self.ctrl.last_status = "success"
            return True, self.ctrl.steps_taken, self.ctrl.last_status
        self.ctrl.last_status = "final_success_check_failed"
        return False, self.ctrl.steps_taken, self.ctrl.last_status

    def solve_episode(self, *, recorder, max_episode_steps: int) -> tuple[bool, int, str]:
        self.ctrl.reset_episode(max_episode_steps)
        self.last_debug_info = {}
        self.first_pick_debug = {}
        previous_success_gate_enabled = self._set_success_gate_enabled(False)

        try:
            if self.ctrl.settle_grippers_to_targets(
                recorder,
                left_target=1.0,
                right_target=1.0,
                fail_status="initial_gripper_open_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "initial_gripper_open",
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "left_gripper_joint_qpos": np.asarray(self.ctrl.current_gripper_joint_qpos(0), dtype=np.float64).round(4).tolist(),
                    "right_gripper_joint_qpos": np.asarray(self.ctrl.current_gripper_joint_qpos(1), dtype=np.float64).round(4).tolist(),
                    "left_gripper_joint_target": np.asarray(self.ctrl.target_gripper_joint_qpos(0, 1.0), dtype=np.float64).round(4).tolist(),
                    "right_gripper_joint_target": np.asarray(self.ctrl.target_gripper_joint_qpos(1, 1.0), dtype=np.float64).round(4).tolist(),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            pick_candidate = self.ctrl.choose_grasp_candidate(
                desired_pose=self._box_pose(),
                arm_index=0,
                contact_ids=LEFT_GRASP_CONTACT_IDS,
                pre_distance=self.PRE_GRASP_DISTANCE,
                grasp_distance=self.LEFT_FIRST_GRASP_DISTANCE,
                pose_builder=self._grasp_pose_from_box_pose,
                preferred_quat=self._preferred_grasp_quat(0),
                preferred_top_down_quat=TOP_DOWN_PREFERRED_GRASP_QUATS[0],
                pre_pose_candidates_builder=self._left_first_grasp_pre_pose_candidates_from_box_pose,
                selection_mode="prefer_side",
            )
            if pick_candidate is None:
                self.ctrl.last_status = "left_pick_plan_failed"
                self.last_debug_info = {
                    "stage": "left_pick",
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_root_pose": np.asarray(self.base_env.get_arm_root_pose_array(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                    "attempts": list(self.ctrl.last_grasp_attempts),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            self.first_pick_debug = {
                "stage": "left_pick",
                "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                "left_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64).round(4).tolist(),
                "left_root_pose": np.asarray(self.base_env.get_arm_root_pose_array(0), dtype=np.float64).round(4).tolist(),
                "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                "selected_contact_id": int(pick_candidate.contact_id),
                "selected_pre_pose": np.asarray(pick_candidate.pre_pose, dtype=np.float64).round(4).tolist(),
                "selected_grasp_pose": np.asarray(pick_candidate.grasp_pose, dtype=np.float64).round(4).tolist(),
                "selected_pre_plan_len": int(len(pick_candidate.pre_plan["position"])),
                "selected_grasp_plan_len": int(len(pick_candidate.grasp_plan["position"])),
                "selected_grasp_plan_source": str(pick_candidate.grasp_plan_source),
                "attempts": list(self.ctrl.last_grasp_attempts),
            }

            left_approach_pose = self.ctrl.gripper_pose_to_planner_pose(
                0,
                self._get_left_first_grasp_approach_pose(pick_candidate.pre_pose),
            )
            if self.ctrl.move_to_pose(
                0,
                left_approach_pose,
                recorder,
                fail_status="left_pick_approach_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                if self.ALLOW_LEFT_FIRST_GRASP_APPROACH_FALLBACK:
                    self.first_pick_debug["left_pick_approach_fallback"] = {
                        "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                        "left_approach_pose": np.asarray(left_approach_pose, dtype=np.float64).round(4).tolist(),
                        "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                        "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                        "plan_debug": dict(self.ctrl.last_plan_debug),
                    }
                    self.ctrl.last_status = "running"
                else:
                    self.last_debug_info = {
                        "stage": "left_pick_approach",
                        "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                        "left_approach_pose": np.asarray(left_approach_pose, dtype=np.float64).round(4).tolist(),
                        "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                        "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                        "plan_debug": dict(self.ctrl.last_plan_debug),
                    }
                    return False, self.ctrl.steps_taken, self.ctrl.last_status

            left_pre_pose = self.ctrl.gripper_pose_to_planner_pose(0, pick_candidate.pre_pose)
            if self.ctrl.move_to_pose(
                0,
                left_pre_pose,
                recorder,
                prefer_screw=True,
                fail_status="left_pick_pre_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "left_pick_pre",
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "left_pre_pose": np.asarray(left_pre_pose, dtype=np.float64).round(4).tolist(),
                    "selected_contact_id": int(pick_candidate.contact_id),
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            left_grasp_pose = self.ctrl.gripper_pose_to_planner_pose(0, pick_candidate.grasp_pose)
            if self.ctrl.execute_plan(0, pick_candidate.grasp_plan, recorder):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "left_pick_grasp",
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "left_grasp_pose": np.asarray(left_grasp_pose, dtype=np.float64).round(4).tolist(),
                    "selected_contact_id": int(pick_candidate.contact_id),
                    "selected_grasp_plan_source": str(pick_candidate.grasp_plan_source),
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            self.first_pick_debug["post_grasp_box_pose"] = np.asarray(
                self._box_pose(),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_grasp_left_ee_pose"] = np.asarray(
                self.ctrl.current_ee_pose(0),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_grasp_left_gripper_pose"] = np.asarray(
                self.ctrl.current_gripper_pose(0),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_grasp_left_qpos"] = np.asarray(
                self.ctrl.current_qpos(0),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_grasp_left_gripper_joint_qpos"] = np.asarray(
                self.ctrl.current_gripper_joint_qpos(0),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_grasp_left_gripper_value"] = round(
                float(self.ctrl.current_gripper_value(0)),
                4,
            )

            if self.LEFT_FIRST_GRASP_CLOSE_STEPS > 0:
                success = self.ctrl.hold_pose(
                    recorder,
                    steps=self.LEFT_FIRST_GRASP_CLOSE_STEPS,
                    left_gripper=0.0,
                    allow_success_early=False,
                )
                self.first_pick_debug["post_close_box_pose"] = np.asarray(
                    self._box_pose(),
                    dtype=np.float64,
                ).round(4).tolist()
                self.first_pick_debug["post_close_left_ee_pose"] = np.asarray(
                    self.ctrl.current_ee_pose(0),
                    dtype=np.float64,
                ).round(4).tolist()
                self.first_pick_debug["post_close_left_gripper_pose"] = np.asarray(
                    self.ctrl.current_gripper_pose(0),
                    dtype=np.float64,
                ).round(4).tolist()
                self.first_pick_debug["post_close_left_qpos"] = np.asarray(
                    self.ctrl.current_qpos(0),
                    dtype=np.float64,
                ).round(4).tolist()
                self.first_pick_debug["post_close_left_gripper_joint_qpos"] = np.asarray(
                    self.ctrl.current_gripper_joint_qpos(0),
                    dtype=np.float64,
                ).round(4).tolist()
                self.first_pick_debug["post_close_left_gripper_joint_target"] = np.asarray(
                    self.ctrl.target_gripper_joint_qpos(0, 0.0),
                    dtype=np.float64,
                ).round(4).tolist()
                self.first_pick_debug["post_close_left_gripper_value"] = round(
                    float(self.ctrl.current_gripper_value(0)),
                    4,
                )
                self.first_pick_debug["post_close_alignment"] = self._box_gripper_alignment_debug(
                    arm_index=0,
                    planned_gripper_pose=pick_candidate.grasp_pose,
                )
                if success:
                    return True, self.ctrl.steps_taken, self.ctrl.last_status
                if self.ctrl.last_status != "running":
                    self.last_debug_info = {
                        "stage": "left_pick_close",
                        "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                        "left_grasp_pose": np.asarray(left_grasp_pose, dtype=np.float64).round(4).tolist(),
                        "selected_contact_id": int(pick_candidate.contact_id),
                        "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                        "left_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64).round(4).tolist(),
                        "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                        "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                        "plan_debug": dict(self.ctrl.last_plan_debug),
                    }
                    return False, self.ctrl.steps_taken, self.ctrl.last_status

            if self.ctrl.move_by_displacement_via_ik(
                0,
                recorder,
                z=self.LEFT_FIRST_GRASP_CLAMP_LIFT_DISTANCE,
                steps=self.LEFT_FIRST_GRASP_CLAMP_LIFT_QPOS_STEPS,
                settle_steps=self.LEFT_FIRST_GRASP_CLAMP_LIFT_SETTLE_STEPS,
                fail_status="left_clamp_lift_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "left_clamp_lift",
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "left_clamp_lift_distance": round(float(self.LEFT_FIRST_GRASP_CLAMP_LIFT_DISTANCE), 4),
                    "left_clamp_lift_mode": "ik_qpos",
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_gripper_pose": np.asarray(self.ctrl.current_gripper_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            self.first_pick_debug["post_clamp_lift_box_pose"] = np.asarray(
                self._box_pose(),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_clamp_lift_left_ee_pose"] = np.asarray(
                self.ctrl.current_ee_pose(0),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_clamp_lift_left_gripper_pose"] = np.asarray(
                self.ctrl.current_gripper_pose(0),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_clamp_lift_left_qpos"] = np.asarray(
                self.ctrl.current_qpos(0),
                dtype=np.float64,
            ).round(4).tolist()
            self.first_pick_debug["post_clamp_lift_left_gripper_value"] = round(
                float(self.ctrl.current_gripper_value(0)),
                4,
            )
            self.first_pick_debug["post_clamp_lift_mode"] = "ik_qpos"
            self.first_pick_debug["post_clamp_lift_distance"] = round(
                float(self.LEFT_FIRST_GRASP_CLAMP_LIFT_DISTANCE),
                4,
            )
            self.first_pick_debug["post_clamp_lift_alignment"] = self._box_gripper_alignment_debug(
                arm_index=0,
                planned_gripper_pose=pick_candidate.grasp_pose,
            )

            if self.ctrl.move_by_displacement(
                0,
                recorder,
                z=self.LEFT_LIFT_DISTANCE - self.LEFT_FIRST_GRASP_CLAMP_LIFT_DISTANCE,
                fail_status="left_lift_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "left_lift",
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            planned_handover_box_pose = self._solve_box_pose_from_functional_target(
                np.asarray(BLOCK_MIDDLE_POSE, dtype=np.float64),
                functional_id=0,
            )
            handover_box_pose = planned_handover_box_pose
            left_handover_pose = self._place_pose_from_current_grasp(
                arm_index=0,
                target_pose=np.asarray(BLOCK_MIDDLE_POSE, dtype=np.float64),
                functional_id=0,
                pre_dis=0.0,
                pre_dis_axis="grasp",
            )
            if self.ctrl.move_to_pose(
                0,
                self.ctrl.gripper_pose_to_planner_pose(0, left_handover_pose),
                recorder,
                prefer_screw=True,
                fail_status="left_handover_move_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "left_handover_move",
                    "handover_box_pose": np.asarray(handover_box_pose, dtype=np.float64).round(4).tolist(),
                    "left_handover_pose": np.asarray(left_handover_pose, dtype=np.float64).round(4).tolist(),
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status
            self.first_pick_debug["post_left_handover_alignment"] = self._box_gripper_alignment_debug(
                arm_index=0,
                planned_box_pose=planned_handover_box_pose,
                planned_gripper_pose=left_handover_pose,
            )
            actual_handover_box_pose = np.asarray(self._box_pose(), dtype=np.float64).copy()
            handover_box_pose = actual_handover_box_pose
            self.first_pick_debug["right_handover_reference"] = {
                "source": "actual_box_pose_after_left_handover",
                "planned_handover_box_pose": planned_handover_box_pose.round(4).tolist(),
                "actual_handover_box_pose": actual_handover_box_pose.round(4).tolist(),
                "actual_minus_planned_xyz": (
                    actual_handover_box_pose[:3] - planned_handover_box_pose[:3]
                ).round(4).tolist(),
            }

            (
                handover_candidate,
                selected_place_candidate,
                right_handover_selection_debug,
                right_handover_selection_reason,
                right_handover_options,
            ) = self._select_right_handover_candidate(handover_box_pose)
            if handover_candidate is None:
                self.ctrl.last_status = "right_handover_plan_failed"
                self.last_debug_info = {
                    "stage": "right_handover",
                    "planned_handover_box_pose": np.asarray(
                        planned_handover_box_pose,
                        dtype=np.float64,
                    ).round(4).tolist(),
                    "handover_box_pose": np.asarray(handover_box_pose, dtype=np.float64).round(4).tolist(),
                    "selection_reason": right_handover_selection_reason,
                    "selection_debug": right_handover_selection_debug,
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "attempts": list(self.ctrl.last_grasp_attempts),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status
            viable_branch_options = [item for item in right_handover_options if item["future_place_ok"]]
            trial_branch_options: list[dict[str, object]] = []
            for option in viable_branch_options:
                future_place_candidates = list(option.get("future_place_candidates") or [])
                if not future_place_candidates and option.get("future_place_candidate") is not None:
                    future_place_candidates = [option["future_place_candidate"]]
                seen_variant_indices: set[int] = set()
                for candidate_option in future_place_candidates[: self.RIGHT_HANDOVER_PLACE_VARIANT_TRIAL_LIMIT]:
                    variant_index = int(candidate_option["variant_index"])
                    if variant_index in seen_variant_indices:
                        continue
                    seen_variant_indices.add(variant_index)
                    trial_branch_options.append(
                        {
                            **option,
                            "future_place_candidate": candidate_option,
                            "future_place_variant_index": variant_index,
                            "future_place_path_len": float(candidate_option["path_len"]),
                        }
                    )
            if self._allow_right_handover_branch_trial() and len(trial_branch_options) > 1:
                planner_contact_id = int(handover_candidate.contact_id)
                planner_place_variant_index = (
                    None if selected_place_candidate is None else int(selected_place_candidate["variant_index"])
                )
                trial_branch_options.sort(
                    key=lambda item: (
                        float(item["future_place_path_len"]),
                        float(item["candidate"].score),
                        int(item.get("future_place_variant_index") or 0),
                    )
                )
                capped_trial_branch_options: list[dict[str, object]] = []
                for option in trial_branch_options:
                    if (
                        int(option["contact_id"]) == planner_contact_id
                        and option.get("future_place_variant_index") == planner_place_variant_index
                    ):
                        capped_trial_branch_options.append(option)
                        break
                for option in trial_branch_options:
                    if len(capped_trial_branch_options) >= self.RIGHT_HANDOVER_TOTAL_TRIAL_BRANCH_LIMIT:
                        break
                    if any(
                        int(existing["contact_id"]) == int(option["contact_id"])
                        and existing.get("future_place_variant_index") == option.get("future_place_variant_index")
                        for existing in capped_trial_branch_options
                    ):
                        continue
                    capped_trial_branch_options.append(option)
                trial_branch_options = capped_trial_branch_options

            if self._allow_right_handover_branch_trial() and len(trial_branch_options) > 1:
                snapshot = self._capture_rollout_snapshot()
                trial_results: list[dict[str, object]] = []
                best_branch_option = None
                best_branch_score = None
                best_promising_branch_option = None
                best_promising_branch_score = None
                planner_branch_score = None
                trial_recorder = _NullRecorder()
                for option in trial_branch_options:
                    self._restore_rollout_snapshot(snapshot)
                    previous_success_detection = self.ctrl.success_detection_enabled
                    self.ctrl.success_detection_enabled = False
                    try:
                        trial_success, trial_steps, trial_status = self._execute_right_branch(
                            recorder=trial_recorder,
                            handover_box_pose=handover_box_pose,
                            handover_candidate=option["candidate"],
                            selected_place_candidate=option["future_place_candidate"],
                            right_handover_selection_debug=right_handover_selection_debug,
                            right_handover_selection_reason=right_handover_selection_reason,
                        )
                    finally:
                        self.ctrl.success_detection_enabled = previous_success_detection
                    trial_debug_info = copy.deepcopy(self.last_debug_info)
                    branch_score = self._trial_branch_score(
                        success=trial_success,
                        status=trial_status,
                        debug_info=trial_debug_info,
                        steps_taken=trial_steps,
                    )
                    trial_reached_final_check = trial_debug_info.get("stage") == "final_success_check"
                    trial_results.append(
                        {
                            "contact_id": int(option["contact_id"]),
                            "place_variant_index": None if option.get("future_place_candidate") is None else int(option["future_place_candidate"]["variant_index"]),
                            "status": str(trial_status),
                            "steps_taken": int(trial_steps),
                            "score": list(branch_score),
                            "stage": trial_debug_info.get("stage"),
                            "reached_final_check": bool(trial_reached_final_check),
                            "xy_aligned": bool(trial_debug_info.get("xy_aligned", False)),
                            "z_aligned": bool(trial_debug_info.get("z_aligned", False)),
                            "spatial_success": bool(trial_debug_info.get("spatial_success", False)),
                            "box_to_target_xy": trial_debug_info.get("box_to_target_xy"),
                        }
                    )
                    if best_branch_score is None or branch_score > best_branch_score:
                        best_branch_score = branch_score
                        best_branch_option = option
                    if trial_reached_final_check and (
                        best_promising_branch_score is None or branch_score > best_promising_branch_score
                    ):
                        best_promising_branch_score = branch_score
                        best_promising_branch_option = option
                    if (
                        int(option["contact_id"]) == planner_contact_id
                        and option.get("future_place_variant_index") == planner_place_variant_index
                    ):
                        planner_branch_score = branch_score
                self._restore_rollout_snapshot(snapshot)
                trial_selection_reason = "no_promising_trial"
                selected_trial_branch_option = None
                if best_promising_branch_option is not None:
                    if self._trial_branch_supports_switch(
                        planner_score=planner_branch_score,
                        candidate_score=best_promising_branch_score,
                    ):
                        selected_trial_branch_option = best_promising_branch_option
                        trial_selection_reason = "promising_trial_selected"
                    else:
                        trial_selection_reason = "kept_planner_selection_margin"
                elif best_branch_option is not None:
                    trial_selection_reason = "kept_planner_selection"

                if selected_trial_branch_option is not None:
                    handover_candidate = selected_trial_branch_option["candidate"]
                    selected_place_candidate = selected_trial_branch_option["future_place_candidate"]
                    right_handover_selection_reason = f"{right_handover_selection_reason}_trial_selected"

                right_handover_selection_debug = list(right_handover_selection_debug) + [
                    {
                        "trial_selection_reason": trial_selection_reason,
                        "trial_branch_count": int(len(trial_branch_options)),
                        "trial_branch_results": trial_results,
                    }
                ]

            return self._execute_right_branch(
                recorder=recorder,
                handover_box_pose=handover_box_pose,
                handover_candidate=handover_candidate,
                selected_place_candidate=selected_place_candidate,
                right_handover_selection_debug=right_handover_selection_debug,
                right_handover_selection_reason=right_handover_selection_reason,
            )
        except EpisodeLimitReached:
            self.ctrl.last_status = "max_episode_steps_exceeded"
            return False, self.ctrl.steps_taken, self.ctrl.last_status
        finally:
            if previous_success_gate_enabled is not None:
                self._set_success_gate_enabled(previous_success_gate_enabled)
