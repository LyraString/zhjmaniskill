from __future__ import annotations

import numpy as np

from sim.maniskill.expert.primitives import (
    EpisodeLimitReached,
    FixedRateMotionPrimitives,
    matrix_to_pose,
    pose_to_matrix,
    scaled_local_matrix,
    to_numpy,
)
from sim.maniskill.tasks.handover_block_env import BLOCK_MIDDLE_POSE, BOX_HALF_SIZE, TARGET_HALF_SIZE


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
LEFT_GRASP_CONTACT_IDS = (0, 1)
RIGHT_GRASP_CONTACT_IDS = (4, 5, 6, 7)


class HandoverBlockSolver:
    PRE_GRASP_DISTANCE = 0.12
    HANDOVER_PRE_GRASP_DISTANCE = 0.08
    GRASP_DISTANCE = 0.0
    ROTATE_NUM = 10
    LEFT_ROTATE_LIM = (-0.6, 0.6)
    RIGHT_ROTATE_LIM = (-0.6, 0.6)
    LEFT_FIRST_GRASP_YAW_VARIANTS = (-0.15, -0.075, 0.0, 0.075, 0.15)
    LEFT_LIFT_DISTANCE = 0.16
    LEFT_HANDOVER_CLEARANCE_Z = 0.18
    LEFT_RELEASE_LIFT_DISTANCE = 0.12
    LEFT_FIRST_GRASP_APPROACH_OFFSET = np.asarray([0.0, -0.08, 0.08], dtype=np.float64)
    LEFT_FIRST_GRASP_BOX_AXIS_SHIFT = 0.02
    ABOVE_TARGET_OFFSET = np.asarray([0.05, -0.12, 0.20], dtype=np.float64)
    HANDOVER_APPROACH_OFFSET = np.asarray([0.0, -0.07, 0.10], dtype=np.float64)
    RIGHT_CARRY_LIFT_DISTANCE = 0.14
    RIGHT_RELEASE_LIFT_DISTANCE = 0.12
    FINAL_RELEASE_SETTLE_STEPS = 12
    FINAL_SETTLE_STEPS = 12

    def __init__(self, env_backend):
        self.backend = env_backend
        self.base_env = env_backend.env.unwrapped
        self.ctrl = FixedRateMotionPrimitives(env_backend)
        self.box_half_size = to_numpy(BOX_HALF_SIZE).reshape(3)
        self.target_half_size = to_numpy(TARGET_HALF_SIZE).reshape(3)
        self.last_debug_info: dict[str, object] = {}

    @property
    def last_status(self) -> str:
        return self.ctrl.last_status

    def _box_pose(self) -> np.ndarray:
        return np.concatenate(
            [
                to_numpy(self.base_env.box.pose.p).reshape(-1, 3)[0],
                to_numpy(self.base_env.box.pose.q).reshape(-1, 4)[0],
            ],
            axis=0,
        ).astype(np.float64, copy=False)

    def _target_pose(self) -> np.ndarray:
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

    def _offset_pose_world(self, pose: np.ndarray, *, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
        shifted = np.asarray(pose, dtype=np.float64).copy()
        shifted[:3] += np.asarray([x, y, z], dtype=np.float64)
        return shifted

    def solve_episode(self, *, recorder, max_episode_steps: int) -> tuple[bool, int, str]:
        self.ctrl.reset_episode(max_episode_steps)
        self.last_debug_info = {}

        try:
            if self.ctrl.hold_pose(
                recorder,
                steps=self.ctrl.HOLD_SETTLE_STEPS,
                left_gripper=1.0,
                right_gripper=1.0,
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status

            pick_candidate = self.ctrl.choose_grasp_candidate(
                desired_pose=self._box_pose(),
                arm_index=0,
                contact_ids=LEFT_GRASP_CONTACT_IDS,
                pre_distance=self.PRE_GRASP_DISTANCE,
                grasp_distance=self.GRASP_DISTANCE,
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
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "left_full_qpos": np.asarray(self.ctrl.current_full_qpos(0), dtype=np.float64).round(4).tolist(),
                    "attempts": list(self.ctrl.last_grasp_attempts),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

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
                self.last_debug_info = {
                    "stage": "left_pick_approach",
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "left_approach_pose": np.asarray(left_approach_pose, dtype=np.float64).round(4).tolist(),
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            if self.ctrl.move_to_pose(
                0,
                self.ctrl.gripper_pose_to_planner_pose(0, pick_candidate.pre_pose),
                recorder,
                fail_status="left_pick_plan_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            if self.ctrl.move_to_pose(
                0,
                self.ctrl.gripper_pose_to_planner_pose(0, pick_candidate.grasp_pose),
                recorder,
                prefer_screw=True,
                allow_pose_fallback=False,
                fail_status="left_pick_plan_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            success = self.ctrl.close_gripper(0, recorder)
            if success:
                return True, self.ctrl.steps_taken, self.ctrl.last_status

            if self.ctrl.move_by_displacement(
                0,
                recorder,
                z=self.LEFT_LIFT_DISTANCE,
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

            handover_box_pose = self._solve_box_pose_from_functional_target(
                np.asarray(BLOCK_MIDDLE_POSE, dtype=np.float64),
                functional_id=0,
            )
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

            handover_candidate = self.ctrl.choose_grasp_candidate(
                desired_pose=handover_box_pose,
                arm_index=1,
                contact_ids=RIGHT_GRASP_CONTACT_IDS,
                pre_distance=self.HANDOVER_PRE_GRASP_DISTANCE,
                grasp_distance=self.GRASP_DISTANCE,
                pose_builder=self._grasp_pose_from_box_pose,
                preferred_quat=self._preferred_grasp_quat(1),
                preferred_top_down_quat=TOP_DOWN_PREFERRED_GRASP_QUATS[1],
                pre_pose_candidates_builder=self._grasp_pre_pose_candidates_from_box_pose,
                selection_mode="auto",
            )
            if handover_candidate is None:
                self.ctrl.last_status = "right_handover_plan_failed"
                self.last_debug_info = {
                    "stage": "right_handover",
                    "handover_box_pose": np.asarray(handover_box_pose, dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "right_full_qpos": np.asarray(self.ctrl.current_full_qpos(1), dtype=np.float64).round(4).tolist(),
                    "attempts": list(self.ctrl.last_grasp_attempts),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            approach_pose = self.ctrl.gripper_pose_to_planner_pose(
                1,
                self._get_handover_approach_pose(handover_candidate.pre_pose),
            )
            if self.ctrl.move_to_pose(1, approach_pose, recorder, fail_status="right_handover_approach_failed"):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            handover_candidate, success = self.ctrl.grasp_actor(
                desired_pose=handover_box_pose,
                arm_index=1,
                contact_ids=(handover_candidate.contact_id,),
                pre_grasp_dis=self.HANDOVER_PRE_GRASP_DISTANCE,
                grasp_dis=self.GRASP_DISTANCE,
                pose_builder=self._grasp_pose_from_box_pose,
                preferred_quat=self._preferred_grasp_quat(1),
                preferred_top_down_quat=TOP_DOWN_PREFERRED_GRASP_QUATS[1],
                pre_pose_candidates_builder=self._grasp_pre_pose_candidates_from_box_pose,
                recorder=recorder,
                fail_status="right_handover_plan_failed",
                selection_mode="auto",
            )
            if handover_candidate is None:
                return False, self.ctrl.steps_taken, self.ctrl.last_status
            if success:
                return True, self.ctrl.steps_taken, self.ctrl.last_status

            if self.ctrl.open_gripper(0, recorder, fail_status="left_open_failed"):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                return False, self.ctrl.steps_taken, self.ctrl.last_status

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
                    "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                    "left_qpos": np.asarray(self.ctrl.current_qpos(0), dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            if self.ctrl.back_to_origin(0, recorder, fail_status="left_back_to_origin_failed"):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            if self.ctrl.move_by_displacement(
                1,
                recorder,
                z=self.RIGHT_CARRY_LIFT_DISTANCE,
                move_axis="world",
                fail_status="right_carry_lift_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_carry_lift",
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            target_fp_pose = self._target_functional_world_pose(1)
            current_right_pose = self.ctrl.current_ee_pose(1).copy()
            above_target_pose = current_right_pose.copy()
            above_target_pose[0] = float(target_fp_pose[0] + self.ABOVE_TARGET_OFFSET[0])
            above_target_pose[1] = float(target_fp_pose[1] + self.ABOVE_TARGET_OFFSET[1])
            above_target_pose[2] = float(max(current_right_pose[2], target_fp_pose[2] + self.ABOVE_TARGET_OFFSET[2]))
            if self.ctrl.move_to_pose(1, above_target_pose, recorder, fail_status="right_above_target_failed"):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_above_target",
                    "target_fp_pose": np.asarray(target_fp_pose, dtype=np.float64).round(4).tolist(),
                    "above_target_pose": np.asarray(above_target_pose, dtype=np.float64).round(4).tolist(),
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
                }
                return False, self.ctrl.steps_taken, self.ctrl.last_status

            right_pre_place_pose = self._place_pose_from_current_grasp(
                arm_index=1,
                target_pose=target_fp_pose,
                functional_id=0,
                pre_dis=0.08,
                pre_dis_axis="fp",
            )
            right_place_pose = self._place_pose_from_current_grasp(
                arm_index=1,
                target_pose=target_fp_pose,
                functional_id=0,
                pre_dis=self.GRASP_DISTANCE,
                pre_dis_axis="fp",
            )
            if self.ctrl.place_actor(
                arm_index=1,
                pre_pose=self.ctrl.gripper_pose_to_planner_pose(1, right_pre_place_pose),
                place_pose=self.ctrl.gripper_pose_to_planner_pose(1, right_place_pose),
                recorder=recorder,
                fail_status="right_place_failed",
                open_after=True,
                prefer_screw_final=True,
                allow_success_on_release=False,
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
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

            if self.ctrl.move_to_pose(
                1,
                above_target_pose,
                recorder,
                fail_status="right_release_lift_failed",
            ):
                return True, self.ctrl.steps_taken, self.ctrl.last_status
            if self.ctrl.last_status != "running":
                self.last_debug_info = {
                    "stage": "right_release_lift",
                    "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                    "right_qpos": np.asarray(self.ctrl.current_qpos(1), dtype=np.float64).round(4).tolist(),
                    "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                    "above_target_pose": np.asarray(above_target_pose, dtype=np.float64).round(4).tolist(),
                    "plan_debug": dict(self.ctrl.last_plan_debug),
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

            if self.ctrl.current_success():
                self.ctrl.last_status = "success"
                return True, self.ctrl.steps_taken, self.ctrl.last_status

            final_info = self.base_env.evaluate()
            self.last_debug_info = {
                "stage": "final_success_check",
                "box_pose": np.asarray(self._box_pose(), dtype=np.float64).round(4).tolist(),
                "right_ee_pose": np.asarray(self.ctrl.current_ee_pose(1), dtype=np.float64).round(4).tolist(),
                "left_ee_pose": np.asarray(self.ctrl.current_ee_pose(0), dtype=np.float64).round(4).tolist(),
                "xy_aligned": bool(to_numpy(final_info["xy_aligned"]).reshape(-1)[0]),
                "z_aligned": bool(to_numpy(final_info["z_aligned"]).reshape(-1)[0]),
                "spatial_success": bool(to_numpy(final_info["spatial_success"]).reshape(-1)[0]),
                "right_gripper_open": bool(to_numpy(final_info["right_gripper_open"]).reshape(-1)[0]),
                "right_ee_clear": bool(to_numpy(final_info["right_ee_clear"]).reshape(-1)[0]),
                "box_static": bool(to_numpy(final_info["box_static"]).reshape(-1)[0]),
            }
            self.ctrl.last_status = "final_success_check_failed"
            return False, self.ctrl.steps_taken, self.ctrl.last_status
        except EpisodeLimitReached:
            self.ctrl.last_status = "max_episode_steps_exceeded"
            return False, self.ctrl.steps_taken, self.ctrl.last_status
