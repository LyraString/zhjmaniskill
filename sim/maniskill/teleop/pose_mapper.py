"""Map PICO controller motion to bounded bimanual EEF targets."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .xr_client import XRState


DEFAULT_XR_TO_WORLD_ROTATION = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Quaternion must be finite and non-zero.")
    quaternion /= norm
    return quaternion


def _rotation_from_wxyz(quaternion: np.ndarray) -> Rotation:
    w, x, y, z = _normalize_quaternion_wxyz(quaternion)
    return Rotation.from_quat([x, y, z, w])


def _wxyz_from_rotation(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return _normalize_quaternion_wxyz(np.asarray([w, x, y, z]))


def _pose_world_from_root(root_pose: np.ndarray, root_pose_target: np.ndarray) -> np.ndarray:
    root_pose = np.asarray(root_pose, dtype=np.float64).reshape(7)
    root_pose_target = np.asarray(root_pose_target, dtype=np.float64).reshape(7)
    root_rotation = _rotation_from_wxyz(root_pose[3:])
    target_rotation = root_rotation * _rotation_from_wxyz(root_pose_target[3:])
    target_position = root_pose[:3] + root_rotation.apply(root_pose_target[:3])
    return np.concatenate([target_position, _wxyz_from_rotation(target_rotation)])


def _pose_root_from_world(root_pose: np.ndarray, world_pose_target: np.ndarray) -> np.ndarray:
    root_pose = np.asarray(root_pose, dtype=np.float64).reshape(7)
    world_pose_target = np.asarray(world_pose_target, dtype=np.float64).reshape(7)
    root_rotation_inv = _rotation_from_wxyz(root_pose[3:]).inv()
    target_position = root_rotation_inv.apply(world_pose_target[:3] - root_pose[:3])
    target_rotation = root_rotation_inv * _rotation_from_wxyz(world_pose_target[3:])
    return np.concatenate([target_position, _wxyz_from_rotation(target_rotation)])


def _bounded_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= float(max_norm) or norm <= 0.0:
        return vector
    return vector * (float(max_norm) / norm)


class BimanualXRMapper:
    """Clutched target-pose mapper with the repo's delta-action semantics."""

    def __init__(
        self,
        *,
        xr_to_world_rotation: np.ndarray = DEFAULT_XR_TO_WORLD_ROTATION,
        translation_scale: float = 1.0,
        grip_threshold: float = 0.9,
        max_translation_step: float = 0.015,
        max_rotation_step: float = 0.10,
    ) -> None:
        rotation = np.asarray(xr_to_world_rotation, dtype=np.float64).reshape(3, 3)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("xr_to_world_rotation must be orthonormal.")
        if float(np.linalg.det(rotation)) < 0.999:
            raise ValueError("xr_to_world_rotation must be a proper rotation matrix.")
        if translation_scale <= 0.0:
            raise ValueError("translation_scale must be positive.")
        if max_translation_step <= 0.0 or max_rotation_step <= 0.0:
            raise ValueError("Per-step motion limits must be positive.")

        self.xr_to_world_rotation = rotation
        self.translation_scale = float(translation_scale)
        self.grip_threshold = float(grip_threshold)
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.desired_ee_pose_root: np.ndarray | None = None
        self._active = np.zeros(2, dtype=np.bool_)
        self._controller_anchor_world = np.zeros((2, 7), dtype=np.float64)
        self._ee_anchor_world = np.zeros((2, 7), dtype=np.float64)

    @property
    def clutch_active(self) -> np.ndarray:
        return self._active.copy()

    def reset(self, current_ee_pose_root: np.ndarray) -> None:
        current_ee_pose_root = (
            np.asarray(current_ee_pose_root, dtype=np.float64).reshape(2, 7).copy()
        )
        for arm_index in range(2):
            current_ee_pose_root[arm_index, 3:] = _normalize_quaternion_wxyz(
                current_ee_pose_root[arm_index, 3:]
            )
        self.desired_ee_pose_root = current_ee_pose_root.copy()
        self._active.fill(False)

    def _mapped_controller_pose(self, pose_xyzw: np.ndarray) -> np.ndarray:
        pose_xyzw = np.asarray(pose_xyzw, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(pose_xyzw)):
            raise ValueError("XR controller pose contains NaN or Inf.")
        quaternion_xyzw = pose_xyzw[3:]
        quaternion_norm = float(np.linalg.norm(quaternion_xyzw))
        if quaternion_norm < 1e-8:
            raise ValueError("XR controller quaternion is zero.")
        controller_rotation = Rotation.from_quat(quaternion_xyzw / quaternion_norm)
        axis_rotation = Rotation.from_matrix(self.xr_to_world_rotation)
        mapped_rotation = axis_rotation * controller_rotation * axis_rotation.inv()
        mapped_position = self.xr_to_world_rotation @ pose_xyzw[:3]
        return np.concatenate([mapped_position, _wxyz_from_rotation(mapped_rotation)])

    def update_targets(
        self,
        xr_state: XRState,
        *,
        root_pose_world: np.ndarray,
        current_ee_pose_root: np.ndarray,
        allow_controller_motion: bool = True,
    ) -> np.ndarray:
        if self.desired_ee_pose_root is None:
            self.reset(current_ee_pose_root)
        root_pose_world = np.asarray(root_pose_world, dtype=np.float64).reshape(2, 7)
        current_ee_pose_root = np.asarray(current_ee_pose_root, dtype=np.float64).reshape(2, 7)

        for arm_index in range(2):
            requested_active = bool(xr_state.grip[arm_index] > self.grip_threshold)
            if not allow_controller_motion:
                requested_active = False

            if requested_active:
                try:
                    controller_world = self._mapped_controller_pose(
                        xr_state.controller_pose_xyzw[arm_index]
                    )
                except ValueError:
                    requested_active = False

            if requested_active and not self._active[arm_index]:
                self._controller_anchor_world[arm_index] = controller_world
                self._ee_anchor_world[arm_index] = _pose_world_from_root(
                    root_pose_world[arm_index],
                    current_ee_pose_root[arm_index],
                )

            if requested_active:
                controller_anchor = self._controller_anchor_world[arm_index]
                ee_anchor = self._ee_anchor_world[arm_index]
                delta_position = self.translation_scale * (
                    controller_world[:3] - controller_anchor[:3]
                )
                controller_delta_rotation = (
                    _rotation_from_wxyz(controller_world[3:])
                    * _rotation_from_wxyz(controller_anchor[3:]).inv()
                )
                target_world = np.concatenate(
                    [
                        ee_anchor[:3] + delta_position,
                        _wxyz_from_rotation(
                            controller_delta_rotation * _rotation_from_wxyz(ee_anchor[3:])
                        ),
                    ]
                )
                self.desired_ee_pose_root[arm_index] = _pose_root_from_world(
                    root_pose_world[arm_index], target_world
                )

            self._active[arm_index] = requested_active

        return self.desired_ee_pose_root.copy()

    def build_delta_action(
        self,
        *,
        current_ee_pose_root: np.ndarray,
        trigger: np.ndarray,
    ) -> np.ndarray:
        if self.desired_ee_pose_root is None:
            raise RuntimeError("reset(...) or update_targets(...) must be called first.")
        current_ee_pose_root = np.asarray(current_ee_pose_root, dtype=np.float64).reshape(2, 7)
        trigger = np.asarray(trigger, dtype=np.float64).reshape(2)
        action = np.zeros(14, dtype=np.float32)
        for arm_index, offset in ((0, 0), (1, 7)):
            current_pose = current_ee_pose_root[arm_index]
            desired_pose = self.desired_ee_pose_root[arm_index]
            delta_position = _bounded_vector(
                desired_pose[:3] - current_pose[:3],
                self.max_translation_step,
            )
            delta_rotation = (
                _rotation_from_wxyz(desired_pose[3:])
                * _rotation_from_wxyz(current_pose[3:]).inv()
            ).as_euler("XYZ")
            delta_rotation = _bounded_vector(delta_rotation, self.max_rotation_step)
            action[offset : offset + 3] = delta_position
            action[offset + 3 : offset + 6] = delta_rotation
            # XR trigger: 0=open, 1=closed. RoboTwin gripper action: 1=open, 0=closed.
            action[offset + 6] = float(np.clip(1.0 - trigger[arm_index], 0.0, 1.0))
        return action
