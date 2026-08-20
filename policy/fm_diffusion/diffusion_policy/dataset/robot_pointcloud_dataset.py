import copy
import numbers
from pathlib import Path
from typing import Dict

import numba
import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset


_ARM_OFFSETS = (0, 7)
_DEFAULT_UR5_WSG_URDF = (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "embodiments"
    / "ur5-wsg"
    / "ur5_wsg_gripper.urdf"
)


def _reconstruct_absolute_eef_actions(state, recorded_action, urdf_path):
    """Recover old controller targets and actual EEF poses in each arm root."""

    import pytorch_kinematics as pk

    state = np.asarray(state, dtype=np.float32)
    recorded_action = np.asarray(recorded_action, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(f"Expected state shape [T, 14], got {state.shape}.")
    if recorded_action.shape != state.shape:
        raise ValueError(
            "Expected recorded action shape to match state shape, got "
            f"state={state.shape} action={recorded_action.shape}."
        )
    if not np.isfinite(state).all() or not np.isfinite(recorded_action).all():
        raise ValueError("State and recorded actions must contain only finite values.")

    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"UR5-Wsg URDF not found: {urdf_path}")
    with urdf_path.open("r", encoding="utf-8") as stream:
        chain = pk.build_serial_chain_from_urdf(
            stream.read(),
            end_link_name="ee_link",
            root_link_name="world",
        ).to(dtype=torch.float32)
    expected_joints = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    if chain.get_joint_parameter_names() != expected_joints:
        raise ValueError(
            "UR5-Wsg FK joint order does not match the recorded state layout."
        )

    actual_eef_pose = np.empty((len(state), 2, 12), dtype=np.float32)
    absolute_action = recorded_action.copy()
    fk_batch_size = 131072
    with torch.inference_mode():
        for arm_index, offset in enumerate(_ARM_OFFSETS):
            for start in range(0, len(state), fk_batch_size):
                end = min(start + fk_batch_size, len(state))
                qpos = torch.from_numpy(state[start:end, offset : offset + 6])
                matrix = chain.forward_kinematics(qpos).get_matrix()
                actual_position = matrix[:, :3, 3]
                actual_rotation = matrix[:, :3, :3]
                delta_position = torch.from_numpy(
                    recorded_action[start:end, offset : offset + 3]
                )
                delta_euler = torch.from_numpy(
                    recorded_action[start:end, offset + 3 : offset + 6]
                )
                target_rotation = torch.matmul(
                    euler_angles_to_matrix(delta_euler, "XYZ"),
                    actual_rotation,
                )

                actual_eef_pose[start:end, arm_index, :3] = actual_position.numpy()
                actual_eef_pose[start:end, arm_index, 3:] = (
                    actual_rotation.reshape(-1, 9).numpy()
                )
                absolute_action[start:end, offset : offset + 3] = (
                    actual_position + delta_position
                ).numpy()
                absolute_action[start:end, offset + 3 : offset + 6] = (
                    matrix_to_euler_angles(target_rotation, "XYZ").numpy()
                )

    if not np.isfinite(actual_eef_pose).all() or not np.isfinite(absolute_action).all():
        raise ValueError("Reconstructed EEF poses contain non-finite values.")
    return actual_eef_pose, absolute_action


def _sequence_source_indices(indices, sequence_length):
    """Return replay-buffer indices after SequenceSampler edge padding."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError(f"Expected sampler indices shape [N, 4], got {indices.shape}.")
    if int(sequence_length) <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}.")
    steps = np.arange(sequence_length, dtype=np.int64)[None, :]
    buffer_start = indices[:, 0:1]
    sample_start = indices[:, 2:3]
    source_length = indices[:, 1:2] - buffer_start
    if np.any(source_length <= 0):
        raise ValueError("Sampler indices contain an empty replay-buffer range.")
    source_offset = np.clip(steps - sample_start, 0, source_length - 1)
    return buffer_start + source_offset


def _absolute_actions_to_anchor_relative(absolute_action, anchor_eef_pose):
    """Express an absolute target chunk relative to one actual EEF anchor."""

    if not isinstance(absolute_action, torch.Tensor) or not isinstance(
        anchor_eef_pose, torch.Tensor
    ):
        raise TypeError("absolute_action and anchor_eef_pose must be torch tensors.")
    squeeze_batch = absolute_action.ndim == 2
    if squeeze_batch:
        absolute_action = absolute_action.unsqueeze(0)
        anchor_eef_pose = anchor_eef_pose.unsqueeze(0)
    if absolute_action.ndim != 3 or absolute_action.shape[-1] != 14:
        raise ValueError(
            f"Expected absolute action shape [B, H, 14], got {absolute_action.shape}."
        )
    if anchor_eef_pose.shape != (absolute_action.shape[0], 2, 12):
        raise ValueError(
            "Expected anchor EEF pose shape [B, 2, 12], got "
            f"{anchor_eef_pose.shape}."
        )
    if not torch.isfinite(absolute_action).all() or not torch.isfinite(
        anchor_eef_pose
    ).all():
        raise ValueError("Absolute targets and anchor EEF poses must be finite.")

    relative_action = absolute_action.clone()
    for arm_index, offset in enumerate(_ARM_OFFSETS):
        anchor_position = anchor_eef_pose[:, arm_index, :3]
        anchor_rotation = anchor_eef_pose[:, arm_index, 3:].reshape(-1, 3, 3)
        target_rotation = euler_angles_to_matrix(
            absolute_action[:, :, offset + 3 : offset + 6],
            "XYZ",
        )
        relative_rotation = torch.matmul(
            target_rotation,
            anchor_rotation[:, None].transpose(-1, -2),
        )
        relative_action[:, :, offset : offset + 3] = (
            absolute_action[:, :, offset : offset + 3]
            - anchor_position[:, None]
        )
        relative_action[:, :, offset + 3 : offset + 6] = matrix_to_euler_angles(
            relative_rotation,
            "XYZ",
        )

    if squeeze_batch:
        relative_action = relative_action[0]
    return relative_action


class RobotPointCloudDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["point_cloud", "state", "action"],
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "point_cloud": self.replay_buffer["point_cloud"]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        point_cloud = sample["point_cloud"].astype(np.float32)

        data = {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        point_cloud = samples["point_cloud"].to(device, non_blocking=True)
        agent_pos = samples["state"].to(device, non_blocking=True)
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": {
                "point_cloud": point_cloud,
                "agent_pos": agent_pos,
            },
            "action": action,
        }


class RobotMultiViewPointCloudDataset(BaseImageDataset):
    POINT_CLOUD_KEYS = (
        "head_point_cloud",
        "left_wrist_point_cloud",
        "right_wrist_point_cloud",
    )

    def __init__(
        self,
        zarr_path,
        horizon=1,
        obs_steps=None,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        normalize_point_clouds=True,
    ):

        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[*self.POINT_CLOUD_KEYS, "state", "action"],
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.obs_steps = horizon if obs_steps is None else int(obs_steps)
        if not 1 <= self.obs_steps <= self.horizon:
            raise ValueError(
                f"obs_steps must be in [1, {self.horizon}], got {self.obs_steps}."
            )
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.normalize_point_clouds = bool(normalize_point_clouds)

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros(
                (
                    batch_size,
                    self.obs_steps if k != "action" else sequence_length,
                    *v.shape[1:],
                ),
                dtype=v.dtype,
            )
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        if self.normalize_point_clouds:
            for key in self.POINT_CLOUD_KEYS:
                data[key] = self.replay_buffer[key]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if not self.normalize_point_clouds:
            for key in self.POINT_CLOUD_KEYS:
                normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = {
            "agent_pos": sample["state"][:self.obs_steps].astype(np.float32),
        }
        for key in self.POINT_CLOUD_KEYS:
            obs[key] = sample[key][:self.obs_steps].astype(np.float32)

        return {
            "obs": obs,
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            for key in (*self.POINT_CLOUD_KEYS, "state"):
                sample[key] = sample[key][:self.obs_steps]
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                if k == "action":
                    batch_sample_sequence(
                        self.buffers[k],
                        v,
                        self.sampler.indices,
                        idx,
                        self.sampler.sequence_length,
                    )
                else:
                    batch_sample_sequence_prefix(
                        self.buffers[k],
                        v,
                        self.sampler.indices,
                        idx,
                        self.obs_steps,
                    )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        obs = {
            "agent_pos": samples["state"].to(device, non_blocking=True),
        }
        for key in self.POINT_CLOUD_KEYS:
            obs[key] = samples[key].to(device, non_blocking=True)
        return {
            "obs": obs,
            "action": samples["action"].to(device, non_blocking=True),
        }


class RobotMultiViewPointCloudChunkAnchorDataset(RobotMultiViewPointCloudDataset):
    """Train on fixed EEF target chunks expressed from one observation anchor."""

    def __init__(
        self,
        zarr_path,
        horizon=1,
        obs_steps=None,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        normalize_point_clouds=True,
        urdf_path=None,
    ):
        if obs_steps is None:
            raise ValueError("obs_steps is required for chunk-anchor action labels.")
        super().__init__(
            zarr_path=zarr_path,
            horizon=horizon,
            obs_steps=obs_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=val_ratio,
            batch_size=batch_size,
            max_train_episodes=max_train_episodes,
            normalize_point_clouds=normalize_point_clouds,
        )
        self.action_anchor_index = self.obs_steps - 1
        if int(self.pad_before) != self.action_anchor_index:
            raise ValueError(
                "pad_before must equal obs_steps - 1 for chunk-anchor labels, got "
                f"pad_before={self.pad_before} obs_steps={self.obs_steps}."
            )

        actual_eef_pose, absolute_action = _reconstruct_absolute_eef_actions(
            self.replay_buffer["state"],
            self.replay_buffer["action"],
            _DEFAULT_UR5_WSG_URDF if urdf_path is None else urdf_path,
        )
        self._actual_eef_pose = actual_eef_pose
        self.replay_buffer.data["action"] = absolute_action

    def get_validation_dataset(self):
        """Build a validation view without sharing mutable batch output buffers."""

        val_set = super().get_validation_dataset()
        val_set.buffers = {
            key: np.zeros_like(value) for key, value in self.buffers.items()
        }
        val_set.buffers_torch = {
            key: torch.from_numpy(value) for key, value in val_set.buffers.items()
        }
        return val_set

    def _source_indices(self, idx):
        if isinstance(idx, numbers.Integral):
            sampler_indices = self.sampler.indices[int(idx) : int(idx) + 1]
        else:
            sampler_indices = self.sampler.indices[np.asarray(idx, dtype=np.int64)]
        return _sequence_source_indices(sampler_indices, self.horizon)

    def _convert_sample_action(self, action, source_indices):
        anchor_indices = source_indices[:, self.action_anchor_index]
        anchor_eef_pose = torch.from_numpy(self._actual_eef_pose[anchor_indices])
        if action.ndim == 2:
            anchor_eef_pose = anchor_eef_pose[0]
        return _absolute_actions_to_anchor_relative(action, anchor_eef_pose)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        scalar_index = isinstance(idx, numbers.Integral)
        if not scalar_index and not isinstance(idx, np.ndarray):
            return super().__getitem__(idx)
        normalized_idx = int(idx) if scalar_index else idx
        sample = super().__getitem__(normalized_idx)
        relative_action = self._convert_sample_action(
            sample["action"],
            self._source_indices(normalized_idx),
        )
        return {**sample, "action": relative_action}

    def _materialize_anchor_relative_actions(self):
        num_samples = len(self.sampler)
        if num_samples == 0:
            raise ValueError("Cannot fit an action normalizer without training samples.")
        labels = np.empty(
            (num_samples, self.horizon, 14),
            dtype=np.float32,
        )
        materialize_batch_size = 8192
        for start in range(0, num_samples, materialize_batch_size):
            end = min(start + materialize_batch_size, num_samples)
            source_indices = _sequence_source_indices(
                self.sampler.indices[start:end],
                self.horizon,
            )
            absolute_action = torch.from_numpy(
                self.replay_buffer["action"][source_indices]
            )
            anchor_eef_pose = torch.from_numpy(
                self._actual_eef_pose[
                    source_indices[:, self.action_anchor_index]
                ]
            )
            labels[start:end] = _absolute_actions_to_anchor_relative(
                absolute_action,
                anchor_eef_pose,
            ).numpy()
        return labels

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "agent_pos": self.replay_buffer["state"],
        }
        if self.normalize_point_clouds:
            for key in self.POINT_CLOUD_KEYS:
                data[key] = self.replay_buffer[key]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self._materialize_anchor_relative_actions(),
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        if not self.normalize_point_clouds:
            for key in self.POINT_CLOUD_KEYS:
                normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def _batch_sample_sequence_prefix(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    prefix_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        prefix_start_idx = min(sample_start_idx, prefix_length)
        prefix_end_idx = min(sample_end_idx, prefix_length)

        if prefix_start_idx > 0:
            data[i, :prefix_start_idx] = input_arr[buffer_start_idx]
        if prefix_end_idx > prefix_start_idx:
            source_end_idx = buffer_start_idx + prefix_end_idx - prefix_start_idx
            data[i, prefix_start_idx:prefix_end_idx] = input_arr[buffer_start_idx:source_end_idx]
        if prefix_end_idx < prefix_length:
            data[i, prefix_end_idx:] = input_arr[buffer_end_idx - 1]


_batch_sample_sequence_prefix_sequential = numba.jit(
    _batch_sample_sequence_prefix,
    nopython=True,
    parallel=False,
)
_batch_sample_sequence_prefix_parallel = numba.jit(
    _batch_sample_sequence_prefix,
    nopython=True,
    parallel=True,
)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)


def batch_sample_sequence_prefix(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    prefix_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, prefix_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes >= 2**20:
        _batch_sample_sequence_prefix_parallel(data, input_arr, indices, idx, prefix_length)
    else:
        _batch_sample_sequence_prefix_sequential(data, input_arr, indices, idx, prefix_length)
