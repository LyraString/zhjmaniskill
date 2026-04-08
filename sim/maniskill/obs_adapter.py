"""Observation adapters for ManiSkill backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.schemas.observation import Observation


DEFAULT_POINT_CLOUD_SIZE = 512


def _convert_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if isinstance(value, Mapping):
        return {key: _convert_value(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_convert_value(item) for item in value]
    return value


def _to_xyz(pointcloud: Any) -> np.ndarray:
    pointcloud_np = np.asarray(pointcloud)
    if pointcloud_np.ndim >= 1 and pointcloud_np.shape[-1] >= 3:
        return pointcloud_np[..., :3].astype(np.float32, copy=False)
    return pointcloud_np.astype(np.float32, copy=False)


def _squeeze_single_env(value: Any) -> np.ndarray:
    value_np = np.asarray(value)
    if value_np.ndim >= 1 and value_np.shape[0] == 1:
        value_np = value_np[0]
    return value_np


def _resize_point_cloud(point_cloud_xyz: np.ndarray, target_points: int) -> np.ndarray:
    if point_cloud_xyz.shape[0] == 0:
        return np.zeros((target_points, 3), dtype=np.float32)

    if point_cloud_xyz.shape[0] >= target_points:
        idx = np.linspace(0, point_cloud_xyz.shape[0] - 1, target_points, dtype=np.int64)
        return point_cloud_xyz[idx].astype(np.float32, copy=False)

    pad_count = target_points - point_cloud_xyz.shape[0]
    pad = np.repeat(point_cloud_xyz[:1], pad_count, axis=0)
    return np.concatenate([point_cloud_xyz, pad], axis=0).astype(np.float32, copy=False)


def _extract_point_cloud_from_sensor_data(
    payload: Mapping[str, Any],
    target_points: int = DEFAULT_POINT_CLOUD_SIZE,
) -> np.ndarray | None:
    sensor_data = payload.get("sensor_data")
    sensor_param = payload.get("sensor_param")
    if not isinstance(sensor_data, Mapping) or not isinstance(sensor_param, Mapping):
        return None

    point_sets: list[np.ndarray] = []
    for cam_uid, images in sensor_data.items():
        if not isinstance(images, Mapping):
            continue
        if "position" not in images or "segmentation" not in images:
            continue
        cam_param = sensor_param.get(cam_uid)
        if not isinstance(cam_param, Mapping) or "cam2world_gl" not in cam_param:
            continue

        position = np.asarray(images["position"], dtype=np.float32)
        segmentation = np.asarray(images["segmentation"])
        cam2world = np.asarray(cam_param["cam2world_gl"], dtype=np.float32)

        if position.ndim >= 4 and position.shape[0] == 1:
            position = position[0]
        if segmentation.ndim >= 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]
        if cam2world.ndim == 3 and cam2world.shape[0] == 1:
            cam2world = cam2world[0]

        if position.shape[-1] < 3:
            continue

        position_xyz = position[..., :3] / 1000.0
        valid = np.asarray(segmentation != 0).reshape(-1)
        xyzw = np.concatenate(
            [
                position_xyz.reshape(-1, 3),
                valid[:, None].astype(np.float32),
            ],
            axis=-1,
        )
        world_xyzw = xyzw @ cam2world.T
        xyz = world_xyzw[world_xyzw[:, 3] > 0, :3]
        if xyz.shape[0] > 0:
            point_sets.append(xyz.astype(np.float32, copy=False))

    if not point_sets:
        return None

    merged = np.concatenate(point_sets, axis=0)
    return _resize_point_cloud(merged, target_points)


def _extract_point_cloud(payload: Mapping[str, Any], target_points: int = DEFAULT_POINT_CLOUD_SIZE) -> np.ndarray | None:
    pointcloud = payload.get("pointcloud")
    if isinstance(pointcloud, Mapping) and "xyzw" in pointcloud:
        xyzw = _squeeze_single_env(pointcloud["xyzw"])
        xyzw = np.asarray(xyzw, dtype=np.float32).reshape(-1, xyzw.shape[-1])
        if xyzw.shape[-1] >= 4:
            valid = xyzw[:, 3] > 0
            xyz = xyzw[valid, :3]
        else:
            xyz = xyzw[:, :3]
        return _resize_point_cloud(xyz, target_points)
    if pointcloud is not None:
        return _to_xyz(_squeeze_single_env(pointcloud))

    point_cloud = payload.get("point_cloud")
    if point_cloud is not None:
        return _to_xyz(_squeeze_single_env(point_cloud))

    point_cloud = _extract_point_cloud_from_sensor_data(payload, target_points=target_points)
    if point_cloud is not None:
        return point_cloud

    return None


def _extract_agent_pos(payload: Mapping[str, Any]) -> np.ndarray | None:
    agent = payload.get("agent")
    if isinstance(agent, Mapping) and "qpos" in agent:
        qpos = _squeeze_single_env(agent["qpos"])
        return np.asarray(qpos, dtype=np.float32).reshape(-1)

    state = payload.get("state")
    if state is not None:
        state_np = _squeeze_single_env(state)
        return np.asarray(state_np, dtype=np.float32).reshape(-1)

    return None


def convert_maniskill_obs(raw_obs: Any, metadata: dict[str, Any] | None = None) -> Observation:
    """Convert a raw ManiSkill observation into the shared ``Observation`` schema."""

    converted = _convert_value(raw_obs)
    if isinstance(converted, Mapping):
        payload = dict(converted)
        point_cloud = _extract_point_cloud(payload)
        if point_cloud is not None:
            payload["point_cloud"] = point_cloud

        agent_pos = _extract_agent_pos(payload)
        if agent_pos is not None:
            payload["agent_pos"] = agent_pos
    else:
        payload = {"raw_obs": converted}
    return Observation.from_mapping(payload, metadata=metadata)
