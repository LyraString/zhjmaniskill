"""Observation adapters for ManiSkill backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.schemas.observation import Observation


DEFAULT_POINT_CLOUD_SIZE = 512
DEFAULT_POINT_CLOUD_PRIMARY_CAMERA = "head_camera"
DEFAULT_POINT_CLOUD_CROP_MIN = np.asarray([-0.6, -0.35, 0.7401], dtype=np.float32)
DEFAULT_POINT_CLOUD_CROP_MAX = np.asarray([0.6, 0.35, 2.0], dtype=np.float32)
DEFAULT_POINT_CLOUD_SAMPLING = "fps"
DEFAULT_MULTIVIEW_POINT_CLOUD_VIEWS = (
    {"key": "head_point_cloud", "camera": "head_camera", "points": 1024},
    {"key": "left_wrist_point_cloud", "camera": "left_wrist_camera", "points": 512},
    {"key": "right_wrist_point_cloud", "camera": "right_wrist_camera", "points": 512},
)
ENV_POINT_CLOUD_EXCLUDED_SEG_IDS: dict[str, tuple[int, ...]] = {
    "HandoverBlock-v1": (3,),
}


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


def _crop_point_cloud(point_cloud_xyz: np.ndarray) -> np.ndarray:
    if point_cloud_xyz.shape[0] == 0:
        return point_cloud_xyz.astype(np.float32, copy=False)

    inside_bounds = np.logical_and(
        point_cloud_xyz >= DEFAULT_POINT_CLOUD_CROP_MIN,
        point_cloud_xyz <= DEFAULT_POINT_CLOUD_CROP_MAX,
    ).all(axis=1)
    return point_cloud_xyz[inside_bounds].astype(np.float32, copy=False)


def _farthest_point_sample(point_cloud_xyz: np.ndarray, target_points: int) -> np.ndarray:
    num_points = point_cloud_xyz.shape[0]
    if num_points <= target_points:
        return point_cloud_xyz.astype(np.float32, copy=False)

    sampled = np.empty((target_points, 3), dtype=np.float32)
    sampled[0] = point_cloud_xyz[0]
    min_dist_sq = np.sum((point_cloud_xyz - sampled[0]) ** 2, axis=1)
    farthest_idx = int(np.argmax(min_dist_sq))

    for sample_idx in range(1, target_points):
        sampled[sample_idx] = point_cloud_xyz[farthest_idx]
        dist_sq = np.sum((point_cloud_xyz - sampled[sample_idx]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        farthest_idx = int(np.argmax(min_dist_sq))

    return sampled


def _postprocess_point_cloud(
    point_cloud_xyz: np.ndarray,
    target_points: int,
    *,
    sampling: str = DEFAULT_POINT_CLOUD_SAMPLING,
) -> np.ndarray:
    point_cloud_xyz = np.asarray(point_cloud_xyz, dtype=np.float32).reshape(-1, 3)
    cropped = _crop_point_cloud(point_cloud_xyz)
    if cropped.shape[0] == 0:
        return np.zeros((target_points, 3), dtype=np.float32)

    sampling = str(sampling).lower()
    if sampling in {"fast", "resize", "linspace"}:
        return _resize_point_cloud(cropped, target_points)

    sampled = _farthest_point_sample(cropped, target_points)
    return _resize_point_cloud(sampled, target_points)


def _filter_segmentation_ids(
    xyz: np.ndarray,
    segmentation: np.ndarray | None,
    *,
    env_id: str | None,
) -> np.ndarray:
    if segmentation is None or env_id is None:
        return xyz

    excluded_seg_ids = ENV_POINT_CLOUD_EXCLUDED_SEG_IDS.get(env_id)
    if not excluded_seg_ids:
        return xyz

    seg_flat = np.asarray(segmentation).reshape(-1)
    keep = ~np.isin(seg_flat, np.asarray(excluded_seg_ids, dtype=seg_flat.dtype))
    if not np.any(keep):
        return xyz
    return xyz[keep]


def _extract_world_xyz_from_camera(
    images: Mapping[str, Any],
    cam_param: Mapping[str, Any],
    *,
    env_id: str | None,
) -> np.ndarray | None:
    if "position" not in images:
        return None
    if "cam2world_gl" not in cam_param:
        return None

    position = np.asarray(images["position"], dtype=np.float32)
    cam2world = np.asarray(cam_param["cam2world_gl"], dtype=np.float32)
    segmentation = images.get("segmentation")

    if position.ndim >= 4 and position.shape[0] == 1:
        position = position[0]
    if cam2world.ndim == 3 and cam2world.shape[0] == 1:
        cam2world = cam2world[0]
    if segmentation is not None:
        segmentation = np.asarray(segmentation)
        if segmentation.ndim >= 4 and segmentation.shape[0] == 1:
            segmentation = segmentation[0]

    if position.shape[-1] < 3:
        return None

    position_xyz = position[..., :3] / 1000.0
    if position.shape[-1] >= 4:
        valid = position[..., 3] < 1.0
    elif segmentation is not None:
        valid = segmentation != 0
    else:
        valid = np.ones(position_xyz.shape[:-1], dtype=bool)

    points_xyzw = np.concatenate(
        [
            position_xyz.reshape(-1, 3),
            valid.reshape(-1, 1).astype(np.float32),
        ],
        axis=-1,
    )
    world_xyzw = points_xyzw @ cam2world.T
    xyz = world_xyzw[world_xyzw[:, 3] > 0, :3]
    if xyz.shape[0] == 0:
        return None
    xyz = _filter_segmentation_ids(
        xyz,
        segmentation.reshape(-1) if segmentation is not None else None,
        env_id=env_id,
    )
    if xyz.shape[0] == 0:
        return None
    return xyz.astype(np.float32, copy=False)


def _extract_point_cloud_from_sensor_data(
    payload: Mapping[str, Any],
    target_points: int = DEFAULT_POINT_CLOUD_SIZE,
    *,
    env_id: str | None = None,
    sampling: str = DEFAULT_POINT_CLOUD_SAMPLING,
) -> np.ndarray | None:
    sensor_data = payload.get("sensor_data")
    sensor_param = payload.get("sensor_param")
    if not isinstance(sensor_data, Mapping) or not isinstance(sensor_param, Mapping):
        return None

    head_images = sensor_data.get(DEFAULT_POINT_CLOUD_PRIMARY_CAMERA)
    head_param = sensor_param.get(DEFAULT_POINT_CLOUD_PRIMARY_CAMERA)
    if isinstance(head_images, Mapping) and isinstance(head_param, Mapping):
        head_xyz = _extract_world_xyz_from_camera(head_images, head_param, env_id=env_id)
        if head_xyz is not None:
            return _postprocess_point_cloud(head_xyz, target_points, sampling=sampling)

    point_sets: list[np.ndarray] = []
    for cam_uid, images in sensor_data.items():
        if not isinstance(images, Mapping):
            continue
        cam_param = sensor_param.get(cam_uid)
        if not isinstance(cam_param, Mapping):
            continue
        xyz = _extract_world_xyz_from_camera(images, cam_param, env_id=env_id)
        if xyz is not None:
            point_sets.append(xyz)

    if not point_sets:
        return None

    merged = np.concatenate(point_sets, axis=0)
    return _postprocess_point_cloud(merged, target_points, sampling=sampling)


def _normalize_point_cloud_views(views: Any) -> list[dict[str, Any]]:
    if views in (None, "", False):
        return []
    if views is True:
        return [dict(view) for view in DEFAULT_MULTIVIEW_POINT_CLOUD_VIEWS]
    if not isinstance(views, Sequence) or isinstance(views, (str, bytes, bytearray)):
        raise TypeError("point_cloud_views must be a sequence of view specs.")

    normalized = []
    for view in views:
        if not isinstance(view, Mapping):
            raise TypeError(f"Invalid point cloud view spec: {view!r}")
        key = str(view.get("key", "")).strip()
        camera = str(view.get("camera", "")).strip()
        points = int(view.get("points", DEFAULT_POINT_CLOUD_SIZE))
        if not key or not camera:
            raise ValueError(f"View spec requires non-empty `key` and `camera`: {view!r}")
        if points <= 0:
            raise ValueError(f"View spec points must be positive: {view!r}")
        normalized.append({"key": key, "camera": camera, "points": points})
    return normalized


def _extract_multiview_point_clouds_from_sensor_data(
    payload: Mapping[str, Any],
    views: Any,
    *,
    env_id: str | None = None,
    sampling: str = DEFAULT_POINT_CLOUD_SAMPLING,
) -> dict[str, np.ndarray]:
    view_specs = _normalize_point_cloud_views(views)
    if len(view_specs) == 0:
        return {}

    sensor_data = payload.get("sensor_data")
    sensor_param = payload.get("sensor_param")
    if not isinstance(sensor_data, Mapping) or not isinstance(sensor_param, Mapping):
        return {}

    result: dict[str, np.ndarray] = {}
    for view in view_specs:
        camera_name = str(view["camera"])
        images = sensor_data.get(camera_name)
        cam_param = sensor_param.get(camera_name)
        if not isinstance(images, Mapping) or not isinstance(cam_param, Mapping):
            continue
        xyz = _extract_world_xyz_from_camera(images, cam_param, env_id=env_id)
        if xyz is None:
            continue
        result[str(view["key"])] = _postprocess_point_cloud(
            xyz,
            int(view["points"]),
            sampling=sampling,
        )
    return result


def _extract_point_cloud(
    payload: Mapping[str, Any],
    target_points: int = DEFAULT_POINT_CLOUD_SIZE,
    *,
    env_id: str | None = None,
    sampling: str = DEFAULT_POINT_CLOUD_SAMPLING,
) -> np.ndarray | None:
    point_cloud = _extract_point_cloud_from_sensor_data(
        payload,
        target_points=target_points,
        env_id=env_id,
        sampling=sampling,
    )
    if point_cloud is not None:
        return point_cloud

    pointcloud = payload.get("pointcloud")
    if isinstance(pointcloud, Mapping) and "xyzw" in pointcloud:
        xyzw = _squeeze_single_env(pointcloud["xyzw"])
        xyzw = np.asarray(xyzw, dtype=np.float32).reshape(-1, xyzw.shape[-1])
        if xyzw.shape[-1] >= 4:
            valid = xyzw[:, 3] > 0
            xyz = xyzw[valid, :3]
        else:
            xyz = xyzw[:, :3]
        return _postprocess_point_cloud(xyz, target_points, sampling=sampling)
    if pointcloud is not None:
        return _postprocess_point_cloud(
            _to_xyz(_squeeze_single_env(pointcloud)),
            target_points,
            sampling=sampling,
        )

    point_cloud = payload.get("point_cloud")
    if point_cloud is not None:
        return _postprocess_point_cloud(
            _to_xyz(_squeeze_single_env(point_cloud)),
            target_points,
            sampling=sampling,
        )

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
        env_id = None if metadata is None else metadata.get("env_id")
        sampling = (
            DEFAULT_POINT_CLOUD_SAMPLING
            if metadata is None
            else metadata.get("point_cloud_sampling", DEFAULT_POINT_CLOUD_SAMPLING)
        )
        point_cloud_views = None if metadata is None else metadata.get("point_cloud_views")
        multiview_point_clouds = _extract_multiview_point_clouds_from_sensor_data(
            payload,
            point_cloud_views,
            env_id=env_id,
            sampling=sampling,
        )
        if multiview_point_clouds:
            payload.update(multiview_point_clouds)
        else:
            point_cloud = _extract_point_cloud(payload, env_id=env_id, sampling=sampling)
            if point_cloud is not None:
                payload["point_cloud"] = point_cloud

        agent_pos = _extract_agent_pos(payload)
        if agent_pos is not None:
            payload["agent_pos"] = agent_pos
    else:
        payload = {"raw_obs": converted}
    return Observation.from_mapping(payload, metadata=metadata)
