"""Observation adapters for ManiSkill backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.schemas.observation import Observation


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


def convert_maniskill_obs(raw_obs: Any, metadata: dict[str, Any] | None = None) -> Observation:
    """Convert a raw ManiSkill observation into the shared ``Observation`` schema."""

    converted = _convert_value(raw_obs)
    if isinstance(converted, Mapping):
        payload = dict(converted)
        if "pointcloud" in payload and "point_cloud" not in payload:
            payload["point_cloud"] = _to_xyz(payload["pointcloud"])
        elif "point_cloud" in payload:
            payload["point_cloud"] = _to_xyz(payload["point_cloud"])
    else:
        payload = {"raw_obs": converted}
    return Observation.from_mapping(payload, metadata=metadata)
