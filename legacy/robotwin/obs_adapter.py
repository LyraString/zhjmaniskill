"""Observation adapters for the legacy RoboTwin runtime."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.schemas.observation import Observation


def convert_robotwin_obs(
    raw_obs: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Observation:
    """Convert a RoboTwin raw observation into the shared schema.

    The current training stack only depends on point-cloud observations and
    low-dimensional agent state, so this adapter intentionally exposes the
    minimal learner-facing payload.
    """

    point_cloud = np.asarray(raw_obs["pointcloud"])[..., :3].astype(np.float32)
    agent_pos = np.asarray(raw_obs["joint_action"]["vector"]).astype(np.float32)

    return Observation.from_mapping(
        {
            "point_cloud": point_cloud,
            "agent_pos": agent_pos,
        },
        metadata=metadata,
    )
