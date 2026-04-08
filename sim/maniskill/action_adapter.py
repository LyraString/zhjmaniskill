"""Action adapters for ManiSkill backends."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.schemas.action import ActionSpec


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def convert_action_to_maniskill(action: ActionSpec) -> np.ndarray:
    """Convert a shared ``ActionSpec`` into a ManiSkill action array."""

    return _to_numpy(action.value).astype(np.float32, copy=False)
