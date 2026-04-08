"""Action adapters for the legacy RoboTwin runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from core.schemas.action import ActionSpec


def convert_action_to_robotwin(action: ActionSpec) -> Any:
    """Convert the shared action schema into a legacy RoboTwin action payload."""

    return np.asarray(action.value)
