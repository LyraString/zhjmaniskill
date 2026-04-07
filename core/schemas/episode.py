"""Episode-level schema definitions shared across simulator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.schemas.action import ActionSpec
from core.schemas.observation import Observation


@dataclass(slots=True)
class EpisodeRecord:
    """Lightweight container for one collected episode or rollout fragment."""

    observations: List[Observation] = field(default_factory=list)
    actions: List[ActionSpec] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
