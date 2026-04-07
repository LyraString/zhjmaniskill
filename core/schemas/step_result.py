"""Environment step result schema shared across simulator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from core.schemas.observation import Observation


@dataclass(slots=True)
class StepResult:
    """Result returned by one backend step."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        observation: Observation,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any] | None = None,
    ) -> "StepResult":
        return cls(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )
