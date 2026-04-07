"""Observation schema definitions shared across simulator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(slots=True)
class Observation:
    """Backend-agnostic observation container.

    `data` stores the raw learner-facing observation payload, e.g.
    `{"point_cloud": ..., "agent_pos": ...}` for the current point-cloud stack.
    """

    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> "Observation":
        return cls(
            data=dict(data),
            metadata=dict(metadata or {}),
        )
