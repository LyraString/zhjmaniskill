"""Action schema definitions shared across simulator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass(slots=True)
class ActionSpec:
    """Backend-agnostic action container.

    `value` stores the concrete action payload, while `action_type` keeps the
    control mode explicit (e.g. `qpos`, `joint_delta`, `ee_delta`).
    """

    value: Any
    action_type: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        value: Any,
        action_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionSpec":
        return cls(
            value=value,
            action_type=action_type,
            metadata=dict(metadata or {}),
        )
