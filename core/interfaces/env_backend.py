"""Simulator-agnostic single-environment backend interface."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from core.schemas.action import ActionSpec
from core.schemas.observation import Observation
from core.schemas.step_result import StepResult


@runtime_checkable
class EnvBackend(Protocol):
    """Minimal single-environment runtime contract used by learners/runners."""

    def reset(self, *, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the initial observation."""

    def step(self, action: ActionSpec) -> StepResult:
        """Apply one control action and return the resulting transition."""

    def close(self) -> None:
        """Release simulator resources."""
