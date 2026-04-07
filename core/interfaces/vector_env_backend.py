"""Simulator-agnostic vectorized environment backend interface."""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, runtime_checkable

from core.schemas.action import ActionSpec
from core.schemas.observation import Observation
from core.schemas.step_result import StepResult


@runtime_checkable
class VectorEnvBackend(Protocol):
    """Minimal vector-environment contract used by rollout code."""

    num_envs: int

    def reset(self, seeds: Optional[Sequence[int]] = None) -> Sequence[Observation]:
        """Reset all managed environments and return initial observations."""

    def step(self, actions: Sequence[ActionSpec]) -> Sequence[StepResult]:
        """Step all managed environments once."""

    def close(self) -> None:
        """Release simulator resources."""
