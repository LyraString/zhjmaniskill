"""Factory interface for constructing simulator backends for a task."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.interfaces.env_backend import EnvBackend
from core.interfaces.vector_env_backend import VectorEnvBackend


@runtime_checkable
class TaskFactory(Protocol):
    """Factory for single-env and vector-env task backends."""

    def create_env(self, task_name: str, **kwargs: Any) -> EnvBackend:
        """Build one task environment backend."""

    def create_vector_env(
        self,
        task_name: str,
        num_envs: int,
        **kwargs: Any,
    ) -> VectorEnvBackend:
        """Build a vectorized task environment backend."""
