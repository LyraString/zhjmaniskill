"""Backend-agnostic single-environment runner."""

from __future__ import annotations

import contextlib
import os
from typing import Any, Dict

from core.interfaces.task_factory import TaskFactory
from core.schemas.action import ActionSpec


@contextlib.contextmanager
def _temporary_cwd(path: str | None):
    if not path:
        yield
        return
    prev_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)


class OnlineEnvRunner:
    """Thin runner wrapper around an ``EnvBackend`` instance.

    The public API intentionally mirrors the existing RoboTwin runner contract:
    ``reset() -> (obs, info)`` and ``step(action) -> (obs, reward, terminated,
    truncated, info)``.
    """

    def __init__(
        self,
        *,
        task_factory: TaskFactory,
        task_name: str,
        action_type: str = "qpos",
        **backend_kwargs: Any,
    ) -> None:
        self.task_factory = task_factory
        self.task_name = task_name
        self.action_type = action_type
        self.backend_kwargs = dict(backend_kwargs)
        bootstrap_cwd = self.backend_kwargs.pop("bootstrap_cwd", None)
        try:
            with _temporary_cwd(bootstrap_cwd):
                self.backend = self.task_factory.create_env(
                    task_name=task_name,
                    action_type=action_type,
                    **self.backend_kwargs,
                )
        except ModuleNotFoundError as exc:
            if exc.name == "mplib.sapien_utils":
                raise RuntimeError(
                    "Legacy RoboTwin runtime requires an mplib build that provides "
                    "`mplib.sapien_utils`. The current Python environment has an "
                    "incompatible mplib package. Run this legacy smoke test in the "
                    "`zhjRoboTwin` environment or install the RoboTwin-compatible "
                    "mplib version into the current environment."
                ) from exc
            raise

    def reset(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        obs = self.backend.reset()
        return dict(obs.data), dict(obs.metadata)

    def step(self, action: Any) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        result = self.backend.step(
            ActionSpec.from_value(action, action_type=self.action_type)
        )
        return (
            dict(result.observation.data),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            dict(result.info),
        )

    def close(self) -> None:
        self.backend.close()
