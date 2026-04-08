"""Task factory for ManiSkill backends."""

from __future__ import annotations

from typing import Any

from core.interfaces.env_backend import EnvBackend
from core.interfaces.task_factory import TaskFactory
from core.interfaces.vector_env_backend import VectorEnvBackend
from sim.maniskill.runtime_env import ManiSkillEnvBackend


class ManiSkillTaskFactory(TaskFactory):
    """Factory that creates ManiSkill single-environment backends."""

    def create_env(self, task_name: str, **kwargs: Any) -> EnvBackend:
        action_type = kwargs.pop("action_type", None)
        obs_mode = str(kwargs.pop("obs_mode", "state"))
        control_mode = kwargs.pop("control_mode", None)
        render_mode = kwargs.pop("render_mode", None)

        if control_mode is None and action_type not in (None, "", "continuous"):
            control_mode = str(action_type)

        return ManiSkillEnvBackend(
            env_id=task_name,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            env_kwargs=dict(kwargs),
        )

    def create_vector_env(
        self,
        task_name: str,
        num_envs: int,
        **kwargs: Any,
    ) -> VectorEnvBackend:
        raise NotImplementedError(
            "ManiSkill vector backend is not part of stage 6."
        )
