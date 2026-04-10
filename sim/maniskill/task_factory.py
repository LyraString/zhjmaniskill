"""Task factory for ManiSkill backends."""

from __future__ import annotations

from typing import Any

from core.interfaces.env_backend import EnvBackend
from core.interfaces.task_factory import TaskFactory
from core.interfaces.vector_env_backend import VectorEnvBackend
from sim.maniskill.runtime_env import ManiSkillEnvBackend
from sim.maniskill.task_profiles import apply_task_profile, resolve_runtime_control_mode
from sim.maniskill.vector_runtime_env import ManiSkillVectorEnvBackend


class ManiSkillTaskFactory(TaskFactory):
    """Factory that creates ManiSkill single-environment backends."""

    def create_env(self, task_name: str, **kwargs: Any) -> EnvBackend:
        env_id, resolved_kwargs, _profile_name = apply_task_profile(task_name, kwargs)
        action_type = resolved_kwargs.pop("action_type", None)
        obs_mode = str(resolved_kwargs.pop("obs_mode", "state"))
        control_mode = resolved_kwargs.pop("control_mode", None)
        render_mode = resolved_kwargs.pop("render_mode", None)
        sim_freq = resolved_kwargs.pop("sim_freq", None)
        control_freq = resolved_kwargs.pop("control_freq", None)
        sim_config = resolved_kwargs.pop("sim_config", None)

        _resolved_action_type, control_mode = resolve_runtime_control_mode(
            action_type=action_type,
            control_mode=control_mode,
        )

        return ManiSkillEnvBackend(
            env_id=env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            sim_freq=int(sim_freq) if sim_freq is not None else None,
            control_freq=int(control_freq) if control_freq is not None else None,
            sim_config=dict(sim_config or {}),
            env_kwargs=dict(resolved_kwargs),
        )

    def create_vector_env(
        self,
        task_name: str,
        num_envs: int,
        **kwargs: Any,
    ) -> VectorEnvBackend:
        env_id, resolved_kwargs, _profile_name = apply_task_profile(task_name, kwargs)
        action_type = resolved_kwargs.pop("action_type", None)
        obs_mode = str(resolved_kwargs.pop("obs_mode", "state"))
        control_mode = resolved_kwargs.pop("control_mode", None)
        render_mode = resolved_kwargs.pop("render_mode", None)
        sim_freq = resolved_kwargs.pop("sim_freq", None)
        control_freq = resolved_kwargs.pop("control_freq", None)
        sim_config = resolved_kwargs.pop("sim_config", None)

        _resolved_action_type, control_mode = resolve_runtime_control_mode(
            action_type=action_type,
            control_mode=control_mode,
        )

        return ManiSkillVectorEnvBackend(
            env_id=env_id,
            num_envs=int(num_envs),
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            sim_freq=int(sim_freq) if sim_freq is not None else None,
            control_freq=int(control_freq) if control_freq is not None else None,
            sim_config=dict(sim_config or {}),
            env_kwargs=dict(resolved_kwargs),
        )
