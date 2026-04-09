"""ManiSkill task profile registry used to keep smoke-task defaults in one place."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


RUNTIME_DEFAULT_KEYS = (
    "action_type",
    "obs_mode",
    "control_mode",
    "render_mode",
    "reward_mode",
    "robot_uids",
    "sim_freq",
    "control_freq",
)


DEFAULT_MANISKILL_RUNTIME: dict[str, Any] = {
    "action_type": "pd_joint_delta_pos",
    "obs_mode": "pointcloud+state_dict",
    "control_mode": None,
    "render_mode": None,
    "reward_mode": "normalized_dense",
    "robot_uids": "panda",
    "sim_freq": 200,
    "control_freq": 20,
}


MANISKILL_TASK_PROFILES: dict[str, dict[str, Any]] = {
    "pickcube_smoke": {
        "env_id": "PickCube-v1",
        **DEFAULT_MANISKILL_RUNTIME,
        "shape_meta": {
            "point_cloud": [512, 3],
            "agent_pos": [9],
            "action": [8],
        },
        "dataset": {
            "zarr_path": "data/PickCube-v1-smoke_npz-1.zarr",
        },
    },
}


_ENV_ID_TO_PROFILE = {
    str(profile["env_id"]): profile_name
    for profile_name, profile in MANISKILL_TASK_PROFILES.items()
}


def get_task_profile(profile_name: str) -> dict[str, Any]:
    if profile_name not in MANISKILL_TASK_PROFILES:
        raise KeyError(f"Unknown ManiSkill task profile: {profile_name}")
    profile = deepcopy(MANISKILL_TASK_PROFILES[profile_name])
    profile["profile_name"] = profile_name
    return profile


def resolve_task_profile(task_name_or_profile: str | None) -> dict[str, Any] | None:
    if not task_name_or_profile:
        return None
    if task_name_or_profile in MANISKILL_TASK_PROFILES:
        return get_task_profile(task_name_or_profile)
    if task_name_or_profile in _ENV_ID_TO_PROFILE:
        return get_task_profile(_ENV_ID_TO_PROFILE[task_name_or_profile])
    return None


def apply_task_profile(
    task_name: str,
    runtime_kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any], str | None]:
    resolved_kwargs = dict(runtime_kwargs)
    profile = resolve_task_profile(task_name)
    resolved_env_id = task_name
    profile_name = None

    for key, value in DEFAULT_MANISKILL_RUNTIME.items():
        if resolved_kwargs.get(key) is None:
            resolved_kwargs[key] = deepcopy(value)

    if profile is None:
        return resolved_env_id, resolved_kwargs, profile_name

    resolved_env_id = str(profile["env_id"])
    profile_name = str(profile["profile_name"])
    for key in RUNTIME_DEFAULT_KEYS:
        if runtime_kwargs.get(key) is None and key in profile:
            resolved_kwargs[key] = deepcopy(profile[key])
    return resolved_env_id, resolved_kwargs, profile_name
