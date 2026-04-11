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
    # ``action_type`` is preserved as a RoboTwin-style compatibility alias.
    # Internally we resolve it to ManiSkill's canonical ``control_mode`` when
    # no explicit control_mode is provided.
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
    "handover_block_smoke": {
        "env_id": "HandoverBlock-v1",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "continuous",
        "robot_uids": "none",
        "shape_meta": {
            "point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/handover_block-demo_clean-50.zarr",
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


def resolve_runtime_task_target(
    *,
    task_name: str | None = None,
    task_profile: str | None = None,
    env_id: str | None = None,
) -> tuple[str, str | None]:
    """Resolve the canonical ManiSkill env identifier for runtime creation.

    Priority is intentionally aligned with ManiSkill-native usage:
    explicit ``env_id`` first, then optional profile indirection, and finally
    ``task_name`` as a backward-compatible fallback.
    """

    normalized_task_name = str(task_name) if task_name not in (None, "") else None
    normalized_task_profile = str(task_profile) if task_profile not in (None, "") else None
    normalized_env_id = str(env_id) if env_id not in (None, "") else None

    resolved_profile_name: str | None = None
    profile = None
    if normalized_task_profile is not None:
        profile = resolve_task_profile(normalized_task_profile)
        if profile is None:
            raise KeyError(f"Unknown ManiSkill task profile: {normalized_task_profile}")
        resolved_profile_name = str(profile["profile_name"])
        profile_env_id = str(profile["env_id"])
        if normalized_env_id is not None and normalized_env_id != profile_env_id:
            raise ValueError(
                "Conflicting ManiSkill task identifiers: "
                f"profile `{resolved_profile_name}` resolves to env_id `{profile_env_id}`, "
                f"but explicit env_id `{normalized_env_id}` was also provided."
            )

    if normalized_env_id is not None:
        return normalized_env_id, resolved_profile_name

    if profile is not None:
        return str(profile["env_id"]), resolved_profile_name

    if normalized_task_name is None:
        raise ValueError("At least one of task_name, task_profile, or env_id must be provided.")

    fallback_profile = resolve_task_profile(normalized_task_name)
    if fallback_profile is not None:
        return str(fallback_profile["env_id"]), str(fallback_profile["profile_name"])

    return normalized_task_name, None


def resolve_runtime_control_mode(
    *,
    action_type: str | None = None,
    control_mode: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve external RoboTwin-style action naming into ManiSkill semantics."""

    normalized_action_type = None if action_type in (None, "") else str(action_type)
    normalized_control_mode = None if control_mode in (None, "") else str(control_mode)

    if normalized_control_mode is not None:
        if normalized_action_type in (None, "continuous"):
            normalized_action_type = normalized_control_mode
        return normalized_action_type, normalized_control_mode

    if normalized_action_type in (None, "continuous"):
        return normalized_action_type, None

    return normalized_action_type, normalized_action_type


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
