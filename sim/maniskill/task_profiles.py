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
    "point_cloud_sampling",
    "point_cloud_views",
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
    "point_cloud_views": None,
}


UR5_V2_MULTIVIEW_POINT_CLOUD_VIEWS: list[dict[str, Any]] = [
    {"key": "head_point_cloud", "camera": "head_camera", "points": 1024},
    {"key": "left_wrist_point_cloud", "camera": "left_wrist_camera", "points": 512},
    {"key": "right_wrist_point_cloud", "camera": "right_wrist_camera", "points": 512},
]

UR5_REAL_LAYOUT_ARMS_DISTANCE = 0.50
UR5_REAL_LAYOUT_BASE_Z = 0.74 + 0.84
UR5_REAL_LAYOUT_BASE_Y = -0.65
UR5_REAL_LAYOUT_ROBOT_POSE: list[list[float]] = [
    # Quaternion is wxyz. The two base z-axes are mirrored in the separation-z
    # plane with a 45 degree tilt from vertical.
    [0.0, UR5_REAL_LAYOUT_BASE_Y, UR5_REAL_LAYOUT_BASE_Z, 0.9238795325, 0.0, -0.3826834324, 0.0],
    [0.0, UR5_REAL_LAYOUT_BASE_Y, UR5_REAL_LAYOUT_BASE_Z, 0.9238795325, 0.0, 0.3826834324, 0.0],
]
UR5_REAL_LAYOUT_HOMESTATE: list[list[float]] = [
    [0.5893980265, -2.9735164666, -1.7383480072, 0.8828423005, -0.4102165532, 1.3963156032],
    [-0.3893829882, -0.1912880859, 1.7997839451, 1.5604257839, 0.3802834806, -0.8096333503],
]


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
        "point_cloud_sampling": "fast",
        "shape_meta": {
            "point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/handover_block-demo_clean-50.zarr",
        },
    },
    "handover_block_native_legacy_rng_smoke": {
        "env_id": "HandoverBlock-v1",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "continuous",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "shape_meta": {
            "point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/HandoverBlock-v1-smoke_npz-1.zarr",
        },
        "teacher_dataset": {
            "zarr_path": "policy/FlowMatching/data/handover_block-demo_clean-50.zarr",
            "seed_path": "data/handover_block/demo_clean/seed.txt",
        },
        "env_kwargs": {
            "legacy_rng_compat": True,
        },
    },
    "handover_block_native_smoke": {
        "env_id": "HandoverBlock-v1",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "continuous",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "shape_meta": {
            "point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/HandoverBlock-v1-smoke_npz-1.zarr",
        },
        "teacher_dataset": {
            "zarr_path": "policy/FlowMatching/data/handover_block-demo_clean-50.zarr",
            "seed_path": "data/handover_block/demo_clean/seed.txt",
        },
    },
    "handover_block_ur5_v2_smoke": {
        "env_id": "HandoverBlockUR5-v2",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "continuous",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "shape_meta": {
            "point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/HandoverBlockUR5-v2-smoke_npz-1.zarr",
        },
    },
    "handover_block_ur5_v2_multiview_probe": {
        "env_id": "HandoverBlockUR5-v2",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "continuous",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "shape_meta": {
            "point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/HandoverBlockUR5-v2-smoke_npz-1.zarr",
        },
        "env_kwargs": {
            "enable_wrist_cameras": True,
        },
    },
    "handover_block_ur5_v2_real_layout_probe": {
        "env_id": "HandoverBlockUR5-v2",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "continuous",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "point_cloud_views": UR5_V2_MULTIVIEW_POINT_CLOUD_VIEWS,
        "shape_meta": {
            "head_point_cloud": [1024, 3],
            "left_wrist_point_cloud": [512, 3],
            "right_wrist_point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/HandoverBlockUR5-v2-real_layout_probe.zarr",
        },
        "env_kwargs": {
            "arms_distance": UR5_REAL_LAYOUT_ARMS_DISTANCE,
            "robot_pose_override": UR5_REAL_LAYOUT_ROBOT_POSE,
            "homestate_override": UR5_REAL_LAYOUT_HOMESTATE,
            "planner_srdf_override": "./ur5_wsg_real_layout.srdf",
            "enable_wrist_cameras": True,
            "clamp_gripper_drive_targets": True,
            "disable_arm_self_collisions": True,
            "workspace_table_y": -0.45,
            "object_y_offset": -0.45,
            "box_x_offset": -0.38,
            "target_x_offset": 0.225,
            # Keep the first real-layout dataset inside the validated place workspace.
            "target_x_range_override": [0.13, 0.13],
            "target_y_range_override": [0.165, 0.165],
            "expert_grasp_variant": "real_layout_v1",
        },
    },
    "handover_block_ur5_v2_multiview_train": {
        "env_id": "HandoverBlockUR5-v2",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "continuous",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "point_cloud_views": UR5_V2_MULTIVIEW_POINT_CLOUD_VIEWS,
        "shape_meta": {
            "head_point_cloud": [1024, 3],
            "left_wrist_point_cloud": [512, 3],
            "right_wrist_point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": "data/HandoverBlockUR5-v2-handover_block_ur5_v2_multiview_train-1.zarr",
        },
        "env_kwargs": {
            "enable_wrist_cameras": True,
        },
    },
    "handover_block_ur5_v2_multiview_delta_pose_actual": {
        "env_id": "HandoverBlockUR5-v2",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "bimanual_pd_ee_delta_pose",
        "control_mode": "bimanual_pd_ee_delta_pose",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "point_cloud_views": UR5_V2_MULTIVIEW_POINT_CLOUD_VIEWS,
        "shape_meta": {
            "head_point_cloud": [1024, 3],
            "left_wrist_point_cloud": [512, 3],
            "right_wrist_point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "dataset": {
            "zarr_path": (
                "data/HandoverBlockUR5-v2-"
                "handover_block_ur5_v2_multiview_delta_pose_native_actual_100_v1-100.zarr"
            ),
        },
        "env_kwargs": {
            "enable_wrist_cameras": True,
        },
    },
    "handover_block_ur5_v2_vr_robotwin_clean": {
        "env_id": "HandoverBlockUR5-v2",
        **DEFAULT_MANISKILL_RUNTIME,
        "action_type": "bimanual_pd_ee_delta_pose",
        "control_mode": "bimanual_pd_ee_delta_pose",
        "robot_uids": "none",
        "point_cloud_sampling": "fast",
        "point_cloud_views": UR5_V2_MULTIVIEW_POINT_CLOUD_VIEWS,
        "shape_meta": {
            "head_point_cloud": [1024, 3],
            "left_wrist_point_cloud": [512, 3],
            "right_wrist_point_cloud": [512, 3],
            "agent_pos": [14],
            "action": [14],
        },
        "env_kwargs": {
            "enable_wrist_cameras": True,
            "wrist_camera_fovy_deg": 37.0,
            "initial_gripper_value": 1.0,
            "box_x_range_override": [-0.25, -0.05],
            "box_y_range_override": [0.0, 0.25],
            "box_yaw_limit_override": 0.2,
            "target_x_range_override": [0.1, 0.25],
            "target_y_range_override": [0.15, 0.2],
            "block_middle_pose_override": [0.0, 0.1, 0.9, 0.0, 1.0, 0.0, 0.0],
            "handover_reward_center": [-0.023, -0.092, 1.011],
            "handover_reward_plateau_radius": 0.16,
        },
    },
}


_ENV_ID_TO_PROFILE: dict[str, str] = {}
for _profile_name, _profile in MANISKILL_TASK_PROFILES.items():
    _ENV_ID_TO_PROFILE.setdefault(str(_profile["env_id"]), _profile_name)


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
    for key, value in profile.get("env_kwargs", {}).items():
        if runtime_kwargs.get(key) is None:
            resolved_kwargs[key] = deepcopy(value)
    return resolved_env_id, resolved_kwargs, profile_name
