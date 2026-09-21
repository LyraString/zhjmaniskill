#!/usr/bin/env python3
"""Export head-only vs fused camera point clouds near handover stage4.

This script is a standalone diagnostic probe. It does not change the training or
evaluation observation path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import dill
import imageio.v2 as imageio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.schemas.action import ActionSpec
from sim.maniskill import ManiSkillTaskFactory, apply_task_profile
from sim.maniskill.action_adapter import convert_action_to_maniskill
from sim.maniskill.obs_adapter import (
    DEFAULT_POINT_CLOUD_CROP_MAX,
    DEFAULT_POINT_CLOUD_CROP_MIN,
    ENV_POINT_CLOUD_EXCLUDED_SEG_IDS,
    _extract_world_xyz_from_camera,
    _postprocess_point_cloud,
    convert_maniskill_obs,
)
from sim.maniskill.task_profiles import resolve_runtime_control_mode, resolve_runtime_task_target


ADAPTER_MODULES = {
    "flowmatching": "policy.FlowMatching.deploy_maniskill",
    "fpo": "policy.FPO.deploy_maniskill",
}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    return np.asarray(value)


def _to_cpu_tree(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu_tree(item) for item in value)
    return value


def _first_env_xyz(value: Any) -> np.ndarray:
    array = _to_numpy(value).astype(np.float32)
    if array.ndim >= 2 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1)[:3].astype(np.float32)


def _float_info(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    if key not in info:
        return float(default)
    array = _to_numpy(info[key]).reshape(-1)
    if array.size == 0:
        return float(default)
    return float(array[0])


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    else:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if colors.shape[0] != points.shape[0]:
            raise ValueError("PLY colors must match point count.")

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _marker_points(center: np.ndarray, radius: float = 0.012) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32).reshape(3)
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float32,
    )
    return center[None, :] + offsets


def _write_annotated_ply(path: Path, points: np.ndarray, target: np.ndarray, box: np.ndarray, right_ee: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    base_colors = np.full((points.shape[0], 3), 175, dtype=np.uint8)
    target_marker = _marker_points(target)
    box_marker = _marker_points(box)
    right_marker = _marker_points(right_ee)
    marker_points = np.concatenate([target_marker, box_marker, right_marker], axis=0)
    marker_colors = np.concatenate(
        [
            np.tile(np.asarray([[255, 30, 30]], dtype=np.uint8), (target_marker.shape[0], 1)),
            np.tile(np.asarray([[30, 255, 30]], dtype=np.uint8), (box_marker.shape[0], 1)),
            np.tile(np.asarray([[30, 80, 255]], dtype=np.uint8), (right_marker.shape[0], 1)),
        ],
        axis=0,
    )
    _write_ply(
        path,
        np.concatenate([points, marker_points], axis=0),
        np.concatenate([base_colors, marker_colors], axis=0),
    )


def _save_camera_rgb_images(raw_obs: dict[str, Any], prefix: Path) -> list[str]:
    sensor_data = raw_obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        return []

    saved_paths: list[str] = []
    rgb_images: list[np.ndarray] = []
    for camera_name in sorted(sensor_data.keys()):
        camera_data = sensor_data.get(camera_name)
        if not isinstance(camera_data, dict) or "rgb" not in camera_data:
            continue
        rgb = _to_numpy(camera_data["rgb"])
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            continue
        path = prefix.with_name(prefix.name + f"_{camera_name}_rgb.png")
        imageio.imwrite(path, rgb)
        saved_paths.append(str(path))
        rgb_images.append(rgb)

    if rgb_images:
        min_height = min(image.shape[0] for image in rgb_images)
        resized = [image[:min_height] for image in rgb_images]
        contact_sheet = np.concatenate(resized, axis=1)
        path = prefix.with_name(prefix.name + "_rgb_contact_sheet.png")
        imageio.imwrite(path, contact_sheet)
        saved_paths.append(str(path))
    return saved_paths


def _camera_rgb(raw_obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    sensor_data = raw_obs.get("sensor_data")
    if not isinstance(sensor_data, dict):
        return None
    camera_data = sensor_data.get(camera_name)
    if not isinstance(camera_data, dict) or "rgb" not in camera_data:
        return None
    rgb = _to_numpy(camera_data["rgb"])
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        return None
    return rgb


def _squeeze_env_axis(array: Any) -> np.ndarray:
    array = _to_numpy(array)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _camera_pixels_for_filtered_points(
    raw_obs: dict[str, Any],
    camera_name: str,
    *,
    env_id: str,
) -> np.ndarray | None:
    sensor_data = raw_obs.get("sensor_data")
    sensor_param = raw_obs.get("sensor_param")
    if not isinstance(sensor_data, dict) or not isinstance(sensor_param, dict):
        return None
    images = sensor_data.get(camera_name)
    params = sensor_param.get(camera_name)
    if not isinstance(images, dict) or not isinstance(params, dict):
        return None
    if "position" not in images or "cam2world_gl" not in params:
        return None

    position = np.asarray(_squeeze_env_axis(images["position"]), dtype=np.float32)
    cam2world = np.asarray(_squeeze_env_axis(params["cam2world_gl"]), dtype=np.float32)
    segmentation = images.get("segmentation")
    if segmentation is not None:
        segmentation = _squeeze_env_axis(segmentation)

    if position.ndim != 3 or position.shape[-1] < 3:
        return None
    if cam2world.shape != (4, 4):
        return None

    if position.shape[-1] >= 4:
        valid = position[..., 3] < 1.0
    elif segmentation is not None:
        valid = np.asarray(segmentation).reshape(position.shape[:2] + (-1,))[..., 0] != 0
    else:
        valid = np.ones(position.shape[:2], dtype=bool)

    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return None

    camera_xyz = position[ys, xs, :3] / 1000.0
    world_xyz = (camera_xyz @ cam2world[:3, :3].T) + cam2world[:3, 3]
    keep = np.logical_and(
        world_xyz >= DEFAULT_POINT_CLOUD_CROP_MIN,
        world_xyz <= DEFAULT_POINT_CLOUD_CROP_MAX,
    ).all(axis=1)

    excluded_seg_ids = ENV_POINT_CLOUD_EXCLUDED_SEG_IDS.get(env_id)
    if excluded_seg_ids and segmentation is not None:
        seg_values = np.asarray(segmentation).reshape(position.shape[:2] + (-1,))[..., 0][ys, xs]
        keep &= ~np.isin(seg_values, np.asarray(excluded_seg_ids, dtype=seg_values.dtype))

    if not np.any(keep):
        return None
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.int32, copy=False)


def _camera_intrinsic(params: dict[str, Any]) -> np.ndarray | None:
    for key in ("intrinsic_cv", "intrinsic", "camera_intrinsic", "K"):
        if key not in params:
            continue
        intrinsic = np.asarray(_squeeze_env_axis(params[key]), dtype=np.float32)
        if intrinsic.shape == (3, 3):
            return intrinsic
    return None


def _project_world_points_to_camera(
    points: np.ndarray,
    raw_obs: dict[str, Any],
    camera_name: str,
    image_shape: tuple[int, int],
) -> np.ndarray | None:
    sensor_param = raw_obs.get("sensor_param")
    if not isinstance(sensor_param, dict):
        return None
    params = sensor_param.get(camera_name)
    if not isinstance(params, dict) or "cam2world_gl" not in params:
        return None
    intrinsic = _camera_intrinsic(params)
    if intrinsic is None:
        return None

    cam2world = np.asarray(_squeeze_env_axis(params["cam2world_gl"]), dtype=np.float32)
    if cam2world.shape != (4, 4):
        return None
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return None

    world_to_cam = np.linalg.inv(cam2world)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam_gl = (homogeneous @ world_to_cam.T)[:, :3]
    cam_cv = np.stack([cam_gl[:, 0], -cam_gl[:, 1], -cam_gl[:, 2]], axis=1)
    positive_depth = cam_cv[:, 2] > 1e-6
    if not np.any(positive_depth):
        return None

    cam_cv = cam_cv[positive_depth]
    projected = cam_cv @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:3]
    pixels = np.rint(uv).astype(np.int32)
    height, width = image_shape
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    if not np.any(inside):
        return None
    return pixels[inside]


def _sample_overlay_pixels(pixels: np.ndarray, max_points: int = 5000) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 2)
    if pixels.shape[0] <= max_points:
        return pixels
    idx = np.linspace(0, pixels.shape[0] - 1, max_points, dtype=np.int64)
    return pixels[idx]


def _alpha_blend_points(
    image: np.ndarray,
    pixels: np.ndarray | None,
    color: tuple[int, int, int],
    *,
    radius: int = 1,
    alpha: float = 0.65,
) -> np.ndarray:
    if pixels is None:
        return image
    pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 2)
    if pixels.shape[0] == 0:
        return image

    out = image.astype(np.float32, copy=True)
    color_arr = np.asarray(color, dtype=np.float32).reshape(1, 3)
    height, width = out.shape[:2]
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            xs = pixels[:, 0] + dx
            ys = pixels[:, 1] + dy
            keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            if not np.any(keep):
                continue
            out[ys[keep], xs[keep]] = (1.0 - alpha) * out[ys[keep], xs[keep]] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _anchor_overlay_points(raw_obs: dict[str, Any], camera_name: str, anchors: dict[str, np.ndarray], image: np.ndarray) -> dict[str, np.ndarray | None]:
    return {
        name: _project_world_points_to_camera(
            np.asarray(point, dtype=np.float32).reshape(1, 3),
            raw_obs,
            camera_name,
            image.shape[:2],
        )
        for name, point in anchors.items()
    }


def _draw_anchor_markers(
    image: np.ndarray,
    raw_obs: dict[str, Any],
    camera_name: str,
    anchors: dict[str, np.ndarray],
) -> np.ndarray:
    anchor_pixels = _anchor_overlay_points(raw_obs, camera_name, anchors, image)
    image = _alpha_blend_points(image, anchor_pixels["target"], (255, 30, 30), radius=5, alpha=0.95)
    image = _alpha_blend_points(image, anchor_pixels["box"], (30, 255, 30), radius=5, alpha=0.95)
    image = _alpha_blend_points(image, anchor_pixels["right_ee"], (30, 80, 255), radius=5, alpha=0.95)
    return image


def _save_rgb_pointcloud_overlay_images(
    raw_obs: dict[str, Any],
    point_sets: dict[str, np.ndarray],
    variants: dict[str, np.ndarray],
    prefix: Path,
    *,
    target: np.ndarray,
    box: np.ndarray,
    right_ee: np.ndarray,
    env_id: str,
) -> list[str]:
    camera_colors = {
        "head_camera": (255, 190, 30),
        "left_wrist_camera": (20, 220, 255),
        "right_wrist_camera": (255, 60, 220),
    }
    anchors = {"target": target, "box": box, "right_ee": right_ee}
    saved_paths: list[str] = []
    own_overlays: list[np.ndarray] = []

    for camera_name in sorted(point_sets.keys()):
        rgb = _camera_rgb(raw_obs, camera_name)
        if rgb is None:
            continue
        pixels = _camera_pixels_for_filtered_points(raw_obs, camera_name, env_id=env_id)
        pixels = _sample_overlay_pixels(pixels) if pixels is not None else None
        overlay = _alpha_blend_points(
            rgb,
            pixels,
            camera_colors.get(camera_name, (255, 255, 0)),
            radius=1,
            alpha=0.58,
        )
        overlay = _draw_anchor_markers(overlay, raw_obs, camera_name, anchors)
        path = prefix.with_name(prefix.name + f"_{camera_name}_rgb_own_pointcloud_overlay.png")
        imageio.imwrite(path, overlay)
        saved_paths.append(str(path))
        own_overlays.append(overlay)

    if own_overlays:
        min_height = min(image.shape[0] for image in own_overlays)
        contact_sheet = np.concatenate([image[:min_height] for image in own_overlays], axis=1)
        path = prefix.with_name(prefix.name + "_rgb_own_pointcloud_overlay_contact_sheet.png")
        imageio.imwrite(path, contact_sheet)
        saved_paths.append(str(path))

    head_rgb = _camera_rgb(raw_obs, "head_camera")
    fused_points = variants.get("balanced_fused_2048")
    if head_rgb is not None and fused_points is not None:
        fused_pixels = _project_world_points_to_camera(
            fused_points,
            raw_obs,
            "head_camera",
            head_rgb.shape[:2],
        )
        fused_pixels = _sample_overlay_pixels(fused_pixels, max_points=2048) if fused_pixels is not None else None
        fused_overlay = _alpha_blend_points(head_rgb, fused_pixels, (255, 235, 40), radius=1, alpha=0.62)
        fused_overlay = _draw_anchor_markers(fused_overlay, raw_obs, "head_camera", anchors)
        path = prefix.with_name(prefix.name + "_head_camera_rgb_balanced_fused_2048_overlay.png")
        imageio.imwrite(path, fused_overlay)
        saved_paths.append(str(path))

    return saved_paths


def _save_projection_png(
    path: Path,
    variants: dict[str, np.ndarray],
    *,
    target: np.ndarray,
    box: np.ndarray,
    right_ee: np.ndarray,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    anchors = np.stack([target, box, right_ee], axis=0)
    all_variant_xy = [
        np.asarray(points, dtype=np.float32).reshape(-1, 3)[:, :2]
        for points in variants.values()
    ]
    all_xy = np.concatenate([*all_variant_xy, anchors[:, :2]], axis=0)
    xy_min = np.min(all_xy, axis=0) - 0.04
    xy_max = np.max(all_xy, axis=0) + 0.04

    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 5), dpi=150)
    if len(variants) == 1:
        axes = [axes]
    for ax, (title, points) in zip(axes, variants.items()):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=8, cmap="viridis", alpha=0.85)
        ax.scatter([target[0]], [target[1]], c="red", s=80, marker="x", label="target")
        ax.scatter([box[0]], [box[1]], c="lime", s=70, marker="o", label="box")
        ax.scatter([right_ee[0]], [right_ee[1]], c="blue", s=70, marker="^", label="right EE")
        ax.set_title(title.replace("_", " "))
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
        ax.set_xlim(float(xy_min[0]), float(xy_max[0]))
        ax.set_ylim(float(xy_min[1]), float(xy_max[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def _sample_camera_points(points: np.ndarray, target_points: int, sampling: str) -> np.ndarray:
    return _postprocess_point_cloud(points, int(target_points), sampling=sampling)


def _balanced_fused_point_cloud(
    point_sets: dict[str, np.ndarray],
    *,
    total_points: int,
    sampling: str,
) -> np.ndarray:
    required = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    missing = [name for name in required if name not in point_sets]
    if missing:
        return _sample_camera_points(
            np.concatenate([point_sets[name] for name in sorted(point_sets.keys())], axis=0),
            total_points,
            sampling,
        )

    head_count = int(total_points) // 2
    wrist_count = (int(total_points) - head_count) // 2
    right_count = int(total_points) - head_count - wrist_count
    return np.concatenate(
        [
            _sample_camera_points(point_sets["head_camera"], head_count, sampling),
            _sample_camera_points(point_sets["left_wrist_camera"], wrist_count, sampling),
            _sample_camera_points(point_sets["right_wrist_camera"], right_count, sampling),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _build_point_cloud_variants(
    point_sets: dict[str, np.ndarray],
    *,
    sampling: str,
) -> dict[str, np.ndarray]:
    camera_names = sorted(point_sets.keys())
    fused_raw = np.concatenate([point_sets[name] for name in camera_names], axis=0)
    return {
        "head_512": _sample_camera_points(point_sets["head_camera"], 512, sampling),
        "naive_fused_512": _sample_camera_points(fused_raw, 512, sampling),
        "balanced_fused_512": _balanced_fused_point_cloud(point_sets, total_points=512, sampling=sampling),
        "balanced_fused_1024": _balanced_fused_point_cloud(point_sets, total_points=1024, sampling=sampling),
        "balanced_fused_2048": _balanced_fused_point_cloud(point_sets, total_points=2048, sampling=sampling),
    }


def _count_near(points: np.ndarray, center: np.ndarray, radius: float) -> int:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if points.shape[0] == 0:
        return 0
    dist = np.linalg.norm(points - np.asarray(center, dtype=np.float32).reshape(1, 3), axis=1)
    return int(np.sum(dist <= float(radius)))


def _camera_point_sets(raw_obs: dict[str, Any], env_id: str) -> dict[str, np.ndarray]:
    sensor_data = raw_obs.get("sensor_data")
    sensor_param = raw_obs.get("sensor_param")
    if not isinstance(sensor_data, dict) or not isinstance(sensor_param, dict):
        return {}

    point_sets: dict[str, np.ndarray] = {}
    for camera_name, images in sensor_data.items():
        params = sensor_param.get(camera_name)
        if not isinstance(images, dict) or not isinstance(params, dict):
            continue
        xyz = _extract_world_xyz_from_camera(images, params, env_id=env_id)
        if xyz is not None and xyz.shape[0] > 0:
            point_sets[str(camera_name)] = xyz.astype(np.float32, copy=False)
    return point_sets


def _save_probe(
    *,
    output_dir: Path,
    episode_idx: int,
    step_idx: int,
    raw_obs: dict[str, Any],
    env_backend,
    env_id: str,
    sampling: str,
    info: dict[str, Any],
) -> dict[str, Any]:
    point_sets = _camera_point_sets(raw_obs, env_id=env_id)
    camera_names = sorted(point_sets.keys())
    if "head_camera" not in point_sets:
        raise RuntimeError(f"head_camera not found. Available cameras: {camera_names}")

    head_raw = point_sets["head_camera"]
    fused_raw = np.concatenate([point_sets[name] for name in camera_names], axis=0)
    variants = _build_point_cloud_variants(point_sets, sampling=sampling)

    unwrapped = env_backend.env.unwrapped
    target = _first_env_xyz(unwrapped.target_box.pose.p)
    box = _first_env_xyz(unwrapped.box.pose.p)
    right_ee = _first_env_xyz(unwrapped.get_arm_end_effector_link(arm_index=1).pose.p)

    prefix = output_dir / f"episode{episode_idx:03d}_step{step_idx:04d}"
    rgb_paths = _save_camera_rgb_images(raw_obs, prefix)
    overlay_paths = _save_rgb_pointcloud_overlay_images(
        raw_obs,
        point_sets,
        variants,
        prefix,
        target=target,
        box=box,
        right_ee=right_ee,
        env_id=env_id,
    )
    _write_ply(prefix.with_name(prefix.name + "_head_raw.ply"), head_raw)
    _write_ply(prefix.with_name(prefix.name + "_fused_raw.ply"), fused_raw)
    for variant_name, points in variants.items():
        _write_annotated_ply(
            prefix.with_name(prefix.name + f"_{variant_name}_annotated.ply"),
            points,
            target,
            box,
            right_ee,
        )
    projection_saved = _save_projection_png(
        prefix.with_name(prefix.name + "_multiview_xy_compare.png"),
        variants,
        target=target,
        box=box,
        right_ee=right_ee,
    )

    radii = (0.03, 0.05, 0.08)
    variant_summaries = {}
    raw_count_by_variant = {
        "head_512": int(head_raw.shape[0]),
        "naive_fused_512": int(fused_raw.shape[0]),
        "balanced_fused_512": int(fused_raw.shape[0]),
        "balanced_fused_1024": int(fused_raw.shape[0]),
        "balanced_fused_2048": int(fused_raw.shape[0]),
    }
    for variant_name, points in variants.items():
        variant_summary = {
            "raw_point_count": raw_count_by_variant.get(variant_name, int(points.shape[0])),
            "sampled_point_count": int(points.shape[0]),
        }
        for label, center in (("target", target), ("box", box), ("right_ee", right_ee)):
            for radius in radii:
                variant_summary[f"points_near_{label}_{int(radius * 100)}cm"] = _count_near(
                    points,
                    center,
                    radius,
                )
        variant_summaries[variant_name] = variant_summary

    per_camera_summaries = {}
    for camera_name, raw_points in point_sets.items():
        sampled_points = _sample_camera_points(raw_points, 512, sampling)
        camera_summary = {
            "raw_point_count": int(raw_points.shape[0]),
            "sampled_point_count": int(sampled_points.shape[0]),
        }
        for point_prefix, points in (("raw", raw_points), ("sampled", sampled_points)):
            for label, center in (("target", target), ("box", box), ("right_ee", right_ee)):
                for radius in radii:
                    camera_summary[f"{point_prefix}_points_near_{label}_{int(radius * 100)}cm"] = _count_near(
                        points,
                        center,
                        radius,
                    )
        per_camera_summaries[camera_name] = camera_summary

    summary = {
        "episode": int(episode_idx),
        "step": int(step_idx),
        "camera_names": camera_names,
        "head_raw_points": int(head_raw.shape[0]),
        "fused_raw_points": int(fused_raw.shape[0]),
        "reward_stage": _float_info(info, "reward_stage", default=-1.0),
        "box_to_target": _float_info(info, "reward_box_to_target", default=np.nan),
        "target": target.tolist(),
        "box": box.tolist(),
        "right_ee": right_ee.tolist(),
        "rgb_paths": rgb_paths,
        "overlay_paths": overlay_paths,
        "projection_png_saved": bool(projection_saved),
        "per_camera": per_camera_summaries,
        "variants": variant_summaries,
    }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize head-only vs fused point clouds at stage4.")
    parser.add_argument("policy_kind", choices=sorted(ADAPTER_MODULES.keys()))
    parser.add_argument("ckpt_path")
    parser.add_argument("task_name")
    parser.add_argument("--task-profile", default="handover_block_ur5_v2_multiview_probe")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--max-episode-steps", type=int, default=900)
    parser.add_argument("--max-probes", type=int, default=3)
    parser.add_argument("--target-stage", type=int, default=4)
    parser.add_argument("--output-dir", default="/home/idiot/zhj/zhj_maniskill/tmp/handover/multiview_pointcloud_stage4")
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--render-backend", default="sapien_cuda")
    parser.add_argument("--point-cloud-sampling", default="fast")
    parser.add_argument("--action-type", default=None)
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--control-mode", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    adapter = importlib.import_module(ADAPTER_MODULES[args.policy_kind])
    payload = torch.load(open(args.ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    ckpt_cfg = payload["cfg"]
    model = adapter.get_model({"ckpt_path": args.ckpt_path})

    runtime_task_name, resolved_profile_name = resolve_runtime_task_target(
        task_name=args.task_name,
        task_profile=args.task_profile,
        env_id=args.task_name,
    )
    runtime_kwargs = {
        "action_type": args.action_type,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "max_episode_steps": args.max_episode_steps,
        "point_cloud_sampling": args.point_cloud_sampling,
    }
    _action_type, control_mode = resolve_runtime_control_mode(
        action_type=runtime_kwargs["action_type"],
        control_mode=runtime_kwargs["control_mode"],
    )
    runtime_kwargs["control_mode"] = control_mode
    profile_runtime_key = resolved_profile_name or runtime_task_name
    _, resolved_runtime, _profile_name = apply_task_profile(profile_runtime_key, runtime_kwargs)

    env_backend = ManiSkillTaskFactory().create_env(task_name=runtime_task_name, **resolved_runtime)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    try:
        for episode_idx in range(int(args.episodes)):
            raw_obs, _reset_info = env_backend.env.reset(seed=int(args.seed) + episode_idx)
            raw_obs = dict(_to_cpu_tree(raw_obs))
            camera_names = sorted(_camera_point_sets(raw_obs, env_id=env_backend.env_id).keys())
            print(f"episode={episode_idx} camera_names={camera_names}")
            if len(camera_names) <= 1:
                print(
                    "state=single_camera_only "
                    "message='Current environment observation exposes no wrist camera point clouds.'"
                )
                break

            obs = convert_maniskill_obs(
                raw_obs,
                metadata={
                    "backend": "maniskill",
                    "env_id": env_backend.env_id,
                    "obs_mode": env_backend.obs_mode,
                    "control_mode": env_backend.control_mode,
                    "point_cloud_sampling": args.point_cloud_sampling,
                },
            )
            encoded_obs = adapter.encode_obs(dict(obs.data))
            expected_obs_dim = int(ckpt_cfg.task.shape_meta.obs.agent_pos.shape[0])
            expected_action_dim = int(ckpt_cfg.task.shape_meta.action.shape[0])
            if expected_obs_dim != int(np.asarray(encoded_obs["agent_pos"]).shape[-1]):
                raise ValueError("Checkpoint observation shape does not match runtime observation.")
            if expected_action_dim != int(env_backend.action_space.shape[0]):
                raise ValueError("Checkpoint action shape does not match runtime action space.")

            adapter.reset_model(model)
            step_idx = 0
            done = False
            probes_this_episode = 0
            while step_idx < int(args.max_episode_steps) and not done:
                if len(model.obs_cache) == 0:
                    model.update_obs(encoded_obs)
                action_chunk = model.get_action()
                for action in action_chunk:
                    mani_action = convert_action_to_maniskill(
                        ActionSpec.from_value(action, action_type=resolved_runtime["action_type"])
                    )
                    raw_obs, _reward, terminated, truncated, info = env_backend.env.step(mani_action)
                    raw_obs = dict(_to_cpu_tree(raw_obs))
                    info = dict(_to_cpu_tree(info))
                    step_idx += 1
                    stage = int(round(_float_info(dict(info), "reward_stage", default=-1.0)))
                    if stage >= int(args.target_stage):
                        summary = _save_probe(
                            output_dir=output_dir,
                            episode_idx=episode_idx,
                            step_idx=step_idx,
                            raw_obs=raw_obs,
                            env_backend=env_backend,
                            env_id=env_backend.env_id,
                            sampling=args.point_cloud_sampling,
                            info=dict(info),
                        )
                        summaries.append(summary)
                        probes_this_episode += 1
                        variants = summary.get("variants", {})
                        head_target_5cm = variants.get("head_512", {}).get("points_near_target_5cm", 0)
                        naive_target_5cm = variants.get("naive_fused_512", {}).get("points_near_target_5cm", 0)
                        balanced_target_5cm = variants.get("balanced_fused_512", {}).get("points_near_target_5cm", 0)
                        balanced1024_target_5cm = variants.get("balanced_fused_1024", {}).get("points_near_target_5cm", 0)
                        print(
                            "saved_probe "
                            f"episode={episode_idx} step={step_idx} "
                            f"stage={stage} cameras={summary['camera_names']} "
                            f"head512_target_5cm={head_target_5cm} "
                            f"naive512_target_5cm={naive_target_5cm} "
                            f"balanced512_target_5cm={balanced_target_5cm} "
                            f"balanced1024_target_5cm={balanced1024_target_5cm}"
                        )
                        if len(summaries) >= int(args.max_probes):
                            done = True
                            break
                    obs = convert_maniskill_obs(
                        raw_obs,
                        metadata={
                            "backend": "maniskill",
                            "env_id": env_backend.env_id,
                            "obs_mode": env_backend.obs_mode,
                            "control_mode": env_backend.control_mode,
                            "point_cloud_sampling": args.point_cloud_sampling,
                        },
                    )
                    encoded_obs = adapter.encode_obs(dict(obs.data))
                    model.update_obs(encoded_obs)
                    done = bool(terminated or truncated or step_idx >= int(args.max_episode_steps))
                    if done:
                        break
                if probes_this_episode >= int(args.max_probes):
                    break
            if len(summaries) >= int(args.max_probes):
                break
    finally:
        env_backend.close()

    aggregate_path = output_dir / "summary.json"
    aggregate_path.write_text(json.dumps({"probes": summaries}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"output_dir={output_dir}")
    print(f"probe_count={len(summaries)}")


if __name__ == "__main__":
    main()
