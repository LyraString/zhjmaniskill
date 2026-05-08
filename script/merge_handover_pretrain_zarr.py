#!/usr/bin/env python3
"""Merge legacy HandoverBlock demos with manual-intervention ManiSkill demos."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import zarr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.converters import (  # noqa: E402
    extract_maniskill_episode_arrays,
    load_maniskill_npz_episode,
)


DEFAULT_OLD_ZARR = (
    REPO_ROOT / "policy/FlowMatching/data/handover_block-demo_clean-50.zarr"
)
DEFAULT_MANUAL_ROOT = REPO_ROOT / "data/HandoverBlock-v1"
DEFAULT_OUTPUT_ZARR = (
    REPO_ROOT / "policy/FlowMatching/data/handover_block-old50_manual10.zarr"
)


def _manual_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.fullmatch(r"policy_manual_intervention_collect(?:_(\d+))?", path.name)
    if match is None:
        return (1, path.name)
    suffix = match.group(1)
    return (0, int(suffix) if suffix is not None else 0)


def _load_old_zarr(old_zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not old_zarr_path.is_dir():
        raise FileNotFoundError(f"old zarr path does not exist: {old_zarr_path}")

    root = zarr.open(str(old_zarr_path), mode="r")
    point_cloud = np.asarray(root["data/point_cloud"][:], dtype=np.float32)
    state = np.asarray(root["data/state"][:], dtype=np.float32)
    action = np.asarray(root["data/action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)

    if not (len(point_cloud) == len(state) == len(action)):
        raise ValueError(
            "old zarr length mismatch: "
            f"point_cloud={len(point_cloud)} state={len(state)} action={len(action)}"
        )
    if len(episode_ends) == 0 or int(episode_ends[-1]) != len(action):
        raise ValueError(
            "old zarr episode_ends mismatch: "
            f"last={episode_ends[-1] if len(episode_ends) else None} action_len={len(action)}"
        )
    return point_cloud, state, action, episode_ends


def _find_manual_episode_paths(manual_root: Path, pattern: str) -> list[Path]:
    if not manual_root.is_dir():
        raise FileNotFoundError(f"manual root does not exist: {manual_root}")

    episode_paths: list[Path] = []
    for episode_dir in sorted(manual_root.glob(pattern), key=_manual_sort_key):
        episode_path = episode_dir / "data" / "episode0.npz"
        if episode_path.is_file():
            episode_paths.append(episode_path)

    if not episode_paths:
        raise FileNotFoundError(
            f"no manual episode npz found under {manual_root} with pattern {pattern!r}"
        )
    return episode_paths


def _load_manual_episodes(
    episode_paths: list[Path],
    *,
    expected_point_cloud_shape: tuple[int, ...],
    expected_state_shape: tuple[int, ...],
    expected_action_shape: tuple[int, ...],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    point_cloud_arrays: list[np.ndarray] = []
    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []
    lengths: list[int] = []

    for episode_path in episode_paths:
        episode = load_maniskill_npz_episode(episode_path)
        point_cloud, state, action = extract_maniskill_episode_arrays(episode)

        if point_cloud.shape[1:] != expected_point_cloud_shape:
            raise ValueError(
                f"{episode_path} point_cloud shape mismatch: "
                f"{point_cloud.shape[1:]} != {expected_point_cloud_shape}"
            )
        if state.shape[1:] != expected_state_shape:
            raise ValueError(
                f"{episode_path} state shape mismatch: "
                f"{state.shape[1:]} != {expected_state_shape}"
            )
        if action.shape[1:] != expected_action_shape:
            raise ValueError(
                f"{episode_path} action shape mismatch: "
                f"{action.shape[1:]} != {expected_action_shape}"
            )

        point_cloud_arrays.append(point_cloud.astype(np.float32))
        state_arrays.append(state.astype(np.float32))
        action_arrays.append(action.astype(np.float32))
        lengths.append(int(len(action)))

    return point_cloud_arrays, state_arrays, action_arrays, lengths


def _episode_ends_from_lengths(lengths: list[int], *, offset: int = 0) -> np.ndarray:
    if any(length <= 0 for length in lengths):
        raise ValueError(f"episode lengths must be positive: {lengths}")
    return np.asarray(offset + np.cumsum(lengths), dtype=np.int64)


def _write_zarr(
    *,
    output_zarr_path: Path,
    point_cloud: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    episode_ends: np.ndarray,
    overwrite: bool,
    attrs: dict[str, object],
) -> None:
    if output_zarr_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"output zarr already exists: {output_zarr_path} "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_zarr_path)

    zarr_root = zarr.group(str(output_zarr_path))
    zarr_root.attrs.update(attrs)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud,
        chunks=(100, *point_cloud.shape[1:]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state,
        chunks=(100, state.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action,
        chunks=(100, action.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the legacy HandoverBlock zarr and manual-intervention npz demos "
            "into one FlowMatching-compatible zarr."
        )
    )
    parser.add_argument("--old-zarr-path", type=Path, default=DEFAULT_OLD_ZARR)
    parser.add_argument("--manual-root", type=Path, default=DEFAULT_MANUAL_ROOT)
    parser.add_argument(
        "--manual-pattern",
        type=str,
        default="policy_manual_intervention_collect*",
    )
    parser.add_argument("--output-zarr-path", type=Path, default=DEFAULT_OUTPUT_ZARR)
    parser.add_argument(
        "--manual-repeat",
        type=int,
        default=1,
        help="Repeat manual episodes after the old demos. Default keeps a 50+10 merge.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output zarr if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manual_repeat < 1:
        raise ValueError("--manual-repeat must be >= 1")

    old_pc, old_state, old_action, old_episode_ends = _load_old_zarr(args.old_zarr_path)
    manual_episode_paths = _find_manual_episode_paths(args.manual_root, args.manual_pattern)
    manual_pc, manual_state, manual_action, manual_lengths = _load_manual_episodes(
        manual_episode_paths,
        expected_point_cloud_shape=old_pc.shape[1:],
        expected_state_shape=old_state.shape[1:],
        expected_action_shape=old_action.shape[1:],
    )

    repeated_manual_pc = manual_pc * args.manual_repeat
    repeated_manual_state = manual_state * args.manual_repeat
    repeated_manual_action = manual_action * args.manual_repeat
    repeated_manual_lengths = manual_lengths * args.manual_repeat

    point_cloud = np.concatenate([old_pc, *repeated_manual_pc], axis=0).astype(np.float32)
    state = np.concatenate([old_state, *repeated_manual_state], axis=0).astype(np.float32)
    action = np.concatenate([old_action, *repeated_manual_action], axis=0).astype(np.float32)
    manual_episode_ends = _episode_ends_from_lengths(
        repeated_manual_lengths,
        offset=int(old_episode_ends[-1]),
    )
    episode_ends = np.concatenate([old_episode_ends, manual_episode_ends], axis=0)

    _write_zarr(
        output_zarr_path=args.output_zarr_path,
        point_cloud=point_cloud,
        state=state,
        action=action,
        episode_ends=episode_ends,
        overwrite=args.overwrite,
        attrs={
            "old_zarr_path": str(args.old_zarr_path),
            "manual_root": str(args.manual_root),
            "manual_pattern": args.manual_pattern,
            "manual_repeat": int(args.manual_repeat),
            "old_episode_count": int(len(old_episode_ends)),
            "manual_episode_count": int(len(manual_episode_paths)),
            "output_episode_count": int(len(episode_ends)),
        },
    )

    print(f"old: episodes={len(old_episode_ends)} steps={len(old_action)}")
    print(
        "manual: "
        f"episodes={len(manual_episode_paths)} steps={sum(manual_lengths)} "
        f"repeat={args.manual_repeat}"
    )
    print(f"output: episodes={len(episode_ends)} steps={len(action)}")
    print(f"saved: {args.output_zarr_path}")


if __name__ == "__main__":
    main()
