"""Convert raw VR episodes to RoboTwin DP3's point_cloud/state/next-qpos Zarr."""
import argparse
import json
from pathlib import Path

import h5py
import hydra
import numpy as np
import zarr
from omegaconf import OmegaConf

from train_maniskill import DP3_ROOT
from diffusion_policy_3d.common.maniskill_pointcloud import (
    encode_raw_observations, raw_observation_from_h5,
)


def convert(dataset, output, preprocessing, num_episodes, device, *, resume=False):
    folders = sorted((dataset / "episodes").glob("episode_*"))
    if len(folders) < num_episodes or num_episodes < 1:
        raise ValueError(f"Requested {num_episodes} episodes; found {len(folders)}.")
    folders = folders[:num_episodes]
    partial = output.with_name(output.name + ".partial")
    if output.exists() or (partial.exists() and not resume):
        raise FileExistsError(f"Refusing to overwrite {output} or {partial}.")
    lengths = []
    for folder in folders:
        meta = json.loads((folder / "episode.json").read_text())
        if meta["control_freq_hz"] != 20 or meta["env_id"] != "HandoverBlockUR5-v2":
            raise ValueError(f"Unexpected task or control rate: {folder}")
        with h5py.File(folder / "data.hdf5", "r") as h5:
            n = len(h5["actions/control/absolute_ee_target"])
            q = h5["observations/robot/agent_qpos"][:]
            if q.shape != (n + 1, 14) or not np.isfinite(q).all():
                raise ValueError(f"Invalid observation/action alignment: {folder}")
            if np.any(np.abs(q[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]) > np.pi):
                raise ValueError(f"Joint state exceeds runtime action bounds: {folder}")
            if np.any((q[:, [6, 13]] < 0) | (q[:, [6, 13]] > 1)):
                raise ValueError(f"Gripper state outside [0, 1]: {folder}")
            lengths.append(n)
    metadata = {"action_semantics": "next_observation_joint_position",
                "preprocessing": preprocessing, "source": str(dataset.resolve()),
                "control_freq": 20, "trim_idle_prefix": False}
    total, count = sum(lengths), int(preprocessing["points"])
    if partial.exists():
        root = zarr.open_group(str(partial), mode="a")
        if any(root.attrs.get(key) != value for key, value in metadata.items()):
            raise ValueError("Partial dataset belongs to another conversion configuration.")
        np.testing.assert_array_equal(root["meta/episode_ends"][:], np.cumsum(lengths))
        np.testing.assert_array_equal(root["meta/episode_ids"][:],
                                      [int(f.name.split("_")[-1]) for f in folders])
        data = root["data"]
        completed = int(root.attrs.get("completed_episodes", 0))
    else:
        root = zarr.open_group(str(partial), mode="w")
        root.attrs.update(metadata)
        data, meta = root.create_group("data"), root.create_group("meta")
        data.create_dataset("point_cloud", shape=(total, count, 3), chunks=(64, count, 3), dtype="f4")
        for key in ("state", "action"):
            data.create_dataset(key, shape=(total, 14), chunks=(1024, 14), dtype="f4")
        meta.create_dataset("episode_ends", data=np.cumsum(lengths, dtype=np.int64))
        meta.create_dataset("episode_ids", data=np.asarray([int(f.name.split("_")[-1]) for f in folders]))
        completed = 0
    start = sum(lengths[:completed])
    for episode in range(completed, len(folders)):
        folder, n = folders[episode], lengths[episode]
        with h5py.File(folder / "data.hdf5", "r") as h5:
            q = h5["observations/robot/agent_qpos"][:]
            # Exactly the temporal shift used by DP3/scripts/process_data.py.
            data["state"][start:start+n] = q[:-1]
            data["action"][start:start+n] = q[1:]
            for first in range(0, n, 64):
                last = min(n, first + 64)
                raws = [raw_observation_from_h5(h5, t, preprocessing["cameras"])
                        for t in range(first, last)]
                clouds = encode_raw_observations(raws, preprocessing, device=device)
                data["point_cloud"][start+first:start+last] = clouds["point_cloud"]
        start += n
        root.attrs["completed_episodes"] = episode + 1
        print(f"converted {episode+1}/{len(folders)} episodes, {start}/{total} frames", flush=True)
    partial.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with hydra.initialize_config_dir(config_dir=str(DP3_ROOT / "diffusion_policy_3d/config"), version_base="1.2"):
        cfg = hydra.compose(config_name="robotwin_dp3")
        OmegaConf.resolve(cfg)
    convert(Path(cfg.task.raw_dataset), args.output or Path(cfg.task.dataset.zarr_path),
            OmegaConf.to_container(cfg.task.preprocessing, resolve=True), args.num_episodes, args.device, resume=args.resume)


if __name__ == "__main__":
    main()
