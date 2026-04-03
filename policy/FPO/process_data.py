import os
import numpy as np
import zarr
import shutil
import argparse
import h5py


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at path: {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        vector = root["/joint_action/vector"][()]
        pointcloud = root["/pointcloud"][()]

    return vector, pointcloud

def to_xyz(pointcloud):
    if pointcloud.ndim == 2 and pointcloud.shape[1] == 0:
        raise ValueError("Raw HDF5 contains empty /pointcloud with shape (T, 0). ")

    pointcloud = np.asarray(pointcloud)

    if pointcloud.ndim != 3:
        raise ValueError(f"Expected pointcloud ndim=3, got shape={pointcloud.shape}")
    
    if pointcloud.shape[-1] < 3:
        raise ValueError(f"Expected last dim >= 3, got shape={pointcloud.shape}")
    
    return pointcloud[..., :3].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = "../../data/" + str(task_name) + "/" + str(task_config)

    save_dir = f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    point_cloud_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    total_count = 0

    for ep_idx in range(num):
        print(f"processing episode: {ep_idx + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{ep_idx}.hdf5")
        vector_all, pointcloud_all = load_hdf5(load_path)

        pointcloud_all = to_xyz(pointcloud_all)

        if len(vector_all) != len(pointcloud_all):
            raise ValueError(
                f"Length mismatch in {load_path}:"
                f"len(vector) = {len(vector_all)}, len(pointcloud)={len(pointcloud_all)}"
            )
        
        T = len(vector_all)
        if T < 2:
            raise ValueError(f"Episode too short in {load_path}, T={T}")
        
        for t in range(T - 1):
            state_arrays.append(vector_all[t].astype(np.float32))
            point_cloud_arrays.append(pointcloud_all[t])
            action_arrays.append(vector_all[t + 1].astype(np.float32))

        total_count += T - 1
        episode_ends_arrays.append(total_count)

    print()

    point_cloud_arrays = np.asarray(point_cloud_arrays, dtype=np.float32)
    state_arrays = np.asarray(state_arrays, dtype=np.float32)
    action_arrays = np.asarray(action_arrays, dtype=np.float32)
    episode_ends_arrays = np.asarray(episode_ends_arrays, dtype=np.int64)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, *point_cloud_arrays.shape[1:])
    action_chunk_size = (100, action_arrays.shape[1])

    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "point_cloud",
        data=point_cloud_arrays,
        chunks=point_cloud_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


if __name__ == "__main__":
    main()
