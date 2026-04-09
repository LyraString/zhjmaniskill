import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.converters import (
    convert_maniskill_npz_dir_to_zarr,
    convert_robotwin_hdf5_dir_to_zarr,
)


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "--backend",
        choices=("robotwin", "maniskill"),
        default="robotwin",
        help="Input trajectory backend. Default keeps legacy RoboTwin behavior.",
    )
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

    load_dir = REPO_ROOT / "data" / task_name / task_config
    save_dir = Path(__file__).resolve().parent / "data" / f"{task_name}-{task_config}-{num}.zarr"

    if args.backend == "maniskill":
        convert_maniskill_npz_dir_to_zarr(
            load_dir=load_dir,
            save_dir=save_dir,
            num_episodes=num,
        )
    else:
        convert_robotwin_hdf5_dir_to_zarr(
            load_dir=load_dir,
            save_dir=save_dir,
            num_episodes=num,
        )


if __name__ == "__main__":
    main()
