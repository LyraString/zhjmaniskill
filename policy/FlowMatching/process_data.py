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
    parser.add_argument(
        "--filter-static",
        action="store_true",
        help="Optionally compress long near-static segments in ManiSkill episodes before writing zarr.",
    )
    parser.add_argument("--static-action-eps", type=float, default=0.05)
    parser.add_argument("--static-state-eps", type=float, default=0.05)
    parser.add_argument("--static-gripper-eps", type=float, default=1.0e-3)
    parser.add_argument("--static-keep-boundary", type=int, default=2)
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the generated zarr name, e.g. `clean` -> `...-clean.zarr`.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = REPO_ROOT / "data" / task_name / task_config
    output_suffix = str(args.output_suffix).strip()
    if args.filter_static and output_suffix == "":
        output_suffix = "clean"
    output_stem = f"{task_name}-{task_config}-{num}"
    if output_suffix:
        output_stem = f"{output_stem}-{output_suffix}"
    save_dir = Path(__file__).resolve().parent / "data" / f"{output_stem}.zarr"

    if args.backend == "maniskill":
        if args.filter_static:
            print(
                "processing with static filtering: "
                f"action_eps={args.static_action_eps} "
                f"state_eps={args.static_state_eps} "
                f"gripper_eps={args.static_gripper_eps} "
                f"keep_boundary={args.static_keep_boundary}"
            )
        convert_maniskill_npz_dir_to_zarr(
            load_dir=load_dir,
            save_dir=save_dir,
            num_episodes=num,
            filter_static=args.filter_static,
            static_action_eps=args.static_action_eps,
            static_state_eps=args.static_state_eps,
            static_gripper_eps=args.static_gripper_eps,
            static_keep_boundary=args.static_keep_boundary,
        )
    else:
        convert_robotwin_hdf5_dir_to_zarr(
            load_dir=load_dir,
            save_dir=save_dir,
            num_episodes=num,
        )

    print(f"saved zarr to: {save_dir}")


if __name__ == "__main__":
    main()
