# VR training

Use one entry point for clean PICO HandoverBlockUR5 data, regardless of episode count.

| Setting | File under `diffusion_policy/config/` |
| --- | --- |
| Epochs, checkpoint interval, run name | `maniskill_fm_vr.yaml` |
| Dataset path and validation split | `task/maniskill_handover_ur5_vr_task.yaml` |
| Shared model architecture | `maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch.yaml` |
| Shared training defaults | `maniskill_fm_base.yaml` |

The default dataset is the existing clean80 Zarr: 80 episodes, 48,471 steps after
488 leading idle steps were removed. It uses a 72/8 train/validation episode split
and uniform sampling. Do not regenerate it for another run on the same data.

## Prepare another dataset

Activate `zhjrl` and run from the worktree root. For example, after collecting
100 completed successful episodes in `pico_handover_clean_v2`:

```bash
conda activate zhjrl
cd /home/idiot/zhj/RoboTwin-vr-teleop
python script/convert_vr_teleop_dataset.py \
  data/HandoverBlockUR5-v2/pico_handover_clean_v2 pico-clean-v2 \
  --output-format zarr \
  --output-path policy/FlowMatching/data/HandoverBlockUR5-pico-clean-v2-100.zarr \
  --num-episodes 100 --trim-idle-prefix
```

The converter selects the first N sorted episode directories; it does not filter
on success. Use the successful clean collection, not the debug collection.
It checks action shape, finite values, radial bounds, observation/action alignment,
profile and frequencies. Point clouds are reconstructed from lossless HDF5 camera
positions and extrinsics; this XYZ policy does not decode the RGB videos.
Only the leading idle segment is removed; internal pauses and 20 Hz timing remain.
The converter refuses to overwrite an existing Zarr and renames `.partial` only
after all episodes have been written successfully.

After conversion, change **only** `maniskill.zarr_path` in
`task/maniskill_handover_ur5_vr_task.yaml` to
`data/HandoverBlockUR5-pico-clean-v2-100.zarr`. Do not create another count-specific
YAML. To repeat clean10, select `data/HandoverBlockUR5-pico-clean-v2-10.zarr` instead.

## Inspect and train

Print the merged configuration without starting training:

```bash
cd /home/idiot/zhj/RoboTwin-vr-teleop/policy/FlowMatching
python train.py --config-name maniskill_fm_vr --cfg job --resolve
```

Check `task.dataset.zarr_path`, `training.resume`, epochs, batch size and AMP.
Then launch on GPUs 1 and 3:

```bash
bash train_fm.sh 114514 1,3 --backend maniskill \
  --config-name maniskill_fm_vr -- logging.mode=offline exp_name=vr_fm
```

The shell launcher overrides logging mode and experiment name, so those two
overrides are intentional. Other settings come from YAML. Defaults are 1000
epochs, BF16, checkpoints every 250 epochs, horizon 8 and 4 executed actions.
Checkpoint directories under `checkpoints/` include the date/time and seed;
DDP logs under `data/outputs/` also include a timestamp.

## Configuration cleanup (2026-09-08)

The former clean10/clean80 entry/task pairs were consolidated into the two VR
files above. Training parameters and the current clean80 data selection are
unchanged; config names and output labels changed. Old datasets, checkpoints,
recorded run configurations and environment/controller code were not changed.

The age cutoff is 2026-06-08. Last edit time alone is not last use: the
`native_teacher` preset was last edited in April but has a July 8 run in the main
worktree, so it stays. The July/August multiview/DiT baselines also stay. Generic
DP/FM/eval presets still have script/default dependencies and are retained.

Removed from the FlowMatching config directory:

| File | Last Git edit | Available evidence |
| --- | --- | --- |
| `robot_fpo_14.yaml` | 2026-04-01 | No source/default references or visible recent run; the dedicated FPO package retains its own preset. |
| `task/maniskill_handover_block_task.yaml` | 2026-04-11 | Unreferenced old smoke task, no visible recent run. |

Usage history is incomplete because old outputs were previously cleaned. These
are conservative obsolete-file candidates, not a claim that all executions are
audited. Tracked removed files remain recoverable from Git history. Future
cleanup should inspect saved run configs, static references, dynamically built
shell config names and Hydra inheritance before deleting anything.
