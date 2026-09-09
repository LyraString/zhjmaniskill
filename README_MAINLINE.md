<div align="center">

# Multiview Bimanual Flow Matching

**A ManiSkill-first research pipeline for multiview point-cloud-conditioned
bimanual manipulation**

[Method](#method) | [Quick Start](#quick-start) | [Training](#flow-matching-pretraining) |
[Evaluation](#evaluation) | [FPO](#fpo-online-fine-tuning) | [Reproducibility](#reproducibility)

</div>

This branch develops a bimanual manipulation pipeline around the
`HandoverBlockUR5-v2` task. The current experimental path combines three
world-frame point clouds, native actual-relative end-effector delta-pose
control, Flow Matching behavior cloning, and optional FPO online fine-tuning.

> **Research status.** This is an active research branch built on top of
> [RoboTwin 2.0](https://robotwin-platform.github.io/) and
> [ManiSkill 3](https://maniskill.readthedocs.io/). It is not the upstream
> RoboTwin repository. Patch-token Action DiT and real-robot-layout support are
> experimental; fixed-seed benchmark results and downloadable checkpoints are
> not yet released.

## Highlights

- **Native delta-pose control.** The environment executes the same 14-D
  actual-relative action that is recorded in the dataset and predicted at
  evaluation time.
- **Three-view 3D observations.** A head camera and two wrist cameras are
  transformed into a common world frame and sampled independently.
- **Spatial patch tokens.** The experimental encoder retains 12 head-camera
  tokens and 6 tokens from each wrist camera instead of collapsing every view
  immediately to one global vector.
- **Action DiT denoiser.** A condition Transformer fuses visual, temporal, and
  proprioceptive tokens; a 12-block Action DiT predicts the Flow Matching
  velocity field over a bimanual action chunk.
- **End-to-end research workflow.** The repository contains expert collection,
  NPZ-to-Zarr conversion, FM pretraining, rollout diagnostics, and CUDA-vector
  FPO fine-tuning.
- **Baseline preservation.** Existing single-view, shared-PointNet multiview,
  joint-position, and U-Net paths remain available for controlled comparisons.

## Method

### System overview

<p align="center">
  <img src="./action_dit_patch_architecture.png" width="100%" alt="Multiview patch-token Action DiT architecture">
</p>

The timing annotation in the figure is an internal single-step training profile,
not a task success rate. Runtime depends on GPU, precision, compilation mode,
batch size, and thermal state.

### Task and data contract

`HandoverBlockUR5-v2` requires the left UR5 arm to pick up a block, transfer it
to the right arm, and place it on a target. The current multiview delta-pose
dataset uses the following schema:

| Key | Shape per step | Semantics |
| --- | ---: | --- |
| `head_point_cloud` | `[1024, 3]` | Head-camera XYZ transformed to world coordinates |
| `left_wrist_point_cloud` | `[512, 3]` | Left wrist-camera XYZ transformed to world coordinates |
| `right_wrist_point_cloud` | `[512, 3]` | Right wrist-camera XYZ transformed to world coordinates |
| `agent_pos` | `[14]` | Current left/right arm joint and gripper state |
| `action` | `[14]` | Actual-relative bimanual delta pose and absolute gripper commands |

The action is ordered as:

```text
[left_dx, left_dy, left_dz, left_droll, left_dpitch, left_dyaw, left_gripper,
 right_dx, right_dy, right_dz, right_droll, right_dpitch, right_dyaw, right_gripper]
```

For each `env.step()`, translation and Euler XYZ rotation deltas are interpreted
relative to the current measured end-effector pose at the start of that control
step. Translation uses the robot-root-aligned frame, gripper commands are
absolute, and the native controller enforces per-axis limits of `0.015 m` and
`0.10 rad`. The default simulation and control frequencies are 200 Hz and 20 Hz.

### Multiview patch-token encoder

For each observation time step:

1. The three cameras are transformed into the same world coordinate frame.
2. The point clouds remain separate; they are not concatenated at point level.
3. Local neighborhoods are selected around 12/6/6 patch centers for
   head/left-wrist/right-wrist views.
4. A local PointNet maps each patch to a 512-D token.
5. The 14-D proprioceptive state is split into one 7-D token per arm.

This produces 26 condition tokens per observation step. With three observation
steps, the current model receives 78 condition tokens.

### Condition Transformer

The Condition Transformer is an observation encoder, not the action generator.
It contains four self-attention blocks at width 768. These blocks exchange
information across cameras, arms, and observation times while preserving all 78
tokens as a condition memory.

### Action DiT and Flow Matching

The training horizon is eight steps. Each bimanual action step is split into a
left-arm token and a right-arm token, producing 16 noisy action tokens. The
12-block Action DiT uses:

- self-attention among action tokens;
- cross-attention from action queries to the 78 condition-memory tokens;
- Flow Matching time conditioning;
- a final projection back to the 14-D action sequence.

The current patch configuration has approximately 212 million trainable
parameters. During closed-loop evaluation, `n_action_steps=1`: the policy
re-observes the environment after every executed action even though it predicts
an eight-step training horizon.

## Repository layout

| Path | Purpose |
| --- | --- |
| `sim/maniskill/tasks/` | ManiSkill environments and native bimanual controllers |
| `sim/maniskill/expert/` | Scripted expert solver and delta-pose motion primitives |
| `sim/maniskill/obs_adapter.py` | Learner-facing point-cloud and proprioception extraction |
| `sim/maniskill/task_profiles.py` | Centralized observation, control, and task profiles |
| `data/converters/` | Raw episode to Zarr conversion |
| `script/collect_maniskill_data.py` | Expert and intervention data collection |
| `script/eval_maniskill_policy.py` | FM/FPO closed-loop evaluation and video export |
| `script/check_maniskill_dataset.py` | Dataset validation and dynamics replay diagnostics |
| `script/handover/` | Stage-specific task probes and visualization tools |
| `policy/fm_diffusion/` | Shared datasets, encoders, policies, workspaces, and runners |
| `policy/FlowMatching/` | Flow Matching configurations and training entry point |
| `policy/FPO/` | FPO configurations and online fine-tuning entry point |
| `legacy/robotwin/` | Legacy RoboTwin runtime compatibility layer |

## Quick Start

### Requirements

- Linux with an NVIDIA GPU
- CUDA-compatible PyTorch
- Python 3.10 (the validated development environment uses Conda `zhjrl`)
- ManiSkill 3 / SAPIEN and the planner dependencies listed in
  `requirements.txt`

```bash
git clone <repository-url>
cd RoboTwin

conda create -n zhjrl python=3.10 -y
conda activate zhjrl
python -m pip install -r requirements.txt
python -m pip install -e policy/fm_diffusion
```

`requirements.txt` is currently a pinned workstation snapshot. Before an
external release, private editable URLs and machine-local package references
must be removed or replaced with public packages.

### Collect multiview delta-pose demonstrations

```bash
CUDA_VISIBLE_DEVICES=0 python script/collect_maniskill_data.py \
  HandoverBlockUR5-v2 delta_mv_actual_100_v1 100 \
  --task-profile handover_block_ur5_v2_multiview_delta_pose_actual \
  --collector scripted_delta_pose_expert \
  --seed 33550336 \
  --max-attempts 500 \
  --max-episode-steps 900 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --save-only-success
```

Raw episodes are written to:

```text
data/HandoverBlockUR5-v2/delta_mv_actual_100_v1/
```

The delta-pose collector only saves episodes that complete the task, contain
finite in-range actions, and report zero controller failures and zero qpos
fallbacks. Dataset replay is still a dynamics diagnostic rather than a guarantee
of bitwise deterministic contact behavior.

### Convert episodes to Zarr

```bash
python policy/FlowMatching/process_data.py \
  --backend maniskill \
  HandoverBlockUR5-v2 delta_mv_actual_100_v1 100
```

The generated dataset is placed at:

```text
policy/FlowMatching/data/HandoverBlockUR5-v2-delta_mv_actual_100_v1-100.zarr
```

Static-frame filtering is optional and disabled by default. Do not enable
`--filter-static` when reproducing an unfiltered baseline.

### Validate a dataset

```bash
python script/check_maniskill_dataset.py \
  policy/FlowMatching/data/HandoverBlockUR5-v2-delta_mv_actual_100_v1-100.zarr \
  --task-profile handover_block_ur5_v2_multiview_delta_pose_actual \
  --smoke-replay \
  --replay-episodes 3 \
  --replay-steps 900 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --save-video-dir tmp/handover/dataset_replay
```

## Flow Matching pretraining

### Patch-token Action DiT

```bash
cd policy/FlowMatching

bash train_fm.sh 114514 0 \
  --backend maniskill \
  --config-name maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch \
  -- \
  task.dataset.zarr_path=/path/to/dataset.zarr \
  task.maniskill.zarr_path=/path/to/dataset.zarr \
  exp_name=delta_mv_actual_patch_dit_fm \
  training.checkpoint_run_name=delta_mv_actual_patch_dit-114514 \
  training.resume=False \
  training.num_epochs=1000 \
  training.checkpoint_every=100 \
  logging.mode=offline
```

The positional arguments to `train_fm.sh` are the random seed and physical GPU
ID. Checkpoints are saved under:

```text
policy/FlowMatching/checkpoints/<checkpoint_run_name>/<epoch>.ckpt
```

Training logs are written below `policy/FlowMatching/data/outputs/` and can be
inspected with:

```bash
tensorboard --logdir policy/FlowMatching/data/outputs \
  --port 16006 --host 127.0.0.1
```

### Main comparison configurations

| Configuration | Observation encoder | Denoiser | Status |
| --- | --- | --- | --- |
| `maniskill_fm_handover_native_teacher` | Single-view PointNet | 1D U-Net | Preserved baseline |
| `maniskill_fm_handover_ur5_v2_multiview` | Shared PointNet + feature fusion | 1D U-Net | Multiview joint-action baseline |
| `maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual` | Shared PointNet + feature fusion | 1D U-Net | Multiview native delta-pose baseline |
| `maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual_action_dit_large` | Global multiview features | Action DiT | Experimental |
| `maniskill_fm_handover_ur5_v2_multiview_delta_pose_actual_action_dit_patch` | 12/6/6 patch tokens | Action DiT | Current experimental path |

Clean PICO training uses `maniskill_fm_vr` for all dataset sizes. See the
[VR training guide](policy/FlowMatching/VR_TRAINING.md) for conversion, dataset
selection and launch commands; episode counts belong in data paths, not new YAML files.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python script/eval_maniskill_policy.py \
  flowmatching \
  /path/to/checkpoint.ckpt \
  HandoverBlockUR5-v2 \
  --task-profile handover_block_ur5_v2_multiview_delta_pose_actual \
  --episodes 100 \
  --seed 999 \
  --max-episode-steps 1500 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --save-video-dir tmp/handover/eval_patch_dit \
  --video-fps 10
```

Use the same checkpoint, seeds, episode count, time limit, control profile, and
hold-filter setting for model comparisons. In addition to success rate, inspect
stage transitions, truncation rate, final box-to-target error, action magnitude,
and rollout videos.

The evaluator provides optional deployment-side drift diagnostics:

```text
--delta-pose-hold-mode {off,deadband,anchor}
```

The default is `off`. `deadband` and `anchor` change executed actions and must be
reported as separate evaluation conditions; they are not part of the trained
policy.

## FPO online fine-tuning

The CUDA FPO runner supports the shared-PointNet multiview actual-delta pipeline.
Patch-token Action DiT checkpoint compatibility has not yet been validated and
should not be assumed from the FM configuration alone.

```bash
cd policy/FPO

bash train_fpo.sh 114514 0 \
  /path/to/pretrained_fm.ckpt \
  --backend maniskill \
  --config-name maniskill_fpo_handover_ur5_v2_multiview_delta_pose_actual \
  -- \
  training.resume=True \
  training.online_iters=300 \
  training.num_envs=96 \
  training.timesteps_per_batch=73728 \
  training.num_fpo_samples=4 \
  training.max_total_fpo_samples_per_call=384 \
  training.n_updates_per_iteration=1 \
  training.num_minibatches=32 \
  training.max_timesteps_per_episode=1500 \
  training.sim_backend=physx_cuda \
  training.render_backend=sapien_cuda \
  training.point_cloud_sampling=fast \
  logging.mode=offline
```

Track success and stage metrics together with ratio clipping, action magnitude,
KL-related statistics, and gradient norms. A faster rollout is not by itself
evidence of a better policy.

## Reproducibility

For every experiment, retain:

- Git commit and branch name;
- raw episode directory and processed Zarr path;
- task profile and full Hydra overrides;
- action semantics, coordinate frame, units, control frequency, and horizon;
- random seeds and train/validation episode split;
- checkpoint name, EMA usage, precision, and `torch.compile` settings;
- fixed-seed evaluation command, scalar metrics, action traces, and videos;
- GPU model, driver/CUDA versions, and runtime profiling conditions.

Large datasets, checkpoints, videos, and `tmp/` diagnostics are intentionally not
tracked by Git. Never infer an experiment configuration from a checkpoint folder
name alone; the serialized Hydra configuration is the source of truth.

## Current scope and limitations

- The current research target is one bimanual handover task; multi-task claims
  are outside the validated scope.
- The 12/6/6 patch allocation and Action DiT scale are experimental design
  choices, not established optima.
- Dynamics replay can diverge at contact events even when collection succeeded;
  inspect robot and object state traces before attributing failure to action
  semantics.
- The real-layout profile is simulation-side work in progress and must not be
  treated as a validated real-robot controller.
- Eval-side hold filters suppress inactive-arm drift but do not repair gripper
  timing or action errors learned by the policy.
- FPO support for the Patch Action DiT path remains to be validated.

## Citation and acknowledgements

This branch uses RoboTwin task assets and ManiSkill simulation infrastructure.
Please cite the corresponding upstream projects when using this code.

```bibtex
@article{chen2025robotwin,
  title={RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong
         Domain Randomization for Robust Bimanual Robotic Manipulation},
  author={Chen, Tianxing and Chen, Zanxin and Chen, Baijun and Cai, Zijian and
          Liu, Yibin and Li, Zixuan and Liang, Qiwei and Lin, Xianliang and
          Ge, Yiheng and Gu, Zhenyu and others},
  journal={arXiv preprint arXiv:2506.18088},
  year={2025}
}

@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for
         Generalizable Embodied AI},
  author={Tao, Stone and Xiang, Fanbo and Shukla, Arth and Qin, Yuzhe and
          Hinrichsen, Xander and Yuan, Xiaodi and Bao, Chen and Lin, Xinsong and
          Liu, Yulin and others},
  journal={Robotics: Science and Systems},
  year={2025}
}
```

## License

This repository is released under the [MIT License](./LICENSE). Third-party
components and assets retain their original licenses.
