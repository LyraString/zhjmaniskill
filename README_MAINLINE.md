# RoboTwin ManiSkill Mainline

This document describes the ManiSkill-based mainline in this repository. The
upstream RoboTwin documentation is still available in [README.md](README.md),
but the workflow below is the recommended entry point for the current branch.

The current research target is `HandoverBlock-v1`: a dual-arm handover and
placement task used to collect demonstrations and train policies with
Flow Matching / FPO.

## Current Status

This branch contains three overlapping workflows:

- Upstream RoboTwin code and assets.
- A ManiSkill runtime for RoboTwin-style bimanual tasks.
- Policy training and evaluation through Flow Matching and FPO.

For `HandoverBlock-v1`, the most reliable data path at the moment is:

```text
policy checkpoint -> policy_manual_intervention -> accepted .npz demos -> zarr -> FM/FPO training
```

`scripted_expert` remains useful for debugging solver logic, but it is not the
recommended source of final training demonstrations because its success rate is
not stable enough across seeds.

## Repository Layout

Important files and directories:

- [mainline.sh](mainline.sh): top-level wrapper for process/train/eval commands.
- [script/collect_maniskill_data.py](script/collect_maniskill_data.py): main ManiSkill data collection CLI.
- [script/eval_maniskill_policy.py](script/eval_maniskill_policy.py): direct ManiSkill policy evaluation CLI.
- [script/check_maniskill_dataset.py](script/check_maniskill_dataset.py): zarr schema and replay smoke checks.
- [sim/maniskill/](sim/maniskill): ManiSkill runtime, task profiles, action/observation adapters, recorders.
- [sim/maniskill/tasks/](sim/maniskill/tasks): custom ManiSkill task implementations.
- [sim/maniskill/expert/](sim/maniskill/expert): scripted expert primitives and handover solver.
- [sim/maniskill/policy_manual_intervention_collector.py](sim/maniskill/policy_manual_intervention_collector.py): policy rollout with keyboard/manual correction.
- [policy/FlowMatching/](policy/FlowMatching): Flow Matching data processing, training, deployment, evaluation.
- [policy/FPO/](policy/FPO): FPO data processing, training, deployment, evaluation.
- [data/](data): raw collected episodes.
- [policy/FlowMatching/data/](policy/FlowMatching/data): Flow Matching zarr datasets.
- [policy/FPO/data/](policy/FPO/data): FPO zarr datasets.
- [eval_videos/](eval_videos): evaluation videos when enabled.

## Environment

The examples below assume:

- Repository root: `/home/idiot/zhj/RoboTwin`
- Python environment: `/home/idiot/anaconda3/envs/zhjrl/bin/python3.10`
- Main task profile: `handover_block_native_smoke`
- Main env id: `HandoverBlock-v1`

Install the upstream dependencies and assets first by following
[README.md](README.md). Additional Python requirements used by scripts are in
[script/requirements.txt](script/requirements.txt).

For GPU simulation/rendering, keep the simulation backend and render backend on
the same CUDA device:

```bash
--sim-backend physx_cuda:3 --render-backend sapien_cuda:3
```

Do not mix devices such as `physx_cuda:1` with `sapien_cuda:3`; SAPIEN/ManiSkill
will fail because physics and render buffers live on different GPUs.

## Quick Start

### 1. Collect One Manual-Intervention Demo

Use a trained Flow Matching checkpoint as the base policy, then intervene only
when the policy needs correction.

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/collect_maniskill_data.py \
  HandoverBlock-v1 policy_manual_intervention_collect_002 1 \
  --task-profile handover_block_native_smoke \
  --collector policy_manual_intervention \
  --policy-kind flowmatching \
  --ckpt-path /home/idiot/zhj/RoboTwin/policy/FlowMatching/checkpoints/HandoverBlock-v1-legacy_replay_native_dp3-50-clean-114514/2000.ckpt \
  --seed 44 \
  --max-episode-steps 1000 \
  --render-mode human \
  --sim-backend physx_cuda:3 \
  --render-backend sapien_cuda:3 \
  --save-video-dir /home/idiot/zhj/RoboTwin/tmp/collect_videos_card3/policy_manual_intervention_collect_002 \
  --video-fps 10 \
  --debug-policy-manual-intervention
```

The raw episode is saved to:

```text
data/HandoverBlock-v1/policy_manual_intervention_collect_002/data/episode0.npz
```

The saved `.npz` contains training arrays only:

- `point_cloud`
- `agent_pos`
- `actions`

The success flag and debug information are printed in the terminal log, so keep
the terminal log together with the video when deciding whether to accept a demo.

### 2. Process Accepted Demos to Zarr

After collecting and accepting a set of demos under one `task_config`, convert
them to Flow Matching zarr:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 policy/FlowMatching/process_data.py \
  --backend maniskill \
  HandoverBlock-v1 policy_manual_intervention_clean 10
```

Output:

```text
policy/FlowMatching/data/HandoverBlock-v1-policy_manual_intervention_clean-10.zarr
```

To generate an FPO-compatible zarr:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 policy/FPO/process_data.py \
  --backend maniskill \
  HandoverBlock-v1 policy_manual_intervention_clean 10
```

Optional static filtering:

```bash
/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 policy/FlowMatching/process_data.py \
  --backend maniskill \
  --filter-static \
  --static-action-eps 0.05 \
  --static-state-eps 0.05 \
  --static-gripper-eps 1e-3 \
  --static-keep-boundary 2 \
  HandoverBlock-v1 policy_manual_intervention_clean 10
```

This writes:

```text
policy/FlowMatching/data/HandoverBlock-v1-policy_manual_intervention_clean-10-clean.zarr
```

### 3. Train Flow Matching

```bash
cd /home/idiot/zhj/RoboTwin

bash mainline.sh train-fm 42 3 \
  --backend maniskill \
  --config-name maniskill_fm_handover_native \
  -- task.maniskill.zarr_path=/home/idiot/zhj/RoboTwin/policy/FlowMatching/data/HandoverBlock-v1-policy_manual_intervention_clean-10.zarr \
     logging.mode=offline
```

The `maniskill_fm_handover_native` config currently defaults to long training
(`training.num_epochs: 3000`). Override it only when doing a quick smoke test.

### 4. Evaluate Flow Matching

```bash
cd /home/idiot/zhj/RoboTwin

bash mainline.sh eval-fm HandoverBlock-v1 3 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --ckpt-path /ABS/PATH/TO/FM.ckpt \
  --episodes 20 \
  --max-episode-steps 1000 \
  -- --sim-backend physx_cuda:3 --render-backend sapien_cuda:3
```

You can also call the direct evaluator:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/eval_maniskill_policy.py \
  flowmatching /ABS/PATH/TO/FM.ckpt HandoverBlock-v1 \
  --task-profile handover_block_native_smoke \
  --episodes 20 \
  --max-episode-steps 1000 \
  --sim-backend physx_cuda:3 \
  --render-backend sapien_cuda:3 \
  --save-video-dir /home/idiot/zhj/RoboTwin/eval_videos/flowmatching/manual_clean_eval \
  --video-fps 10
```

## Data Collection Modes

[script/collect_maniskill_data.py](script/collect_maniskill_data.py) supports:

| Collector | Status | Use case |
| --- | --- | --- |
| `random` | Working | Pipeline smoke tests only. |
| `legacy_replay` | Diagnostic | Replay old zarr action sequences in the current ManiSkill env. Useful for compatibility checks, not recommended as final demo source when replay fails. |
| `scripted_expert` | Experimental | Debug solver and task logic. Success is seed-sensitive for `HandoverBlock-v1`. |
| `policy_intervention` | Experimental | Policy rollout with automatic intervention hooks. |
| `policy_manual_intervention` | Recommended for current demos | Policy rollout with user correction and saved successful trajectories. |

### Manual-Intervention Controls

During `policy_manual_intervention` collection:

- `m`: enter manual mode from policy rollout.
- `1` / `2`: select left / right arm.
- Arrow keys or `w/a/s/d`: planar motion.
- `u/j` or `r/f`: up / down.
- `g`: auto-align selected gripper to the current grasp target.
- `o` / `c`: open / close selected gripper.
- `p`: print status.
- `n`: return control to the policy.
- `q`: abort current episode.

Practical notes:

- Use unique `task_config` names for separate one-episode collection runs.
- Re-running `episode_num=1` with the same `task_config` overwrites `episode0.npz`.
- If you want one clean dataset, copy accepted demos into a single directory:

```text
data/HandoverBlock-v1/policy_manual_intervention_clean/data/episode0.npz
data/HandoverBlock-v1/policy_manual_intervention_clean/data/episode1.npz
...
```

Then process that directory with `expert_data_num` equal to the accepted episode
count.

## Dataset Validation

Check a generated zarr:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/check_maniskill_dataset.py \
  /home/idiot/zhj/RoboTwin/policy/FlowMatching/data/HandoverBlock-v1-policy_manual_intervention_clean-10.zarr \
  --task-profile handover_block_native_smoke
```

Run a short replay smoke test when the zarr contains action data:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/check_maniskill_dataset.py \
  /home/idiot/zhj/RoboTwin/policy/FlowMatching/data/HandoverBlock-v1-policy_manual_intervention_clean-10.zarr \
  --task-profile handover_block_native_smoke \
  --smoke-replay \
  --replay-episodes 3 \
  --replay-steps 1000 \
  --sim-backend physx_cuda:3 \
  --render-backend sapien_cuda:3 \
  --save-video-dir /home/idiot/zhj/RoboTwin/tmp/replay_smoke_manual_clean
```

## Legacy Replay

The old RoboTwin/FlowMatching demonstration zarr can be replayed for debugging:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/collect_maniskill_data.py \
  HandoverBlock-v1 legacy_replay_legacy_rng_check 1 \
  --task-profile handover_block_native_legacy_rng_smoke \
  --collector legacy_replay \
  --teacher-zarr-path /home/idiot/zhj/RoboTwin/policy/FlowMatching/data/handover_block-demo_clean-50.zarr \
  --teacher-seed-path /home/idiot/zhj/RoboTwin/data/handover_block/demo_clean/seed.txt \
  --max-episode-steps 800 \
  --render-mode human \
  --sim-backend physx_cuda:3 \
  --render-backend sapien_cuda:3
```

Known limitation: old demonstrations may not replay successfully in the current
environment due to simulator, contact, controller, and geometry differences.
Use this path to diagnose compatibility, not as proof that a zarr is a valid
current-branch training dataset.

## Task Profiles and Training Configs

There are two layers of configuration:

- ManiSkill task profiles in [sim/maniskill/task_profiles.py](sim/maniskill/task_profiles.py).
- Policy training task YAMLs under:
  - [policy/FlowMatching/diffusion_policy/config/task/](policy/FlowMatching/diffusion_policy/config/task)
  - [policy/FPO/diffusion_policy/config/task/](policy/FPO/diffusion_policy/config/task)

Task profiles define runtime defaults such as environment id, observation mode,
action shape, and smoke zarr path.

Training YAMLs define the dataset loaded by the policy trainer. For real
training, always override:

```text
task.maniskill.zarr_path=/ABS/PATH/TO/YOUR.zarr
```

Do not rely on the default smoke zarr for experiments.

Common task profiles:

| Profile | Env | Purpose |
| --- | --- | --- |
| `pickcube_smoke` | `PickCube-v1` | Small pipeline smoke test. |
| `handover_block_native_smoke` | `HandoverBlock-v1` | Current native ManiSkill handover runtime. |
| `handover_block_native_legacy_rng_smoke` | `HandoverBlock-v1` | Legacy RNG compatibility diagnostics. |

## FPO Workflow

FPO is normally run after obtaining a reasonable Flow Matching checkpoint.

Process FPO zarr:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 policy/FPO/process_data.py \
  --backend maniskill \
  HandoverBlock-v1 policy_manual_intervention_clean 10
```

Train FPO from an FM checkpoint:

```bash
cd /home/idiot/zhj/RoboTwin

bash mainline.sh train-fpo 42 3 /ABS/PATH/TO/FM.ckpt \
  --backend maniskill \
  --config-name maniskill_fpo_handover_native \
  -- task.maniskill.zarr_path=/home/idiot/zhj/RoboTwin/policy/FPO/data/HandoverBlock-v1-policy_manual_intervention_clean-10.zarr \
     training.sim_backend=physx_cuda:3 \
     training.render_backend=sapien_cuda:3 \
     logging.mode=offline
```

Evaluate FPO:

```bash
cd /home/idiot/zhj/RoboTwin

bash mainline.sh eval-fpo HandoverBlock-v1 3 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --ckpt-path /ABS/PATH/TO/FPO.ckpt \
  --episodes 20 \
  --max-episode-steps 1000 \
  -- --sim-backend physx_cuda:3 --render-backend sapien_cuda:3
```

## Reproducibility Checklist

For each dataset used in an experiment, keep:

- The raw `.npz` directory under `data/HandoverBlock-v1/<task_config>/`.
- The generated zarr path.
- The accepted seed list.
- The terminal logs containing `success=1`.
- The videos used for quality control.
- The exact checkpoint path used as the base policy for manual intervention.
- The exact training command and Hydra overrides.

Minimum quality control before training:

- Every accepted episode should end with `success=1`.
- Replay smoke should not show obvious desynchronization.
- Demos with severe box slip, arm collisions, or excessive manual jitter should be excluded.
- Keep failed attempts outside the clean training directory.

## Troubleshooting

### Renderer cannot create a window

Use the same GPU for simulation and rendering:

```bash
--sim-backend physx_cuda:3 --render-backend sapien_cuda:3
```

If running headless video capture, omit `--render-mode human` and pass
`--save-video-dir`; the collector will use `rgb_array` rendering.

### GPU utilization is low but the viewer looks slow

Human rendering is often synchronization-bound rather than pure GPU-compute
bound. For data collection, prefer saved videos over real-time visual inspection
when possible.

### Manual-intervention demo looks successful but training quality is poor

Check the video and replay first. Successful terminal status only means the
environment success predicate fired; it does not guarantee the demonstration is
smooth enough for behavior cloning.

### Old zarr replays fail

This is expected for some legacy demonstrations. Initial object poses can be
aligned, but controller/contact/geometry differences can still make old action
sequences diverge.

### `scripted_expert` success varies by seed

This is currently a known limitation. Use `scripted_expert` to debug task logic,
but prefer `policy_manual_intervention` for accepted training demonstrations.

## Extending the Project

When modifying functionality, start from these files:

- Collection CLI: [script/collect_maniskill_data.py](script/collect_maniskill_data.py)
- Manual intervention: [sim/maniskill/policy_manual_intervention_collector.py](sim/maniskill/policy_manual_intervention_collector.py)
- Handover solver: [sim/maniskill/expert/handover_block_solver.py](sim/maniskill/expert/handover_block_solver.py)
- Motion primitives: [sim/maniskill/expert/primitives.py](sim/maniskill/expert/primitives.py)
- Task definition and success predicate: [sim/maniskill/tasks/handover_block_env.py](sim/maniskill/tasks/handover_block_env.py)
- Runtime profiles: [sim/maniskill/task_profiles.py](sim/maniskill/task_profiles.py)
- Raw trajectory format: [sim/maniskill/trajectory_recorder.py](sim/maniskill/trajectory_recorder.py)
- Flow Matching configs: [policy/FlowMatching/diffusion_policy/config/](policy/FlowMatching/diffusion_policy/config)
- FPO configs: [policy/FPO/diffusion_policy/config/](policy/FPO/diffusion_policy/config)

Keep changes small and tied to one stage of the pipeline. For dataset work, do
not mix failed and accepted episodes in the same training directory.

## Citation

If you use the upstream RoboTwin components, cite the original RoboTwin project
according to [README.md](README.md). Add any branch-specific paper citation here
once the project paper is finalized.
