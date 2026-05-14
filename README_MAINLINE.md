# RoboTwin ManiSkill Mainline

This branch is the ManiSkill-based training line for RoboTwin-style bimanual
manipulation. The current task is `HandoverBlock-v1`, a dual-arm handover task
used for demonstration collection, Flow Matching pretraining, policy evaluation,
and FPO fine-tuning.

The upstream RoboTwin README is still available in [README.md](README.md). This
document is the recommended entry point for the current ManiSkill workflow.

## Status

- Main environment: `HandoverBlock-v1`
- Main task profile: `handover_block_native_smoke`
- Observation used by policies: point cloud `[512, 3]` plus robot state `[14]`
- Action space: continuous joint/gripper action `[14]`
- Current data strategy: old clean demos plus a small number of successful
  manual-intervention ManiSkill demos
- FPO fine-tuning is experimental. Always evaluate checkpoints early because an
  unstable FPO run can diverge within tens of iterations.

## Repository Map

| Path | Purpose |
| --- | --- |
| [sim/maniskill/](sim/maniskill) | ManiSkill runtime, task profiles, obs/action adapters |
| [sim/maniskill/tasks/](sim/maniskill/tasks) | Custom ManiSkill environments |
| [sim/maniskill/policy_manual_intervention_collector.py](sim/maniskill/policy_manual_intervention_collector.py) | Human correction collector |
| [script/collect_maniskill_data.py](script/collect_maniskill_data.py) | ManiSkill data collection CLI |
| [script/eval_maniskill_policy.py](script/eval_maniskill_policy.py) | Direct FM/FPO ManiSkill evaluator |
| [script/merge_handover_pretrain_zarr.py](script/merge_handover_pretrain_zarr.py) | Merge old HandoverBlock demos with manual demos |
| [script/check_maniskill_dataset.py](script/check_maniskill_dataset.py) | Zarr validation and replay checks |
| [policy/FlowMatching/](policy/FlowMatching) | Flow Matching training/eval code |
| [policy/FPO/](policy/FPO) | FPO fine-tuning/eval code |
| [policy/fm_diffusion/](policy/fm_diffusion) | Shared policy/workspace implementation used by FM and FPO |
| [data/HandoverBlock-v1/](data/HandoverBlock-v1) | Raw ManiSkill `.npz` episodes |
| [policy/FlowMatching/data/](policy/FlowMatching/data) | Flow Matching zarr datasets |
| [policy/FPO/data/](policy/FPO/data) | FPO zarr datasets and FPO outputs |

## Environment Assumptions

The commands below assume:

```bash
REPO=/home/idiot/zhj/RoboTwin
PY=/home/idiot/anaconda3/envs/zhjrl/bin/python3.10
```

Activate the `zhjrl` environment before running scripts:

```bash
conda activate zhjrl
cd /home/idiot/zhj/RoboTwin
```

GPU notes:

- For direct Python commands, either expose one GPU with `CUDA_VISIBLE_DEVICES=1`
  and pass `physx_cuda` / `sapien_cuda`, or pass explicit matching device ids
  such as `physx_cuda:3` / `sapien_cuda:3`.
- Do not mix physics and render devices, e.g. `physx_cuda:1` with
  `sapien_cuda:3`.
- The training wrappers take a physical GPU id as their second positional
  argument and internally set `CUDA_VISIBLE_DEVICES=<gpu_id>`. Inside the Python
  process this GPU becomes `cuda:0`.

## Data

### Raw Manual-Intervention Episodes

Successful manual-intervention episodes are stored as one episode per directory:

```text
data/HandoverBlock-v1/policy_manual_intervention_collect/data/episode0.npz
data/HandoverBlock-v1/policy_manual_intervention_collect_001/data/episode0.npz
...
```

Each `.npz` stores:

- `point_cloud`
- `agent_pos`
- `actions`

Keep the terminal log and video for each accepted episode. The environment
success flag does not guarantee the trajectory is smooth enough for training.

### Old Plus Manual Pretraining Zarr

The current pretraining dataset combines:

- 50 old clean demos:
  `policy/FlowMatching/data/handover_block-demo_clean-50.zarr`
- 10 manual-intervention ManiSkill demos under:
  `data/HandoverBlock-v1/policy_manual_intervention_collect*`

Build the merged zarr:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/merge_handover_pretrain_zarr.py \
  --old-zarr-path /home/idiot/zhj/RoboTwin/policy/FlowMatching/data/handover_block-demo_clean-50.zarr \
  --manual-root /home/idiot/zhj/RoboTwin/data/HandoverBlock-v1 \
  --manual-pattern 'policy_manual_intervention_collect*' \
  --output-zarr-path /home/idiot/zhj/RoboTwin/policy/FlowMatching/data/handover_block-old50_manual10.zarr \
  --overwrite
```

The default output is:

```text
policy/FlowMatching/data/handover_block-old50_manual10.zarr
```

## Collect Manual-Intervention Data

Use this when a learned policy can roughly do the task but needs human
correction.

```bash
cd /home/idiot/zhj/RoboTwin

CUDA_VISIBLE_DEVICES=3 /home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/collect_maniskill_data.py \
  HandoverBlock-v1 policy_manual_intervention_collect_010 1 \
  --task-profile handover_block_native_smoke \
  --collector policy_manual_intervention \
  --policy-kind flowmatching \
  --ckpt-path /ABS/PATH/TO/FLOWMATCHING.ckpt \
  --seed 56 \
  --max-episode-steps 1200 \
  --render-mode human \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --debug-policy-manual-intervention
```

To save a video instead of opening the viewer:

```bash
CUDA_VISIBLE_DEVICES=1 /home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/collect_maniskill_data.py \
  HandoverBlock-v1 policy_manual_intervention_collect_010 1 \
  --task-profile handover_block_native_smoke \
  --collector policy_manual_intervention \
  --policy-kind flowmatching \
  --ckpt-path /ABS/PATH/TO/FLOWMATCHING.ckpt \
  --seed 56 \
  --max-episode-steps 1200 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --save-video-dir /home/idiot/zhj/RoboTwin/tmp/manual_collect_videos \
  --video-fps 10 \
  --debug-policy-manual-intervention
```

Important controls:

| Key | Action |
| --- | --- |
| `m` | enter manual mode |
| `1` / `2` | select left / right arm |
| arrow keys or `w/a/s/d` | planar motion |
| `u/j` or `r/f` | up / down |
| `g` | auto-align selected gripper to the current target |
| `o` / `c` | open / close gripper |
| `p` | print status |
| `n` | return to policy control |
| `q` | abort episode |

Use a new `task_config` directory for each one-episode collection run. Reusing
the same directory with `episode_num=1` overwrites `episode0.npz`.

## Process Raw `.npz` Episodes to Zarr

For a normal ManiSkill dataset directory:

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

The FPO data processor has the same interface:

```bash
/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 policy/FPO/process_data.py \
  --backend maniskill \
  HandoverBlock-v1 policy_manual_intervention_clean 10
```

Optional static-segment compression:

```bash
/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 policy/FlowMatching/process_data.py \
  --backend maniskill \
  --filter-static \
  --static-action-eps 0.05 \
  --static-state-eps 0.05 \
  --static-gripper-eps 1e-3 \
  --static-keep-boundary 2 \
  --output-suffix clean \
  HandoverBlock-v1 policy_manual_intervention_clean 10
```

## Train Flow Matching

Use `train_fm.sh` from `policy/FlowMatching`. The first positional argument is
the seed, and the second is the physical GPU id.

### From Scratch on the Merged Old50+Manual10 Dataset

```bash
cd /home/idiot/zhj/RoboTwin/policy/FlowMatching

bash train_fm.sh 114514 1 \
  --backend maniskill \
  --config-name maniskill_fm_handover_native_teacher \
  -- \
  task.dataset.zarr_path=/home/idiot/zhj/RoboTwin/policy/FlowMatching/data/handover_block-old50_manual10.zarr \
  exp_name=handover_block_old50_manual10_scratch_fm \
  training.resume=False \
  training.checkpoint_run_name=handover_block-old50_manual10_scratch-114514 \
  training.num_epochs=3000 \
  training.checkpoint_every=100 \
  training.rollout_every=50 \
  training.sample_every=50 \
  logging.mode=offline
```

### Continue Training From an Existing FM Checkpoint

```bash
cd /home/idiot/zhj/RoboTwin/policy/FlowMatching

bash train_fm.sh 114514 1 \
  /home/idiot/zhj/RoboTwin/policy/FlowMatching/checkpoints/HandoverBlock-v1-legacy_replay_native_dp3-50-clean-114514/2000.ckpt \
  --backend maniskill \
  --config-name maniskill_fm_handover_native_teacher \
  -- \
  task.dataset.zarr_path=/home/idiot/zhj/RoboTwin/policy/FlowMatching/data/handover_block-old50_manual10.zarr \
  exp_name=handover_block_old50_manual10_resume_fm \
  training.resume=True \
  training.checkpoint_run_name=handover_block-old50_manual10-114514 \
  training.num_epochs=5000 \
  training.checkpoint_every=100 \
  training.rollout_every=50 \
  training.sample_every=50 \
  logging.mode=offline
```

Checkpoints are saved under:

```text
policy/FlowMatching/checkpoints/<training.checkpoint_run_name>/<epoch>.ckpt
```

Training logs are under:

```text
policy/FlowMatching/data/outputs/<date>/<time>_<config>_<task>/
```

## Evaluate Flow Matching

Direct evaluator:

```bash
cd /home/idiot/zhj/RoboTwin

CUDA_VISIBLE_DEVICES=1 /home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/eval_maniskill_policy.py \
  flowmatching \
  /ABS/PATH/TO/FLOWMATCHING.ckpt \
  HandoverBlock-v1 \
  --task-profile handover_block_native_smoke \
  --episodes 50 \
  --seed 0 \
  --max-episode-steps 1000 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --save-video-dir /home/idiot/zhj/RoboTwin/tmp/eval_fm_videos \
  --video-fps 10
```

The printed `success_rate` is the main metric. Save videos for failed and
successful episodes when changing data or checkpoints.

## Fine-Tune With FPO

FPO should start from a Flow Matching checkpoint that already produces a
reasonable policy. Fine-tuning directly from a weak or unstable checkpoint tends
to collapse.

Use `train_fpo.sh` from `policy/FPO`:

```bash
cd /home/idiot/zhj/RoboTwin/policy/FPO

bash train_fpo.sh 114514 1 \
  /home/idiot/zhj/RoboTwin/policy/FlowMatching/checkpoints/handover_block-old50_manual10-114514/4900.ckpt \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --config-name maniskill_fpo_handover_native \
  -- \
  training.resume=True \
  training.num_envs=96 \
  training.timesteps_per_batch=73728 \
  training.num_fpo_samples=4 \
  training.max_total_fpo_samples_per_call=384 \
  training.n_updates_per_iteration=1 \
  training.num_minibatches=64 \
  training.gamma=0.995 \
  training.lmbda=0.99 \
  training.clip_coef=0.01 \
  training.clamp_logratio=0.2 \
  training.rho_max=2.0 \
  training.max_grad_norm=0.5 \
  training.max_timesteps_per_episode=1000 \
  training.sim_backend=physx_cuda \
  training.render_backend=sapien_cuda \
  logging.mode=offline \
  training.point_cloud_sampling=fast
```

Parameter interpretation:

```text
steps/env = training.timesteps_per_batch / training.num_envs
```

For the command above:

```text
73728 / 96 = 768 steps/env
```

Key FPO stability checks:

| Log field | Expected use |
| --- | --- |
| `clipped_ratio_frac` | If it quickly approaches `1.0`, the actor update is too aggressive. |
| `rho_mean` | Large spikes indicate policy ratio instability. |
| `rollout_action_abs_max` | If it jumps from about `1-2` to `5+`, the policy is likely diverging. |
| `rollout_reward_stage_mean` | Shows which task stage the rollout reaches. |
| `rollout_reward_stage_frac_*` | Stage distribution during rollout. |

If FPO diverges early, reduce update strength before increasing data size:

```text
training.n_updates_per_iteration=1
training.clip_coef=0.01
training.clamp_logratio=0.2
training.rho_max=2.0
training.max_grad_norm=0.5
```

FPO checkpoints are saved under:

```text
policy/FPO/checkpoints/online_fpo-pretrain_ep<pretrain_epoch>-<seed>/<pretrain_epoch>-<online_iter>.ckpt
```

Warning: repeated runs with the same pretrain epoch and seed can overwrite the
same checkpoint filenames. Rename or move useful checkpoints before launching a
new run.

## Evaluate FPO

```bash
cd /home/idiot/zhj/RoboTwin

CUDA_VISIBLE_DEVICES=3 /home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/eval_maniskill_policy.py \
  fpo \
  /ABS/PATH/TO/FPO.ckpt \
  HandoverBlock-v1 \
  --task-profile handover_block_native_smoke \
  --episodes 50 \
  --seed 0 \
  --max-episode-steps 1000 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --save-video-dir /home/idiot/zhj/RoboTwin/tmp/eval_fpo_videos \
  --video-fps 10
```

Do not only trust the scalar success rate. Watch several videos, especially
after FPO fine-tuning, because unstable policies may satisfy isolated reward
signals while producing physically poor behavior.

## TensorBoard

Flow Matching:

```bash
cd /home/idiot/zhj/RoboTwin/policy/FlowMatching
tensorboard --logdir data/outputs --port 16006 --host 127.0.0.1
```

FPO:

```bash
cd /home/idiot/zhj/RoboTwin/policy/FPO
tensorboard --logdir data/outputs --port 16006 --host 127.0.0.1
```

FPO logs include `online/actor_loss`, `online/critic_loss`,
`online/rho_mean`, `online/clipped_ratio_frac`, and rollout diagnostics such as
`online/rollout_action_abs_max` and `online/rollout_reward_stage_mean`.

## Dataset Validation and Replay

Validate a generated zarr:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/check_maniskill_dataset.py \
  /ABS/PATH/TO/DATASET.zarr \
  --task-profile handover_block_native_smoke
```

Replay a few episodes:

```bash
/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/check_maniskill_dataset.py \
  /ABS/PATH/TO/DATASET.zarr \
  --task-profile handover_block_native_smoke \
  --smoke-replay \
  --replay-episodes 3 \
  --replay-steps 1000 \
  --sim-backend physx_cuda:3 \
  --render-backend sapien_cuda:3 \
  --save-video-dir /home/idiot/zhj/RoboTwin/tmp/replay_videos
```

## Legacy Replay

Old RoboTwin demonstrations can be replayed for diagnostics:

```bash
cd /home/idiot/zhj/RoboTwin

/home/idiot/anaconda3/envs/zhjrl/bin/python3.10 script/collect_maniskill_data.py \
  HandoverBlock-v1 legacy_replay_check 1 \
  --task-profile handover_block_native_legacy_rng_smoke \
  --collector legacy_replay \
  --teacher-zarr-path /home/idiot/zhj/RoboTwin/policy/FlowMatching/data/handover_block-demo_clean-50.zarr \
  --teacher-seed-path /home/idiot/zhj/RoboTwin/data/handover_block/demo_clean/seed.txt \
  --max-episode-steps 800 \
  --render-mode human \
  --sim-backend physx_cuda:3 \
  --render-backend sapien_cuda:3
```

This is not a guarantee that old actions are valid in the current environment.
Controller, contact, and geometry differences can make old trajectories diverge.

## Common Failure Modes

### Renderer Cannot Create a Window

Human rendering may only work on display-capable GPUs. For headless video,
remove `--render-mode human` and use `--save-video-dir`.

### Simulation and Rendering Device Mismatch

Use matching backends:

```bash
--sim-backend physx_cuda:3 --render-backend sapien_cuda:3
```

or expose one GPU:

```bash
CUDA_VISIBLE_DEVICES=1 ... --sim-backend physx_cuda --render-backend sapien_cuda
```

### FPO Checkpoint Produces Random Motion

Inspect the training log before evaluating more checkpoints. If the run shows
`rollout_action_abs_max` jumping to `5+` and `clipped_ratio_frac` near `1.0`,
the checkpoint is already diverged.

### FPO OOM After Rollout

This usually happens during actor update, not simulation. Increase
`training.num_minibatches` or reduce `training.num_fpo_samples` /
`training.n_updates_per_iteration`.

### Old Data Replays Fail

Expected for some old demonstrations. Use old data for pretraining if it helps
the policy learn coarse behavior, but do not assume old action replay is a valid
success check in the current ManiSkill environment.

## Reproducibility Checklist

For every experiment, record:

- dataset zarr path and generation command
- raw accepted `.npz` paths
- base checkpoint path
- exact train/eval command
- seed and GPU id
- TensorBoard or `logs.json.txt` directory
- evaluation videos
- success rate and qualitative notes

Recommended minimum evaluation:

- FM checkpoints: at least 50 episodes with saved videos
- FPO checkpoints: evaluate early checkpoints first, e.g. 10-50 online
  iterations, before running long fine-tuning

## Citation

If you use upstream RoboTwin components, cite the original RoboTwin project
according to [README.md](README.md). Add the project-specific citation here when
the paper is finalized.
