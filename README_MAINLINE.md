# ManiSkill Bimanual Manipulation Mainline

This repository contains a ManiSkill-based training pipeline for bimanual robotic
manipulation built on the [RoboTwin 2.0](https://robotwin-platform.github.io/)
platform. We target fixed-frequency joint-position control with Flow Matching
pretraining and FPO online fine-tuning. The primary task is **HandoverBlock**, a
dual-arm handover task where the left arm grasps a block from the table, transfers
it to the right arm at a designated handover pose, and the right arm places it on
a target.

## Setup

```bash
conda activate zhjrl
cd /home/idiot/zhj/RoboTwin
```

The environment uses `CUDA_VISIBLE_DEVICES` for GPU assignment. Physics and
rendering backends must target the same device.

## Repository Structure

| Path | Description |
| --- | --- |
| `sim/maniskill/tasks/` | ManiSkill task environments (`HandoverBlock-v1`, `HandoverBlockUR5-v2`) |
| `sim/maniskill/expert/` | Scripted expert solver and motion primitives |
| `sim/maniskill/task_profiles.py` | Centralized runtime defaults per task profile |
| `script/collect_maniskill_data.py` | Data collection CLI (scripted expert, manual intervention, legacy replay) |
| `script/eval_maniskill_policy.py` | Policy evaluation CLI |
| `script/check_maniskill_dataset.py` | Zarr validation and replay diagnostics |
| `script/handover/` | Stage-wise diagnostic probes for the handover task |
| `policy/FlowMatching/` | Flow Matching pretraining |
| `policy/FPO/` | FPO online fine-tuning |
| `policy/fm_diffusion/` | Shared policy and workspace modules |
| `runners/` | CUDA batched rollout runners for multi-environment training |
| `core/` | Abstract interfaces (action/observation schemas, env backends) |
| `legacy/robotwin/` | RoboTwin legacy runtime compatibility adapters |
| `data/` | Raw `.npz` episode directories |
| `policy/FlowMatching/data/` | Processed Flow Matching `.zarr` datasets |
| `policy/FPO/data/` | FPO datasets and outputs |

## Task Environments

### HandoverBlock-v1

The original ManiSkill port of the RoboTwin handover block task with Dual UR5-WSG
arms. Box spawns on the left side of the table; the target is on the right.

- Action space: 14-D absolute joint positions (6-D left arm + 1-D left gripper + 6-D right arm + 1-D right gripper)
- Observation: point cloud `[512, 3]` + robot state `[14]`
- Control frequency: 20 Hz
- Task profile: `handover_block_native_smoke`

### HandoverBlockUR5-v2

A narrower spawn/goal region variant optimized for UR5 workspace reachability.
Otherwise identical interface to HandoverBlock-v1.

- Task profile: `handover_block_ur5_v2_smoke`
- Box spawn range: x ∈ [-0.12, -0.09], y ∈ [0.07, 0.11]
- Target range: x ∈ [0.08, 0.16], y ∈ [0.11, 0.19]
- Handover pose: [0.02, 0.11, 0.92]

## Data Collection

### Scripted Expert

The scripted expert uses a multi-stage planner-based solver that executes:

1. Left arm first grasp and lift
2. Left arm move to handover pose
3. Right arm static-receiver handover grasp
4. Right arm carry to target and place

```bash
CUDA_VISIBLE_DEVICES=0 python script/collect_maniskill_data.py \
  HandoverBlockUR5-v2 handover_block_ur5_v2_collect 50 \
  --task-profile handover_block_ur5_v2_smoke \
  --collector scripted_expert \
  --seed 108 \
  --max-attempts 200 \
  --max-episode-steps 900 \
  --sim-backend physx_cuda:0 \
  --save-only-success
```

**Positional arguments:**

| Arg | Description |
| --- | --- |
| `HandoverBlockUR5-v2` | ManiSkill environment ID |
| `handover_block_ur5_v2_collect` | Output subdirectory under `data/HandoverBlockUR5-v2/` |
| `50` | Number of successful episodes to collect |

**Key options:**

| Option | Description |
| --- | --- |
| `--task-profile` | Pre-registered profile with default obs/action/reward settings |
| `--collector` | Collection strategy: `scripted_expert`, `policy_manual_intervention`, `legacy_replay`, `random` |
| `--seed` | Base reset seed; each attempt increments by 1 |
| `--max-attempts` | Cap on total episodes (including failures) before giving up |
| `--max-episode-steps` | Max steps per episode before automatic truncation |
| `--sim-backend` | `physx_cpu` or `physx_cuda[:<device_id>]` |
| `--save-only-success` | Skip saving failed episodes; successful ones are also action-replay-validated |
| `--debug-scripted-expert` | Print solver debug info (grasp candidate selection, placement decisions) |
| `--save-video-dir` | If set, save an `.mp4` video per episode alongside the `.npz` |
| `--video-fps` | Frame rate for saved videos (default 10) |

Successful episodes are automatically action-replay-validated before saving. Each
episode produces an `episode<N>.npz` and `episode<N>.json` metadata file under
`data/HandoverBlockUR5-v2/<task_config>/data/`.

## Data Processing

Convert raw `.npz` episodes to Flow Matching zarr format:

```bash
python policy/FlowMatching/process_data.py \
  --backend maniskill \
  HandoverBlockUR5-v2 handover_block_ur5_v2_collect 50
```

**Positional arguments:**

| Arg | Description |
| --- | --- |
| `HandoverBlockUR5-v2` | Environment ID matching the data directory under `data/` |
| `handover_block_ur5_v2_collect` | Subdirectory name under `data/<env_id>/` |
| `50` | Expected number of episodes (validates that the directory has at least this many) |

**Key options:**

| Option | Description |
| --- | --- |
| `--backend` | `maniskill` for this pipeline (`legacy` for old RoboTwin HDF5 data) |
| `--filter-static` | Remove near-static frames (gripper unchanged, action/state below eps) |
| `--static-action-eps` | Action change threshold for static-frame detection (default 0.05) |
| `--static-state-eps` | State change threshold for static-frame detection (default 0.05) |
| `--static-gripper-eps` | Gripper change threshold for static-frame detection (default 1e-3) |
| `--static-keep-boundary` | Frames to keep before/after each transition to avoid over-trimming |
| `--output-suffix` | Appended to the output zarr name (e.g., `clean` produces `...-clean.zarr`) |

Output: `policy/FlowMatching/data/HandoverBlockUR5-v2-handover_block_ur5_v2_collect-50.zarr`

Validate a zarr dataset:

```bash
python script/check_maniskill_dataset.py \
  /path/to/dataset.zarr \
  --task-profile handover_block_ur5_v2_smoke \
  --smoke-replay --replay-episodes 3 --replay-steps 1000 \
  --sim-backend physx_cuda:0 --render-backend sapien_cuda:0 \
  --save-video-dir /tmp/replay_videos
```

| Option | Description |
| --- | --- |
| `--smoke-replay` | Replay saved actions in the environment to check validity |
| `--replay-episodes` | Number of episodes to replay (default 3) |
| `--replay-steps` | Max steps per replay episode |
| `--save-video-dir` | Save replay videos for visual inspection |

## Training

### Flow Matching Pretraining

```bash
cd policy/FlowMatching

bash train_fm.sh 42 0 \
  --backend maniskill \
  --config-name maniskill_fm_handover_native_teacher \
  -- \
  task.dataset.zarr_path=/path/to/dataset.zarr \
  exp_name=handover_block_ur5_v2_50_fm \
  training.resume=False \
  training.checkpoint_run_name=handover_block_ur5_v2_50-42 \
  training.num_epochs=3000 \
  training.checkpoint_every=100 \
  training.rollout_every=50 \
  training.sample_every=50 \
  logging.mode=offline
```

**Script arguments:**

| Arg | Description |
| --- | --- |
| `42` | Random seed for training |
| `0` | Physical GPU ID (sets `CUDA_VISIBLE_DEVICES` internally) |

**Key options (passed after `--`):**

| Option | Description |
| --- | --- |
| `task.dataset.zarr_path` | Path to the processed `.zarr` training dataset |
| `exp_name` | Experiment name used in log directory naming |
| `training.resume` | `True` to continue from an existing checkpoint, `False` for fresh training |
| `training.checkpoint_run_name` | Subdirectory under `checkpoints/` for this run |
| `training.num_epochs` | Total training epochs |
| `training.checkpoint_every` | Save a checkpoint every N epochs |
| `training.rollout_every` | Run an evaluation rollout every N epochs |
| `training.sample_every` | Log sample predictions every N epochs |
| `logging.mode` | `offline` for disk-only logging, `online` for W&B or similar |

Checkpoints are saved under `policy/FlowMatching/checkpoints/<run_name>/<epoch>.ckpt`.
Monitor training with:

```bash
tensorboard --logdir policy/FlowMatching/data/outputs --port 16006
```

### FPO Online Fine-Tuning

Start from a converged Flow Matching checkpoint:

```bash
cd policy/FPO

bash train_fpo.sh 42 0 \
  /path/to/pretrained_fm.ckpt \
  --backend maniskill \
  --task-profile handover_block_ur5_v2_smoke \
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

**Script arguments:**

| Arg | Description |
| --- | --- |
| `42` | Random seed |
| `0` | Physical GPU ID |
| `/path/to/pretrained_fm.ckpt` | Path to the Flow Matching checkpoint to initialize from |

**Key training options (passed after `--`):**

| Option | Description |
| --- | --- |
| `training.num_envs` | Number of parallel simulation environments |
| `training.timesteps_per_batch` | Total environment steps collected per batch; steps/env = `timesteps_per_batch / num_envs` |
| `training.num_fpo_samples` | Number of online samples per policy call (higher = more exploration) |
| `training.n_updates_per_iteration` | Number of PPO-style updates per collected batch |
| `training.num_minibatches` | Minibatch count for actor/critic updates (increase if OOM) |
| `training.gamma` / `training.lmbda` | Discount factor and GAE lambda |
| `training.clip_coef` | PPO clipping epsilon — lower = more conservative update |
| `training.clamp_logratio` | Cap on the log-ratio between new and old policy |
| `training.rho_max` | Max importance sampling ratio before clipping |
| `training.max_grad_norm` | Global gradient norm clipping |
| `training.max_timesteps_per_episode` | Max steps per rollout episode before truncation |
| `training.point_cloud_sampling` | `fast` (GPU FPS) or `random` — point cloud downsampling method |

Key stability indicators in TensorBoard:

| Metric | Warning Sign |
| --- | --- |
| `clipped_ratio_frac` | Approaching 1.0: actor update too aggressive |
| `rollout_action_abs_max` | Jumping from ~1-2 to 5+: policy likely diverging |
| `rho_mean` | Large spikes: ratio instability |

If FPO diverges early, reduce `clip_coef`, `clamp_logratio`, and `rho_max`
before increasing data volume.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python script/eval_maniskill_policy.py \
  flowmatching \
  /path/to/checkpoint.ckpt \
  HandoverBlockUR5-v2 \
  --task-profile handover_block_ur5_v2_smoke \
  --episodes 50 \
  --seed 0 \
  --max-episode-steps 1000 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda \
  --save-video-dir /tmp/eval_videos \
  --video-fps 10
```

**Positional arguments:**

| Arg | Description |
| --- | --- |
| `flowmatching` | Policy kind: `flowmatching` or `fpo` |
| `/path/to/checkpoint.ckpt` | Path to the checkpoint to evaluate |
| `HandoverBlockUR5-v2` | Environment ID |

**Key options:**

| Option | Description |
| --- | --- |
| `--episodes` | Number of evaluation episodes |
| `--seed` | Starting seed (incremented per episode) |
| `--max-episode-steps` | Max steps per episode |
| `--save-video-dir` | Save rollout videos for visual inspection (strongly recommended) |
| `--video-fps` | Frame rate of saved videos |

For FPO checkpoints, replace `flowmatching` with `fpo`. Always inspect rollout
videos alongside scalar success rates; unstable policies may satisfy isolated
reward signals while producing poor physical behavior.

## Reproducibility Checklist

For each experiment, record:

- Dataset zarr path and generation command
- Raw `.npz` source directories
- Base checkpoint path (for FPO)
- Exact train/eval commands with seeds and GPU assignments
- TensorBoard log directory
- Evaluation videos
- Success rate and qualitative observations

## Citation

If you use upstream RoboTwin components, cite the original works:

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

@InProceedings{Mu_2025_CVPR,
  author    = {Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and
               Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and
               Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and
               Ding, Mingyu and Luo, Ping},
  title     = {RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  year      = {2025},
  pages     = {27649-27660}
}
```
