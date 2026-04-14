# RoboTwin Mainline Guide

这份文档描述的是当前这个仓库里已经完成改造、并且实际可用的主线工作流。

- 上游 RoboTwin 官方说明保留在 [README.md](/home/idiot/zhj/RoboTwin/README.md)
- 这份文档只解释当前这个分支里的实际行为
- 当前主环境默认是 `zhjrl`
- 当前主 simulator 默认是 `ManiSkill`
- RoboTwin 原运行时仍然保留，但应视为 `legacy`

## 1. 当前仓库到底是什么

这个仓库现在同时包含两套东西：

1. 上游 RoboTwin 2.0 原始代码和文档
2. 一条新的 ManiSkill 主线

当前 ManiSkill 主线的目标是：

- 保留 RoboTwin 的任务和 embodiment 语义
- 把环境执行、采集、训练、评估统一到 ManiSkill 侧
- 支持 GPU 并行和固定频率运行

当前已经落地的真实任务是：

- `PickCube-v1`
- `HandoverBlock-v1`

其中 `HandoverBlock-v1` 是这次改造的重点任务。

## 2. 目录总览

### 2.1 核心目录

- `core/`
  - simulator-agnostic schema 和接口
- `legacy/robotwin/`
  - RoboTwin 原运行时 adapter
- `sim/maniskill/`
  - ManiSkill backend、vector backend、task profile、trajectory recorder
- `sim/maniskill/tasks/`
  - 自定义 ManiSkill env 注册入口
  - `BaseBimanualEnv` 是双臂任务基类
  - `HandoverBlock-v1` 是当前第一个完整迁移任务
- `data/converters/`
  - RoboTwin HDF5 -> zarr
  - ManiSkill NPZ -> zarr
- `policy/FlowMatching/`
  - FM 训练、处理、评估入口
- `policy/FPO/`
  - FPO 训练、处理、评估入口

### 2.2 当前最重要的几个文件

- [mainline.sh](/home/idiot/zhj/RoboTwin/mainline.sh)
  - 根目录统一入口
- [sim/maniskill/task_profiles.py](/home/idiot/zhj/RoboTwin/sim/maniskill/task_profiles.py)
  - ManiSkill task profile 注册表
- [README.md](/home/idiot/zhj/RoboTwin/README.md)
  - 上游 RoboTwin 官方说明
- [README_MAINLINE.md](/home/idiot/zhj/RoboTwin/README_MAINLINE.md)
  - 当前 ManiSkill 主线说明

## 3. 当前主线使用原则

### 3.1 推荐入口

统一使用根目录入口：

```bash
bash mainline.sh <command> ...
```

当前支持：

- `collect`
- `process`
- `process-fm`
- `process-fpo`
- `train-fm`
- `train-fpo`
- `eval-fm`
- `eval-fpo`

### 3.2 `task_profile` 和 `task yaml` 的区别

当前仓库里有两层任务身份：

1. `task_profile`
   - 运行时默认值和 smoke 数据路径
   - 定义在 [task_profiles.py](/home/idiot/zhj/RoboTwin/sim/maniskill/task_profiles.py)
2. `task yaml`
   - 算法真正读取的 shape、dataset、obs/action 配置
   - 定义在 `policy/*/diffusion_policy/config/task/*.yaml`

这两层没有完全合并，原因是为了尽量贴近原 RoboTwin 的显式配置风格。

### 3.3 `legacy` 和 `native` 的区别

当前 `handover_block` 有两条数据链：

- `legacy`
  - 来自原 RoboTwin 时代的 handover zarr
  - 对应 task:
    - `maniskill_handover_block_task`
  - 默认数据：
    - `handover_block-demo_clean-50.zarr`

- `native`
  - 来自当前 ManiSkill collect/process 流水线
  - 对应 task:
    - `maniskill_handover_block_native_task`
  - 默认数据：
    - `HandoverBlock-v1-smoke_npz-1.zarr`

当前最重要的现实判断是：

- `legacy` 数据可用于预训练 / warm start
- `native smoke` 目前只够验证链路，不够做真正的 FM 预训练

## 4. 当前可用的 task profile

定义见 [task_profiles.py](/home/idiot/zhj/RoboTwin/sim/maniskill/task_profiles.py)。

| profile | env_id | action / state | 默认数据 | 用途 |
| --- | --- | --- | --- | --- |
| `pickcube_smoke` | `PickCube-v1` | `8 / 9` | `PickCube-v1-smoke_npz-1.zarr` | pickcube smoke 验证 |
| `handover_block_smoke` | `HandoverBlock-v1` | `14 / 14` | `handover_block-demo_clean-50.zarr` | legacy handover 路线 |
| `handover_block_native_smoke` | `HandoverBlock-v1` | `14 / 14` | `HandoverBlock-v1-smoke_npz-1.zarr` | native smoke 链路验证 |

## 5. 数据与输出路径

### 5.1 原始采集数据

ManiSkill collect 默认写到：

```text
data/<env_id>/<task_config>/data/episode*.npz
```

例如：

```text
data/HandoverBlock-v1/smoke_npz/data/episode0.npz
```

### 5.2 处理后的 zarr

FM 和 FPO 仍然各自维护自己的 zarr 目录：

- FM:
  - `policy/FlowMatching/data/*.zarr`
- FPO:
  - `policy/FPO/data/*.zarr`

共享入口只是统一了 `process` 命令，不代表 zarr 输出已经合并。

### 5.3 训练日志

Hydra 输出目录仍在各算法目录下：

- `policy/FlowMatching/data/outputs/...`
- `policy/FPO/data/outputs/...`

### 5.4 checkpoint

当前 FM / FPO checkpoint 保存路径是：

```text
<repo_root>/checkpoints/...
```

不是：

- `policy/FlowMatching/checkpoints/`
- `policy/FPO/checkpoints/`

这是当前代码的真实行为，后续如果需要再单独收口。

## 6. 快速开始

### 6.1 ManiSkill collect

推荐写法：

```bash
bash mainline.sh collect 1 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --episode-num 1 \
  --max-episode-steps 50 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda
```

兼容旧写法：

```bash
bash mainline.sh collect HandoverBlock-v1 smoke_npz 1 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --episode-num 1 \
  --max-episode-steps 50
```

### 6.2 process

推荐写法：

```bash
bash mainline.sh process fm --backend maniskill --task-profile handover_block_native_smoke
bash mainline.sh process fpo --backend maniskill --task-profile handover_block_native_smoke
```

兼容别名：

```bash
bash mainline.sh process-fm --backend maniskill --task-profile handover_block_native_smoke
bash mainline.sh process-fpo --backend maniskill --task-profile handover_block_native_smoke
```

### 6.3 FM 训练

ManiSkill FM 的短写法是：

```bash
bash mainline.sh train-fm <seed> <gpu_id> \
  --backend maniskill \
  [--config-name CONFIG] \
  -- <hydra overrides>
```

例如：

```bash
bash mainline.sh train-fm 42 1 \
  --backend maniskill \
  -- task=maniskill_handover_block_task \
     logging.mode=offline \
     exp_name=handover_block_legacy_fm
```

### 6.4 FPO 训练

ManiSkill FPO 的短写法是：

```bash
bash mainline.sh train-fpo <seed> <gpu_id> [resume_ckpt_path] \
  --backend maniskill \
  [--config-name CONFIG] \
  -- <hydra overrides>
```

例如：

```bash
bash mainline.sh train-fpo 42 1 \
  --backend maniskill \
  -- task=maniskill_handover_block_native_task \
     training.sim_backend=physx_cuda \
     training.render_backend=sapien_cuda \
     logging.mode=offline
```

### 6.5 eval

FM:

```bash
bash mainline.sh eval-fm handover_block_smoke 1 \
  --backend maniskill \
  --task-profile handover_block_smoke \
  --ckpt-path /ABS/PATH/TO/FM.ckpt \
  --episodes 1 \
  --max-episode-steps 50
```

FPO:

```bash
bash mainline.sh eval-fpo handover_block_native_smoke 1 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --ckpt-path /ABS/PATH/TO/FPO.ckpt \
  --episodes 1 \
  --max-episode-steps 50 \
  -- --sim-backend physx_cuda --render-backend sapien_cuda
```

## 7. 当前 handover 推荐工作流

### 7.1 不推荐的路线

当前不推荐：

- 用 `native smoke` 数据直接做正式 FM 预训练

原因很简单：

- 当前这份 `HandoverBlock-v1-smoke_npz-1.zarr` 只有 1 个 episode、5 步数据
- 只够验证链路，不够训练

### 7.2 当前推荐路线

当前推荐的训练路线是：

1. 用 `legacy handover` 数据做 FM 预训练
2. 在当前 ManiSkill handover 环境里做 FPO finetune

也就是：

- `legacy 数据 -> 行为先验`
- `ManiSkill online RL -> 环境适配`

### 7.3 推荐命令

#### 第一步：FM 预训练，吃 legacy handover 数据

```bash
cd /home/idiot/zhj/RoboTwin
bash mainline.sh train-fm 42 1 \
  --backend maniskill \
  --config-name maniskill_fm \
  -- task=maniskill_handover_block_task \
     dataloader.batch_size=64 \
     val_dataloader.batch_size=64 \
     training.num_epochs=3000 \
     training.gradient_accumulate_every=2 \
     training.rollout_every=50 \
     training.checkpoint_every=100 \
     training.sample_every=5 \
     logging.mode=offline \
     exp_name=handover_block_legacy_fm
```

#### 第二步：FPO finetune，运行在当前 ManiSkill 环境

```bash
cd /home/idiot/zhj/RoboTwin
bash mainline.sh train-fpo 42 1 /ABS/PATH/TO/FM.ckpt \
  --backend maniskill \
  --config-name maniskill_fpo_finetune \
  -- task=maniskill_handover_block_task \
     training.sim_backend=physx_cuda \
     training.render_backend=sapien_cuda \
     logging.mode=offline \
     exp_name=handover_block_legacy_fpo_finetune
```

#### 第三步：评估

```bash
cd /home/idiot/zhj/RoboTwin
bash mainline.sh eval-fpo handover_block_smoke 1 \
  --backend maniskill \
  --task-profile handover_block_smoke \
  --ckpt-path /ABS/PATH/TO/FPO.ckpt \
  --episodes 1 \
  --max-episode-steps 50 \
  -- --sim-backend physx_cuda --render-backend sapien_cuda
```

## 8. 当前已经验证过的主线

当前已经实际验证通过的内容包括：

- `HandoverBlock-v1` 能正常 `gym.make / reset / step`
- `collect -> process -> train-fpo -> eval-fpo` 的 native smoke 闭环已通
- ManiSkill `sim_backend / render_backend` 已打通
- FM / FPO / eval / process / collect 的命令入口已经收口

但要区分两件事：

1. 工程链路已通
2. 策略已经能学会任务

前者已经成立，后者目前还没有。

## 9. 当前训练相关配置

### 9.1 通用 ManiSkill 配置

- FM:
  - [maniskill_fm.yaml](/home/idiot/zhj/RoboTwin/policy/FlowMatching/diffusion_policy/config/maniskill_fm.yaml)
  - [maniskill_fm_base.yaml](/home/idiot/zhj/RoboTwin/policy/FlowMatching/diffusion_policy/config/maniskill_fm_base.yaml)
- FPO:
  - [maniskill_fpo.yaml](/home/idiot/zhj/RoboTwin/policy/FPO/diffusion_policy/config/maniskill_fpo.yaml)
  - [maniskill_fpo_base.yaml](/home/idiot/zhj/RoboTwin/policy/FPO/diffusion_policy/config/maniskill_fpo_base.yaml)
  - [maniskill_fpo_finetune.yaml](/home/idiot/zhj/RoboTwin/policy/FPO/diffusion_policy/config/maniskill_fpo_finetune.yaml)

### 9.2 当前 handover 专用配置

- FM native 训练 preset:
  - [maniskill_fm_handover_native.yaml](/home/idiot/zhj/RoboTwin/policy/FlowMatching/diffusion_policy/config/maniskill_fm_handover_native.yaml)
- FPO native 在线训练 preset:
  - [maniskill_fpo_handover_native.yaml](/home/idiot/zhj/RoboTwin/policy/FPO/diffusion_policy/config/maniskill_fpo_handover_native.yaml)

这些 config 存在的意义是：

- 减少长串 override
- 保持 handover 和默认 pickcube 配置解耦

但要注意：

- `maniskill_fm_handover_native.yaml` 当前只是“能跑”的配置
- 不代表当前 native smoke 数据已经足够支撑真正的 FM 预训练

## 10. Handover 任务当前和原 RoboTwin 的关系

当前 `HandoverBlock-v1` 已经尽量向原任务靠拢：

- 双臂 `UR5-WSG` embodiment
- `14` 维动作 / `14` 维 `agent_pos`
- legacy handover 动作语义对齐
- `box` / `target_box` 尺寸和 reset 范围对齐
- `box` 姿态随机化已改回“小范围 yaw 扰动”
- 默认传感器已切到更接近原 `demo_clean` 的 `head_camera`

仍然没有完全逐项复刻的地方包括：

- ManiSkill 固定频率执行模型和原 RoboTwin 时序并不相同
- 当前没有迁原 RoboTwin scripted handover expert
- native demonstration 数据还没有建起来

## 11. 常见问题

### 11.1 为什么 `human` 渲染在 GPU 1 上会失败

如果 GPU 1 不是接显示器的卡，`render_mode=human` 会失败。当前更稳的方式是：

- `sim_backend="physx_cuda:1"`
- `render_backend="sapien_cuda:3"`

也就是：

- 卡 1 跑仿真
- 卡 3 出窗口

### 11.2 为什么 FM 训练会“立刻结束”

要区分两种情况：

1. 配置没切过去
   - 例如没传 `--config-name`
   - 还在用 `maniskill_fm` 默认的 `num_epochs=1`
2. 数据太少
   - 当前 native smoke 数据只有 5 步
   - 大 batch 会导致 dataloader 直接为空

### 11.3 为什么 native smoke 不适合正式 FM 预训练

因为当前它只有 smoke 级数据量，而且 collect 逻辑也不是 expert demonstration。

所以当前更现实的路线是：

- 先用 legacy handover 数据做预训练
- 再用 ManiSkill FPO finetune 去适配当前环境

## 12. 当前建议

如果你的目标是继续推进训练，而不是再改工程结构，那么当前推荐优先级是：

1. 先跑 `legacy FM pretrain + ManiSkill FPO finetune`
2. 暂时不要继续用 `native smoke` 作为正式预训练数据
3. 后面如果要把 native 路线做成正式主线，再单独解决 demonstration 数据来源

如果你的目标是继续改仓库，而不是继续训练，那么下一批更合适的工作会是：

1. checkpoint 路径收口
2. 文档和命令示例进一步统一
3. native expert 数据链建设
