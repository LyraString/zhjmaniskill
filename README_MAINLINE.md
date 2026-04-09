# RoboTwin Mainline Notes

这份文档描述的是当前这个仓库的实际主线用法。

- 官方 RoboTwin 说明保留在 [README.md](/home/idiot/zhj/RoboTwin/README.md)
- 当前仓库的主环境是 `zhjrl`
- 当前仓库的主 simulator 是 `ManiSkill`
- RoboTwin 原运行时仍然保留，但应视为 `legacy`

## 当前结构

- `core/`
  - simulator-agnostic schema 和接口
- `legacy/robotwin/`
  - RoboTwin 旧运行时 adapter
- `sim/maniskill/`
  - ManiSkill backend、vector backend、task profile、trajectory recorder
- `data/converters/`
  - RoboTwin HDF5 converter 和 ManiSkill NPZ converter
- `runners/`
  - 抽离后的 online runner

## 推荐环境

- `zhjrl`
  - 主开发环境
  - 用于 ManiSkill collect / process / FM / FPO / eval
- `zhjRoboTwin`
  - legacy 对照环境
  - 只在需要验证 RoboTwin 原运行时时使用

## 当前主线命令

### 1. ManiSkill 采集

根目录入口：

```bash
bash collect_data.sh PickCube-v1 smoke_npz 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --episode-num 1 \
  --max-episode-steps 5
```

会落到：

```text
data/PickCube-v1/smoke_npz/data/episode0.npz
```

### 2. ManiSkill 转 zarr

Flow Matching：

```bash
cd policy/FlowMatching
bash process_data.sh PickCube-v1 smoke_npz 1 --backend maniskill
```

FPO：

```bash
cd policy/FPO
bash process_data.sh PickCube-v1 smoke_npz 1 --backend maniskill
```

### 3. ManiSkill Flow Matching smoke

```bash
cd policy/FlowMatching
bash train_fm.sh PickCube-v1 smoke_npz 1 42 8 0 \
  --backend maniskill \
  --task-profile pickcube_smoke
```

默认配置：

- `policy/FlowMatching/diffusion_policy/config/maniskill_fm_pickcube.yaml`

### 4. ManiSkill FPO smoke

```bash
cd policy/FPO
bash train_fpo.sh PickCube-v1 smoke_npz 1 42 8 0 \
  --backend maniskill \
  --task-profile pickcube_smoke
```

默认配置：

- `policy/FPO/diffusion_policy/config/maniskill_fpo_pickcube.yaml`

### 5. ManiSkill FM -> FPO finetune smoke

```bash
cd policy/FPO
bash train_fpo.sh PickCube-v1 smoke_npz 1 42 8 0 /ABS/PATH/TO/FM.ckpt \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --config-name maniskill_fpo_pickcube_finetune
```

### 6. ManiSkill 评估

Flow Matching：

```bash
cd policy/FlowMatching
bash eval.sh PickCube-v1 smoke_npz dummy 1 42 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --ckpt-path /ABS/PATH/TO/FM.ckpt \
  --episodes 1 \
  --max-episode-steps 10
```

FPO：

```bash
cd policy/FPO
bash eval.sh PickCube-v1 smoke_npz dummy 1 42 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --ckpt-path /ABS/PATH/TO/FPO.ckpt \
  --episodes 1 \
  --max-episode-steps 10
```

## 当前边界

- `PickCube-v1` 目前只是 ManiSkill smoke task，不是正式研究任务
- 训练、评估、数据链已经能走 ManiSkill 主线
- RoboTwin 相关脚本和运行时仍然在仓库内，但不应继续作为主依赖

## 后续建议

- 继续把正式研究任务迁到 ManiSkill
- RoboTwin 只保留 legacy 对照链
- 之后再考虑 Isaac Lab / MuJoCo 的第二层 simulator profile
