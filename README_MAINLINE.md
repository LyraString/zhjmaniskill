# RoboTwin ManiSkill Mainline

这份文档描述的是当前分支里真实可用的主线工作流，而不是上游 RoboTwin 官方默认工作流。

- 上游项目介绍、论文信息和原始 RoboTwin 使用说明保留在 [README.md](README.md)
- 当前分支的 ManiSkill 主线说明以这份文档为准
- 当前主要验证对象是 `HandoverBlock-v1`

## 项目定位

当前仓库同时包含两套东西：

1. 上游 RoboTwin 2.0 代码与文档
2. 一条新的 ManiSkill 主线

这条 ManiSkill 主线的目标是：

- 保留 RoboTwin 的任务语义和 embodiment 语义
- 用 ManiSkill 环境统一采集、训练和评测
- 把策略训练链路收敛到 `collect -> process -> train -> eval`

当前分支里，最值得关注的任务是：

- `HandoverBlock-v1`

## 环境准备

这份文档默认你已经完成了上游 RoboTwin 的基础安装。

安装与资产下载请先参考：

- [README.md](README.md)
- 上游 RoboTwin 官方文档

当前分支的额外假设是：

- 默认 Python 环境是 `zhjrl`
- 当前主 simulator 是 ManiSkill
- 调试 `scripted_expert` 时通常使用：
  - `--sim-backend physx_cpu`
  - `--render-backend sapien_cuda`

如果你只想快速开始，不必先理解所有目录，直接按下面这条链走即可：

1. 采 raw episode
2. 处理成 zarr
3. 训练 FM / FPO
4. 评测

## 仓库结构

下面这些目录是当前主线最重要的部分：

- [`mainline.sh`](mainline.sh)
  - 根目录统一入口
- [`script/`](script)
  - 采集、评测和桥接脚本
- [`sim/maniskill/`](sim/maniskill)
  - ManiSkill runtime、task profile、trajectory recorder、expert solver
- [`sim/maniskill/tasks/`](sim/maniskill/tasks)
  - 当前自定义 ManiSkill 任务实现
- [`policy/FlowMatching/`](policy/FlowMatching)
  - Flow Matching 数据处理、训练、评测
- [`policy/FPO/`](policy/FPO)
  - FPO 数据处理、训练、评测
- [`data/`](data)
  - 原始采集数据、训练输出
- [`checkpoints/`](checkpoints)
  - 保存的模型权重
- [`eval_videos/`](eval_videos)
  - 评测视频

如果你只想理解主链路，先看这几个文件：

- [`script/collect_maniskill_data.py`](script/collect_maniskill_data.py)
- [`sim/maniskill/expert/handover_block_solver.py`](sim/maniskill/expert/handover_block_solver.py)
- [`sim/maniskill/expert/primitives.py`](sim/maniskill/expert/primitives.py)
- [`sim/maniskill/tasks/handover_block_env.py`](sim/maniskill/tasks/handover_block_env.py)
- [`sim/maniskill/task_profiles.py`](sim/maniskill/task_profiles.py)

## 当前支持的主工作流

主入口是：

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

但有一个当前分支特有的限制需要明确：

- `scripted_expert` 采集目前还没有完整接回 `mainline.sh collect`
- 原因是 [`script/collect_data.py`](script/collect_data.py) 目前只接受 `random|legacy_replay`
- 所以 `scripted_expert` 请直接调用 [`script/collect_maniskill_data.py`](script/collect_maniskill_data.py)

## 任务配置的两层含义

当前主线里有两层任务配置，作用不同。

### `task profile`

定义在 [`sim/maniskill/task_profiles.py`](sim/maniskill/task_profiles.py)。

它负责：

- 运行时默认参数
- smoke 数据路径
- `env_id`
- `shape_meta`

当前常用 profile：

- `pickcube_smoke`
- `handover_block_smoke`
- `handover_block_native_smoke`

### task yaml

定义在：

- [`policy/FlowMatching/diffusion_policy/config/task/`](policy/FlowMatching/diffusion_policy/config/task)
- [`policy/FPO/diffusion_policy/config/task/`](policy/FPO/diffusion_policy/config/task)

它负责：

- 训练阶段真正读取的 `zarr_path`
- 数据形状
- 数据集类

这意味着：

- `task profile` 更偏运行时默认值
- task yaml 更偏训练配置

对于你自己采出来的新数据，训练时通常应该显式覆盖：

```bash
task.dataset.zarr_path=/ABS/PATH/TO/YOUR.zarr
```

而不是依赖默认的 smoke zarr。

## 从采集到训练再到评测

推荐把当前分支理解成四个阶段：

1. 收 raw episode
2. 处理成 zarr
3. 训练 FM / FPO
4. 评测并保存视频

下面按这个顺序说明。

## 1. 数据采集

### 1.1 当前支持的采集模式

[`script/collect_maniskill_data.py`](script/collect_maniskill_data.py) 当前支持：

- `random`
- `legacy_replay`
- `scripted_expert`

其中：

- `random`
  - 只用于链路验证
- `legacy_replay`
  - 用 legacy zarr 里的动作序列重放到当前 ManiSkill 环境
- `scripted_expert`
  - 当前用于生成 native handover demonstration 的主模式

### 1.2 `scripted_expert` 的推荐入口

可视化调试：

```bash
cd /path/to/RoboTwin
CUDA_VISIBLE_DEVICES=3 /home/idiot/anaconda3/envs/zhjrl/bin/python script/collect_maniskill_data.py \
  HandoverBlock-v1 scripted_expert_npz 1 \
  --task-profile handover_block_native_smoke \
  --collector scripted_expert \
  --seed 0 \
  --max-episode-steps 800 \
  --sim-backend physx_cpu \
  --render-backend sapien_cuda \
  --render-mode human
```

无头采集：

```bash
cd /path/to/RoboTwin
CUDA_VISIBLE_DEVICES=1 /home/idiot/anaconda3/envs/zhjrl/bin/python script/collect_maniskill_data.py \
  HandoverBlock-v1 scripted_expert_npz 10 \
  --task-profile handover_block_native_smoke \
  --collector scripted_expert \
  --save-only-success \
  --seed 0 \
  --max-episode-steps 800 \
  --sim-backend physx_cpu \
  --render-backend sapien_cuda
```

### 1.3 `episode_num` 现在是什么意思

对 `scripted_expert` 而言：

- `episode_num` 表示最终要保存的成功轨迹数
- 不是尝试的 seed 数

例如：

```bash
... scripted_expert_npz 10 --collector scripted_expert --save-only-success ...
```

表示：

- 会持续尝试新的 seed
- 直到真的保存出 `10` 条成功 episode

### 1.4 原始数据写到哪里

采集成功后，原始 episode 会写到：

```text
data/<env_id>/<task_config>/data/episode*.npz
```

例如：

```text
data/HandoverBlock-v1/scripted_expert_npz/data/episode0.npz
```

当前 `.npz` 默认包含：

- `point_cloud`
- `agent_pos`
- `actions`

对应实现见：

- [`sim/maniskill/trajectory_recorder.py`](sim/maniskill/trajectory_recorder.py)

### 1.5 当前推荐的采集后端

对 `scripted_expert`，当前推荐：

- `--sim-backend physx_cpu`
- `--render-backend sapien_cuda`

原因是：

- `physx_cpu` 路径下，当前 motion planning / solver 行为更稳定
- `physx_cuda` 之前存在 planner 集成不一致的问题

## 2. 数据处理

处理入口统一在：

```bash
bash mainline.sh process fm ...
bash mainline.sh process fpo ...
```

### 2.1 处理成 FM 用的 zarr

```bash
cd /path/to/RoboTwin
bash mainline.sh process fm HandoverBlock-v1 scripted_expert_npz 10 --backend maniskill
```

### 2.2 处理成 FPO 用的 zarr

```bash
cd /path/to/RoboTwin
bash mainline.sh process fpo HandoverBlock-v1 scripted_expert_npz 10 --backend maniskill
```

### 2.3 处理后的输出路径

FM 输出：

```text
policy/FlowMatching/data/HandoverBlock-v1-scripted_expert_npz-10.zarr
```

FPO 输出：

```text
policy/FPO/data/HandoverBlock-v1-scripted_expert_npz-10.zarr
```

### 2.4 smoke profile 和真实训练数据的关系

`handover_block_native_smoke` 目前默认仍然指向：

```text
data/HandoverBlock-v1-smoke_npz-1.zarr
```

所以：

- smoke profile 适合链路验证
- 真正训练时，建议显式覆盖成你自己的 zarr 路径

## 3. Flow Matching 训练

Flow Matching 的统一入口：

```bash
bash mainline.sh train-fm <seed> <gpu_id> --backend maniskill ...
```

推荐直接使用 handover native preset：

- [`maniskill_fm_handover_native.yaml`](policy/FlowMatching/diffusion_policy/config/maniskill_fm_handover_native.yaml)

示例：

```bash
cd /path/to/RoboTwin
bash mainline.sh train-fm 42 1 \
  --backend maniskill \
  --config-name maniskill_fm_handover_native \
  -- task.dataset.zarr_path=/path/to/RoboTwin/policy/FlowMatching/data/HandoverBlock-v1-scripted_expert_npz-10.zarr \
     logging.mode=offline
```

说明：

- 这里显式覆盖了 `task.dataset.zarr_path`
- 这样不会误用默认 smoke 数据

训练输出默认在：

```text
data/outputs/<exp_name>_seed<seed>/
```

checkpoint 默认在：

```text
checkpoints/
```

## 4. FPO 训练

FPO 的统一入口：

```bash
bash mainline.sh train-fpo <seed> <gpu_id> [resume_ckpt_path] --backend maniskill ...
```

推荐使用：

- [`maniskill_fpo_handover_native.yaml`](policy/FPO/diffusion_policy/config/maniskill_fpo_handover_native.yaml)

示例：

```bash
cd /path/to/RoboTwin
bash mainline.sh train-fpo 42 1 /ABS/PATH/TO/FM.ckpt \
  --backend maniskill \
  --config-name maniskill_fpo_handover_native \
  -- task.dataset.zarr_path=/path/to/RoboTwin/policy/FPO/data/HandoverBlock-v1-scripted_expert_npz-10.zarr \
     training.sim_backend=physx_cuda \
     training.render_backend=sapien_cuda \
     logging.mode=offline
```

这里的典型用法是：

- 先用 FM 学到一个 behavior prior
- 再用 FPO 在 ManiSkill 环境里做在线适配

## 5. 评测

评测统一入口：

- `bash mainline.sh eval-fm ...`
- `bash mainline.sh eval-fpo ...`

### 5.1 评测 FM

```bash
cd /path/to/RoboTwin
bash mainline.sh eval-fm handover_block_native_smoke 1 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --ckpt-path /ABS/PATH/TO/FM.ckpt \
  --episodes 20 \
  --max-episode-steps 800 \
  -- --sim-backend physx_cuda --render-backend sapien_cuda
```

### 5.2 评测 FPO

```bash
cd /path/to/RoboTwin
bash mainline.sh eval-fpo handover_block_native_smoke 1 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --ckpt-path /ABS/PATH/TO/FPO.ckpt \
  --episodes 20 \
  --max-episode-steps 800 \
  -- --sim-backend physx_cuda --render-backend sapien_cuda
```

### 5.3 评测视频保存位置

如果各算法目录下的 `maniskill_eval.yaml` 打开了视频保存，视频会默认写到：

```text
eval_videos/<algo>/<task_label>/<ckpt_parent>-<ckpt_stem>/
```

例如：

```text
eval_videos/flowmatching/handover_block_native_smoke/<ckpt_parent>-<ckpt_stem>/episode0.mp4
```

## 当前推荐的完整工作流

如果你要从零开始复现实验，建议按这个顺序走：

1. 用 `scripted_expert` 采集 native raw episode
2. 处理成 FM / FPO 各自的 zarr
3. 用 FM 先做离线训练
4. 用 FPO 接着做在线训练
5. 用 `eval-fm` / `eval-fpo` 评测并录视频

对应命令链大致是：

```bash
python script/collect_maniskill_data.py ...
bash mainline.sh process fm ...
bash mainline.sh process fpo ...
bash mainline.sh train-fm ...
bash mainline.sh train-fpo ...
bash mainline.sh eval-fm ...
bash mainline.sh eval-fpo ...
```

## 如果你要自己改功能，应该改哪里

按职责分可以这样找文件：

- 改采集 CLI、seed 遍历方式、保存逻辑
  - [`script/collect_maniskill_data.py`](script/collect_maniskill_data.py)
- 改 handover 任务阶段顺序、waypoint、handover 行为
  - [`sim/maniskill/expert/handover_block_solver.py`](sim/maniskill/expert/handover_block_solver.py)
- 改 motion primitive、规划 fallback、开合夹爪、撤离逻辑
  - [`sim/maniskill/expert/primitives.py`](sim/maniskill/expert/primitives.py)
- 改成功判定、target 容差、是否必须撤离 box
  - [`sim/maniskill/tasks/handover_block_env.py`](sim/maniskill/tasks/handover_block_env.py)
- 改 raw `.npz` 落盘内容
  - [`sim/maniskill/trajectory_recorder.py`](sim/maniskill/trajectory_recorder.py)
- 改默认 profile 和 smoke 数据路径
  - [`sim/maniskill/task_profiles.py`](sim/maniskill/task_profiles.py)

## 当前已知限制

这几个限制建议在开始前就知道：

- `scripted_expert` 目前请直接调用 [`script/collect_maniskill_data.py`](script/collect_maniskill_data.py)
  - 还没有完整接回 `mainline.sh collect`
- `handover_block_native_smoke` 默认只指向 smoke zarr
  - 训练真实模型时请显式覆盖 `task.dataset.zarr_path`
- `scripted_expert` 当前更推荐 `physx_cpu`
- 目前分支的重点是 ManiSkill 主线，不是完全复刻上游 RoboTwin 的全部运行方式

## 参考

- 上游项目说明： [README.md](README.md)
- ManiSkill 主线入口： [mainline.sh](mainline.sh)
- 采集入口： [script/collect_maniskill_data.py](script/collect_maniskill_data.py)
