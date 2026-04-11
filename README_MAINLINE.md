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
- `sim/maniskill/bimanual_embodiment.py`
  - RoboTwin2.0 风格的双臂 embodiment 加载层
- `sim/maniskill/tasks/`
  - RoboTwin 自定义 ManiSkill env 注册入口
  - `BaseBimanualEnv` 负责通用双臂 env 骨架
  - `DualUR5WSG-v1` 只是一个默认 `UR5-WSG` 验证实例
  - `HandoverBlock-v1` 是第一个真实 RoboTwin 任务 env
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

## 推荐入口

优先使用根目录统一入口：

```bash
bash mainline.sh <command> ...
```

这里的设计原则是：

- `mainline.sh` 只负责把命令转发到现有脚本
- 训练 / 评估 / 数据处理参数仍然以命令行显式参数为准
- `task_profile` 目前只作为 ManiSkill 侧的辅助信息和一致性提示，不会替你自动补齐那串运行参数
- 如果你后面要接自己的任务，可以在 shell 命令末尾加 `--`，把额外 override 原样透传给底层入口
- 为了减少和 RoboTwin 现有使用习惯的差异，外部配置和命令仍然可以继续写 `action_type`
- 但在 ManiSkill 内部，这一层会被解释成官方文档里的 `control_mode` 兼容别名

目前支持：

- `collect`
- `process-fm`
- `process-fpo`
- `train-fm`
- `train-fpo`
- `eval-fm`
- `eval-fpo`

### 1. ManiSkill 采集

根目录入口：

```bash
bash mainline.sh collect PickCube-v1 smoke_npz 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --episode-num 1 \
  --max-episode-steps 5
```

会落到：

```text
data/PickCube-v1/smoke_npz/data/episode0.npz
```

根目录的 [collect_data.sh](/home/idiot/zhj/RoboTwin/collect_data.sh) 现在也已经改成对当前工作目录不敏感，可以从仓库外部目录直接调用。

### 2. ManiSkill 转 zarr

Flow Matching：

```bash
bash mainline.sh process-fm PickCube-v1 smoke_npz 1 --backend maniskill
```

FPO：

```bash
bash mainline.sh process-fpo PickCube-v1 smoke_npz 1 --backend maniskill
```

### 3. ManiSkill Flow Matching smoke

```bash
bash mainline.sh train-fm 42 0 \
  --backend maniskill \
  --task-profile pickcube_smoke
```

默认配置：

- `policy/FlowMatching/diffusion_policy/config/maniskill_fm.yaml`

这里的顶层算法配置现在是通用命名，`pickcube` 只保留在 task yaml 里。

如果你要显式覆盖 task/config 里的字段，可以这样写：

```bash
bash mainline.sh train-fm 42 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  -- task.name=my_task task.dataset.zarr_path=data/my_task.zarr exp_name=my_exp
```

### 4. ManiSkill FPO smoke

```bash
bash mainline.sh train-fpo 42 0 \
  --backend maniskill \
  --task-profile pickcube_smoke
```

默认配置：

- `policy/FPO/diffusion_policy/config/maniskill_fpo.yaml`

同样支持透传 Hydra override：

```bash
bash mainline.sh train-fpo 42 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  -- training.num_envs=1 task.name=my_task
```

### 5. ManiSkill FM -> FPO finetune smoke

```bash
bash mainline.sh train-fpo 42 0 /ABS/PATH/TO/FM.ckpt \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --config-name maniskill_fpo_finetune
```

### 6. ManiSkill 评估

Flow Matching：

```bash
bash mainline.sh eval-fm pickcube_smoke 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --ckpt-path /ABS/PATH/TO/FM.ckpt \
  --episodes 1 \
  --max-episode-steps 10
```

FPO：

```bash
bash mainline.sh eval-fpo pickcube_smoke 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --ckpt-path /ABS/PATH/TO/FPO.ckpt \
  --episodes 1 \
  --max-episode-steps 10
```

如果你要给评估入口继续传额外参数，也可以在末尾加 `--`：

```bash
bash mainline.sh eval-fm pickcube_smoke 0 \
  --backend maniskill \
  --task-profile pickcube_smoke \
  --ckpt-path /ABS/PATH/TO/FM.ckpt \
  -- --render-mode human
```

## 自定义 ManiSkill 任务

当前默认的 ManiSkill task 文件虽然内容还是 `pickcube_smoke`，但文件名已经改成通用的 `maniskill_task.yaml`，不代表仓库以后只能用 `PickCube-v1`。

当前推荐的组织方式是：

- 顶层算法配置使用通用名字
  - `maniskill_fm`
  - `maniskill_fpo`
  - `maniskill_fpo_finetune`
- 具体任务差异放在 task yaml 里

如果你要接自己的 ManiSkill 任务，最小做法是：

1. 复制一份 task yaml
   - 例如从：
     - `policy/FlowMatching/diffusion_policy/config/task/maniskill_task.yaml`
     - `policy/FPO/diffusion_policy/config/task/maniskill_task.yaml`
   - 分别复制成你自己的 task 文件

2. 改这几项任务描述
   - `name`
   - `maniskill.profile`
   - `maniskill.env_id`
   - `maniskill.zarr_path`
   - `shape_meta.obs.agent_pos.shape`
   - `shape_meta.action.shape`

3. 保持 ManiSkill 运行参数和你的任务一致
   - `action_type`
   - `obs_mode`
   - `reward_mode`
   - `robot_uids`
   - `sim_freq`
   - `control_freq`

4. 用 shell 入口透传 override 切到你的 task

Flow Matching：

```bash
bash mainline.sh train-fm 42 0 \
  --backend maniskill \
  -- task=my_custom_maniskill_task exp_name=my_exp
```

FPO：

```bash
bash mainline.sh train-fpo 42 0 \
  --backend maniskill \
  -- task=my_custom_maniskill_task exp_name=my_exp
```

说明：

- ManiSkill 运行时创建环境时，当前主线会优先使用 `task.maniskill.env_id`。这更符合 ManiSkill 原生用法。
- `task.name` 可以继续作为你自己的训练任务别名，不必强行等于 env id。
- `maniskill.profile` 现在是可选的默认参数模板层，不是创建 ManiSkill 环境的必需字段。
- 你可以继续沿用当前仓库里的 `action_type` 写法；只有在直接参考 ManiSkill 官方文档时，记住它在本仓库内部对应的是 `control_mode`。
- collect / eval 的 ManiSkill 专用脚本现在也和 FPO 在线链保持一致，统一按 `env_id > profile > task_name` 的顺序解析任务身份。
- FM 这边主要由 `shape_meta` 驱动。
- FPO 这边的 critic 输入维度现在也会根据 `shape_meta` 自动推导，不再写死在 `pickcube_smoke` 的 9 维 `agent_pos` 上。
- 也就是说，只要你的 task yaml 里的 observation / action 维度写对，后面这两条主线都能跟着适配。

## HandoverBlock 配置入口

当前已经新增了 `handover_block_smoke` 的配置入口：

- ManiSkill profile:
  - `sim/maniskill/task_profiles.py`
- FM task:
  - `policy/FlowMatching/diffusion_policy/config/task/maniskill_handover_block_task.yaml`
- FPO task:
  - `policy/FPO/diffusion_policy/config/task/maniskill_handover_block_task.yaml`

当前约定是：

- `env_id: HandoverBlock-v1`
- `agent_pos: [14]`
- `action: [14]`
- `action_type: continuous`
- `robot_uids: none`

这里的 `continuous` 是刻意保守的写法。因为 `HandoverBlock-v1` 目前走的是仓库内自定义双臂动作接口，不是 ManiSkill 官方 Panda controller 路径。

你现在可以直接通过 override 切到它：

Flow Matching：

```bash
bash mainline.sh train-fm 42 0 \
  --backend maniskill \
  -- task=maniskill_handover_block_task exp_name=handover_block_smoke
```

FPO：

```bash
bash mainline.sh train-fpo 42 0 \
  --backend maniskill \
  -- task=maniskill_handover_block_task exp_name=handover_block_smoke
```

另外现在单独增加了一套 `handover_block_native_smoke` 配置入口，用来走 ManiSkill 原生 `collect -> process -> train` 数据链：

- ManiSkill profile:
  - `sim/maniskill/task_profiles.py`
- FM task:
  - `policy/FlowMatching/diffusion_policy/config/task/maniskill_handover_block_native_task.yaml`
- FPO task:
  - `policy/FPO/diffusion_policy/config/task/maniskill_handover_block_native_task.yaml`

这套 native smoke 的默认 zarr 路径是：

- `data/HandoverBlock-v1-smoke_npz-1.zarr`

推荐命令模板：

collect：

```bash
bash mainline.sh collect HandoverBlock-v1 smoke_npz 1 \
  --backend maniskill \
  --task-profile handover_block_native_smoke \
  --episode-num 1 \
  --max-episode-steps 50 \
  --sim-backend physx_cuda \
  --render-backend sapien_cuda
```

process：

```bash
bash mainline.sh process-fm HandoverBlock-v1 smoke_npz 1 --backend maniskill
```

Flow Matching：

```bash
bash mainline.sh train-fm 42 0 \
  --backend maniskill \
  -- task=maniskill_handover_block_native_task exp_name=handover_block_native_smoke
```

FPO：

```bash
bash mainline.sh train-fpo 42 0 \
  --backend maniskill \
  -- task=maniskill_handover_block_native_task exp_name=handover_block_native_smoke
```

评估也可以显式控制 ManiSkill backend：

```bash
bash mainline.sh eval-fpo handover_block_smoke 1 \
  --backend maniskill \
  --task-profile handover_block_smoke \
  --ckpt-path /ABS/PATH/TO/FPO.ckpt \
  --episodes 1 \
  --max-episode-steps 50 \
  -- --sim-backend physx_cuda --render-backend sapien_cuda
```

这里 ManiSkill 评估入口现在推荐的 2 个位置参数是：

```bash
task_name gpu_id
```

如果还想显式指定评估 seed，可以继续传第 3 个位置参数：

```bash
task_name gpu_id seed
```

旧的 legacy 兼容写法仍然保留：

```bash
task_name task_config ckpt_setting expert_data_num gpu_id [seed]
```

如果你要显式控制 ManiSkill 环境 backend，现在也可以直接透传：

```bash
bash mainline.sh train-fpo 42 1 \
  --backend maniskill \
  -- task=maniskill_handover_block_task \
     training.sim_backend=physx_cuda \
     training.render_backend=sapien_cuda \
     exp_name=handover_block_smoke
```

这里 ManiSkill 训练入口现在推荐的 2 个位置参数是：

```bash
seed gpu_id
```

如果需要继续传恢复训练或 finetune 的 checkpoint，可以追加第 3 个位置参数：

```bash
seed gpu_id resume_ckpt_path
```

旧的 legacy 兼容写法仍然保留：

```bash
task_name task_config expert_data_num seed action_dim gpu_id [resume_ckpt_path]
```

说明：

- `training.device` 控制的是 PyTorch 训练设备
- `training.sim_backend` / `training.render_backend` 控制的是 ManiSkill 环境创建 backend
- 这两层现在已经在 FPO 的 ManiSkill single/vector 路径里显式打通

当前两套 handover 配置的区别是：

- `maniskill_handover_block_task`
  - 默认吃现有 legacy zarr：`data/handover_block-demo_clean-50.zarr`
- `maniskill_handover_block_native_task`
  - 默认吃 ManiSkill 原生 collect/process 生成的 zarr：`data/HandoverBlock-v1-smoke_npz-1.zarr`

现在 `train-fm` / `train-fpo` 在使用 `task=...` 切到 handover task 时，自动 `process` 也会优先跟随 task yaml 里的默认 zarr 路径，而不是继续按位置参数里的 `PickCube-v1` 去猜。

## 当前边界

- `PickCube-v1` 目前只是 ManiSkill smoke task，不是正式研究任务
- 训练、评估、数据链已经能走 ManiSkill 主线
- RoboTwin 相关脚本和运行时仍然在仓库内，但不应继续作为主依赖
- `DualUR5WSG-v1` 目前只是 embodiment 验证 env
  - 它用来确认 RoboTwin 的双臂 embodiment 能否在 ManiSkill 的固定频率 / GPU 并行语义下落地
  - 当前默认实例是 `UR5-WSG + UR5-WSG`
- `HandoverBlock-v1` 目前只迁移了场景布局、reset 分布和空间成功条件
  - 还没有迁移 RoboTwin 的 expert script
  - 当前已经补了最小 `14` 维动作接口：左臂 `6 + 1`、右臂 `6 + 1`
  - 臂动作现在按 RoboTwin 现有数据语义解释为绝对 joint qpos 目标
  - gripper 动作是 `[0,1]` 归一化开合目标
  - `is_right_gripper_open()` 已经接回 ManiSkill success 判定
  - batched env 现在会按 env 维度分别维护 gripper 状态，不再把并行环境压成一个全局标量
  - 当前也已经开通 `pointcloud+state_dict` 观测模式，用来对接 ManiSkill 主线数据链
  - 当前 `agent.qpos` 也已经对齐成 `14` 维：左臂 `6` + 左 gripper `1` + 右臂 `6` + 右 gripper `1`
  - 它还不是 `HandoverBlock-v1`，也没有迁移 RoboTwin 的脚本式 handover 逻辑
  - 任务逻辑和 embodiment 现在开始重新分层，对齐 RoboTwin2.0 原本结构

## 后续建议

- 继续把正式研究任务迁到 ManiSkill
- RoboTwin 只保留 legacy 对照链
- 之后再考虑 Isaac Lab / MuJoCo 的第二层 simulator profile
