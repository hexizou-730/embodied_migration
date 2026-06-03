# 实验进度报告：LLM 机器人程序迁移

更新时间：2026-06-03
当前阶段：`PullCube-v1` 已完成 Panda → xArm6 自动迁移成功案例；`PickCube-v1` 已验证 Panda 源端 baseline 成功，但 Panda → xArm6 迁移仍是 hard case，用于分析真实抓取迁移的瓶颈。
报告用途：组会汇报 + 后续 workshop 论文实验记录。

## 0. 组会速览

### 0.1 一句话进展

本项目已经从简单高层代码迁移推进到真实 ManiSkill 控制迁移：DeepSeek V4-Pro 已能为 xArm6 自动生成 `PullCube-v1` 目标 adapter 并成功执行；`PickCube-v1` 的 Panda source baseline 也已验证成功，但迁移到 xArm6 后 LLM adapter generation 仍无法稳定形成 force-closure grasp，说明抓取迁移需要结构化物理探针和更深层接触/控制建模。

### 0.2 当前实验结果对比

| Case | 任务 | 迁移方向 | 结果 | 关键证据 | 结论 |
|---|---|---|---|---|---|
| Case 02 | `PullCube-v1` 接触拖拽 | Panda → xArm6 | 成功 | `target_success=True`, `ret_val=True`, `elapsed_steps=460` | LLM 可以生成可执行 xArm6 target adapter |
| Case 03 | `PickCube-v1` 真实抓取 | Panda → xArm6 | Panda 源端成功；xArm6 未成功，作为 hard case | Panda baseline `ret_val=True`, `elapsed_steps=40`; xArm6 probe 32 组参数均 `is_grasping=False`; 最近失败 `tcp_grasp_z=0.0820` | 源程序有效，但 prompt-only target adapter synthesis 对 force-closure grasp 不够 |
| Case 01 | `PullCube-v1` | Panda → Fetch | 保留为诊断失败 | 9D action space、移动底盘、接触侧可达性问题 | embodiment 差异会造成非高层代码层面的失败 |

### 0.3 PullCube 成功说明了什么

`PullCube-v1` 的高层程序不变：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")
ret_val = robot.pull(cube, goal)
```

LLM 生成的是目标端 adapter，不是改高层任务代码。成功说明：对于接触拖拽类任务，LLM 能根据失败日志和 embodiment 约束修改 target adapter 的动作空间映射、接触侧、拖拽策略和执行参数，并通过真实 ManiSkill `env.step(action)` 达到成功。

### 0.4 PickCube 为什么定义为 hard case

`PickCube-v1` 的高层程序同样保持不变：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")

grasp_ok = robot.grasp(cube)
ret_val = robot.place(cube, goal) if grasp_ok else False
```

Panda baseline 已验证成功：

```text
robot_uid = panda
method = source-copy
success = true
message = ret_val=True
elapsed_steps = 40
is_grasped = true
is_obj_placed = true
```

因此 PickCube 的问题不是源端程序错误，而是目标机器人迁移后的真实抓取执行问题。当前失败已经从“代码格式错误”逐步定位到物理执行层：

| 阶段 | 观察 | 含义 |
|---|---|---|
| 早期失败 | 方块被撞飞或掉落 | approach/descent 策略破坏性太强 |
| 中期失败 | `tcp_grasp_xy/z` 很小但 `is_grasping=False` | 手到位但夹爪包络没有形成抓取 |
| 侧推失败 | `cube_disp_xy` 接近或超过 3 cm | 闭爪时把方块横向挤走 |
| 最新失败 | `tcp_grasp_z=0.0820` | adapter 没下降到有效抓取高度就进入失败判断 |

因此 PickCube 不是简单“再调 prompt 就能好”的问题，而是 force-closure grasp 迁移瓶颈。

### 0.5 自动探针结果

为避免无限 prompt，已加入自动抓取参数探针：

```bash
python scripts/xarm6_pick_grasp_probe.py \
  --sim-backend auto \
  --render-backend gpu
```

探针固定 XY，枚举少量：

```text
grasp_z_offset / close_steps / close_command / settle_steps
```

结果：

```text
total_probe_cases = 32
grasping_cases = 0
best_probe_case:
  grasp_z_offset=0.016
  close_steps=12
  close_command=-0.6
  cube_disp_xy=0.00458
  tcp_grasp_xy=0.00239
  tcp_grasp_z=0.00152
  is_grasping_after_close=False
```

结论：probe 没有给出成功答案，但证明简单 close-envelope 参数调节不足。后续 LLM 需要处理更结构性的 grasp primitive，而不是继续堆更多 z-offset/close 参数。

### 0.6 当前可讲的研究结论

```text
LLM-generated adapter migration works for contact-based manipulation
but exposes clear limits on real force-closure grasping.
Structured physical probing helps diagnose the failure space,
but robust grasp transfer likely needs constraint-aware repair and
deeper controller/contact modeling.
```

中文表述：

```text
LLM 可以迁移接触拖拽类机器人程序；
但在真实抓取任务中，仅靠 prompt 和 adapter 参数搜索不够。
结构化探针能把失败分解为可解释约束，
下一步应把 LLM 迁移和约束处理 / 学习引导优化结合起来。
```

### 0.7 下一阶段计划

| 优先级 | 内容 | 目的 |
|---|---|---|
| 高 | 固定当前 prompt，不再无限手工追日志 | 收束实验，避免调参化 |
| 高 | 跑 PullCube 多 seed，统计成功率 | 形成主实验正结果 |
| 中 | 将 PickCube 作为 hard case，整理失败类型 | 形成负结果和研究动机 |
| 中 | 扩展 probe 为 constraint-aware repair | 与 constraint handling / learning-guided optimization 方向对齐 |
| 低 | 后续再考虑 Fetch 或其他机器人 | 不分散当前主线 |

## 1. 项目目标

本项目研究的问题是：**同一个高层机器人程序能否从源机器人迁移到目标机器人，并在目标机器人上真实执行成功。**

当前重点不再是简单的 PyBullet 方块放盘子任务，而是转向 ManiSkill 中更接近真实控制问题的接触任务：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")

ret_val = robot.pull(cube, goal)
```

这个程序在高层看起来不复杂，但底层执行涉及：

- 机器人动作空间差异
- 控制器接口差异
- TCP 接触位置
- 移动底盘与机械臂协调
- 接触几何和可达性
- 失败类型判定

因此它适合作为“代码迁移不是简单重生成代码，而是跨 embodiment 的执行适配问题”的核心案例。

## 2. 当前实验设置

### 2.1 仿真平台

当前使用：

- 仿真环境：ManiSkill
- 任务：`PullCube-v1`
- 观测模式：`state`
- 控制模式：`pd_ee_delta_pos`
- 运行方式：远程服务器无 GUI 运行
- 渲染：远程 GUI 不可用，实验主要使用状态量和日志判断

### 2.2 远程运行平台

当前主要实验在远程 Linux 平台运行：

```text
远程主机：rotule
远程项目路径：~/Embodied/embodied_migration
Conda 环境：em-ms
命令行提示符示例：(em-ms) [rotule embodied_migration]$
```

远程连接方式：

```bash
ssh hexi.zou@rotule.polytechnique.fr
```

进入项目目录：

```bash
cd ~/Embodied/embodied_migration
conda activate em-ms
```

当前远程 GPU 信息：

```text
NVIDIA-SMI 595.80
Driver Version: 595.80
CUDA Version: 13.2
GPU: NVIDIA RTX 4000 Ada Generation
GPU Memory: 20475 MiB
```

`nvidia-smi` 输出中可见 GPU 对远程环境可用，因此当前实验可以使用远程 GPU / CUDA 环境运行 ManiSkill headless 仿真。

远程 GUI 当前报错：

```text
RuntimeError('Create window failed: Renderer does not support display.')
```

因此目前采用 headless 实验是合理的，后续如果需要演示视频，可以在本地可视化环境或支持显示的机器上补录。

### 2.3 LLM 调用设置

当前主实验通过 DeepSeek 官方 API 调用 LLM。

主要使用模型：

```text
deepseek-v4-pro
```

报告中简称为：

```text
DeepSeek V4-Pro
```

早期实验曾通过 OpenRouter 使用 `anthropic/claude-opus-4.6`。在 OpenRouter key 限额不足后，主实验切换为 DeepSeek 官方 API，并设置：

```text
provider = deepseek
model = deepseek-v4-pro
thinking = disabled
```

用途：

- 根据源程序、机器人 profile、capability card 和失败日志生成目标 adapter
- 分析迁移失败层级
- 总结 Panda → Fetch 的执行假设差异
- 生成 migration analysis

需要注意：

**LLM 不是直接控制机器人成功，而是生成或修改 adapter 代码。最终是否成功仍然由 ManiSkill 的真实 `env.step(action)` 执行和环境 `evaluate()` 判定。**

### 2.4 当前机器人

当前保留一个源机器人和两个目标机器人：

| 角色 | 机器人 | 说明 |
|---|---|---|
| Source | Panda | 源机器人，固定机械臂，当前任务可成功 |
| Target success case | xarm6_robotiq | 固定机械臂，DeepSeek 已自动生成成功迁移 adapter |
| Target failure case | Fetch | 移动底盘 + 机械臂，迁移后当前失败 |

### 2.5 当前任务

任务名称：`PullCube-v1`
中文解释：把方块拉/推到目标区域。

seed 0 下的关键位置：

```text
cube_pos = [-0.0007, 0.0536, 0.0200]
goal_pos = [-0.2007, 0.0536, 0.0010]
```

也就是说，方块需要大约沿 `-x` 方向移动 20 cm。

## 3. 当前代码迁移框架

目前项目中把一次机器人程序执行分为三层：

### 3.1 高层程序层

高层程序保持不变：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")
ret_val = robot.pull(cube, goal)
```

这一层体现用户或 LLM 生成的任务程序。

### 3.2 Skill Adapter 层

`robot.pull(cube, goal)` 不直接等于成功执行，它需要被 adapter 翻译为真实控制动作。

例如：

- Panda 的 `pull` 可以直接用固定机械臂接触并拖拽
- Fetch 的 `pull` 需要考虑移动底盘、9 维动作空间、接触侧选择和 TCP 几何

因此真正的迁移难点主要发生在 adapter 层。

### 3.3 Controller / Contact 层

目标机器人最终执行的是：

```python
env.step(action)
```

Fetch 的动作空间为 9 维：

```text
action_space = Box(-1.0, 1.0, (9,), float32)
layout = [arm_xyz(3), gripper(1), body(3), base(2)]
```

也就是：

```text
action[0:3] = 机械臂末端 delta xyz
action[3]   = gripper
action[4:7] = body / torso / head
action[7:9] = mobile base
```

这说明 Panda 的 4 维动作假设不能直接迁移到 Fetch。

## 4. 已完成实验

### 4.1 Panda 源端实验

运行 Panda 源端程序：

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube \
  --robot panda \
  --method source-copy \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 100
```

结果：

```text
success = true
failure_type = success
message = ret_val=True
```

结论：

**Panda 可以成功完成 `PullCube-v1`。源程序和源 adapter 是有效的。**

### 4.2 Fetch 直接迁移实验

将同一个高层程序直接迁移到 Fetch：

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube \
  --robot fetch \
  --method source-copy \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 100
```

首次失败：

```text
RuntimeError:
PullCube adapter expects action_space last dim in {4, 7}, got shape (9,)
```

结论：

**直接复制 Panda 的 adapter 到 Fetch 会失败，因为 Fetch 的控制器动作空间不同。**

失败层级：

```text
program 层：没有问题
adapter / controller interface 层：失败
```

### 4.3 LLM 目标 adapter 生成实验

之后使用 OpenRouter 调用更强的 LLM，例如：

```text
anthropic/claude-opus-4.6
```

让 LLM 根据失败日志生成 Fetch 目标 adapter。

LLM 已经尝试过的改动包括：

- 识别 Fetch 9 维动作空间
- 重写 `_validate_action_space`
- 重写 `_make_action`
- 增加 mobile base 控制
- 调整接触偏移
- 增加接触重试
- 增加步数预算
- 加入 base 逼近策略
- 加入 contact geometry 分析

但是 Fetch 仍未成功。

典型失败结果：

```text
success = false
failure_type = contact execution failure
cube_goal_xy = 0.2000m
tcp_cube_xy 仍然较大
cube 基本没有移动
```

结论：

**LLM 可以提出跨层修改，但目前没有自动找到可成功执行的 Fetch adapter。**

这对论文有价值：说明 program-only generation 不足，需要分析 adapter / controller / contact 层。

### 4.4 手写 Fetch oracle adapter 实验

为了验证问题是否只是 LLM 没写好，我们手写了一个最小 Fetch oracle adapter。

该 adapter 做了：

- 支持 Fetch 9 维动作空间
- 使用 `base[7] > 0` 让移动底盘靠近 cube
- 停止底盘
- 使用机械臂下降接触
- 尝试拖拽 cube 到 goal

运行结果：

```text
success = false
cube_goal_xy = 0.4101
tcp_cube_xy = 0.4615
cube_pos = [0.2093, 0.0555, 0.0200]
goal_pos = [-0.2007, 0.0536, 0.0010]
```

关键现象：

**cube 被推到了 `+x` 方向，而目标在 `-x` 方向。**

这说明 Fetch 当前建立接触的方向是错的。

### 4.5 xarm6_robotiq 目标迁移实验

在 Fetch 被记录为失败案例后，新增目标机器人：

```text
xarm6_robotiq
```

对应迁移 case：

```text
case02_pull_cube_panda_to_xarm6
```

运行命令：

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

使用模型：

```text
anthropic/claude-opus-4.6
```

实验结果：

```text
最终结果：失败
迭代次数：3
主要失败层：skill_adapter / contact geometry
```

xarm6_robotiq 和 Panda 的关键差异：

| 项目 | Panda | xarm6_robotiq |
|---|---|---|
| DoF | 7 | 6 |
| 动作空间 | 4维 | 4维 |
| 夹爪 | Panda parallel jaw | Robotiq 多指夹爪 |
| 底盘 | 固定 | 固定 |
| 主要迁移难点 | 原始成功 | 接触侧选择 + 接触维持 |

LLM 生成的 adapter 主要修改：

- 动作空间诊断确认：xarm6_robotiq 在 `pd_ee_delta_pos` 下实际为 4 维；
- 控制器布局：`action[0:3]` 是 arm delta xyz，`action[3]` 是 active gripper；
- 增加运动步数，减小单步最大位移；
- 调整接触偏移：`contact_x_offset` 和 `contact_z_offset`；
- 增加持续下压力；
- 使用多组接触候选参数重试；
- 增加按压阶段，试图让末端执行器和方块建立稳定接触。

失败证据：

```text
R2: cube_goal_xy = 0.2000m, tcp_cube_xy = 0.1024m
R3: cube_goal_xy = 0.1488m, tcp_cube_xy = 0.0896m
```

解释：

- R2 中方块基本没有移动，说明接触没有有效建立；
- R3 中方块移动了约 5 cm，说明 xarm6 已经能产生部分有效接触；
- 但目标需要约 20 cm 的移动，当前只完成约 25%；
- TCP 与 cube 的 xy 距离仍在约 9 cm，接触保持不够稳定；
- 目前不是 Fetch 那种“完全无法到正确接触侧”的不可行问题，而是接触-拖拽控制不足。

阶段性结论：

**Panda → xarm6_robotiq 比 Panda → Fetch 更接近成功。最初的 LLM-generated adapter 仍未完成任务，失败主要来自 contact primitive / skill adapter 层，因此后续针对 xarm6 做了更明确的接触诊断和手写 oracle adapter。**

### 4.6 xarm6 诊断实验与 oracle 轨迹

随后运行专门的 xarm6 诊断脚本：

```bash
python scripts/xarm6_pull_diagnostics.py \
  --seed 0 \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

诊断确认 xarm6 的真实控制器为：

```text
action_space = Box(-1.0, 1.0, (4,), float32)
arm: action[0:3]
gripper_active: action[3]
```

最重要的成功轨迹：

```text
case: x_plus_down_drag_x_minus
x_plus:       (0.8, 0.0, 0.0), steps=100
down:         (0.0, 0.0, -0.8), steps=80
drag_x_minus: (-0.8, 0.0, -0.05), steps=160
success = true
```

成功后的关键指标：

```text
cube_dx = -0.1099
cube_goal_xy = 0.0905
tcp_cube_xy = 0.0322
success = true
```

这说明：

- xarm6 可以到达正确接触侧；
- xarm6 可以把 cube 往 `-x` 目标方向拖动；
- 原先 LLM 生成的 closed-loop waypoint adapter 失败，不是因为机器人不可行，而是因为接触轨迹设计不合适；
- 直接 raw contact sequence 反而成功。

因此保留一个最小 oracle adapter 作为内部可行性证据：复现 `x_plus → down → drag_x_minus` 成功轨迹，并仍然使用真实 `env.step(action)` 和 ManiSkill success 判断。该 oracle 不作为 LLM 自动迁移结果。

### 4.7 xarm6 oracle adapter 真实执行成功

将诊断得到的成功轨迹独立保存在：

```text
maniskill_backend/oracles/xarm6_pull_cube_oracle.py
```

随后在远程服务器运行真实 xarm6 oracle：

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube \
  --robot xarm6_robotiq \
  --method target-module-generation \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500 \
  --code-file maniskill_backend/case_programs/case01_pull_cube.py \
  --adapter-module maniskill_backend.oracles.xarm6_pull_cube_oracle
```

最终输出：

```text
success = true
failure_type = success
failure_layer = success
message = ret_val=True
elapsed_steps = 191
```

执行日志：

```text
api = pull
args = {"obj": "cube", "target": "goal", "oracle": true}
result = true
ok = true
```

结论：

**Panda → xarm6_robotiq 当前已证明物理上可行，但还不能称为 LLM 自动迁移成功案例。**

这个成功不是通过修改高层程序得到的，高层程序仍然是：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")
ret_val = robot.pull(cube, goal)
```

人工验证过的目标机器人 adapter 变化是：

```text
Panda 的 closed-loop contact pull
→ xarm6 的 measured raw contact sequence
→ x_plus → down → drag_x_minus
```

这说明同一个 LMP 程序可以保持不变，但不同 embodiment 需要不同的执行适配层。下一步需要验证 LLM 能否自动生成这样的目标执行适配层。

### 4.8 当前主线：LLM 自动生成 xarm6 迁移代码

需要特别区分：

```text
xarm6 oracle adapter 成功 != LLM 自动迁移成功
```

当前 xarm6 成功 oracle 是根据诊断结果手写的，因此只能作为：

- 可行性上限；
- 成功 contact primitive 证据；
- 后续对比的 oracle upper bound。

为了优先完成最主要的问题，当前只保留一个 xarm6 主实验：

```text
case02_pull_cube_panda_to_xarm6
```

`case02` 从非 oracle 的 waypoint seed adapter 开始。Prompt 允许给出：

```text
真实 action space：arm[0:3] + gripper[3]；
从 cube 运动方向相反的一侧建立接触；
下降到接触高度；
保持轻微下压力；
沿 goal 方向拖拽。
```

Prompt 不会给出：

```text
人工 oracle 的具体动作数值；
人工 oracle 的具体 step 数；
完整成功轨迹；
oracle 文件内容。
```

这样既不是让 LLM 在完全没有物理先验的情况下盲试，也没有把人工答案直接泄露给模型。如果 `case02` 成功，可以写：

```text
LLM generated a successful target adapter from embodiment constraints,
an abstract contact strategy, and real execution feedback.
```

远程运行命令为：

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

### 4.9 最新 Opus 4.6 自动生成实验：反向推动失败

在修复安全校验误报、限制 LLM 输出 token 并启用模块快照后，Opus 4.6 生成的三轮 adapter 均通过单元测试并进入真实 ManiSkill 仿真。

结果：

| Round | `cube_goal_xy` | cube 最终 x | 结论 |
|---|---:|---:|---|
| R1 | `0.3124m` | `+0.1096m` | 方块被推向错误的 `+x` 方向 |
| R2 | `0.5107m` | `+0.3057m` | 在错误方向上继续累积 |
| R3 | `0.5806m` | `+0.3724m` | TCP 与 cube 脱离，任务进一步恶化 |

本轮说明：

- LLM 已经能够生成可执行、可测试、可进入真实仿真的目标 adapter；
- 失败不再来自代码格式、安全校验或动作空间误判；
- 当前主要问题是 contact primitive 的方向语义：`+x` 只能用于接触前绕到 cube 右侧，建立接触后必须沿 `-x` 拖拽；
- Prompt 将增加反向进展守卫：若 cube x 或 `cube_goal_xy` 增加，应立即停止当前接触尝试。

### 4.10 DeepSeek V4-Pro 自动生成实验：接触脉冲过弱

在 OpenRouter key 限额不足后，主实验切换到 DeepSeek 官方 API：

```text
provider = deepseek
model = deepseek-v4-pro
thinking = disabled
```

关闭 thinking mode 是为了让输出预算优先用于完整 Python adapter，而不是被思考内容消耗。

三轮生成模块均通过安全校验、单元测试和真实 ManiSkill 执行：

| Round | `cube_goal_xy` | `tcp_cube_xy` | cube 是否移动 |
|---|---:|---:|---|
| R1 | `0.2000m` | `0.0295m` | 否 |
| R2 | `0.2000m` | `0.0295m` | 否 |
| R3 | `0.2000m` | `0.0299m` | 否 |

分析 Round 3 代码后发现：

- TCP 已接近 cube，方向守卫有效，未再把 cube 推向 `+x`；
- `_drag_pulse()` 将米制位移直接传给归一化动作空间，最大脉冲约为 `0.021`；
- 接触下压力直接使用 `-0.004`；
- xarm6 的 `pd_ee_delta_pos` 控制器动作范围是 `[-1, 1]`，因此这些脉冲太弱，无法形成有效接触力。

下一轮 Prompt 增加控制语义约束：

```text
metric TCP error
→ divide by max_delta_m
→ clip to [-1, 1]
→ env.step(normalized_action)
```

持续接触脉冲应使用安全但足够明显的归一化动作，并在短脉冲后检查 cube 是否真正向 goal 移动。

### 4.11 DeepSeek V4-Pro 自动生成实验：重复 far-side 接触失败

在补充归一化动作语义后，DeepSeek 三轮生成的模块哈希完全相同：

```text
b96e049... round_01.py
b96e049... round_02.py
b96e049... round_03.py
```

真实执行结果也完全一致：

```text
cube_goal_xy = 0.2000m
tcp_cube_xy = 0.3006m
cube position unchanged
```

代码分析显示：

- 模型使用固定 `contact_x_offset_m=0.065`；
- TCP 到达该接触点后直接执行负 x 拖拽；
- 实际没有形成有效接触，TCP 从 cube 旁边扫走，最终距离扩大到约 `0.30m`；
- `temperature=0` 下，失败反馈没有促使模型改变策略，三轮生成完全相同。

下一轮增加两类约束：

1. 使用更远的正 x 侧 sweep start，再下降并沿负 x 扫过接触区域；
2. 每轮 retry 必须做实质性策略修改；若生成模块与当前失败模块完全相同，runner 直接拒绝该轮。

### 4.12 DeepSeek V4-Pro 自动生成实验：xarm6 首次成功

增加 far-side sweep、接触前检查和 retry 多样性约束后，重新运行：

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

DeepSeek V4-Pro 在第一轮生成新的 xarm6 目标 adapter，并通过真实 ManiSkill 执行验证：

```text
overall_success = True
message = target success reached

ROUND 1
module_valid = True
verification_ok = True
target_success = True
target_message = ret_val=True
final_info = {'elapsed_steps': [460], 'success': [True]}
```

生成模块快照：

```text
results/generated_modules/case02_pull_cube_panda_to_xarm6/round_01.py
```

本轮的关键意义：

- LMP 程序保持不变：`robot.pull(cube, goal)`；
- 成功代码不是手写 oracle，而是 DeepSeek V4-Pro 生成的完整目标 adapter；
- 目标 adapter 针对 xarm6 调整了动作空间验证、控制步数、接触搜索、接触验证、脉冲拖拽和在线进展守卫；
- 最终成功由真实 `env.step(action)` 与 ManiSkill `evaluate()` 判定，而不是由 LLM 文本声明；
- oracle adapter 仍仅作为物理可行性对照，不作为生成答案输入。

#### 4.12.1 最直白的解释

Panda 原来的执行方式比较简单：

```text
去一个固定位置接触方块
→ 默认已经碰到方块
→ 沿着目标方向连续拖动
```

这套动作直接搬到 xarm6 后会失败。原因是 xarm6 的手臂长度、关节结构和夹爪形状与 Panda 不同。即使高层命令仍然是“拉方块”，xarm6 也不一定能在同一个位置真正碰到方块。

DeepSeek V4-Pro 生成的新代码改成：

```text
尝试多个接触位置
→ 检查机械臂是否真的到达方块正确一侧
→ 短距离拖一下
→ 检查方块有没有向目标移动
→ 如果方向错了或没有效果，就换一个接触位置继续尝试
```

因此，这里的“代码迁移”不是把同一段 Panda 代码复制给 xarm6，而是保留相同的任务目标，重新生成一套适合 xarm6 的执行方法。

#### 4.12.2 成功 adapter 的代码差异

| 项目 | Panda / 初始 seed adapter | DeepSeek 生成的 xarm6 成功 adapter | 作用 |
|---|---|---|---|
| 高层命令 | `robot.pull(cube, goal)` | `robot.pull(cube, goal)` | 任务目标保持不变 |
| 接触位置 | 使用一个固定偏移量 | 搜索 `x_offsets=[0.14, 0.12, 0.10, 0.08]` 和 `z_offsets=[0.018, 0.015, 0.012, 0.008]` | 寻找 xarm6 真正可用的接触点 |
| 接触确认 | 默认到达目标点后已经接触 | 检查 `tcp[0] > cube_pos[0] + 0.03` | 确认 TCP 已经绕到方块正确一侧 |
| 拖拽方式 | 沿 waypoint 连续移动 | 使用 `_drag_pulse()` 短脉冲拖拽 | 降低一次动作过大导致接触丢失的风险 |
| 拖拽方向 | 使用固定线性路径 | 使用 `drag_dir=[-1.0, 0.0, -0.15]` | 向目标 `-x` 方向拖动，同时轻微下压维持接触 |
| 进展检查 | 主要在阶段结束后判断 success | 每次脉冲后检查 cube 是否远离 goal | 及时发现错误动作 |
| 回退策略 | 单次路径失败后结束 | 当前候选失败后换一组偏移量 | 自动尝试新的接触方案 |
| 自适应增强 | 无 | 无进展时加大拖拽幅度和下压力 | 尝试恢复有效接触 |

成功 adapter 中最重要的代码片段是：

```python
x_offsets = [0.14, 0.12, 0.10, 0.08]       # 尝试多个水平方向接触距离，单位为米
z_offsets = [0.018, 0.015, 0.012, 0.008]   # 尝试多个接触高度，单位为米

if tcp[0] <= cube_pos[0] + 0.03:
    continue  # TCP 没有到达方块右侧，放弃当前方案，尝试下一个接触点

drag_dir = np.array([-1.0, 0.0, -0.15], dtype=np.float32)  # 向左拖，同时轻微向下压
```

它表达的逻辑非常直接：

```text
先找到能真正碰到方块的位置
→ 再确认自己站在正确一侧
→ 最后向正确方向拖动
```

#### 4.12.3 逐段代码解释

下面只摘录成功 adapter 中最重要的逻辑，并在每行后增加中文注释。这里的注释用于解释报告，不修改真实实验代码。

第一段：搜索不同的接触点。

```python
x_offsets = [0.14, 0.12, 0.10, 0.08]      # 方块右侧的水平距离候选值，单位为米
z_offsets = [0.018, 0.015, 0.012, 0.008]  # TCP 接触高度候选值，单位为米

for x_off in x_offsets:                    # 依次尝试不同的水平距离
    for z_off in z_offsets:                # 对每个水平距离，再尝试不同高度
        contact = cube_pos + np.array(     # 计算本次要尝试的接触位置
            [x_off, 0.0, z_off],           # x 为方块右侧距离，y 不变，z 为接触高度
            dtype=np.float32,
        )
```

直白解释：

```text
Panda 原来只尝试一个固定位置。
xarm6 新代码最多尝试 4 × 4 = 16 个位置，
直到找到一个适合自己手臂和夹爪的位置。
```

第二段：先从上方靠近，再下降。

```python
pre_contact = contact + np.array(          # 在真正接触点的正上方设置一个预接触点
    [0.0, 0.0, 0.08],                      # 比接触点高 8 cm
    dtype=np.float32,
)

self._move_towards(                        # 先移动到方块右侧上方
    pre_contact,
    gripper=self.gripper_close,            # 夹爪保持闭合
    steps=self.move_steps,                 # 最多执行预设数量的控制步
)
self._move_towards(                        # 再从上方向下移动到接触位置
    contact,
    gripper=self.gripper_close,
    steps=self.move_steps,
)
```

直白解释：

```text
机械臂不是贴着桌面横冲过去。
它先到方块右上方，再下降，减少碰撞和错位。
```

第三段：确认 TCP 确实到达正确一侧。

```python
tcp = self._tcp_pos()                      # 读取当前 TCP，也就是夹爪末端的位置

if tcp[0] <= cube_pos[0] + 0.03:          # 如果 TCP 没有比方块更靠右至少 3 cm
    continue                               # 当前接触点无效，换下一组位置继续尝试
```

直白解释：

```text
代码不再假设“发出了移动命令就一定到位”。
它会检查夹爪是否真的绕到了方块右侧。
```

第四段：使用短脉冲向目标方向拖动。

```python
drag_dir = np.array(                       # 设置拖动方向
    [-1.0, 0.0, -0.15],                    # x 为负：向左拖；z 略小于 0：轻微向下压
    dtype=np.float32,
)

self._drag_pulse(                          # 执行一次短距离拖动
    drag_dir,
    magnitude=0.6,                         # 使用 0.6 的归一化动作幅度
    steps=6,                               # 连续执行 6 个环境步
    gripper=self.gripper_close,            # 拖动期间保持夹爪闭合
)
```

直白解释：

```text
不再一次性拖很远。
每次只拖 6 步，然后观察方块是否真的向目标移动。
```

第五段：检查方向是否正确，必要时换方案。

```python
cube_pos = self._actor_pos("cube")         # 重新读取拖动后的方块位置
cube_goal_xy = float(                      # 计算方块和目标之间的水平距离
    np.linalg.norm(goal_pos[:2] - cube_pos[:2])
)

if cube_pos[0] > prev_cube_x + 0.005:      # 如果方块反而向右移动超过 5 mm
    break                                  # 立即停止当前接触方案

if cube_goal_xy > prev_cube_goal_xy + 0.005:  # 如果方块离目标更远超过 5 mm
    break                                     # 立即停止，换下一个接触点
```

直白解释：

```text
拖错方向时，不继续浪费动作步数。
它会停下来，换一个接触位置重试。
```

第六段：如果方向正确但拖不动，则增强动作。

```python
if pulse_idx > 3 and abs(cube_pos[0] - prev_cube_x) < 0.002:
    drag_dir = np.array(                   # 拖动多次后方块仍几乎不动
        [-1.0, 0.0, -0.25],                # 增加向下分量，尝试维持接触
        dtype=np.float32,
    )
    self._drag_pulse(
        drag_dir,
        magnitude=0.8,                     # 将动作幅度从 0.6 提升到 0.8
        steps=6,
        gripper=self.gripper_close,
    )
```

直白解释：

```text
如果轻轻拖不动，就稍微加大力度，并向下多压一点。
```

整段代码可以概括为：

```text
多试几个位置
→ 从上方靠近
→ 确认真的站到方块右侧
→ 向左下方短距离拖动
→ 每次拖动后检查效果
→ 拖错了就换位置，拖不动就稍微加力
```

该案例应表述为：

```text
failure-driven target-adapter generation success
```

即：基于 embodiment 信息、抽象控制约束和历史失败诊断，由 LLM 自动生成可执行的目标机器人 adapter。

## 5. 最新诊断：Fetch 接触侧不可达

为了判断 Fetch 是不是只是 Z 轴高度不够，我们做了 Z 轴下降测试。

结果：

```text
after base
tcp = [-0.1382, 0.0000, 0.2362]
cube = [-0.0007, 0.0536, 0.0200]

step 30 tcp = [-0.0687, -0.0043, 0.0208]
tcp_cube_z = 0.0008
```

结论：

**Fetch 能降到桌面高度，Z 轴不是主要问题。**

随后测试 Fetch 能否绕到 cube 的正确接触侧。

任务目标是把 cube 往 `-x` 方向移动，因此理想接触侧应该是：

```text
tcp.x > cube.x
```

也就是 TCP 应该在 cube 的 `+x` 侧，从右侧向左推/拉。

但是多组诊断结果显示：

```text
far_side = False
```

典型结果：

```text
base_steps=40, torso=0.0
tcp  = [-0.0703, -0.0030, 0.0210]
cube = [-0.0007,  0.0536, 0.0200]
far_side = False
```

即：

- cube 在 `x ≈ -0.0007`
- Fetch TCP 最多只能到 `x ≈ -0.0195` 或 `x ≈ -0.07`
- TCP 始终在 cube 的 `-x` 侧
- TCP 没有到达 `+x` 正确接触侧

最终判断：

**Fetch 在当前 seed 0 场景下无法到达正确接触侧，因此无法把 cube 推向目标方向。**

## 6. 当前核心结论

当前最重要的结论是：

**Panda → Fetch 的 PullCube 迁移不是简单代码生成问题，而是 embodiment 改变后导致的接触侧可达性失败。**

高层程序没有错：

```python
ret_val = robot.pull(cube, goal)
```

但是目标机器人 Fetch 的几何、动作空间、移动底盘和 TCP 可达范围不同，导致它无法复现 Panda 的接触策略。

因此当前失败应记录为：

```text
failure_layer = contact_geometry + reachability
failure_type  = contact-side reachability failure
```

## 7. 对论文方向的意义

这个失败案例对论文是有价值的。

它支持以下观点：

1. 代码迁移不是简单地把同一个 LMP 程序复制到新机器人。
2. LLM 可以生成高层程序，也可以尝试生成 adapter，但仍需要底层物理约束反馈。
3. Capability card / embodiment profile 只描述能力是不够的，还需要执行时诊断。
4. 一些失败不是代码 bug，而是目标 embodiment 在当前任务几何下不可行。
5. 系统应该能判断 infeasible，而不是无限重试生成代码。

这个案例可以作为论文中的失败驱动分析案例：

```text
Panda succeeds → Fetch direct migration fails → LLM adapter migration still fails → oracle adapter confirms contact-side reachability failure
```

## 8. 当前进度总结

| 项目 | 状态 | 说明 |
|---|---|---|
| ManiSkill 环境 | 已可运行 | 远程 headless 可跑 |
| Panda PullCube | 成功 | 源端任务成立 |
| Fetch source-copy | 失败 | 动作空间 9D 不兼容 |
| Fetch LLM adapter generation | 已跑 | 能修改 adapter，但未成功 |
| Fetch oracle adapter | 已跑 | 仍失败，说明不是单纯 LLM 质量问题 |
| 接触侧诊断 | 已完成 | Fetch 无法到正确接触侧 |
| xarm6 module generation | 成功 | DeepSeek V4-Pro 第一轮生成成功目标 adapter，`elapsed_steps=460` |
| xarm6 诊断脚本 | 已跑 | 找到成功 raw contact sequence |
| xarm6 oracle adapter | 成功 | 内部可行性证据：`success=true`, `elapsed_steps=191` |
| xarm6 LLM 自动生成主线 | 成功 | `overall_success=true`, `target_success=true`, `ret_val=True` |
| xarm6 PickCube 抓取迁移 | 首轮远程失败已记录 | LLM adapter 未形成真实抓取，已定位破坏性候选重试问题 |
| 当前案例结论 | 已形成 | Fetch 是 contact-side reachability failure |

## 9. 下一步计划

### 9.0 已验证成功案例：Panda → xarm6_robotiq

在 Fetch 被记录为失败案例后，下一个目标机器人选择：

```text
xarm6_robotiq
```

新的迁移 case：

```text
case02_pull_cube_panda_to_xarm6
```

选择原因：

- xarm6_robotiq 和 Panda 一样是固定基座单臂机器人；
- 不需要处理 Fetch 那样的移动底盘动作空间；
- 仍然和 Panda 有差异，例如 DoF、工作空间、TCP、夹爪和接触参数；
- 成功概率高于 Fetch，但仍能体现 embodiment 迁移；
- 已成为论文中的首个自动迁移成功案例。

当前实验设计变为：

| 用途 | Source | Target | 预期作用 |
|---|---|---|---|
| 成功案例 | Panda | xarm6_robotiq | 证明 adapter/contact/controller 可迁移 |
| 失败案例 | Panda | Fetch | 证明系统需要识别不可行迁移 |

后续默认主实验将优先跑：

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

### 9.1 xarm6 下一步主实验

xarm6 已通过独立 oracle 和 DeepSeek 自动生成 adapter 分别证明物理可行性与自动迁移可行性。下一步：

1. 保存本次成功 adapter 和完整日志；
2. 使用多个 seed 重复运行，统计成功率、平均环境步数和失败类型；
3. 做 ablation：移除 far-side sweep、接触前检查、retry 多样性约束和失败反馈；
4. 比较 source-copy、oracle 和 LLM-generated adapter；
5. 记录 xarm6 成功案例与 Fetch 不可行案例的差异。

### 9.2 固定当前失败案例

接下来应把 Fetch 的失败明确记录为：

```text
target embodiment infeasible under current scene geometry
```

并在代码/日志中把这类失败从普通 execution failure 中区分出来。

### 9.3 扩展目标机器人或新任务设置

在已有一个成功案例和一个失败案例后，后续可扩展实验覆盖范围：

后续可选路线：

- 换一个更接近 Panda 的目标机械臂
- 调整 PullCube 初始布局，使 Fetch 能到正确接触侧
- 选择另一个接触任务，但保证 source 与 target 都有可行解
- 使用同一个任务，比较不同 target robot 的可迁移性

### 9.4 完善实验表格

后续需要补充：

| Case | Source | Target | Method | Success | Failure Layer | Main Evidence |
|---|---|---|---|---|---|---|
| PullCube | Panda | Panda | source | Yes | success | `ret_val=True` |
| PullCube | Panda | Fetch | source-copy | No | controller interface | action dim 9 mismatch |
| PullCube | Panda | Fetch | LLM adapter | No | skill/contact | no effective contact |
| PullCube | Panda | Fetch | oracle adapter | No | reachability/contact side | cannot reach `+x` side |
| PullCube | Panda | xarm6_robotiq | LLM adapter | Yes | success | `ret_val=True`, `elapsed_steps=460` |

### 9.5 后续报告更新规则

之后每次实验更新时，建议追加以下内容：

1. 实验命令
2. 关键输出
3. 是否成功
4. 失败层级
5. 与上一轮相比 LLM 或 adapter 修改了什么
6. 对论文论点有什么帮助

### 9.6 新增抓取式迁移主线：Panda → xarm6_robotiq

在 `PullCube-v1` 接触式推移迁移成功后，下一步新增真正需要夹爪抓取的任务：

```text
case03_pick_cube_panda_to_xarm6
```

任务环境：

```text
PickCube-v1
```

固定高层程序：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")

grasp_ok = robot.grasp(cube)
ret_val = robot.place(cube, goal) if grasp_ok else False
```

与 `PullCube-v1` 的区别：

| 任务 | 物理动作 | 是否必须验证抓取 |
|---|---|---|
| `PullCube-v1` | 闭合夹爪接触方块并推移 | 否 |
| `PickCube-v1` | 夹住方块、抬升、搬运到三维目标点 | 是 |

新增 `ManiSkillPickCubeRobot` adapter。它会：

```text
张开夹爪
→ 从方块上方靠近
→ 下降到抓取位置
→ 闭合夹爪
→ 调用 ManiSkill agent.is_grasping(cube) 验证真实抓取
→ 抬升方块
→ 再次检查是否滑落
→ 搬运到三维目标位置
→ 使用 ManiSkill evaluate() 判断最终成功
```

底层控制器仍然冻结：

```text
control_mode = pd_ee_delta_pos
low-level controller = PDEEPosController
```

LLM 只允许生成 target-side adapter，调整内容包括：

```text
抓取点偏移
接近高度
抬升高度
夹爪开合等待时间
归一化动作幅度
抓取失败后的重试策略
搬运 waypoint
```

远程主实验命令：

```bash
python -m maniskill_backend.module_generation_runner \
  --case case03_pick_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

### 9.7 PickCube 首轮远程结果与下一次修正

远程平台由 `rotule.polytechnique.fr` 切换到 `allemagne.polytechnique.fr`。用户主目录位于学校 NFS，因此项目代码、Conda 环境和实验日志均保留。当前可用 GPU：

```text
NVIDIA RTX A4000
显存：16376 MiB
Driver Version：595.80
CUDA Version：13.2
```

`case03_pick_cube_panda_to_xarm6` 首轮结果：

| Round | 代码校验 | 真实执行 | 关键失败证据 |
|---|---|---|---|
| 1 | 通过 | 失败 | `is_grasping=False`，方块被撞落，`cube_pos.z=-0.8996` |
| 2 | 通过 | 失败 | `is_grasping=False`，方块被推到 `[0.2234, -0.1416, 0.02]` |
| 3 | 未通过 | 未执行 | 生成模块无有效更新 |

LLM 生成的目标 adapter 已经做了多候选 `(dx, dy, dz)` 搜索，但它把 7 个候选放在同一个 episode 中连续尝试。一次失败会真实改变仿真状态；后续候选继续追逐已经偏移甚至掉落的方块，无法得到干净的抓取验证。

因此，下一轮 prompt 增加以下约束：

```text
每个候选抓取前后检查方块位移
仅当方块仍直立且 xy 位移较小时，才允许在同一 episode 尝试下一个候选
方块掉落、明显偏移或离开可达空间时，立即返回真实失败
禁止在同一 episode 中继续追逐已被撞偏的方块
让外层 generation round 在环境 reset 后尝试新策略
```

这不是向 LLM 提供人工成功轨迹，而是加入 failure-driven adaptation 所需的真实失败证据。

### 9.8 PickCube 第二轮远程结果：真实抓取已经出现

加入状态感知重试约束后，DeepSeek V4-Pro 生成了新的 xarm6 adapter。远程结果：

| Round | 代码校验 | 真实执行 | 关键结果 |
|---|---|---|---|
| 1 | 通过 | 失败 | 候选 2 把方块横向推动 `0.2960 m`，位移保护主动终止 |
| 2 | 通过 | 失败 | 候选 2 把方块横向推动 `0.2781 m`，位移保护主动终止 |
| 3 | 通过 | 失败 | `is_grasping=True`，`tcp_cube_xyz=0.0102`，方块已被抬升到 `z=0.1861 m` |

Round 3 是重要进展：xarm6 已经真实夹住方块并完成部分抬升。最终距离三维目标仍为：

```text
cube_goal_xyz=0.1237 m
```

但生成代码错误地返回：

```text
all grasp candidates failed
```

这与 `is_grasping=True` 矛盾。当前问题已经从“无法抓取”缩小为：

```text
抓取后分支处理错误
候选搜索占用过多 episode 步数
未把剩余预算留给 place(cube, goal)
```

下一轮 prompt 增加：

```text
抓取、闭爪、抬升和搬运后检查 episode 是否 terminated/truncated
抬升后 is_grasping=True 时立即记录 held_object 并返回 grasp success
禁止继续松爪或搜索候选
返回所有候选失败前再次检查 is_grasping
如果抓住后 episode 已截断，明确报告预算在搬运前耗尽
优先尝试 1 至 2 个安全候选，把剩余预算留给 place
```

这仍然是 failure-driven prompt adaptation：LLM 得到的是失败证据和正确的状态机约束，没有得到人工编写的成功动作序列。

### 9.9 PickCube 第三轮远程结果：首次接近轨迹仍需修正

加入抓取后状态保留与 transport 预算约束后，新一轮运行结果：

| Round | 代码校验 | 真实执行 | 关键结果 |
|---|---|---|---|
| 1 | 通过 | 失败 | 第一次尝试仍把方块横向推动 `0.1513 m` |
| 2 | 未通过 | 未执行 | LLM 返回与当前失败模块相同的代码 |
| 3 | 未通过 | 未执行 | LLM 再次返回相同代码 |

这次不是 token 截断。Round 2 和 Round 3 的明确错误为：

```text
ValueError('Generated adapter module is unchanged from the current failed module.')
```

失败范围进一步收窄：

```text
第一次接近和下降轨迹存在横向冲击
多候选数量不是当前关键问题
DeepSeek 未根据反馈实质修改下降阶段
```

下一轮 prompt 强制要求：

```text
在方块上方安全高度先完成 xy 对齐
接近物体后只做近似垂直的 z 方向下降
最终下降阶段更严格限制水平动作分量
闭爪前检查 TCP 水平误差
若水平对齐失败，在触碰方块前报告 approach failure
减少候选数量，禁止用更大的候选网格掩盖下降轨迹问题
```

这依旧只提供真实失败证据和适配原则，不包含人工成功动作序列。

### 9.10 PickCube 第四轮远程结果：需要闭爪前阶段诊断

加入 `xy/z` 分离下降约束后，DeepSeek V4-Pro 生成了新的 `_approach_and_close()`：

```text
安全高度完成 xy 对齐
→ 近似垂直下降
→ 水平动作限幅 ±0.3
→ 垂直动作限幅 ±0.8
→ 闭合夹爪
```

运行结果：

```text
is_grasping=False
cube_goal_xyz=0.2793
tcp_cube_xyz=0.1573
```

方块仍在桌面附近：

```text
cube_pos=[-0.0108, 0.0634, 0.02]
```

Round 2 和 Round 3 再次返回相同模块，没有进入仿真。此时不能直接根据 `tcp_cube_xyz=0.1573` 断定下降停得太高，因为这项指标是在候选失败、开爪和撤退后记录的最终距离。

当前缺失的是闭爪时刻的阶段证据：

```text
tcp_pos
intended_grasp_pos
tcp_grasp_xy
tcp_grasp_z
cube_pos
cube_displacement
is_grasping_after_close
```

下一轮 prompt 强制要求：

```text
固定步数下降后重新测量 TCP 到抓取点的残差
只有 xy 和 z 残差足够小时才允许闭爪
若尚未到位，执行有上限的垂直精调或报告 approach-alignment failure
闭爪前后记录阶段级诊断
不要用撤退后的最终 TCP 距离判断闭爪阶段根因
```

这一步的目的不是人工指定成功轨迹，而是让 LLM 获得足够精确的真实失败反馈。

### 9.11 PickCube 第五轮远程结果：转向夹爪包络高度

加入 readiness guard 后，DeepSeek V4-Pro 在 Round 2 生成了更保守的单候选 adapter：

```text
安全高度接近
→ xy 单独对齐
→ z 单独下降
→ 检查闭爪前残差
→ 闭爪
```

关键结果：

```text
is_grasping=False
displacement=0.0015m
cube_pos=[0.0007, 0.0533, 0.02]
```

这表明水平冲击问题基本解决：方块只移动约 `1.5 mm`，没有掉落，也没有被明显推离原位。但夹爪仍未形成真实抓取。

ManiSkill 官方 `PickCube-v1` 配置中，`xarm6_robotiq` 使用：

```text
cube_half_size=0.02m
```

当前 LLM 代码允许：

```text
tcp_grasp_xy <= 0.025m
tcp_grasp_z <= 0.025m
```

该容差比方块半边长还大。由此推断，即使 readiness guard 通过，TCP 也可能仍未进入有效夹持包络。

下一轮 prompt 要求：

```text
使用更严格的闭爪前 xy/z 残差阈值
在固定 xy 下尝试少量有边界的 z 偏移
调整闭爪等待时间和 settle 时间
每个闭爪失败消息必须保留 tcp_grasp_xy 和 tcp_grasp_z
禁止重新扩大水平候选搜索
```

这依旧不包含人工成功轨迹。提示仅加入官方物体尺寸和真实失败证据。

### 9.12 PickCube 第六轮远程结果：失败诊断本身需要约束

最新一轮远程实验结果为：

```text
ROUND 1
target_success=False
target_message=all grasp candidates failed; is_grasping=False,
               cube_goal_xyz=0.2700,
               tcp_cube_xyz=0.0911,
               cube_pos=[0.0091, -0.0173, 0.0200]

ROUND 2
target_success=False
target_message 与 Round 1 完全相同

ROUND 3
module_valid=False
module_error=Generated adapter module is unchanged from the current failed module.
```

这轮结果有两个含义：

1. 方块仍在桌面上，且没有出现早期那种大幅横向撞飞或掉落，说明下降阶段的破坏性已经明显降低。
2. 失败信息仍只给出最终 `tcp_cube_xyz=0.0911`，没有给出闭爪时刻的 `tcp_grasp_xy` 和 `tcp_grasp_z`。该最终距离可能是在失败、开爪、撤退之后记录的，因此不能判断闭爪瞬间到底是 XY 没对准、Z 高度不对，还是 gripper close/settle 时序不合适。

因此，本轮不是继续盲目增加候选点，而是把“失败也必须可解释”写入运行器约束：

```text
如果 xarm6 PickCube 的 grasp 失败，
failure message 必须包含：
  tcp_grasp_xy
  tcp_grasp_z

否则该生成模块虽然 Python 合法、unittest 通过，
也会被视为 diagnostic_contract 不合格。
```

这一步将人工调试经验进一步转化为自动纠正装置的一部分：LLM 不仅要生成可执行 adapter，还要在失败时返回足够结构化的阶段证据，使下一轮 repair 能判断应该调整抓取高度、闭爪等待，还是接近轨迹。

### 9.13 PickCube 第七轮远程结果：接近已到位，闭爪造成侧向挤出

加入 close-time 诊断契约后，最新远程结果终于返回了闭爪阶段残差：

```text
ROUND 1 / ROUND 2
target_success=False
target_message=cube was not grasped;
               is_grasping=False,
               tcp_grasp_xy=0.0076,
               tcp_grasp_z=0.0120,
               cube_goal_xyz=0.4754,
               tcp_cube_xyz≈0.446,
               cube_pos=[-0.0509, -0.3862, 0.0200]

ROUND 3
module_valid=False
module_error=Generated adapter module is unchanged from the current failed module.
```

这轮的关键信息是：

```text
tcp_grasp_xy = 7.6 mm
tcp_grasp_z  = 12.0 mm
```

也就是说，闭爪前或闭爪时 TCP 已经很接近 intended grasp point，接近阶段不再是主要瓶颈。但闭爪后方块被挤到：

```text
cube_pos.y = -0.3862
```

这说明当前失败模式已经从：

```text
approach/descent alignment failure
```

转变为：

```text
gripper-envelope side-push failure
```

直白理解：机器人已经把末端放到比较接近的位置，但 Robotiq gripper 闭合时的包络、接触高度或闭爪时序不对，把方块侧向挤走了，而不是夹住。

下一轮约束更新：

```text
如果 grasp 失败，failure message 必须包含：
  tcp_grasp_xy
  tcp_grasp_z
  cube_disp_xy

若 tcp_grasp_xy/z 已经较小，但 cube_disp_xy 很大：
  不再继续调 approach；
  不扩大 horizontal candidate search；
  将失败归类为 gripper-envelope side push；
  改 close-phase geometry：
    - 固定 xy
    - 尝试略高的抓取高度
    - 闭爪时 xyz command 为 0
    - 增加慢速/分阶段 close settle
    - cube_disp_xy > 0.03m 立即中止本轮
```

这一步进一步体现了自动纠正装置的思想：系统不是只知道“失败了”，而是能把失败从“没到位”细分为“已到位但闭爪接触几何错误”，从而选择不同 repair 方向。

### 9.14 PickCube 第八轮远程结果：到位且无明显侧推，但夹爪未形成抓取

在加入 `cube_disp_xy` 诊断契约后，新的远程结果进一步细化了失败类型：

```text
ROUND 1
module_error=failed xarm6 PickCube grasp modules must report close-time diagnostics
missing tcp_grasp_xy, tcp_grasp_z, cube_disp_xy

ROUND 2
target_success=False
target_message=grasp failed after close;
               tcp_grasp_xy=0.0027,
               tcp_grasp_z=0.0002,
               cube_disp_xy=0.0052,
               cube_pos=[-0.0059, 0.0562, 0.0200]

ROUND 3
module_valid=False
module_error=Generated adapter module is unchanged from the current failed module.
```

这轮最关键的证据是：

```text
tcp_grasp_xy = 2.7 mm
tcp_grasp_z  = 0.2 mm
cube_disp_xy = 5.2 mm
is_grasping  = False
```

这说明：

```text
TCP 已经非常接近目标抓取点；
方块也没有被明显推飞；
但是夹爪闭合后仍没有形成真实抓取。
```

因此失败模式再次细分，从上一轮的：

```text
gripper-envelope side-push failure
```

进一步变成：

```text
good-alignment / no-displacement / no-grasp
```

直白理解：机器人已经把手放到了正确位置，而且没有把方块撞走，但 Robotiq gripper 的闭合高度、闭合时序或接触包络仍不对，所以 `is_grasping=False`。

下一轮约束更新：

```text
若满足：
  tcp_grasp_xy <= 0.005m
  tcp_grasp_z  <= 0.005m
  cube_disp_xy <= 0.01m
  is_grasping=False

则不要继续调 approach；
不要增加 horizontal candidates；
不要重复 centered close；

应改 close envelope：
  - 固定 xy
  - 小范围搜索 grasp_z_offset_m
  - 尝试略高或略低的闭爪高度
  - 闭爪时保持 xyz command = 0
  - 增加 close/settle duration
  - 闭爪后先检查 self._is_grasping('cube')，成功再 lift/transport
```

这一步的意义是：自动 repair 现在已经能把 PickCube 抓取失败分成至少三类：

| 失败类型 | 证据 | 下一步 |
|---|---|---|
| approach/descent alignment failure | `tcp_grasp_xy/z` 大 | 改接近轨迹和下降方式 |
| gripper-envelope side-push failure | `tcp_grasp_xy/z` 小，但 `cube_disp_xy` 大 | 改闭爪接触包络，避免侧推 |
| good-alignment/no-displacement/no-grasp | `tcp_grasp_xy/z` 小，`cube_disp_xy` 小，但 `is_grasping=False` | 改抓取高度、闭爪时序、close settle |

### 9.15 PickCube 第九轮远程结果：0 偏移闭爪导致侧推回归

在加入 close-envelope repair 约束后，新的远程结果反而出现了退步：

```text
ROUND 1
target_success=False
target_message=cube displaced laterally during close (candidate 0);
               tcp_grasp_xy=0.0037,
               tcp_grasp_z=0.0040,
               cube_disp_xy=0.0406,
               is_grasping=False,
               cube_pos=[-0.0089, 0.0152, 0.0200]

ROUND 2 / ROUND 3
module_valid=False
module_error=Generated adapter module is unchanged from the current failed module.
```

这轮的关键问题是：

```text
candidate 0 使用 grasp_z_offset = 0.0
cube_disp_xy = 4.06 cm
```

也就是说，LLM 虽然开始关注 close envelope，但它仍把 **0 偏移 centered close** 放在第一个候选。这个动作已经被实验验证会把方块侧向挤开，因此是一个 regression。

这次失败和上一节不同：

| 轮次 | 证据 | 解释 |
|---|---|---|
| 第八轮 | `tcp_grasp_xy=0.0027`, `tcp_grasp_z=0.0002`, `cube_disp_xy=0.0052` | 到位、没撞飞，但没夹住 |
| 第九轮 | `tcp_grasp_xy=0.0037`, `tcp_grasp_z=0.0040`, `cube_disp_xy=0.0406` | 到位，但 0 偏移闭爪又造成侧推 |

下一轮约束更新：

```text
不要再把 grasp_z_offset=0.0 作为第一个 xarm6 candidate；
如果保留 0 偏移，也必须放在更安全的非零 Z 候选之后；
第一候选必须体现实质 close-envelope 变化：
  - 非零 Z offset
  - 或分阶段 gripper close command
  - 或更长 zero-xyz close settle
  - 或不同 pre-close height
```

直白理解：现在不是“LLM 不知道失败原因”，而是它知道方向后仍然选择了一个已经失败过的默认动作。下一步要把“不要重复 0 偏移第一候选”写成硬约束，让自动 repair 真正探索新的闭爪包络。

### 9.16 PickCube 第十轮远程结果：非零 Z 约束仍不够，0/负 Z 均侧推

进一步更新 prompt 后，新一轮远程结果仍未成功：

```text
ROUND 1
module_error=missing tcp_grasp_xy, tcp_grasp_z, cube_disp_xy
target_message=all grasp candidates failed; is_grasping=False,
               tcp_cube_xyz=0.1042,
               cube_pos=[-0.0234, 0.0636, 0.0200]

ROUND 2
module_error=missing cube_disp_xy
target_message=cube displaced by 0.0359m during close;
               tcp_grasp_xy=0.0015,
               tcp_grasp_z=0.0020
execution_log args: candidate=2, z_offset=0.0

ROUND 3
module_error=missing cube_disp_xy
target_message=cube displaced by 0.0503m during close;
               tcp_grasp_xy=0.0001,
               tcp_grasp_z=0.0002
execution_log args: candidate=1, z_offset=-0.005
```

这轮有两个结论。

第一，物理上仍是侧推失败：

```text
z_offset = 0.0    → cube displacement = 3.59 cm
z_offset = -0.005 → cube displacement = 5.03 cm
```

而且两轮的 TCP 残差都非常小：

```text
tcp_grasp_xy <= 1.5 mm
tcp_grasp_z  <= 2.0 mm
```

这再次说明问题不是“手没有到位”，而是 **0 或负 Z 闭爪高度会把方块挤走**。

第二，工程上出现了日志格式问题。LLM 写的是：

```text
cube displaced by 0.0359m during close
```

这个信息对人是有用的，但没有包含标准字段：

```text
cube_disp_xy=...
```

所以 runner 按诊断契约把它判为缺少 `cube_disp_xy`。本轮后更新为：

```text
runner 可以识别 "cube displaced by Xm" 作为 cube displacement alias；
prompt 仍要求 LLM 输出标准字段 cube_disp_xy=...；
下一轮明确禁止 0 或负 Z 作为首选 close height；
优先尝试正 Z close-height sweep。
```

下一轮约束更新：

```text
不要从 z_offset=0.0 开始；
不要从 z_offset<0 开始；
优先尝试正 Z close-height sweep；
闭爪时保持 xyz command = 0；
使用更慢的 staged close；
失败消息必须精确包含 cube_disp_xy=...
```

### 9.17 下一步方法调整：加入自动抓取参数探针

连续多轮 PickCube 失败说明：继续只靠 prompt 让 LLM 猜 `grasp_z_offset`、闭爪步数和闭爪命令，效率很低。因此项目从“纯 LLM repair”进一步扩展为：

```text
LLM adapter generation
  + structured physical probing
  + failure-guided prompt feedback
```

新增脚本：

```bash
python scripts/xarm6_pick_grasp_probe.py \
  --sim-backend auto \
  --render-backend gpu
```

该脚本做的事情是：

```text
固定 xy；
只枚举少量 close-envelope 参数：
  grasp_z_offset
  close_steps
  close_command
  settle_steps

每组参数都从 fresh reset 开始；
真实执行 ManiSkill env.step(action)；
记录：
  is_grasping_after_close
  is_grasping_after_lift
  cube_disp_xy
  tcp_grasp_xy
  tcp_grasp_z
  cube_lift_delta_z
```

输出文件：

```text
results/xarm6_pick_grasp_probe.json
results/xarm6_pick_grasp_probe.md
results/xarm6_pick_grasp_probe_prompt.txt
```

其中 `xarm6_pick_grasp_probe_prompt.txt` 会被 `module_generation_runner` 自动读取，并加入下一轮 LLM prompt：

```text
# Structured xArm6 PickCube probe feedback
...
best_probe_case:
  grasp_z_offset=...
  close_steps=...
  close_command=...
  cube_disp_xy=...
  is_grasping_after_close=...
```

这样下一轮 LLM 不再凭日志猜，而是基于一组真实仿真扫参结果生成 adapter。

这一步的研究意义是：当迁移任务从 PullCube 的接触拖拽进入 PickCube 的真实抓取时，单纯自然语言反馈不够，需要把失败空间转化为一个小规模约束搜索问题。LLM 负责生成结构化 adapter，探针负责给出物理可行性证据，两者组合形成更稳定的自动纠正装置。

### 9.18 自动探针首轮结果：没有成功参数，但发现 adapter 逻辑错误

自动探针首轮结果如下：

```text
total_probe_cases = 32
grasping_cases = 0

best_probe_case:
  grasp_z_offset=0.016
  close_steps=12
  close_command=-0.6
  settle_steps=8
  tcp_grasp_xy=0.00239
  tcp_grasp_z=0.00152
  cube_disp_xy=0.00458
  is_grasping_after_close=False
  is_grasping_after_lift=False
```

这说明：自动探针没有找到“答案参数”。它只告诉我们，在当前固定 XY、正 Z close-height sweep 和简单 close command 范围内，最好的参数也只是 **低侧推、对齐好，但仍未形成 grasp**。

因此这个 probe 不是把答案给 LLM，而是给出一个约束结论：

```text
简单调 grasp_z_offset / close_steps / close_command 仍不够；
LLM 不应把 best_probe_case 当作成功轨迹；
但它可以用 best_probe_case 作为低破坏性的起点。
```

随后 LLM 迁移轮次结果：

```text
ROUND 1
target_message=all grasp candidates failed;
               is_grasping=True,
               cube_goal_xyz=0.2802,
               tcp_cube_xyz=0.0358,
               cube_pos=[0.0270, 0.0809, 0.0212]

ROUND 2
target_message=all grasp candidates failed;
               is_grasping=False,
               tcp_grasp_xy=0.0011,
               tcp_grasp_z=0.0010,
               cube_disp_xy=0.0541
```

Round 1 的核心不是物理失败，而是 **adapter 逻辑错误**：

```text
message 里已经写 is_grasping=True；
但 adapter 仍返回 all grasp candidates failed。
```

这说明 LLM 生成的 adapter 在最终抓取检查处没有遵守规则：

```text
if self._is_grasping("cube"):
    self.held_object = "cube"
    return grasp success
```

因此本轮后新增硬约束：

```text
如果 grasp failure message 中出现 is_grasping=True，
runner 会判定该 module 违反诊断契约；
adapter 不允许在 is_grasping=True 时返回 grasp failure；
必须保留抓取，继续 lift/place，或报告后续 lift/place 阶段失败。
```

Round 2 则继续说明：即使基于 probe，LLM 仍可能产生侧推策略：

```text
cube_disp_xy=0.0541
```

下一步重点从单纯 close 参数搜索转为：

```text
1. 保留任何出现的 is_grasping=True；
2. 不把 best_probe_case 视为成功答案；
3. 若 fixed-XY close probe 全失败，允许 LLM 改更高层 grasp primitive：
   - finger envelope
   - lift timing
   - close-after-contact order
   - verified grasp preservation
```

### 9.19 Probe 被使用后仍失败：固定 XY close sweep 不足

最新一轮 LLM 已经读取了 probe feedback，并生成了以探针最佳参数为起点的 close-envelope sweep：

```text
候选参数从 probe best case 开始：
  z_offset=0.016
  close_steps=12
  close_command=-0.6
```

但最终仍失败：

```text
target_message=all grasp candidates failed;
               is_grasping=False,
               tcp_grasp_xy=0.0009,
               tcp_grasp_z=0.0015,
               cube_disp_xy=0.0297
```

这组数据非常集中：

```text
TCP 对齐误差 < 1 mm；
Z 误差约 1.5 mm；
但 cube_disp_xy 接近 3 cm；
仍然没有形成 grasp。
```

这说明 LLM 不是没有用 probe，而是 probe 暴露出一个更深的问题：

```text
固定 XY + 调 z_offset / close_steps / close_command
不足以解决 xarm6 + Robotiq 的抓取接触包络。
```

因此下一步不能继续简单增加更多：

```text
z_offset candidates
close_steps candidates
close_command candidates
```

否则只是扩大同一个失败空间。更合理的方向是让 LLM 改更结构性的 grasp primitive：

```text
1. 一旦 self._is_grasping("cube") 为 True，立刻保留抓取并进入 lift/place；
2. 如果 fixed-XY probe 全失败，允许非常小的 finger-envelope micro-offset；
3. micro-offset 必须是诊断性的，不能追逐已移动方块；
4. cube_disp_xy > 0.03m 立即中止；
5. 如果所有真实 close attempt 仍不形成 grasp，报告 close-envelope/force infeasibility。
```

这一步把失败空间从：

```text
parameter tuning
```

推进到：

```text
gripper-envelope / force-closure primitive repair
```

### 9.20 最新回归：未下降到抓取高度就进入失败判断

最新 LLM 迁移分析返回：

```text
target_message=all grasp candidates failed;
               is_grasping=False,
               tcp_grasp_xy=0.0096,
               tcp_grasp_z=0.1328,
               cube_disp_xy=unknown,
               cube_pos=[0.0169, 0.0621, 0.0200]
```

这次失败和前几轮不同。关键不是 close envelope，而是：

```text
tcp_grasp_z = 0.1328m
```

也就是说，闭爪或失败判断时 TCP 仍然离 intended grasp point 约 13.3 cm。这个高度远远超过方块尺寸，说明机器人根本没有下降到可抓取高度。

因此这轮不能解释为：

```text
gripper/force failure
```

而应该解释为：

```text
approach/descent failure
```

直白理解：手还在空中，没到方块附近，就开始说“夹爪没夹住”。这不是夹爪力的问题，而是下降/到位逻辑的问题。

同时，诊断信息中出现：

```text
cube_disp_xy=unknown
```

这说明虽然字段名出现了，但值不可用。为了让自动 repair 更稳定，本轮后更新诊断契约：

```text
tcp_grasp_xy / tcp_grasp_z / cube_disp_xy 必须是数值；
cube_disp_xy=unknown 不再算合格；
如果因为没有接触而没有方块位移，应报告 cube_disp_xy=0.0；
如果 tcp_grasp_z 很大，不应闭爪，应继续安全下降或报告 approach/descent failure。
```

下一步约束更新：

```text
若 tcp_grasp_z 很大：
  不要 close gripper；
  不要报告 force failure；
  先做 bounded vertical refinement；
  若仍不能下降，则报告 phase-specific approach/descent failure；
  failure message 仍必须包含数值 tcp_grasp_xy、tcp_grasp_z、cube_disp_xy。
```

## 10. 当前一句话总结

当前项目已经从简单高层代码迁移推进到真实仿真控制迁移：DeepSeek V4-Pro 已成功为 xarm6 自动生成可执行的 `PullCube-v1` 目标 adapter，并在 `PickCube-v1` 中生成过能够真实夹住并部分抬升方块的 adapter；当前 PickCube 已加入自动抓取参数探针，首轮探针未找到成功 close 参数，且最新回归显示 LLM 有时会在 TCP 仍高于抓取点 13cm 时报告抓取失败，因此下一步同时约束 approach/descent 到位检查与 gripper-envelope repair。
