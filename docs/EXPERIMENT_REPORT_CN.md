# 实验进度报告：LLM 机器人程序迁移

更新时间：2026-06-02
当前阶段：ManiSkill `PullCube-v1` 首个 LLM 自动迁移成功案例已验证
报告用途：作为后续实验记录的基础版本，之后所有实验进展、失败案例、统计结果和论文分析都在此文件上继续更新。

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

## 10. 当前一句话总结

当前项目已经从简单高层代码迁移推进到真实仿真控制迁移：DeepSeek V4-Pro 已成功为 xarm6 自动生成可执行的 `PullCube-v1` 目标 adapter，而 Fetch 因动作空间、移动底盘和接触侧可达性差异迁移失败；两个案例共同说明机器人程序迁移需要跨 program、adapter、controller 和 contact geometry 的系统性诊断。
