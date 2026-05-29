# 实验进度报告：LLM 机器人程序迁移

更新时间：2026-05-29
当前阶段：ManiSkill `PullCube-v1` 迁移案例分析中
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

当前 LLM 调用通过 OpenRouter 完成。

主要使用模型：

```text
anthropic/claude-opus-4.6
```

报告中简称为：

```text
Opus 4.6
```

用途：

- 根据源程序、机器人 profile、capability card 和失败日志生成目标 adapter
- 分析迁移失败层级
- 总结 Panda → Fetch 的执行假设差异
- 生成 migration analysis

需要注意：

**LLM 不是直接控制机器人成功，而是生成或修改 adapter 代码。最终是否成功仍然由 ManiSkill 的真实 `env.step(action)` 执行和环境 `evaluate()` 判定。**

### 2.4 当前机器人

当前只保留两个机器人作为主要对比：

| 角色 | 机器人 | 说明 |
|---|---|---|
| Source | Panda | 源机器人，固定机械臂，当前任务可成功 |
| Target | Fetch | 目标机器人，移动底盘 + 机械臂，迁移后当前失败 |

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

因此已将 xarm6 adapter 更新为最小 oracle adapter：复现 `x_plus → down → drag_x_minus` 成功轨迹，并仍然使用真实 `env.step(action)` 和 ManiSkill success 判断。

### 4.7 xarm6 oracle adapter 真实执行成功

将诊断得到的成功轨迹写入：

```text
maniskill_backend/generated_adapters/case02_xarm6_pull_cube.py
```

随后在远程服务器运行真实 xarm6 目标 adapter：

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
  --adapter-module maniskill_backend.generated_adapters.case02_xarm6_pull_cube
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

**Panda → xarm6_robotiq 当前已经形成第一个成功迁移案例。**

这个成功不是通过修改高层程序得到的，高层程序仍然是：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")
ret_val = robot.pull(cube, goal)
```

真正发生迁移的是目标机器人 adapter：

```text
Panda 的 closed-loop contact pull
→ xarm6 的 measured raw contact sequence
→ x_plus → down → drag_x_minus
```

这说明同一个 LMP 程序可以保持不变，但不同 embodiment 需要不同的执行适配层。

### 4.8 下一步：LLM 生成迁移代码的严格实验

需要特别区分：

```text
xarm6 oracle adapter 成功 != LLM 自动迁移成功
```

当前 xarm6 成功 adapter 是根据诊断结果手写的 oracle，因此只能作为：

- 可行性上限；
- 成功 contact primitive 证据；
- 后续对比的 oracle upper bound。

为了避免“把答案直接喂给 LLM”，后续把 xarm6 分成两个 LLM 条件：

```text
case03_pull_cube_panda_to_xarm6_failure_feedback
case04_pull_cube_panda_to_xarm6_abstract_hint
```

两者都从非 oracle 的失败 seed adapter 开始，且都不提供完整成功轨迹。

| Case | 给 LLM 的信息 | 是否泄露完整答案 | 研究含义 |
|---|---|---|---|
| `case03` | action space、target profile、失败日志、当前失败 adapter | 否 | 测试 LLM 是否能仅凭失败反馈迁移 |
| `case04` | case03 信息 + 抽象策略提示 | 否 | 测试 LLM 是否能把抽象接触策略变成代码 |
| `case02` | 人类诊断出的完整成功轨迹写入 adapter | 是 | oracle upper bound，不算 LLM 自主成功 |

`case04` 允许给出的抽象策略是：

```text
从 cube 的正确接触侧建立接触；
下降到接触高度；
保持轻微下压力；
沿 goal 方向拖拽。
```

但不能给：

```text
具体动作数值
具体 step 数
完整 oracle 轨迹
```

如果 `case03` 成功，可以写：

```text
LLM generated a successful migration adapter from failure feedback.
```

如果 `case04` 成功，可以写：

```text
LLM generated a successful migration adapter from abstract diagnostic guidance.
```

运行命令分别为：

```bash
python -m maniskill_backend.module_generation_runner \
  --case case03_pull_cube_panda_to_xarm6_failure_feedback \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu

python -m maniskill_backend.module_generation_runner \
  --case case04_pull_cube_panda_to_xarm6_abstract_hint \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

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
| LLM adapter generation | 已跑 | 能修改 adapter，但未成功 |
| Fetch oracle adapter | 已跑 | 仍失败，说明不是单纯 LLM 质量问题 |
| 接触侧诊断 | 已完成 | Fetch 无法到正确接触侧 |
| xarm6 module generation | 已跑 | 3轮后仍失败，但方块已向目标方向移动约 5 cm |
| xarm6 诊断脚本 | 已跑 | 找到成功 raw contact sequence |
| xarm6 oracle adapter | 成功 | `success=true`, `elapsed_steps=191` |
| 当前案例结论 | 已形成 | Fetch 是 contact-side reachability failure |

## 9. 下一步计划

### 9.0 新增成功候选：Panda → xarm6_robotiq

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
- 适合作为论文中的成功迁移案例候选。

当前实验设计变为：

| 用途 | Source | Target | 预期作用 |
|---|---|---|---|
| 成功候选 | Panda | xarm6_robotiq | 证明 adapter/contact/controller 可迁移 |
| 失败案例 | Panda | Fetch | 证明系统需要识别不可行迁移 |

后续默认主实验将优先跑：

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

### 9.1 xarm6 下一步诊断

xarm6 当前不是完全不可行，而是接触拖拽不足。下一步应优先做：

1. 用新的最小 oracle adapter 重新跑 `case02_pull_cube_panda_to_xarm6`；
2. 如果成功，把该结果作为当前成功迁移案例；
3. 再让 Opus 4.6 在 prompt 中看到成功轨迹，测试它能否生成等价 adapter；
4. 做 ablation：去掉 `x_plus`、去掉 `down`、缩短 `drag_x_minus`，观察成功率变化；
5. 记录 xarm6 成功案例与 Fetch 不可行案例的差异。

### 9.2 固定当前失败案例

接下来应把 Fetch 的失败明确记录为：

```text
target embodiment infeasible under current scene geometry
```

并在代码/日志中把这类失败从普通 execution failure 中区分出来。

### 9.3 选择新的目标机器人或新任务设置

为了让论文不仅有失败案例，还需要至少一个成功迁移案例。

后续可选路线：

- 换一个更接近 Panda 的目标机械臂
- 调整 PullCube 初始布局，使 Fetch 能到正确接触侧
- 选择另一个接触任务，但保证 source 与 target 都有可行解
- 使用同一个任务，比较不同 target robot 的可迁移性

### 9.3 完善实验表格

后续需要补充：

| Case | Source | Target | Method | Success | Failure Layer | Main Evidence |
|---|---|---|---|---|---|---|
| PullCube | Panda | Panda | source | Yes | success | `ret_val=True` |
| PullCube | Panda | Fetch | source-copy | No | controller interface | action dim 9 mismatch |
| PullCube | Panda | Fetch | LLM adapter | No | skill/contact | no effective contact |
| PullCube | Panda | Fetch | oracle adapter | No | reachability/contact side | cannot reach `+x` side |

### 9.4 后续报告更新规则

之后每次实验更新时，建议追加以下内容：

1. 实验命令
2. 关键输出
3. 是否成功
4. 失败层级
5. 与上一轮相比 LLM 或 adapter 修改了什么
6. 对论文论点有什么帮助

## 10. 当前一句话总结

当前项目已经从简单高层代码迁移推进到真实仿真控制迁移：Panda 可以完成 `PullCube-v1`，但 Fetch 因动作空间、移动底盘和接触侧可达性差异导致迁移失败，该失败案例证明了机器人程序迁移需要跨 program、adapter、controller 和 contact geometry 的系统性诊断。
