# Embodied Migration

LLM 生成机器人程序的跨具身迁移实验平台。

本项目研究一个问题：**同一个自然语言任务或源机器人程序，在换到不同机器人之后，为什么会失败，以及如何利用机器人能力描述和失败反馈让 LLM 自动修改程序。**

当前项目包含两个层次：

1. **PyBullet 原型层**：验证移动底盘、双臂、不同夹爪/吸附器等具身差异会导致 LLM 代码迁移失败。
2. **robosuite/MuJoCo 复杂任务层**：把任务从简单方块搬运推进到双臂抬锅、handover、peg-in-hole 等更接近机器人论文的任务。

推荐论文方向：

```text
Capability-Conditioned Failure-Driven Adaptation of LLM-Generated Robot Programs
面向机器人能力约束的失败驱动式 LLM 机器人程序迁移
```

---

## 1. 项目做了什么

项目的核心不是单纯“让机器人执行一句指令”，而是研究 **代码迁移**：

```text
源机器人程序 / 同一个任务指令
        ↓
换到目标机器人
        ↓
由于目标机器人能力不同，原代码可能失败
        ↓
Capability Card 告诉 LLM 目标机器人能做什么、不能做什么
Failure Report 告诉 LLM 上一次为什么失败
        ↓
LLM 重新生成或修改机器人程序
```

例如：

- 固定双臂机器人可以直接双臂协作抓两个物体。
- 移动双臂机器人必须先导航到桌边，再执行双臂任务。
- robosuite 里的 Dual IIWA 需要显式设置更大的 grip force。
- 单臂 mobile Tiago 不能执行真正的双臂同步任务，应当拒绝或选择替代策略。

---

## 2. 核心概念

### 2.1 LMP Code

LMP Code 是 LLM 生成的 Python 机器人程序。它不是直接输出一句话，而是输出可执行代码，例如：

```python
red_pos = scene.get_object_position("red block")
green_pos = scene.get_object_position("green block")
tray_pos = scene.get_object_position("yellow tray")

lift_ok = robot.lift_two_objects(red_pos, green_pos)
if lift_ok:
    ret_val = robot.place_two_objects(
        tray_pos + np.array([-0.03, -0.03, 0.05]),
        tray_pos + np.array([0.03, 0.03, 0.05]),
        pre_release_height=0.005,
    )
```

### 2.2 Capability Card

Capability Card 是机器人能力卡，描述目标机器人有哪些 API、能力和限制。

它回答的是：

```text
这个机器人有没有移动底盘？
有没有双臂？
能不能 bimanual coordination？
夹爪是什么类型？
需要低高度释放吗？
需要多大的 grip force？
工作空间半径是多少？
```

它类似 skill/API 的说明书，但不是 skill 本身。  
Skill 是 `robot.lift_two_objects()` 这种可执行函数；Capability Card 是告诉 LLM 什么时候该用这些函数。

### 2.3 Failure Report

Failure Report 是失败反馈。代码第一次失败后，系统会把失败原因组织成结构化信息，再让 LLM 重试。

例如：

```text
robot API returned False
failed to move above object
target is outside reachable workspace
grip force is below required threshold
mobile robot must navigate before picking
```

### 2.4 Baseline / B / B+A

项目中常见的对比方式：

| 名称 | 含义 |
|---|---|
| baseline | 不给能力卡，不给失败反馈，只让 LLM 直接生成代码 |
| B | 给失败反馈并允许 retry，但不给 Capability Card |
| B+A | 给 Failure Report，也给 Capability Card |
| source-copy | 直接把源机器人程序复制到目标机器人上执行 |
| oracle | 手写的理想迁移上界，用来确认任务本身可解 |
| llm | 调用 OpenRouter / LLM 自动生成迁移代码 |

论文里真正要证明的是：**B+A 是否比 baseline / B / source-copy 更稳定。**

---

## 3. 项目结构

```text
embodied_migration/
├── main.py                         # PyBullet 交互式 LLM demo
├── llm_client.py                   # OpenRouter / LLM 客户端
├── requirements.txt                # PyBullet 基础依赖
├── requirements-robosuite.txt      # robosuite / MuJoCo 可选依赖
├── .env.example                    # API key 模板
│
├── robots/                         # PyBullet 机器人封装
│   ├── kuka_robot.py
│   ├── franka_robot.py
│   ├── mobile_robot.py
│   ├── dual_arm_robot.py
│   ├── dual_franka_robot.py
│   └── mobile_dual_arm_robot.py
│
├── perception/                     # PyBullet 桌面场景与物体位置接口
├── capabilities/                   # Capability Card 定义
├── prompts/                        # LLM prompt 构造
├── lmp/                            # LMP 代码抽取、执行、失败反馈
│
├── examples/
│   ├── smoke_test.py               # 不调用 LLM 的 PyBullet 物理层烟测
│   └── robosuite_migration_demo.py # robosuite 复杂任务迁移 demo
│
├── robosuite_backend/              # robosuite / MuJoCo 后端
│   ├── profiles.py                 # 目标机器人 profile
│   ├── tasks.py                    # TwoArmLift / Handover / PegInHole 任务
│   ├── symbolic.py                 # 高层 skill API
│   ├── trajectory_robot.py         # robosuite 控制器轨迹桥接
│   └── migration.py                # source-copy / oracle / llm 迁移逻辑
│
├── benchmark/                      # 批量实验、统计、日志、表格
├── scripts/                        # 一键实验脚本
├── docs/                           # 中文说明文档、演示手册、PPT
└── paper/                          # 论文结构和实验小节模板
```

---

## 4. 环境安装

### 4.1 创建 conda 环境

推荐 Python 3.10。

```bash
conda create -n em python=3.10 -y
conda activate em
```

预期输出：

```text
(em) your-user@your-machine ...
```

### 4.2 进入项目目录

```bash
cd /Users/xifan/Downloads/embodied_migration
```

Ubuntu 上按你的实际路径进入，例如：

```bash
cd ~/embodied_migration
```

### 4.3 安装 PyBullet 基础依赖

```bash
pip install -r requirements.txt
```

预期输出：

```text
Successfully installed ...
```

### 4.4 安装 robosuite / MuJoCo 可选依赖

如果只跑 PyBullet，可以先跳过这一步。  
如果要跑复杂双臂任务，执行：

```bash
pip install -r requirements-robosuite.txt
```

预期输出：

```text
Successfully installed mujoco ...
Successfully installed robosuite ...
```

检查 robosuite 是否可用：

```bash
python -c "from robosuite_backend.env_adapter import availability_message; print(availability_message())"
```

预期输出结尾：

```text
robosuite backend is available
```

如果看到下面这些 warning，一般不影响当前 demo：

```text
[robosuite WARNING] No private macro file found
[robosuite WARNING] Could not import robosuite_models
[robosuite WARNING] Could not load the mink-based whole-body IK
```

---

## 5. 配置 LLM API Key

如果只跑 smoke test / oracle，不需要 API key。  
如果要跑 `--planner llm` 或 `main.py` 交互式 LLM demo，需要配置 OpenRouter key。

复制模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```text
OPENROUTER_API_KEY=你的_OpenRouter_Key
EM_MODEL=anthropic/claude-sonnet-4.5
```

检查 key 是否读到：

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); print('OPENROUTER_API_KEY =', 'SET' if os.getenv('OPENROUTER_API_KEY') else 'MISSING')"
```

预期输出：

```text
OPENROUTER_API_KEY = SET
```

---

## 6. 快速运行：PyBullet 稳定演示

这一部分不调用 LLM，用手写 LMP Code 验证机器人和物理层是否正常。

### 6.1 固定双臂机器人

```bash
python -m examples.smoke_test --robot dual_arm --gui
```

预期现象：

- 弹出 PyBullet GUI。
- 看到两个 KUKA 机械臂。
- 两个机械臂同时抓起 red block 和 green block。
- 两个方块被放入 yellow tray。

预期终端输出包含：

```text
✅ Scene ready. Embodiment: Dual-arm Fixed Manipulator (2x KUKA)
[DualArm:both] Coordinated lift complete
[DualArm:both] Coordinated place complete
ret_val = 'success'
physical_success = True
```

### 6.2 移动双臂机器人

```bash
python -m examples.smoke_test --robot mobile_dual_arm --gui
```

预期现象：

- 弹出 PyBullet GUI。
- 看到 Husky 移动底盘 + 双 KUKA 机械臂。
- 移动机器人先导航到桌边。
- 再同时抓起两个方块并放入托盘。

预期终端输出包含：

```text
✅ Scene ready. Embodiment: Mobile Dual-arm Manipulator
[MobileDualArm] 🚐 Navigated base
[DualArm:both] Coordinated lift complete
[DualArm:both] Coordinated place complete
ret_val = 'success'
physical_success = True
```

### 6.3 双 Franka 机器人

```bash
python -m examples.smoke_test --robot dual_franka --gui
```

预期输出包含：

```text
✅ Scene ready. Embodiment: Dual-arm Fixed Manipulator (2x Franka)
ret_val = 'success'
physical_success = True
```

---

## 7. 运行 PyBullet LLM 交互模式

交互模式会真正调用 LLM，让 LLM 根据自然语言生成 LMP Code。

```bash
python main.py --robot mobile_dual_arm --mode ba
```

预期输出：

```text
🌍 Launching PyBullet (robot=mobile_dual_arm)...
✅ Scene ready. Embodiment: Mobile Dual-arm Manipulator ...
⚙️  Config: capability_card=True, retry=True
💬 Enter a natural-language instruction. Type 'exit' to quit.
👉 >
```

可以输入：

```text
pick up the red block and green block at the same time, then place both into the yellow tray
```

如果 LLM 生成正确代码，预期输出包含：

```text
🧠 [Attempt 1/3] Asking LLM to generate code
🧠 LMP Code to execute:
...
[MobileDualArm] 🚐 Navigated base
[DualArm:both] Coordinated lift complete
[DualArm:both] Coordinated place complete
✅ Instruction code executed on attempt 1.
```

如果第一次失败，系统会自动生成 Failure Report 并重试：

```text
⚠️  Attempt 1 failed, will retry with structured feedback.
🧠 [Attempt 2/3] Retrying with failure report...
```

---

## 8. 运行 robosuite / MuJoCo 复杂任务

robosuite 部分用于展示更复杂的代码迁移任务。当前主要任务包括：

| 任务 | 含义 |
|---|---|
| `two_arm_lift` | 双臂抓住锅的两个把手并抬起 |
| `two_arm_handover` | 一只手抓 hammer，交给另一只手，再放到目标区域 |
| `two_arm_peg_in_hole` | 一只手扶板，另一只手插 peg |

目标机器人包括：

| 机器人 | 含义 |
|---|---|
| `rs_dual_panda` | 源机器人，双 Panda |
| `rs_dual_iiwa` | 目标机器人，双 KUKA IIWA |
| `rs_baxter` | 目标机器人，Baxter 双臂 |
| `rs_mobile_tiago` | 目标机器人，移动单臂 Tiago |

### 8.1 不打开 GUI 的 oracle 迁移

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle \
  --real-control \
  --quiet
```

预期输出包含：

```text
Complex Robosuite Program Migration Demo
Task: two_arm_lift
Source robot: rs_dual_panda
Target robot: rs_dual_iiwa
Planner: oracle
[robosuite] set grip force to 0.85
[robosuite] real controller: approaching pot handles
[robosuite] real controller: lifting both arms
success=True reason=success_on_attempt_1
real_physical_success=True
```

### 8.2 对比 source-copy 失败

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner source-copy \
  --real-control \
  --quiet
```

预期结果通常是失败，因为源程序没有针对 Dual IIWA 设置足够 grip force：

```text
❌ grasp_pot_handle: grip_force=0.50 is below required 0.75
success=False reason=action-fail
real_physical_success=False
```

这就是代码迁移的核心展示：

```text
同一个源程序直接复制到目标机器人会失败；
加入目标机器人能力约束后，需要生成不同代码，例如 set_grip_force(0.85)。
```

### 8.3 打开 GUI 观看 robosuite 任务

macOS 上建议使用 `mjpython` 打开 MuJoCo GUI：

```bash
mjpython -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle \
  --show-env \
  --gui \
  --real-control \
  --hold-seconds 120
```

Ubuntu 上通常可以直接：

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle \
  --show-env \
  --gui \
  --real-control \
  --hold-seconds 120
```

预期现象：

- 弹出 MuJoCo / robosuite viewer。
- 双臂靠近锅的两个把手。
- 系统执行高层 skill 对应的低层控制轨迹。
- 终端输出 `success=True`。

说明：当前 robosuite real-control 是一个研究原型。它已经把 high-level skill 接到了 robosuite 控制器轨迹上，但稳定抓取仍使用 assisted grasp constraint，类似 PyBullet 中的 attach constraint。它适合演示程序迁移思想，但还不是完整的接触丰富控制策略。

### 8.4 调用 LLM 做 robosuite 迁移

需要 `.env` 中配置 `OPENROUTER_API_KEY`。

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner llm \
  --attempts 3
```

预期输出：

```text
Planner: llm
Attempt 1:
  exec_ok=...
  checker_success=...
  code:
    ...
```

如果第一次失败，会看到重试过程：

```text
Retrying with failure report
```

---

## 9. 运行 benchmark

### 9.1 PyBullet benchmark

```bash
python -m benchmark.run_benchmark \
  --robots mobile dual_arm \
  --modes api fewshot card failure card_failure \
  --tasks migration \
  --trials 1 \
  --run-id demo_run
```

预期输出：

```text
Running trial ...
Wrote results/runs/demo_run/summary.csv
```

生成统计表：

```bash
python -m benchmark.analyze_results results/runs/demo_run
```

预期生成：

```text
results/runs/demo_run/tables/method_summary.csv
results/runs/demo_run/tables/failure_breakdown.csv
results/runs/demo_run/tables/generated_code_features.csv
```

### 9.2 robosuite benchmark

```bash
python -m benchmark.run_robosuite_migration \
  --tasks two_arm_lift two_arm_handover two_arm_peg_in_hole \
  --targets rs_dual_iiwa rs_baxter rs_mobile_tiago \
  --planners source-copy oracle \
  --run-id robosuite_demo
```

预期输出：

```text
[source-copy] two_arm_lift: rs_dual_panda -> rs_dual_iiwa
  -> success=False reason=...
[oracle] two_arm_lift: rs_dual_panda -> rs_dual_iiwa
  -> success=True reason=success_on_attempt_1
Wrote results/robosuite_runs/robosuite_demo/summary.csv
```

---

## 10. 推荐演示流程

如果要给老师展示目前进度，建议按这个顺序：

### 第一步：说明项目目标

```text
这个项目研究 LLM 生成的机器人代码如何在不同 embodiment 之间迁移。
我们不是只看一个机器人能否完成任务，而是看源程序换到目标机器人后哪里失败，
以及 Capability Card + Failure Report 能不能帮助 LLM 修改代码。
```

### 第二步：跑 PyBullet 双臂成功案例

```bash
python -m examples.smoke_test --robot dual_arm --gui
```

展示点：

```text
固定双臂机器人可以同时抓两个方块并放入托盘。
```

### 第三步：跑移动双臂成功案例

```bash
python -m examples.smoke_test --robot mobile_dual_arm --gui
```

展示点：

```text
同样任务换成移动双臂机器人后，代码必须多一步 navigate_to。
这就是 embodiment difference 导致的代码迁移差异。
```

### 第四步：跑 robosuite source-copy 失败

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner source-copy \
  --real-control \
  --quiet
```

展示点：

```text
源机器人程序直接复制到 Dual IIWA 上失败，因为目标机器人需要更高 grip force。
```

### 第五步：跑 robosuite oracle 成功

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle \
  --real-control \
  --quiet
```

展示点：

```text
迁移后的代码加入 set_grip_force(0.85)，任务成功。
```

### 第六步：解释 LLM 版本

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner llm \
  --attempts 3
```

展示点：

```text
LLM 会根据 Capability Card 和 Failure Report 生成或修改 LMP Code。
```

---

## 11. 当前暂时成果

目前已经完成：

1. PyBullet 中实现了 6 类机器人：
   - `kuka`
   - `franka`
   - `mobile`
   - `dual_arm`
   - `mobile_dual_arm`
   - `dual_franka`
2. 实现了移动机器人导航 API：
   - `navigate_to`
   - `is_reachable`
   - `get_base_position`
3. 实现了双臂协作 API：
   - `choose_arm_for`
   - `pick_with_arm`
   - `place_with_arm`
   - `lift_two_objects`
   - `place_two_objects`
4. 实现了 Capability Card 注入 prompt。
5. 实现了 Failure Report retry。
6. 实现了 LMP Code 执行器和失败检测。
7. 实现了 PyBullet benchmark、日志记录、代码特征分析和统计表格。
8. 新增 robosuite/MuJoCo 后端：
   - `two_arm_lift`
   - `two_arm_handover`
   - `two_arm_peg_in_hole`
9. 新增 robosuite 目标机器人 profile：
   - `rs_dual_panda`
   - `rs_dual_iiwa`
   - `rs_baxter`
   - `rs_mobile_tiago`
10. 已经支持 `source-copy / oracle / llm` 三种迁移 planner。
11. 已经生成中文项目说明、老师演示文档和简要 PPT。

---

## 12. 当前进度判断

当前项目适合定位为：

```text
研究原型 / 课程论文 / 毕设初步系统 / workshop demo
```

如果目标是 ICRA / IROS / CoRL 主会，还需要继续加强：

1. 更严格的实验矩阵：
   - 更多任务
   - 更多 seed
   - 更多机器人目标
   - 更多 LLM 模型
2. 更强 baseline：
   - source-copy
   - prompt-only
   - few-shot
   - capability-card only
   - failure-report only
   - card + failure-report
3. 更可靠的真实控制：
   - 减少 assisted grasp
   - 增强 robosuite 接触控制
   - 或接入 robomimic / demonstration policy
4. 更完整的失败分类：
   - reachability failure
   - API mismatch
   - synchronization failure
   - force/grip mismatch
   - invalid code
   - unsafe/refusal failure
5. 更完整的统计分析：
   - success rate
   - attempts to success
   - code edit distance
   - API-call difference
   - failure distribution

---

## 13. 常见问题

### 13.1 没有 API key 能跑吗？

可以。以下命令不需要 API key：

```bash
python -m examples.smoke_test --robot dual_arm --gui
python -m examples.robosuite_migration_demo --planner oracle
python -m examples.robosuite_migration_demo --planner source-copy
```

只有 `--planner llm` 和 `main.py` 交互式 LLM 模式需要 API key。

### 13.2 robosuite warning 是错误吗？

通常不是。只要最后出现：

```text
robosuite backend is available
```

就说明当前 demo 可以继续跑。

### 13.3 macOS GUI 闪退怎么办？

robosuite/MuJoCo GUI 在 macOS 上建议用：

```bash
mjpython -m examples.robosuite_migration_demo ...
```

Ubuntu 原生系统一般可以直接用：

```bash
python -m examples.robosuite_migration_demo ...
```

### 13.4 代码迁移到底体现在哪里？

不要只看“机器人有没有动”。重点看生成代码是否因为目标机器人不同而改变：

```text
mobile_dual_arm 需要 navigate_to
dual_iiwa 需要 set_grip_force
baxter handover 需要 clearance pose
mobile_tiago 不能做真实双臂同步任务
```

代码迁移的证据应该来自：

```text
同一个任务 / 源程序
不同目标机器人
不同生成代码
不同 API 调用
不同失败类型
不同成功率
```

---

## 14. 下一步建议

短期继续做：

1. 固定 robosuite 作为正式实验平台。
2. 把 `two_arm_lift` 做成最稳定的主 demo。
3. 扩展 `two_arm_handover` 和 `two_arm_peg_in_hole` 的真实控制轨迹。
4. 跑完整 benchmark，生成统计表和失败案例。
5. 把论文主线收紧为：

```text
Capability Card 解决目标机器人能力差异；
Failure Report 解决第一次生成代码后的执行失败；
二者结合提升 LLM 机器人程序跨具身迁移成功率。
```

