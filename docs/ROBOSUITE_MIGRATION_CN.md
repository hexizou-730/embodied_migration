# Robosuite 复杂任务迁移使用说明

这个模块是当前公开仓库里保留的可运行迁移原型，使用
`robosuite + MuJoCo` 后端展示更复杂的双臂操作任务和更清楚的成功/失败。

## 新增内容

- `robosuite_backend/`
  - 复杂任务规格：`TwoArmLift`、`TwoArmHandover`、`TwoArmPegInHole`
  - 机器人 profile：`rs_dual_panda`、`rs_dual_iiwa`、`rs_baxter`、`rs_mobile_tiago`
  - 高层 skill API：抓锅把手、双臂抬锅、锤子交接、peg-in-hole 对齐和插入
  - source program -> target program 的迁移 prompt
- `examples/robosuite_migration_demo.py`
  - 单个复杂任务演示
- `benchmark/run_robosuite_migration.py`
  - 小规模复杂任务迁移 benchmark
- `requirements-robosuite.txt`
  - robosuite / MuJoCo 依赖

## 安装 robosuite 后端

```bash
cd /Users/xifan/Downloads/embodied_migration
conda activate em
python -m pip install -r requirements-robosuite.txt
```

预期输出：

```text
Successfully installed mujoco ... robosuite ...
```

检查：

```bash
python - <<'PY'
import mujoco, robosuite
print("mujoco", mujoco.__version__)
print("robosuite", robosuite.__version__)
PY
```

预期输出类似：

```text
mujoco 3.x.x
robosuite 1.5.x
```

## 不调用 LLM 的快速演示

这个命令使用内置 oracle patch，先验证复杂任务迁移框架：

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle
```

预期输出重点：

```text
Complex Robosuite Program Migration Demo
Task: two_arm_lift
Source robot: rs_dual_panda
Target robot: rs_dual_iiwa
Planner: oracle
LMP Code to execute:
robot.set_grip_force(0.85)
...
[robosuite] lifted pot to 0.16m while level
success=True reason=success_on_attempt_1
```

这说明：源机器人 Panda 的程序迁移到 IIWA 时，目标程序多了
`robot.set_grip_force(0.85)`，体现了目标 embodiment 的 gripper 约束。

## 演示失败对比

执行源代码不改，直接放到目标机器人：

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner source-copy
```

预期失败：

```text
grasp_pot_handle: grip_force=0.50 is below required 0.75
success=False reason=action-fail
```

这个对比很适合向老师解释：

- source-copy：直接复制源代码，失败
- oracle / LLM migration：根据目标能力补充 grip force，成功

## 调用 LLM 做代码迁移

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner llm
```

预期输出：

```text
Source Successful Program
Target Capability Card
Available Target APIs
LMP Code to execute:
...
```

如果第一次失败，系统会生成 Failure Report，再让 LLM 重写目标代码。

## 打开 robosuite/MuJoCo 场景

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle \
  --show-env \
  --gui
```

预期：

- 打开 robosuite/MuJoCo viewer
- 终端继续显示 LMP 迁移代码和 skill 执行日志

如果 macOS 上 `python ... --gui` 打不开 viewer，可以改用 MuJoCo 自带的
`mjpython`：

```bash
mjpython -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle \
  --show-env \
  --gui
```

注意：当前版本的 MuJoCo viewer 用于展示复杂任务场景；真正的迁移判定先由高层 skill 层完成。后续如果继续深化论文，可以把这些 skill API 替换成连续控制器或示教轨迹。

## 真实控制器轨迹模式

如果希望 `grasp_pot_handle` / `lift_pot` 不只是更新高层状态，而是真的通过
robosuite `env.step(action)` 驱动机械臂运动，使用 `--real-control`：

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

预期输出重点：

```text
[robosuite] real controller: approaching pot handles
[robosuite] real controller: handle distances d0=...m d1=...m
[robosuite] real controller: assisted grasp constraint engaged
[robosuite] real controller: lifting both arms
[robosuite] real robosuite lift: controller_ok=True, physical_success=True, pot_bottom_height=0.130m
success=True reason=success_on_attempt_1
```

这里的区别是：

- 没有 `--real-control`：只跑高层 skill 状态机，速度快，适合 benchmark。
- 有 `--real-control`：skill 会发送 robosuite 连续控制动作，GUI 中机械臂会靠近、闭合、上抬。
- `assisted grasp constraint` 是在控制器到达两个把手后启用的抽象抓取约束，作用类似物理仿真里的 attach constraint。它让演示可以稳定显示“抓住并抬起锅”，同时保留真实控制器轨迹。

对比直接复制失败：

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner source-copy \
  --real-control \
  --quiet
```

预期失败：

```text
grasp_pot_handle: grip_force=0.50 is below required 0.75
success=False reason=action-fail
```

迁移后成功：

```bash
python -m examples.robosuite_migration_demo \
  --task two_arm_lift \
  --target rs_dual_iiwa \
  --planner oracle \
  --real-control \
  --quiet
```

预期成功：

```text
set grip force to 0.85
real_physical_success=True
success=True reason=success_on_attempt_1
```

## 小规模 benchmark

```bash
python -m benchmark.run_robosuite_migration \
  --planners source-copy oracle \
  --tasks two_arm_lift two_arm_handover two_arm_peg_in_hole \
  --targets rs_dual_iiwa rs_baxter rs_mobile_tiago \
  --run-id robosuite_complex_demo
```

预期输出：

```text
[source-copy] two_arm_lift: rs_dual_panda -> rs_dual_iiwa
  -> success=False reason=action-fail
[oracle] two_arm_lift: rs_dual_panda -> rs_dual_iiwa
  -> success=True reason=success_on_attempt_1
Wrote results/robosuite_runs/robosuite_complex_demo/summary.csv
```

结果表：

```text
results/robosuite_runs/robosuite_complex_demo/summary.csv
```

表里会记录：

- success
- final_reason
- attempts
- 是否使用 `set_grip_force`
- 是否使用 `move_to_handover_pose`
- 是否使用 `align_peg_to_hole`
- 是否使用 refusal

## 论文表述重点

这个后端让论文方向从简单 pick-and-place 升级为：

**失败反馈引导的跨具身机器人程序迁移，用复杂双臂操作任务验证。**

更清楚的对比是：

- 直接复制源程序会失败
- 能力卡让 LLM 知道目标机器人限制
- failure report 让 LLM 根据执行失败修复代码
- 不同 target embodiment 需要不同代码 patch
