# 反例驱动的机器人 Adapter 合成

## 1. 问题定义

给定一个在源机器人上成立的高层程序：

```python
ret_val = robot.pull(cube, goal)
```

迁移时固定以下四项：

- 高层程序；
- ManiSkill 底层 controller；
- simulator；
- 任务 success signal。

系统只允许生成目标机器人的 adapter。Adapter 负责把 `robot.pull(...)` 或
`robot.grasp(...)` 翻译成目标机器人的 `env.step(action)` 序列。

## 2. 方法

当前方法称为：

> Counterexample-Guided Embodiment Adapter Synthesis

完整闭环为：

```text
中性 seed adapter
→ LLM 生成候选 adapter
→ development seeds 仿真
→ 选择信息最完整的失败反例
→ 定位违反的 embodiment constraint
→ 主动选择少量 probe
→ probe 测量进入下一轮 prompt
→ LLM 生成带状态 guard 的新 adapter
→ development 达标后只在 held-out seeds 上最终验证
```

Held-out seeds 只用于最终评估，不会反馈给 LLM。

## 3. Embodiment Contract

`maniskill_backend/embodiment_contracts.py` 给每个机器人定义机器可读契约：

- action 维度和通道布局；
- fixed/mobile base；
- TCP frame；
- gripper action 语义；
- controller、simulator、success signal 等冻结边界；
- 必须通过仿真测量的 reachability/contact 约束。

例如：

```text
xArm6: action[0:3]=TCP xyz, action[3]=gripper, fixed base
Fetch: action[0:3]=TCP xyz, action[3]=gripper,
       action[4:7]=body, action[7:9]=base
```

Contract 约束 LLM 的可修改范围，但不提供成功轨迹。

## 4. Counterexample

一次失败不再只是：

```text
success=False
```

而会转成结构化反例：

```json
{
  "seed": 5,
  "failure_layer": "contact_geometry",
  "failure_reason": "contact_side_reachability_failure",
  "violated_constraints": [
    "contact_pose_reachable_before_descent",
    "contact_side_selected_from_runtime_geometry"
  ],
  "evidence": {
    "stage": "descent",
    "tcp_stage_error_norm": 0.09094,
    "tcp_cube_xy": 0.02912
  }
}
```

系统按诊断可信度、数值完整度和物理阶段给反例打信息分，优先修复最有解释力的失败。

## 5. Active Probe

Probe 不再默认完整扫描所有参数。系统根据失败约束选择参数：

| 失败 | 主动 probe 参数 |
|---|---|
| far-side 不可达 | contact x/z offset、approach height |
| 已接触但拖动不足 | drag strength、down bias、stages |
| 抓取侧推 | grasp height、close command、close steps |
| 对齐但无法形成抓取 | close envelope 和 settle timing |

如果已有 probe 结果，下一轮围绕高分测量点做局部搜索；如果没有，使用小规模
one-factor-at-a-time 设计。这样 probe 是物理测量，不是把答案直接告诉 LLM。

## 6. Guarded Adapter

目标不是生成一个只适配 seed 0 的固定轨迹，而是生成带状态分支的 adapter：

```python
if far_side_is_reachable:
    use_far_side_contact()
elif near_side_is_reachable:
    use_alternative_contact()
else:
    return measured_infeasible_failure()
```

每个 fallback 必须产生可测量进展，或者返回有证据的失败；不能靠增加 episode
步数反复执行同一无效动作。

## 7. 一条命令

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis
```

无 ManiSkill/GPU 时先检查流程：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis \
  --dry-run
```

## 8. 当前边界

该框架只对已注册 case 有可执行 simulator/probe backend。它不保证任意任务和任意
机器人自动成功。研究目标是用明确约束、反例和有限仿真预算提高 adapter 合成的
泛化能力，并正确识别当前 embodiment 下的不可行情况。
