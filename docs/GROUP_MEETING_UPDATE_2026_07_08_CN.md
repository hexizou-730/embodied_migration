# 组会进度速览 2026-07-08

## 一句话进展

这两周项目从“手动让 LLM 改 adapter”推进到“把 ManiSkill 包装成 autonomous harness，让 Agent 根据仿真结果选择工具和修复路径”。

## 两周内完成了什么

| 方向 | 之前 | 现在 |
|---|---|---|
| 正结果 | PullCube seed 0 成功 | 保留为 Panda -> xArm6 的成功迁移案例 |
| 泛化评估 | 只看单 seed | 加入 PullCube 0-9 multi-seed 统计，暴露泛化不足 |
| 失败分析 | 手动读日志 | 加入 program / skill adapter / controller primitive / contact geometry / infeasibility 五层诊断 |
| 参数探索 | 手动猜参数 | 加入 structured probing，自动扫接触/抓取参数 |
| LLM 输入 | 人写长报告 | 区分 `agent_observation.json` 和 `human_report.md` |
| 自动化 | 人复制命令 | 新增 `auto.py` 和 `demos/simple_harness/demo.py` |

## 当前可展示证据

### 1. Simple harness demo 已在远程跑通

```bash
python demos/simple_harness/demo.py
python demos/simple_harness/demo.py --run
```

结果：

```text
selected_tool = run_multi_seed
executed = true
returncode = 0
```

含义：

```text
agent_observation.json -> agent_plan.json -> selected simulator tool -> tool_result.json
```

这证明 harness 可以在远程调用真实 ManiSkill 工具，而不是只生成静态 prompt。

### 2. PullCube 是正案例

高层程序不变：

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")
ret_val = robot.pull(cube, goal)
```

当前 xArm6 adapter 在 seed 0 上已成功：

```text
target generated adapter already succeeded before regeneration
```

含义：已有 target adapter 能通过真实 ManiSkill 验证。

### 3. Multi-seed 暴露泛化问题

之前 0-9 seeds 的结果显示：

```text
success seeds: 0, 4, 9
failed seeds: 1, 2, 3, 5, 6, 7, 8
```

主要失败被诊断为：

```text
contact_geometry / contact_side_reachability_failure
```

含义：seed 0 成功不是终点，目标是让 harness 自动找出哪些几何失败需要 probe / repair。

### 4. PickCube 是 hard case

Panda baseline 成功，但 xArm6 target 仍失败。Structured probe 发现：

```text
TCP 可以接近抓取点，但 Robotiq 夹爪仍不能形成稳定 grasp。
```

含义：失败不是代码语法问题，而是 embodiment constraint 和接触物理问题。

## 汇报时的主线

1. 我研究的是“同一段高层机器人程序，换机器人后为什么失败”。
2. 我不改高层 program，而是生成 target-side adapter。
3. PullCube 说明 adapter 迁移可行。
4. Multi-seed 和 PickCube 说明 prompt-only LLM 不够。
5. 因此这两周我把项目推进到 Guava-style harness：Agent 看 observation，选择工具，harness 执行仿真并返回结果。

## 现场展示命令

```bash
RUN_DIR=$(cat results/simple_demo/latest.txt)
cat "$RUN_DIR/agent_plan.json"
cat "$RUN_DIR/selected_tool_command.txt"
cat "$RUN_DIR/tool_result.json"
```

如果要展示完整自动闭环入口：

```bash
python auto.py pull --planner fallback --seeds 0-3 --max-cycles 2
```

## 下一步计划

| 下一步 | 目标 |
|---|---|
| PullCube probe + repair | 提高 0-9 seed 泛化能力 |
| PickCube 保留为 hard case | 系统整理抓取接触失败 |
| 更少人工 prompt | 让 Agent 只读 observation 和工具输出 |
| learning-guided optimization | 用 probe 结果指导下一批参数/adapter 搜索 |

## 可以请老师讨论的问题

1. 当前贡献更适合定位为 system / harness，还是 migration benchmark？
2. 下一步应该优先提升 PullCube 泛化成功率，还是深入 PickCube grasp primitive？
3. 论文 framing 是否可以写成：simulation-in-the-loop harness for cross-embodiment robot code migration？
