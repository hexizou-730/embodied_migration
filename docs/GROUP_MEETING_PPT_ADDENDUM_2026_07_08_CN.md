# 组会 PPT 补充页文案 2026-07-08

## 补充页 1：这两周的核心进展

标题：两周内从手动 prompt 修复推进到 autonomous harness

正文：

- PullCube Panda -> xArm6 保留为正案例：当前 adapter 已能在 seed 0 成功。
- Multi-seed 暴露泛化问题：失败集中在 contact geometry / reachability。
- PickCube 被定义为 hard case：TCP 对齐不等于能形成稳定 grasp。
- 新增 simple harness demo：`agent_observation -> agent_plan -> simulator tool -> tool_result` 已在远程跑通。

一句话讲法：

> 这两周的重点不是又调出一个 seed，而是把项目从“人手动喂日志给 LLM”推进到“LLM/Agent 通过 harness 使用真实仿真工具”。

## 补充页 2：Simple harness demo 证明了什么

标题：Harness 已经能调用真实 ManiSkill 工具

正文：

```text
agent_observation.json
-> agent_plan.json
-> selected_tool_command.txt
-> tool_result.json
```

远程结果：

```text
python demos/simple_harness/demo.py --run
selected_tool = run_multi_seed
executed = true
returncode = 0
```

结论：

- Agent 看到的是结构化 observation，不是 human report。
- Harness 暴露的是安全工具，不是 simulator 内部状态。
- 工具真实执行并记录结果，后续可进入下一轮 repair。

## 补充页 3：为什么这不是普通工程脚本

标题：失败来自 embodiment constraint，而不是代码语法

正文：

| 现象 | 含义 |
|---|---|
| PullCube seed 0 成功 | Adapter 迁移可行 |
| PullCube 多 seed 失败 | 接触几何和可达性泛化不足 |
| PickCube TCP 接近但不 grasp | 物理接触和夹爪包络是关键 |
| Probe 没找到成功 close 参数 | prompt-only 参数猜测不够 |

结论：

> 研究问题从“能不能生成代码”变成“如何让 LLM 在真实仿真反馈下处理 embodiment constraints”。

## 补充页 4：下一步

标题：下一步是 simulation-in-the-loop repair

正文：

1. PullCube：用 structured contact probe 修复 0-9 seed 泛化。
2. PickCube：作为 hard case，整理 grasp/contact failure taxonomy。
3. Agent 输入：只保留 `agent_observation.json`，减少人工解释。
4. Optimization：用 probe 分数指导下一轮参数/adapter 搜索。

讨论问题：

- 当前贡献更像 system/harness，还是 migration benchmark？
- 下一步优先提升 PullCube 泛化，还是深入 PickCube grasp primitive？
- 是否可以 framing 为 cross-embodiment robot code migration 的 simulation-in-the-loop harness？
