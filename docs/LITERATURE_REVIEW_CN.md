# 文献综述：LLM 驱动的跨机器人代码迁移与仿真闭环修复

## 1. 项目定位

本项目研究的问题可以概括为：

> 同一个高层机器人程序，例如 `robot.pull(cube, goal)` 或 `robot.grasp(cube); robot.place(cube, goal)`，在换到不同机器人后为什么会失败？能否让 LLM 自动生成目标机器人侧的 adapter，并通过真实 ManiSkill 仿真反馈持续修复？

这里的重点不是让 LLM 重新写任务逻辑，而是保持高层程序不变，只迁移目标机器人的执行层：

```text
high-level program
-> target-side adapter
-> controller action
-> real ManiSkill env.step(action)
-> success / failure / diagnosis
```

因此，本项目处在三个方向的交叉点：

1. 机器人操作 benchmark 与仿真环境。
2. LLM 生成机器人策略代码。
3. 跨 embodiment 迁移与 simulation-in-the-loop repair。

## 2. 过去十年的发展脉络

### 2.1 2016-2021：从单任务控制到标准化机器人 benchmark

早期机器人学习主要关注单任务控制、强化学习、模仿学习和仿真环境搭建。2019-2021 年出现了一批标准化 manipulation benchmark，使“可复现评估”成为可能。

代表工作包括：

- **Meta-World**：包含 50 个桌面机械臂操作任务，用于 multi-task RL 和 meta-RL，强调任务分布和泛化评估。[Meta-World](https://meta-world.github.io/)
- **RLBench**：提供 100 个手工设计的机器人操作任务，支持视觉、深度、分割、proprioception，并提供基于 motion planner 的演示。[RLBench](https://sites.google.com/view/rlbench)
- **robosuite**：基于 MuJoCo 的模块化机器人学习仿真框架，支持多机器人、多任务和可复现实验。[robosuite](https://robosuite.ai/)
- **ManiSkill / ManiSkill2**：面向 generalizable manipulation skills，强调真实物理仿真、多对象变化和大规模 demonstration。ManiSkill2 包含 20 个任务族、2000+ 对象模型和 4M+ demonstration frames。[ManiSkill2 OpenReview](https://openreview.net/forum?id=b_CQDy9vrD1)

这一阶段的核心贡献是：把机器人操作任务做成可跑、可比、可复现的 benchmark。但此时通常默认机器人、controller 和任务接口是固定的；“同一段高层代码换机器人后如何迁移”还不是主要问题。

### 2.2 2022：LLM 进入机器人任务规划与代码生成

2022 年之后，LLM 开始被用于机器人任务规划和策略代码生成。

- **Language Models as Zero-Shot Planners** 证明 LLM 可以把自然语言任务分解成可执行的中层动作，但需要把自由文本计划映射到环境允许的动作空间。[arXiv](https://arxiv.org/abs/2201.07207)
- **SayCan** 把 LLM 的语义规划能力和机器人技能 affordance/value function 结合，解决“LLM 说得通但机器人做不到”的 grounding 问题。[SayCan](https://say-can.github.io/)
- **Code as Policies** 明确提出用 LLM 生成机器人 policy code，即 Language Model Programs，可以组合 perception、几何计算和 control primitive。[Code as Policies](https://code-as-policies.github.io/)
- **ProgPrompt** 用程序式 prompt 描述可用 action、object 和任务上下文，降低 LLM 生成不可执行计划的概率。[ProgPrompt](https://progprompt.github.io/)

这一阶段和本项目最接近的是 **Code as Policies**。但它主要关注“从自然语言生成机器人代码”，而本项目关注的是“已有高层机器人程序在换 embodiment 后，如何迁移底层执行 adapter”。

### 2.3 2023：闭环反馈、3D grounding 和 embodied multimodal models

2023 年的进展开始把 LLM 从开放式规划推进到带环境反馈的闭环机器人系统。

- **Inner Monologue** 让 LLM 使用环境反馈、成功检测、场景描述和人类反馈形成闭环推理，提高 embodied planning 成功率。[PMLR](https://proceedings.mlr.press/v205/huang23c.html)
- **VoxPoser** 使用 LLM/VLM 生成 3D value maps，将语言约束转化为空间 affordance 和 trajectory planning。[VoxPoser](https://voxposer.github.io/)
- **PaLM-E** 把视觉、状态估计和语言统一输入到 embodied multimodal language model 中，并展示多 embodiment 正迁移。[PaLM-E](https://palm-e.github.io/)
- **RT-2** 把 vision-language model 和机器人动作统一成 vision-language-action model，让模型从网络知识迁移到机器人控制。[RT-2](https://robotics-transformer2.github.io/)
- **Voyager** 虽然是在 Minecraft 中，但它提出了很重要的 agent 结构：执行代码、读取环境反馈、修复程序、积累技能库。[Voyager](https://voyager.minedojo.org/)

这一阶段的关键趋势是：LLM 不再只是一次性输出计划，而是开始进入 **feedback loop**。这和本项目的 harness 很契合：失败结果不是人工读日志后手动改 prompt，而是变成结构化 observation，再给 agent/LLM 做下一轮修复。

### 2.4 2023-2025：跨 embodiment 大模型和通用机器人策略

另一个相关方向是直接训练跨机器人通用策略。

- **Open X-Embodiment / RT-X** 汇集 1M+ 真实机器人轨迹，覆盖 22 种 robot embodiments，并展示跨机器人正迁移。[Open X-Embodiment](https://robotics-transformer-x.github.io/)
- **RoboCat** 训练 multi-embodiment, multi-task generalist agent，可以适应新任务和新机器人，并提出自我生成数据的改进循环。[RoboCat](https://arxiv.org/abs/2306.11706)
- **Octo** 是开源 generalist robot policy，基于 Open X-Embodiment 的 800k trajectories 训练，可快速 finetune 到新 observation/action spaces。[Octo](https://octo-models.github.io/)
- **CrossFormer** 进一步关注不同机器人传感器、动作空间和控制频率不一致时，如何训练统一策略。[CrossFormer](https://crossformer-model.github.io/)

这些工作走的是“数据和模型规模化”的路线：收集大量跨机器人数据，训练统一 policy 或 foundation model。本项目走的是另一条轻量路线：不训练大模型，不收集大规模轨迹，而是让 LLM 生成 adapter，通过仿真验证和诊断逐步修复。

### 2.5 2025-2026：agentic harness 与 simulation-in-the-loop repair

最近两年的趋势更接近本项目当前方向：把 LLM 放进一个工具化、可验证、可反馈的 harness 中。

- **Reflexion** 提出 language agent 可以通过任务反馈写入 verbal memory，从而在后续尝试中改进，不需要更新模型权重。[Reflexion](https://arxiv.org/abs/2303.11366)
- **LLM-driven corrective robot operation code generation** 探索用 LLM 对机器人操作代码做模拟和纠错，目标是提高生成代码可靠性。[arXiv](https://arxiv.org/html/2512.02002v3)
- **ASPIRE** 是非常接近当前项目 framing 的新工作：它让 agent 自动写和修复 code-as-policy 机器人控制程序，利用闭环执行引擎、失败诊断、技能库和验证循环来积累可复用技能。[ASPIRE](https://arxiv.org/abs/2607.00272)

这一阶段说明一个趋势：只靠 prompt-only 生成已经不够，必须把 LLM 包进一个有真实执行、有诊断、有安全边界、有工具接口的 harness 中。

## 3. 与本项目最接近的工作对比

| 方向 | 代表工作 | 它们做什么 | 和本项目的关系 | 本项目的区别 |
|---|---|---|---|---|
| LLM 机器人代码生成 | Code as Policies | 从语言生成机器人 policy code | 本项目也生成可执行 Python adapter | 本项目不是从自然语言任务开始，而是迁移已有高层程序 |
| LLM grounded planning | SayCan, ProgPrompt | 让 LLM 只选择可执行 action 或 skill | 本项目也强调可执行 API 和安全边界 | 本项目进一步要求不同机器人 action/controller/contact geometry 适配 |
| 闭环语言反馈 | Inner Monologue, Reflexion | 用环境反馈改善下一次规划 | 本项目也用 failure diagnosis / probe / online trace | 本项目反馈来自 ManiSkill 物理执行，而不是纯文本或高层状态 |
| 3D 约束与轨迹 | VoxPoser | 从语言生成空间约束和轨迹 | 本项目也关心接触点、TCP、goal、几何约束 | 本项目更关注跨机器人 adapter，而不是开放词汇 manipulation |
| 跨 embodiment policy | RT-X, RoboCat, Octo, CrossFormer | 用大规模多机器人数据训练统一策略 | 本项目同样研究 embodiment gap | 本项目不训练大模型，而是做 LLM adapter synthesis 和仿真闭环修复 |
| Agentic robot skill repair | ASPIRE | agent 写 code-as-policy，执行、诊断、修复、积累技能 | 和本项目最接近 | 本项目更聚焦“同一高层程序跨机器人迁移”和五层失败分类 |

## 4. 本项目的研究空缺和贡献点

现有工作已经证明了三件事：

1. LLM 可以写机器人代码。
2. 多机器人数据可以训练跨 embodiment policy。
3. 环境反馈可以帮助 agent 修复行为。

但相对缺少一个细分问题的系统研究：

> 当高层 LMP 程序固定时，机器人换了 embodiment，失败到底发生在哪一层？LLM 能否只通过生成 target-side adapter 来恢复任务成功？

本项目的贡献可以这样表述：

1. **问题定义**：提出 cross-embodiment LMP adapter migration。高层程序不变，迁移目标机器人执行 adapter。
2. **失败分类**：把失败分为 program / skill adapter / controller primitive / contact geometry / infeasibility 五层，比简单 success/fail 更有诊断价值。
3. **仿真闭环**：通过 ManiSkill 真实 `env.step(action)` 验证，不允许直接修改 simulator、controller 或 success signal。
4. **结构化 probe**：对接触点、闭合参数、拖拽参数做自动扫描，减少纯 prompt 猜测。
5. **online harness**：开始支持在同一 episode 中反复观察 TCP/cube/goal，并选择下一段动作，实现“边做边看边改”。

## 5. 过去十年进度总结

| 阶段 | 主流问题 | 方法特点 | 局限 |
|---|---|---|---|
| 2016-2019 | 单任务机器人学习 | RL、IL、仿真控制 | 泛化弱，任务和机器人固定 |
| 2019-2021 | 标准化 manipulation benchmark | Meta-World、RLBench、robosuite、ManiSkill | 多数不处理跨机器人代码迁移 |
| 2022 | LLM 规划和机器人代码生成 | SayCan、Code as Policies、ProgPrompt | 依赖预定义 skill/API，物理失败诊断弱 |
| 2023 | 闭环反馈和 embodied multimodal | Inner Monologue、VoxPoser、PaLM-E、RT-2 | 多数仍不直接研究 adapter migration |
| 2023-2025 | 跨机器人通用策略 | RT-X、RoboCat、Octo、CrossFormer | 需要大规模跨机器人数据和训练 |
| 2025-2026 | Agentic harness 和自修复机器人代码 | Reflexion-style loops、ASPIRE、simulation-in-loop repair | 安全边界、可解释诊断和跨任务泛化仍开放 |

## 6. 对本项目论文 framing 的建议

可以把项目定位成：

> Simulation-in-the-loop adapter synthesis for cross-embodiment robot code migration.

更中文一点：

> 面向跨机器人 embodiment 的 LMP adapter 自动迁移：基于 ManiSkill 仿真反馈的诊断、probe 与在线修复框架。

这个 framing 的好处是：

- 不和 RT-X/Octo 这类大模型路线硬碰硬。
- 不声称任意机器人任意任务都能成功。
- 突出你的实际贡献：adapter、harness、failure diagnosis、structured probing、online observe-act loop。
- 可以把成功案例和 hard case 都合理纳入论文叙事：成功案例证明可行性，PickCube / 多 seed 泛化证明 prompt-only 不够，从而引出 harness 和 constraint-aware repair 的必要性。

## 参考文献与资料

1. Meta-World: https://meta-world.github.io/
2. RLBench: https://sites.google.com/view/rlbench
3. robosuite: https://robosuite.ai/
4. ManiSkill2: https://openreview.net/forum?id=b_CQDy9vrD1
5. Language Models as Zero-Shot Planners: https://arxiv.org/abs/2201.07207
6. SayCan: https://say-can.github.io/
7. Code as Policies: https://code-as-policies.github.io/
8. ProgPrompt: https://progprompt.github.io/
9. Inner Monologue: https://proceedings.mlr.press/v205/huang23c.html
10. VoxPoser: https://voxposer.github.io/
11. PaLM-E: https://palm-e.github.io/
12. RT-2: https://robotics-transformer2.github.io/
13. Voyager: https://voyager.minedojo.org/
14. Open X-Embodiment / RT-X: https://robotics-transformer-x.github.io/
15. RoboCat: https://arxiv.org/abs/2306.11706
16. Octo: https://octo-models.github.io/
17. CrossFormer: https://crossformer-model.github.io/
18. Reflexion: https://arxiv.org/abs/2303.11366
19. ASPIRE: https://arxiv.org/abs/2607.00272
