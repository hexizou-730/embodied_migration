# 相关工作与论文定位

更新日期：2026-09-08。这里不做论文堆砌，只回答两个问题：已有工作解决了什么，
以及本项目还需要证明什么。

## 1. 现有研究路线

| 路线 | 代表工作 | 主要解决的问题 | 与本项目的差别 |
|---|---|---|---|
| LLM 生成机器人程序 | [Code as Policies](https://arxiv.org/abs/2209.07753) | 把语言指令组合成可执行 policy code 和控制 API 调用 | 重点是“从语言写程序”，不是在固定高层程序下迁移 embodiment-specific adapter |
| 跨 embodiment 策略学习 | [Open X-Embodiment / RT-X](https://openreview.net/pdf?id=zraBtFgxT0) | 统一多机器人数据并训练 X-robot policy | 依赖大规模轨迹与训练；本项目研究无需训练的代码合成与修复 |
| 跨硬件基础设施 | [RIO](https://arxiv.org/abs/2605.11564) | 统一 robot I/O、传感器、数据格式和策略部署接口 | 解决“接口能否复用”；本项目继续研究接口统一后任务代码为何仍受接触和可达性约束 |
| 仿真闭环代码生成 | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) | 用 MLLM 和仿真反馈生成专家程序，再生产双臂训练轨迹 | 目标是数据生成和策略训练；本项目直接评估 target adapter 的 held-out 任务成功率 |
| Embodied harness | [Guava](https://arxiv.org/abs/2606.18363) | 比较 observation、action abstraction 和 agent workflow，形成 perception-reasoning-action loop | Guava 研究通用 harness 设计；本项目把 harness 限定为 adapter-only 迁移并冻结不可修改接口 |
| 自主程序调试与技能积累 | [ASPIRE](https://arxiv.org/abs/2607.00272) | 用细粒度执行 trace 修复代码，并将成功经验积累为技能库 | ASPIRE 面向开放式多任务技能发现；本项目强调受控变量、五层归因、等预算 probe 和跨 seed 反例 |
| 反例驱动程序合成 | [CEGIS / SyGuS](https://www.microsoft.com/en-us/research/wp-content/uploads/2013/05/fmcad13.pdf) | 候选程序经过验证器检查，失败反例进入下一轮合成 | 经典验证器通常基于逻辑规格；本项目把 ManiSkill 物理执行当作验证器，反例是不满足 embodiment constraint 的 rollout |

ManiSkill 适合作为当前实验环境，是因为 task card 明确给出随机化、支持机器人和 success
condition，并允许在 Gym 接口下替换部分 robot embodiment。它提供的是可重复的物理执行环境，
不是“真实机器人已经成功”的证明。[ManiSkill task 文档](https://maniskill.readthedocs.io/en/latest/tasks/index.html)

## 2. 核心技术空缺

现有工作已经证明 LLM 能写机器人程序，也能在仿真中根据失败继续修复。仅仅做
“LLM + simulator + retry”已经不足以成为贡献。本项目要研究的是更具体的问题：

> 当高层 Program、底层 controller、simulator 和 success criterion 都固定时，能否只修改
> target-side adapter，利用 embodiment contract、物理反例和主动 probe，合成在未见 seed 上
> 仍然有效的 guarded adapter？

这里的技术空缺有四个：

1. **失败归因**：任务失败不等于程序语义错误，需要区分 program、skill adapter、controller
   primitive、contact geometry 和 infeasibility。
2. **物理信息获取**：原始 success/fail 太粗，需要从 TCP、物体、目标和阶段残差中构造可复算反例。
3. **低成本探测**：固定网格会浪费仿真调用，需要在相同 probe budget 下选择最有信息的物理实验。
4. **跨 seed 合成**：单 seed 固定轨迹容易过拟合，输出应包含基于运行时状态的 guard、分支和显式失败路径。

## 3. 当前 Motivating Examples

下面两类历史远程实验说明，迁移失败不是简单的语法错误，也不能靠无限增加步数解决：

1. **PullCube 的单 seed 成功不能代表迁移完成。** Panda 到 xArm6 的固定远侧接触策略在
   seed 0 成功，但一组 10-seed 测试只有 3 次成功。失败集中在 approach/descent，部分
   TCP 到阶段目标的残差达到 0.6--0.8 m。这说明固定接触点会随初始几何进入不可达区域，
   需要运行时 reachability guard 和反例驱动的接触侧选择。
2. **TCP 对齐不能推出抓取成立。** 一次 PickCube 失败中，`tcp_grasp_xy=0.0027 m`、
   `tcp_grasp_z=0.0002 m`、`cube_disp_xy=0.0052 m`，但 `is_grasping=False`；随后 32 组
   close-envelope probe 也没有得到抓取。这把问题从“路径没走到”进一步定位到夹爪包络、
   接触高度和闭合动力学。

这些数值目前属于历史实验记录，用来建立 motivation，不替代冻结的论文结果。当前仓库中
对应 xArm6 目标 Adapter 与 neutral seed 同哈希，正式结论必须由新的远程 run、完整日志和
Adapter snapshot 重新确认。

## 4. 当前候选贡献

这些是需要实验验证的候选贡献，不能在结果完成前写成既定结论：

- **受控 adapter-only 迁移协议**：固定四类接口，只允许修改目标 adapter，并用 SHA 检查是否越界。
- **反例驱动的 embodiment adapter synthesis**：development rollout 产生反例，反例约束下一轮代码搜索。
- **约束感知 active probe**：将 violated constraint 映射到少量可控参数，再用测量结果选择下一批实验。
- **guarded adapter contract**：机器检查生成代码是否读取状态、检查终止、做数值分支并保留真实失败路径。
- **discovery-first 新任务入口**：从 ManiSkill 环境动态读取 action/controller/entity/evaluate
  接口，先验证并冻结源 Program，再只生成目标 Adapter；每轮 prompt、代码、仿真结果和 SHA
  单独归档。该入口目前通过 dry-run 和单元测试，真实新任务成功率仍待远程实验。
- **可追溯评测**：严格区分 official-supported 与 stress-test case，development 与 held-out seed 不重叠，
  LLM、oracle 和人工代码来源分别记录。

## 5. 与最接近工作的正面比较

| 问题 | Guava | ASPIRE | RoboTwin 2.0 | 本项目需要证明 |
|---|---|---|---|---|
| 生成对象 | agent 动作/工具调用 | 完整技能程序 | 专家数据生成程序 | 仅 target adapter |
| 反馈 | 多模态 observation | 细粒度 primitive trace | 图像与错误反馈 | 状态量、阶段残差、五层诊断、counterexample |
| embodiment 机制 | 语义动作抽象 | 技能迁移与 API 适配 | affordance-aware 候选 | 机器可读 contract + reach/contact constraints |
| 搜索 | harness workflow | 进化式程序搜索 | 迭代代码修复 | 等预算 active probe + guarded synthesis |
| 主要结果 | 任务执行能力 | 持续学习和技能积累 | 数据规模与下游策略 | held-out adapter success、修复成本、诊断准确率 |
| 受控变量 | 不以 adapter-only 为核心 | 多组件可变 | 数据和程序均可变 | Program/controller/simulator/success 固定 |

因此论文不能把卖点写成“搭建了 autonomous harness”。更准确的卖点是：

> 用可机器检查的 embodiment constraints 和主动物理反例，在冻结执行语义的条件下只合成
> target adapter，并量化这种受控迁移是否比 one-shot、raw failure、diagnosis-only 和 fixed-grid
> probe 更有效、更省仿真。

## 6. 仍未完成的证明

- 尚无本地可复算的完整远程 GPU 结果矩阵。
- RQ1/RQ2 必须在 held-out seeds 上比较 Ours、B2 和 B5；不能只展示 seed 0。
- RQ3 需要独立人工盲标，不能用诊断器自己生成的标签评估自己。
- RQ4 必须把官方支持 case 和强制加载的 stress-test case 分开报告。
- RQ5 需要同时报告 simulator calls、environment steps、LLM calls/tokens 和 wall-clock time。
- 在扩大到更多任务和机器人前，先完成当前五个 case 的 pilot，确认方法差异确实能被测出来。
- 未注册任务入口还需要至少一个真实 ManiSkill 正例；dry-run 只能证明流程可启动，不能证明
  LLM 已经完成新任务迁移。
