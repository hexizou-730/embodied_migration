# StackCube 起步验证

2026-10-23 更正：本目录早期文档将夹爪开闭方向解释反了。实测确认
`pd_ee_delta_pose` 下 `0` 张开、`0.81` 闭合；原始运行文件保留原样。
使用错误配置的第三、四阶段结果仅保留为调试记录，不能用于方法优劣比较。
补充测量和正确接口下的新生成运行见 `../stackcube_followup_20261023/README.md`。

核对日期：2026-10-22。本轮只做源端验证、目标接口检查和无接触动作探针，
没有调用 LLM，也没有运行目标迁移代码。

## 实测结果

环境：`StackCube-v1`；源机器人：`panda`；源控制模式：`pd_joint_pos`。
使用远程现有冻结程序的原始字节，未修改代码，也未重新签发多 seed 冻结记录。
源程序副本见 `source_stack_cube.py`；SHA-256 记录在 `preflight_20261021_224739/summary.json`。

| 源端场景 | 官方成功 | action 数 |
|---|---|---:|
| seed 0 | True | 103 |
| seed 1 | True | 89 |

两个场景是起步检查，不是泛化结论。原冻结记录需要重验证，不能将本次 2/2 写成当前版本 10/10。
每次执行的采样状态轨迹、官方判断和环境记录位于 `preflight_20261021_224739/t07_stack_cube/`。
轨迹是每十步采样，并非逐阶段标注或逐步完整轨迹。

目标：`xarm6_robotiq`，以下模式均可创建环境并 reset：

| 控制模式 | action 维度 | arm 通道 | 夹爪通道及实测范围 |
|---|---:|---|---|
| `pd_ee_delta_pos` | 4 | 0:3 | 3，[-1, 1] |
| `pd_ee_delta_pose` | 7 | 0:6 | 6，[0, 0.81] |

动作探针从相同 seed 的重置状态分别执行。`pd_ee_delta_pose` 中前三个通道沿相应世界轴
产生约 39 mm 的 TCP 位移；第六通道实测改变 TCP 姿态约 2.24 度。
夹爪端点当时被错误解释为 `0` 收拢、`0.81` 张开；10-23 通过指垫间距纠正为
`0` 张开、`0.81` 收拢。原始关节和 TCP 变化见
`action_probe_20261021_225142/summary.json`。这些结论只适用于当前安装版本和机器人模型。

由于堆叠需要调整夹爪姿态，后续固定采用 `pd_ee_delta_pose`。在不接触物体的条件下，
xArm6 到达 cubeA 和 cubeB 上方 12 cm 的位置误差分别为 2.1 mm 和 2.7 mm，执行
12 和 8 个动作；见 `reachability_20261021_225229/summary.json`。这只证明安全预位姿可达，
不证明抓取、搬运或释放可行。

xArm6 不在本任务声明的支持机器人列表中，初始化成功不等于任务兼容性已验证。
本研究将其作为跨 embodiment 的非原生目标；该限制必须在实验和论文中明确报告。

## 本轮发现的诊断问题

原始日志中 official success=True，但 failure_diagnosis 却记录 contact_not_established：
通用诊断把保持静止的底座 cubeB 当作应该移动的物体。
原始证据保持不变。本地已添加成功结果优先分支及回归测试，不改变任务成功标准。
修复未部署远程；失败任务的对象角色识别仍需要阶段信息，不能认为这个补丁解决了全部误诊。

## 下一步的阶段草案

以下为人工检查源码得到的设计草案，不是自动提取结果，也不是已验证的新方法。

| 阶段 | 需要保留的任务关系 | 需要适配或测量 |
|---|---|---|
| 接近 | 接近 cubeA 的抓取位姿 | 规划器调用、可达性、坐标系 |
| 抓取 | 夹持 cubeA | 夹爪几何、闭合通道、抓取信号 |
| 提升 | 带着 cubeA 离开支撑面 | 提升位姿、保持夹持 |
| 对齐 | cubeA 移到 cubeB 上方 | 从物体目标到 TCP 目标的变换 |
| 释放 | 释放后 cubeA 稳定堆在 cubeB 上 | 开爪命令、官方稳定与堆叠判断 |

当前成功信号由环境 evaluate() 提供；不得把“到达目标位姿”替代官方堆叠成功。
旧四方法 pilot 不是本方法的结果。

## 第一阶段结论

第一阶段已完成：源程序能在两个起步场景获得官方成功；目标环境、平移、旋转和夹爪
通道均执行了真实 action；两个关键安全位姿可达；任务环境和官方成功标准均未修改。
研究配置冻结在 `benchmark_tasks/t07_stack_cube/selective_migration_study.json`。
本轮接口探针不计作迁移成功；第二阶段结果记录如下。

## 第二阶段：源程序阶段表示

`stages.py` 对源程序执行确定性的 AST 和数据流分析，不调用 LLM。它先识别真实动作调用，
再沿变量的到达定义追溯实体依赖。成功源端重放同时记录每个 action 对应的源码行及动作后状态。

| 阶段 | 源码行 | 实体 | action 数 | 阶段末官方状态 |
|---|---:|---|---:|---|
| approach | 68 | cubeA | 30 | 未抓取 |
| grasp | 73–74 | cubeA | 21 | `is_cubeA_grasped=True` |
| lift | 80 | cubeA | 24 | 保持抓取 |
| align | 88 | cubeA、cubeB | 26 | 已在 cubeB 上方，仍抓取 |
| release | 90 | cubeA、cubeB | 2 | 已释放、静止、官方成功 |

103/103 个仿真 action 均映射到五个阶段。机器生成结果见 `source_stage_program.json`，
带逐 action 状态的成功轨迹见 `stage_state_trace_20261021_225808/seed_000.json`。
人工写入的预期阶段和实体只在提取后验证，提取器不读取。10-23 后续输入审计发现，
目标端首轮提示的初版完整序列化附带了这些验收元数据；现已过滤，并另行运行净输入对照。
历史运行文件保持原样，不能将它们描述为从未包含验收元数据。

第二阶段已完成，但结论限定为 StackCube 这一种源程序结构：尚未证明提取器能覆盖其他任务。

## 第三阶段：选择性迁移

`selective.py` 将五阶段顺序固化为运行接口。每个阶段执行时记录开始、结束、action 数、
已完成阶段和首个失败阶段。生成代码必须保留阶段接口；失败后的下一轮只能修改对应
`_stage_*` 方法，其他方法通过 AST 哈希锁定。

远程有界 pilot 使用 `deepseek/deepseek-v4.1-flash`、seed 0、最多三轮：

| 轮次 | 静态校验 | 目标 action | 已通过阶段 | 失败阶段 | 结果 |
|---:|---|---:|---|---|---|
| 1 | 通过 | 6 | approach | grasp | 未形成官方抓取 |
| 2 | 通过 | 6 | approach | grasp | 只修改 grasp，仍失败 |
| 3 | 拒绝 | 0 | - | - | grasp 方法没有实质变化 |

第一轮到第二轮间，`_stage_grasp` 的 AST 哈希发生变化；approach、lift、align、release
及固定入口哈希均保持不变。approach 因通过而永久锁定。完整 prompt、候选代码、trial、
阶段事件和哈希见 `selective_migration_phase3/`。

第三阶段完成的是“定位失败阶段并只修该阶段”的机制，不是 xArm6 StackCube 成功。
当前物理瓶颈仍是 grasp；transport、align 和 release 尚未在目标端执行。

## 第四阶段：最小方法对照

第四阶段固定 `StackCube-v1`、Panda 源程序、`xarm6_robotiq`、
`pd_ee_delta_pose`、seed 0、`deepseek/deepseek-v4.1-flash` 和每种方法三轮预算。
环境、控制器与官方成功信号均不修改。三种方法得到相同的目标接口与运行时 API 文档：

注意：该轮目标接口中的夹爪方向配置错误，因此以下数字不是有效方法对照结果。

| 方法 | 可见的源端信息 | 修复范围 |
|---|---|---|
| target_only | 无源程序；只有任务定义和目标接口 | 每轮重写完整 Adapter |
| whole_rewrite | 完整 Panda 源程序和同一份源阶段表示 | 每轮重写完整 Adapter |
| selective | 同一份源阶段表示 | 固定阶段顺序，只改首个失败阶段 |

公平性检查后，三种方法都能进入同一个真实仿真验收链路。单次 pilot 结果如下：

| 方法 | 有效候选 | 有动作 trial | 目标 action 总数 | 最后一次有动作失败 | 已确认阶段 | 官方成功 |
|---|---:|---:|---:|---|---|---|
| target_only | 3 | 3 | 201 | contact_not_established | 无阶段证据 | False |
| whole_rewrite | 2 | 1 | 220 | contact_not_established | 无阶段证据 | False |
| selective | 2 | 2 | 12 | grasp | approach | False |

`whole_rewrite` 的第二轮在首个 action 前发生递归错误，第三轮因代码与已测试候选
语义相同而被拒绝。`selective` 的第三轮同样因没有对 grasp 做实质修改而被拒绝。
这些错误与此前成功执行的物理 trial 分开记录，避免用生成错误覆盖真正的物理失败。

结果表位于 `selective_comparison_phase4/summary.json`，各轮 prompt、生成代码和 trial
均在对应方法目录中。每种方法只有一次独立生成，因此本轮只证明对照协议可执行，
不能说明选择性迁移在成功率、成本或稳定性上优于基线。
