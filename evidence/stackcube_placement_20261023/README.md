# StackCube 对齐与放置诊断

日期：2026-10-23。StackCube-v1，Panda → xarm6_robotiq，目标控制模式
pd_ee_delta_pose，seed 0。沿用纠正过夹爪方向的 LLM 第二轮候选；没有修改环境、
控制器、物体状态或官方成功判据。

## 为什么重新检查对齐

旧候选的 align 返回 True，但阶段末 cubeA 的 z=0.07067 m，cubeB 的 z=0.02000 m。
两者中心高度差约 50.67 mm；官方堆叠条件要求 40 mm，误差不超过 5 mm。
实际偏差约 10.67 mm，官方 `is_cubeA_on_cubeB=False`。源端成功轨迹在同阶段
该状态已为 True。因而“程序认为对齐完成”不足以支持永久锁定 align。

代码新增了源端阶段状态对照：识别源端在某阶段由 False 变 True 的官方谓词，
再检查目标程序声称完成该阶段时的实测值。若有明确反例，就开放该阶段及其后的锁，
本轮仍只允许修改一个阶段。没有对应测量时保持原有定位，不凭缺失数据推断失败。
这是一条由当前成功源轨迹导出的诊断假设，尚未证明能泛化到其他任务。

## LLM 自动修复结果

命令入口 `selective.py --resume-adapter ...` 先从 reset 重放已有 Adapter，再反馈
失败状态；不会恢复或篡改物理状态。此次只开放 `_stage_align`，保留接近、抓取、提升。

| 项目 | 结果 |
|---|---|
| 重放 | 101 actions，官方成功 False |
| 执行停止阶段 | release |
| 状态对照建议修复阶段 | align |
| LLM | DeepSeek v4.1 Flash，3 次请求 |
| 新候选 | 3 次均无 AST 实质变化，被拒绝执行 |
| token / 调用费用 | 54,046 / $0.004751432 |
| 自动迁移成功 | False |

原始 prompt、回复、重放 trial 和校验错误见
`results/selective_migration/support_guard_repair/`。
续修运行不等同于从零生成，对照入口禁止将其复用为从零生成的正式基线结果。

## 无 LLM 的物理诊断对照

`placement_probe.py` 手工替换指定阶段后，每次从相同 seed reset。
只修 align 的版本以“目标方块位置减当前方块位置”为反馈误差，缓慢移到 cubeB
上方并下降，使用官方支撑状态检查对齐。接近、抓取、提升和原 release 代码不变。

| 候选 | 人工修改的方法 | action | 官方 success |
|---|---|---:|---|
| 原 LLM 候选 | 无 | 101 | False |
| 物体位置反馈对齐 | `_stage_align` | 151 | True |
| 物体位置反馈对齐 + 分段开爪 | `_stage_align`、`_stage_release` | 160 | True |

两个成功 trial 均实际执行五个阶段；最终官方状态是 on_cubeB=True、static=True、
grasped=False。原始 trial、每步动作归属、阶段末坐标、代码与哈希见
`results/placement_diagnostic/`。

这证明该场景在现有物理配置下可完成，并支持把本例的修复位置前移到 align。
人工改动的成功不能计作 LLM 自动迁移成功；同一 seed 的两个诊断变体也不能写成
跨场景成功率。当前自动生成的瓶颈是 LLM 未能产生有实质变化的局部修复。
