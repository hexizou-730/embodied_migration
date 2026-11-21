# 论文大纲(工作稿)

状态:2026-11-21 起草。证据以 `evidence/` 已归档文件为准;标注 `[批次待填]` 的数字
等远程批次(留出重放、t05 冻结)返回后补录。所有数字都可由离线复核脚本重算。

## 工作标题(三选一)

1. **Repair, Don't Regenerate: Measured-Evidence Repair Makes LLM Robot-Code Migration Reliable**
2. What Actually Matters in LLM Cross-Embodiment Code Migration: Feedback and Localization, Not Output Scope
3. From One Failure to Near-Certainty: A Difficulty-Stratified Study of LLM Robot-Code Repair

推荐 1:主张清晰、"repair vs regenerate"的对立在标题里完成。

## 一句话主张

跨本体机器人代码迁移中,从零生成不稳定;从实测失败出发、由官方谓词定位的修复闭环
把成功率救回 ~94%;消融证明起作用的是实测反馈与阶段保留,而不是修复输出的粒度
(粒度只影响 ~38% 的 token 开销,对成功率与动作无影响)。

## 核心结果表(论文 Table 1)

| 任务 | 难度 | 从零生成 | 修复闭环(从失败候选) | 留出泛化(seeds 100–109) | 单次成本 |
|---|---|---|---|---|---|
| PickCube(Panda→xArm6,3 阶段) | 易 | **6/6,全部 1 轮**,22–28 actions | 未触发(无需修复) | — | ~$0.002/次 |
| StackCube(Panda→xArm6,5 阶段) | 难 | 历史 2/3×3 方法,不稳定 | **75/80 臂运行(93.8%),中位 1 轮** | **9/10**(修复 Adapter 固定重放) | ~$0.003–0.01/次 |

证据:`evidence/t04_pick_cube_pilot_20261121/`(6 次从零:冒烟 + s1–s5)、
`evidence/granularity_ablation_20261121/`(n=10 + n=30)、历史从零对照见
`evidence/stackcube_automatic_20261023/`。

解读:同一管线覆盖难度谱——简单任务从零即时成功;困难任务从零失败后,修复闭环
近乎必然救回。"什么时候需要修复闭环"由此有了可操作的答案。

## 消融表(论文 Table 2,修复粒度)

| 指标 | method_patch | module_rewrite | 配对差异(module−method) |
|---|---|---|---|
| n=10(seed 0–9) | 9/10 | 10/10 | 动作 −2.8 [−17.3, 8.3];token +2,469 [−42, 4,741] |
| **n=30(seed 10–39,预登记)** | 28/30 | 28/30 | 动作 +5.5 [−14.8, 30.0];**token +3,178 [2,003, 4,458]** |

- 成功率:discordant 对称(1/1),符号检验 p=1.0——粒度不影响可靠性。
- 动作:两轮 CI 均含零——粒度不产生动作节省。
- Token:整模块重写每次多 ~38%——真实的二阶效应,CI 钉紧。

归因分解(论文的关键论证段):历史从零对照中 selective 的 3 倍 action 节省
(366 vs 975/1,329)**未在本消融再现**——两臂动作几乎相同。因此该节省属于
**阶段保留 + 诊断协议**(锁定已过阶段、失败定位到单阶段),不属于输出粒度。
这一段是"我们不仅测了 X,还把 X 的功劳正确归属"的审稿护甲。

## 机制证据(论文 Method/Analysis 章)

1. **源后置条件撤销假通过**:align 自称完成但官方 `is_cubeA_on_cubeB=False`,
   与源轨迹同阶段的 True 翻转对照后撤销锁定——修复闭环可信的前提是定位可信。
   案例:`evidence/stackcube_placement_20261023/`。
2. **确定性阶段归因**:StackCube 103/103、PickCube 70/70 动作全部归入源码行;
   提取零 LLM。t02_pull_cube 的预闭合夹爪乱序(`[grasp, approach, align]`)被
   判定不适用并如实报告——提取器的边界。
3. **任务声明语义(stage_semantics)**:新任务的完成语义由 study 配置声明
   (PickCube 用真实官方谓词 `is_grasped`/`is_obj_placed`),提取器保持任务无关;
   StackCube prompt 字节级不变(fixture 测试钉死)。
4. **证据纪律**:全部 242 项(61+181)离线重算检查通过;预登记 n=30;失败全部
   留分母;供应商变化如实注明(OpenRouter→DeepSeek 直连)。

## 失败分类(论文 Table 3)

| 类别 | 实例 | 发现者 |
|---|---|---|
| 接口假设错误(方向/映射) | 夹爪开闭反向、静止闭爪 no-op | 人工探针(grasp_probe) |
| 阶段假通过 | align 自称完成,官方谓词为 False | 源后置条件对照(自动) |
| 格式脆性 | r08/seed7 嵌套函数被 AST 拒绝(单方法格式特有) | 消融自动记录 |
| 物理失败 | contact_not_established、预算耗尽(从零) | 诊断系统 |

"人修接口、LLM 修策略"的分工是讨论章的观点素材:自动化的真实边界。

## 已补数字(2026-11-21 批次)

- 留出重放:n=10 首个成功修复 Adapter 在 seeds 100–109 上 **9/10** 官方成功
  (仅 seed 100 失败;对照历史 PushCube 固定 Adapter 的 5/10)。
  `evidence/granularity_ablation_20261121/heldout_repair_adapter_t07/`
- t05_place_sphere:候选源程序验证 **0/10**,未冻结,出局(候选不等于成功,
  如实记录);难度阶梯保持 PickCube/StackCube 两点。

## 章节骨架与证据映射

| 章节 | 内容 | 证据来源 |
|---|---|---|
| 1 Introduction | 迁移问题、naive 从零、三个发现 | Table 1 + 消融 |
| 2 Related Work | 跨本体策略迁移(weights);LLM 机器人代码生成(CaP/LMP/Voyager);自动修复/self-debugging | — |
| 3 Method | 阶段程序、重放+定位(含后置条件撤销)、有界修复、契约 | selective_migration.py / repair_granularity.py |
| 4 Setup | 两任务两机器人对、deepseek-flash、seed 策略、证据纪律 | study configs |
| 5.1 难度分层转移 | Table 1 | t04 pilot + ablation |
| 5.2 修复可靠性 + 消融 | Table 2 + 配对图 | granularity README/analysis |
| 5.3 归因分解 | 从零节省=协议≠粒度 | 三组数字对照 |
| 5.4 留出泛化 | 修复 Adapter 9/10(seeds 100–109) | heldout_repair_adapter_t07 |
| 5.5 失败分类 | Table 3 | 各证据目录 |
| 6 Discussion | 何时需要修复闭环;单一场景对、flash 量级、resume 性质的边界;"人修接口"的自动化含义 | 全部 |
| 7 Conclusion | — | — |

## 图表清单

- Fig 1:管线图(源程序→阶段程序→生成→实测失败→定位→单阶段修复→官方验收)
- Fig 2:逐对 token/action 差异散点(n=40,seed 7 异常点标注)
- Table 1/2/3 如上
- Fig 3(可选):align 假通过案例时间线

## 遗留风险与诚实边界(写作时必须保留)

1. 修复数据是 resume 性质(从同一失败候选出发),不是从零成功率;两口径永不混报。
2. n=40 仍属 pilot 量级,所有比较只报效应量与 CI,不做显著优越性宣称。
3. 单一物理场景 × 两任务 × 一对机器人;flash 单模型;不宣称跨任务/跨机器人普适。
4. 修复闭环依赖人工先修好接口假设(夹爪方向);该前提在 Setup 中明说。

## 下一步(按优先级)

1. Fig 2 脚本(从 granularity_analysis.json 画逐对差异)——离线、零成本
2. Section 5 起草(结果章先行,数字全部可溯)
3. StackCube 从零侧若要更强的"难"行,可补 n=5–10 从零(约 $0.05–0.1,可选)
4. t05 已出局(候选 0/10);第三难度点仅在 deadline 富余时重新找可用源程序
