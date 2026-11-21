# StackCube 修复粒度消融

日期:2026-11-21。任务 StackCube-v1,源机器人 Panda,目标 xarm6_robotiq,
目标控制模式 pd_ee_delta_pose。模型为 DeepSeek 官方 API 上的 deepseek-flash
(V4.1 Flash 直连名),temperature=0,thinking disabled,输出上限 6000 tokens。
环境、机器人控制器和官方成功标准保持不变。

本实验回答 PROGRESS.md 第 5 条遗留的问题:固定同一失败候选与实测状态反馈,
只改变修复输出粒度,单方法修复与整体重写是否有差异。

## 起点与协议

所有运行从同一份钉住的失败候选出发(即三次 method_patch 的同一起点):

- 文件:`evidence/stackcube_placement_20261023/results/selective_migration/support_guard_repair/resume_replay/target_adapter.py`
- SHA-256:`fa29a66bd7faab15d73815595dcc5dcb4a779cd0b74fcaeba7df1f38c789cae6`
- 重放:每次先 reset 重放该候选(101 actions,官方成功 False,执行停止在 release,
  源后置条件对照定位 align 假通过);重放动作只记录,不计入新候选累计指标。

| 臂 | LLM 输出 | 修改范围 | 与历史的关系 |
|---|---|---|---|
| method_patch | 单个 `_stage_align` 方法 | Harness 拼回,其余方法 AST 哈希锁定 | 复用 `selective.py --resume-adapter` 代码路径,prompt 逐字节不变 |
| module_rewrite | 完整模块 | 任意 `_stage_*`/`__init__`,受同一五阶段接口约束 | 新臂,证据块与 method_patch 逐字节相同,仅输出范围不同 |

两臂共享:重放、修复定位、实测状态反馈(stage_boundary_states、final_state、
official_evaluation、repair_localization)、环境 seed 0、每运行最多 3 轮、
成对 LLM 采样 seed(0–9)、temperature=0、token 上限 6000。每臂 10 次独立修复
运行,共 20 次。无实质修改(语义不变)的候选在执行前被拒绝。

供应商说明:历史 method_patch 与三方法对照经 OpenRouter(fallback 开启,实际
供应商随请求漂移);本组为 DeepSeek 官方 API 直连。两臂同供应商、同会话,成对
比较不受影响,但不得与本组之外的历史费用数字直接混合比较。DeepSeek 直连不回传
费用字段,记录中 `llm_cost_usd` 为 0;消耗以 tokens 为准,费用可按官方定价折算。

冒烟记录:`granularity_smoke` 因 `.env` 中 OpenRouter key 失效(401 User not
found)记为 llm_unavailable,LLM 费用 0,仅证明失败路径可复现;`granularity_smoke_ds`
为本组正式冒烟(1 对,两臂官方成功)。

## 结果

| 臂 | 官方成功 | 成功率(Wilson 95%) | 成功轮数中位 | 累计新候选 actions | 累计 tokens |
|---|---:|---|---:|---:|---:|
| method_patch | 9/10 | 0.90 [0.60, 0.98] | 1 | 377 | 55,184 |
| module_rewrite | 10/10 | 1.00 [0.72, 1.00] | 1 | 349 | 79,874 |

配对结果(采样 seed 0–9,10 对全部完成,无基础设施失败):

- 成功差异:method 独有失败 0 对,module 独有失败 1 对(seed 7);双侧精确符号检验 p = 1.0。
- 成对动作差(module − method):均值 -2.8,bootstrap 95% [-17.3, 8.3],不排除零。
- 成对 token 差(module − method):均值 +2,469,bootstrap 95% [-42.1, 4,740.5],不排除零。
  逐对点估计显示 module 每次成功运行多消耗约 2.6k tokens(整模块输出的结构性开销),
  但 seed 7 上 method 的多轮失败(共 13,856 tokens)抵消了该差异。
- 证据复核:61 项离线重算检查全部通过。

seed 7 失败解剖(method_patch):第 1、2 轮为 align 的真实物理失败(41/49 actions,
定位与反馈正常);第 3 轮 LLM 在替换方法内定义嵌套函数,被
"Local repair changed the module method set" 拒绝,未执行——这是单方法格式特有的
一种脆性。同 seed 的 module_rewrite 以 1 轮成功(31 actions,7,352 tokens,
仅改 `_stage_align`)。

判读:两臂等成功率、等中位轮数、动作无可分离差异;token 差为正点估计但在 n=10
不显著。结合历史从零生成对照(selective 366 vs 975/1,329 actions),本消融把归因
收窄:大幅 action 节省来自阶段保留与诊断协议,而非单方法输出粒度;输出粒度只影响
每次调用的输出开销。实测证据驱动的修复闭环本身高度可靠:20 次臂运行 19 次官方
成功,中位 1 轮。主张应据此调整为"实测证据修复闭环可靠,输出粒度是二阶因素"。

判读边界:n=10 是 pilot 样本量;单一物理场景、单一失败候选的续修证据;
所有数字只作效应量描述,不做显著性宣称;续修结果不得混入从零生成对照。

## 复核

`analyze_ablation.py` 从原始 cycle 记录离线重算全部汇总,校验与执行时
summary 的漂移、成对采样 seed 一致性、失败候选哈希一致性,输出
`granularity_analysis.json` 与本 README 的表格来源 `granularity_analysis.md`;
不调用 LLM 或仿真,不覆盖原始证据。

结果目录:`results/repair_granularity/granularity_ablation_n10/`(r01–r10
每臂完整 prompt、LLM 响应、trial 与 cycle 记录)。
