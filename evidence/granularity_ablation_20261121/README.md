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

<!-- RESULTS_PENDING:由 analyze_ablation.py 离线重算后填入 -->

| 臂 | 官方成功 | 成功率(Wilson 95%) | 成功轮数中位 | 累计新候选 actions | 累计 tokens |
|---|---:|---|---:|---:|---:|
| method_patch | PENDING | PENDING | PENDING | PENDING | PENDING |
| module_rewrite | PENDING | PENDING | PENDING | PENDING | PENDING |

配对结果(逐采样 seed):

<!-- PAIRED_TABLE_PENDING -->

判读边界:n=10 是 pilot 样本量;单一物理场景、单一失败候选的续修证据;
所有数字只作效应量描述,不做显著性宣称;续修结果不得混入从零生成对照。

## 复核

`analyze_ablation.py` 从原始 cycle 记录离线重算全部汇总,校验与执行时
summary 的漂移、成对采样 seed 一致性、失败候选哈希一致性,输出
`granularity_analysis.json` 与本 README 的表格来源 `granularity_analysis.md`;
不调用 LLM 或仿真,不覆盖原始证据。

结果目录:`results/repair_granularity/granularity_ablation_n10/`(r01–r10
每臂完整 prompt、LLM 响应、trial 与 cycle 记录)。
