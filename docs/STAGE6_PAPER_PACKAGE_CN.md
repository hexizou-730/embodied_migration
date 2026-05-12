# 阶段 6: 论文表格与图表包生成操作手册

阶段 6 的目标是把 Stage 5 的实验 run 目录转换成论文可用材料:

- CSV 统计表
- LaTeX 表格
- SVG 图
- 自动生成的结果小节草稿
- 实验 manifest
- LaTeX `\input{...}` 包含文件

阶段 6 不调用 LLM, 不消耗 API。它只读取已有的 `results/runs/<run_id>/trials/*.json`。

## 0. 前置条件

你需要已经有一个 Stage 5 run, 默认是:

```text
results/runs/stage5_mobile_dual_seeded/
```

如果还没有, 先运行:

```bash
conda activate em
cd /Users/xifan/Downloads/embodied_migration
bash scripts/run_stage5_experiments.sh stage5_mobile_dual_seeded
```

Stage 5 成功后, 预期会看到:

```text
Stage-5 seeded experiment is ready:
  results/runs/stage5_mobile_dual_seeded/summary.csv
  results/runs/stage5_mobile_dual_seeded/tables
  results/runs/stage5_mobile_dual_seeded/audit/audit_report.md
```

## 1. 进入项目环境

命令:

```bash
conda activate em
cd /Users/xifan/Downloads/embodied_migration
```

预期输出:

```text
(em) ...
```

`cd` 命令通常没有输出。如果环境名称显示为 `(em)`, 说明 conda 环境已进入。

## 2. 检查 Stage 5 run 是否存在

命令:

```bash
ls results/runs/stage5_mobile_dual_seeded
```

预期输出大致包含:

```text
metadata.json
summary.csv
trials
prompts
generated_code
raw_responses
tables
audit
```

如果输出:

```text
ls: results/runs/stage5_mobile_dual_seeded: No such file or directory
```

说明还没有 Stage 5 数据, 需要先运行 Stage 5。

## 3. 一键生成 Stage 6 论文材料包

命令:

```bash
bash scripts/build_stage6_paper_package.sh stage5_mobile_dual_seeded
```

预期输出:

```text
Stage-6 paper package input: results/runs/stage5_mobile_dual_seeded
Wrote analysis tables to: results/runs/stage5_mobile_dual_seeded/tables
Main report: results/runs/stage5_mobile_dual_seeded/tables/analysis_report.md
Wrote paper assets to: results/runs/stage5_mobile_dual_seeded/paper_assets
Results-section draft: results/runs/stage5_mobile_dual_seeded/paper_assets/paper_results_section.md
Stage-6 paper package:
  results/runs/stage5_mobile_dual_seeded/tables
  results/runs/stage5_mobile_dual_seeded/paper_assets
Stage-6 verification: OK
```

如果最后没有 `Stage-6 verification: OK`, 说明某个必需文件没生成, 需要看终端里的 `Missing expected Stage-6 output: ...`。

## 4. 检查表格输出

命令:

```bash
ls results/runs/stage5_mobile_dual_seeded/tables
```

预期输出应包含:

```text
analysis_report.md
method_summary.csv
method_summary.tex
robot_method_summary.csv
robot_method_summary.tex
task_family_method_summary.csv
task_family_method_summary.tex
migration_score.csv
migration_score.tex
paired_method_deltas.csv
paired_method_deltas.tex
failure_breakdown.csv
failure_breakdown.tex
generated_code_features.csv
code_changes_after_feedback.csv
code_change_summary.csv
failure_cases.csv
```

最重要的文件:

- `method_summary.tex`: 论文主 ablation 表。
- `robot_method_summary.tex`: mobile vs dual_arm 结果。
- `task_family_method_summary.tex`: mobility vs bimanual 结果。
- `migration_score.tex`: 迁移分数表。
- `paired_method_deltas.tex`: 相对 few-shot baseline 的 matched delta。
- `failure_breakdown.tex`: 失败类型统计。

## 5. 查看主分析报告

命令:

```bash
sed -n '1,120p' results/runs/stage5_mobile_dual_seeded/tables/analysis_report.md
```

预期输出开头类似:

```text
# Benchmark Analysis Report

Run directories:
- `results/runs/stage5_mobile_dual_seeded`

## Method Summary
...
```

这个文件是 Markdown 总结, 用来快速看实验是否符合预期。

## 6. 检查图表与论文草稿输出

命令:

```bash
ls results/runs/stage5_mobile_dual_seeded/paper_assets
```

预期输出:

```text
experiment_manifest.json
paper_results_section.md
figure_index.md
fig_method_success.svg
fig_robot_method_success.svg
fig_task_family_success.svg
fig_migration_score.svg
table_includes.tex
```

各文件含义:

- `fig_method_success.svg`: 五个方法总体成功率图。
- `fig_robot_method_success.svg`: mobile / dual_arm 分机器人图。
- `fig_task_family_success.svg`: mobility / bimanual 分任务族图。
- `fig_migration_score.svg`: cross-embodiment migration score 图。
- `paper_results_section.md`: 自动生成的论文结果小节草稿。
- `experiment_manifest.json`: 实验设置清单。
- `table_includes.tex`: LaTeX 表格 `\input{...}` 集合。

## 7. 查看自动生成的论文结果草稿

命令:

```bash
sed -n '1,160p' results/runs/stage5_mobile_dual_seeded/paper_assets/paper_results_section.md
```

预期输出开头类似:

```text
# Auto-Drafted Results Section

## Main Results

We evaluated ...
```

这个文件不能直接当最终论文, 但可以作为实验结果段落的初稿。

## 8. 查看 LaTeX 表格 include 文件

命令:

```bash
cat results/runs/stage5_mobile_dual_seeded/paper_assets/table_includes.tex
```

预期输出:

```text
% Auto-generated table includes for the paper.
\input{results/runs/stage5_mobile_dual_seeded/tables/method_summary.tex}
\input{results/runs/stage5_mobile_dual_seeded/tables/robot_method_summary.tex}
\input{results/runs/stage5_mobile_dual_seeded/tables/task_family_method_summary.tex}
\input{results/runs/stage5_mobile_dual_seeded/tables/migration_score.tex}
\input{results/runs/stage5_mobile_dual_seeded/tables/paired_method_deltas.tex}
\input{results/runs/stage5_mobile_dual_seeded/tables/failure_breakdown.tex}
```

写论文时可以把这些表格复制进 LaTeX 工程, 或者按路径 `\input`。

## 9. 单步运行方式

如果不想用脚本, 可以手动分两步:

```bash
python -m benchmark.analyze_results results/runs/stage5_mobile_dual_seeded
python -m benchmark.build_paper_assets results/runs/stage5_mobile_dual_seeded
```

预期输出:

```text
Wrote analysis tables to: results/runs/stage5_mobile_dual_seeded/tables
Main report: results/runs/stage5_mobile_dual_seeded/tables/analysis_report.md
Wrote paper assets to: results/runs/stage5_mobile_dual_seeded/paper_assets
Results-section draft: results/runs/stage5_mobile_dual_seeded/paper_assets/paper_results_section.md
```

## 10. 常见错误

### 错误 1: run 目录不存在

输出:

```text
Run directory not found: results/runs/stage5_mobile_dual_seeded
```

解决:

```bash
bash scripts/run_stage5_experiments.sh stage5_mobile_dual_seeded
```

### 错误 2: trials 为空

输出可能是:

```text
No trial JSON files found.
```

解决: Stage 5 没有真正跑完, 检查:

```bash
ls results/runs/stage5_mobile_dual_seeded/trials
cat results/runs/stage5_mobile_dual_seeded/audit/audit_report.md
```

### 错误 3: 缺少某个 Stage 6 输出

输出:

```text
Missing expected Stage-6 output: ...
```

解决: 重新运行:

```bash
bash scripts/build_stage6_paper_package.sh stage5_mobile_dual_seeded
```

如果仍失败, 说明 `benchmark.analyze_results` 或 `benchmark.build_paper_assets` 中途报错, 需要看完整终端日志。

## 11. 阶段 6 完成标准

看到下面这行就算阶段 6 成功:

```text
Stage-6 verification: OK
```

并且以下两个目录存在:

```text
results/runs/stage5_mobile_dual_seeded/tables/
results/runs/stage5_mobile_dual_seeded/paper_assets/
```
