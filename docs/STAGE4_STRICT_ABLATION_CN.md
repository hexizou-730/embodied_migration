# 阶段 4: Mobile + Dual-arm 严格 Baseline / Ablation

阶段 4 的目标是把阶段 3 的任务集变成可复现实验:

- 固定比较对象: `mobile` vs `dual_arm`
- 固定任务集: `migration` = `mobility` + `bimanual`
- 固定 ablation: `api`, `fewshot`, `card`, `failure`, `card_failure`
- 自动输出: trial JSON、summary.csv、统计表、审计报告、论文图表草稿

## 五个方法设置

| mode | 含义 | 尝试次数 |
| --- | --- | --- |
| `api` | 只给 API, 不给 few-shot, 不给 Capability Card, 不给失败反馈 | 1 |
| `fewshot` | API + few-shot 示例 | 1 |
| `card` | API + few-shot + Capability Card | 1 |
| `failure` | API + few-shot + Failure Report 重试, 不给 Capability Card | 最多 3 |
| `card_failure` | API + few-shot + Capability Card + Failure Report 重试 | 最多 3 |

这五个设置对应论文中的 strict baseline / ablation。

## 一键运行

```bash
conda activate em
bash scripts/run_stage4_mobile_dual_ablation.sh
```

默认等价于:

```bash
python -m benchmark.run_benchmark \
  --robots mobile dual_arm \
  --modes api fewshot card failure card_failure \
  --tasks migration \
  --trials 1 \
  --scene-variant fixed \
  --run-id stage4_mobile_dual_ablation
```

脚本会继续自动运行:

```bash
python -m benchmark.analyze_results results/runs/stage4_mobile_dual_ablation
python -m benchmark.audit_run results/runs/stage4_mobile_dual_ablation --fail-on-missing
python -m benchmark.build_paper_assets results/runs/stage4_mobile_dual_ablation
```

## 常用变体

只跑 mobility:

```bash
TASKS=mobility bash scripts/run_stage4_mobile_dual_ablation.sh stage4_mobility
```

只跑 bimanual:

```bash
TASKS=bimanual bash scripts/run_stage4_mobile_dual_ablation.sh stage4_bimanual
```

增加 trial 数:

```bash
TRIALS=3 bash scripts/run_stage4_mobile_dual_ablation.sh stage4_mobile_dual_t3
```

只跑论文主方法:

```bash
MODES="fewshot card_failure" bash scripts/run_stage4_mobile_dual_ablation.sh stage4_pair
```

## 主要输出

运行完成后查看:

```text
results/runs/stage4_mobile_dual_ablation/
├── metadata.json
├── summary.csv
├── trials/
├── prompts/
├── generated_code/
├── raw_responses/
├── tables/
├── audit/
└── paper_assets/
```

最重要的表:

- `tables/method_summary.csv`: 五个方法的总体成功率。
- `tables/robot_method_summary.csv`: mobile 与 dual_arm 各自成功率。
- `tables/task_family_method_summary.csv`: mobility 与 bimanual 分任务族结果。
- `tables/paired_method_deltas.csv`: 相对 `fewshot` 的 matched delta。
- `tables/generated_code_features.csv`: 生成代码是否用了 mobile / dual-arm API。
- `tables/code_changes_after_feedback.csv`: retry 前后代码是否新增关键 API。
- `tables/failure_breakdown.csv`: 失败类型统计。

## 预期验证点

- `card` 是否比 `fewshot` 更常生成 embodiment-aware 代码。
- `failure` 是否能从 action/checker failure 中恢复。
- `card_failure` 是否在 migration score 上最好。
- mobile 的成功是否依赖 `navigate_to` / `is_reachable`。
- dual_arm 的成功是否依赖 `pick_with_arm` / `place_with_arm` / `robot.left` / `robot.right`。
- bimanual 任务中 mobile 的失败应主要体现为能力边界, 不是单纯代码错误。
