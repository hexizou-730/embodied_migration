# 阶段 5: Seeded Mobile + Dual-arm Full Experiment

阶段 5 的目标是把阶段 4 的固定布局 ablation 扩展到多个可复现随机种子。
这样论文结果不只依赖一个摆放位置。

## 阶段 5 做什么

- 使用 `--scene-variant seeded`
- 第 `i` 个 trial 使用 `scene_seed = seed_base + i`
- 所有方法、机器人、任务共享同一组 seed, 便于 paired comparison
- 默认机器人: `mobile dual_arm`
- 默认任务: `migration` = `mobility + bimanual`
- 默认方法: `api fewshot card failure card_failure`
- 跑完自动生成 `summary.csv`, `tables/`, `audit/`

## 运行命令

```bash
conda activate em
bash scripts/run_stage5_experiments.sh
```

默认等价于:

```bash
python -m benchmark.run_benchmark \
  --robots mobile dual_arm \
  --modes api fewshot card failure card_failure \
  --tasks migration \
  --trials 5 \
  --scene-variant seeded \
  --seed-base 0 \
  --run-id stage5_mobile_dual_seeded
```

然后自动运行:

```bash
python -m benchmark.analyze_results results/runs/stage5_mobile_dual_seeded
python -m benchmark.audit_run results/runs/stage5_mobile_dual_seeded --fail-on-missing
```

## Seeded layout 预检查

Stage 5 脚本默认会先运行:

```bash
python -m benchmark.validate_seeded_scenes \
  --robots mobile dual_arm \
  --tasks migration \
  --trials 5 \
  --seed-base 0
```

这个检查不调用 LLM, 只用手写 oracle 验证:

- mobile 在 `mobility` 任务上应能完成。
- dual_arm 在 `mobility` 和 `bimanual` 任务上应能完成。
- mobile 在 `bimanual` 任务上应失败, 因为单臂不能同时保持两个物体悬空。

如果这个预检查失败, 说明随机布局本身不适合做实验, 不应该继续消耗 LLM API。

## 常用参数

换 run id:

```bash
bash scripts/run_stage5_experiments.sh my_stage5_run
```

增加 seed 数:

```bash
TRIALS=10 bash scripts/run_stage5_experiments.sh stage5_mobile_dual_t10
```

换 seed 起点:

```bash
SEED_BASE=100 bash scripts/run_stage5_experiments.sh stage5_seed100
```

只跑 mobility:

```bash
TASKS=mobility bash scripts/run_stage5_experiments.sh stage5_mobility_seeded
```

只跑 bimanual:

```bash
TASKS=bimanual bash scripts/run_stage5_experiments.sh stage5_bimanual_seeded
```

关闭 seed 预检查:

```bash
VALIDATE_SEEDS=0 bash scripts/run_stage5_experiments.sh
```

只允许使用已有 cache, 不产生新的 LLM API 调用:

```bash
OFFLINE_CACHE_ONLY=1 bash scripts/run_stage5_experiments.sh
```

## 主要输出

```text
results/runs/stage5_mobile_dual_seeded/
├── metadata.json
├── summary.csv
├── trials/
├── prompts/
├── generated_code/
├── raw_responses/
├── tables/
└── audit/
```

重点看:

- `metadata.json`: robots, modes, task names, seeds, model, temperature。
- `summary.csv`: 每个 trial 的摘要和代码特征。
- `tables/method_summary.csv`: 总体 ablation。
- `tables/robot_method_summary.csv`: mobile vs dual_arm。
- `tables/task_family_method_summary.csv`: mobility vs bimanual。
- `tables/paired_method_deltas.csv`: 相对 `fewshot` 的 matched delta。
- `tables/generated_code_features.csv`: 是否用了 `navigate_to` 或双臂 API。
- `audit/audit_report.md`: 是否缺 trial, 是否有 incomplete trial。

## 预期结论检查

- `card` 应比 `fewshot` 更 embodiment-aware。
- `failure` 应能恢复部分 action/checker failure。
- `card_failure` 应在 migration score 上最好。
- `mobile` 的 bimanual 失败应是能力边界, 不是随机 IK 失败。
- `dual_arm` 的 bimanual 成功应伴随 `pick_with_arm`, `robot.left/right` 等代码特征。
