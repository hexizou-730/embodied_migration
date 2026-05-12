# Embodied Migration v9: Mobile + Dual-arm 迁移任务集

> v9 = v8 + dual-arm PyBullet 原型与 Mobile/Dual-arm 迁移任务集。

完整中文使用文档见:

```text
docs/PROJECT_USAGE_CN.md
docs/STAGE3_MOBILE_DUAL_TASKS_CN.md
docs/STAGE4_STRICT_ABLATION_CN.md
docs/STAGE5_SEEDED_EXPERIMENT_CN.md
docs/STAGE6_PAPER_PACKAGE_CN.md
docs/ROBOSUITE_MIGRATION_CN.md
```

---

## v8 增量

针对老师反馈「任务太简单 + 迁移模型太简单」,本批次加了:

1. **Mobile Manipulator**(`robots/mobile_robot.py`) — Husky 移动底盘 + KUKA iiwa 7-DOF 臂叠加。新增 `navigate_to`, `is_reachable`, `get_base_position` 三个 API,这是 KUKA / Franka 没有的。
2. **5 个空间几何任务** — `arrange_line`, `arrange_triangle`, `arrange_circle`, `mirror_layout`, `sort_left_to_right`。这些任务要求 LLM 现场写 NumPy 几何计算,Tool Calling 范式做不到。
3. **2 个 refusal 任务** — `refuse_rotate_object`, `refuse_missing_object`。用于评估 Capability Card 是否能让 LLM 正确拒绝不可执行或场景中不存在的请求。
4. **CapabilityCard 扩展** — 新增 `has_mobile_base`, `global_reachable`, `nav_min_clearance_m` 三个字段,LLM 据此自动推导出 mobile-aware 的代码模式。
5. **论文级分析脚本** — `benchmark/analyze_results.py` 会生成 method/robot/task-family/failure/code-diff 的 CSV 与 LaTeX 表格。
6. **Seeded randomized layouts** — `--scene-variant seeded --seed-base N` 可复现生成多初始布局,避免只在单一固定场景上过拟合。
7. **论文指标扩展** — 自动输出 Wilson 95% CI、Migration Score、相对 few-shot baseline 的 paired method delta。
8. **Paper asset builder** — `benchmark/build_paper_assets.py` 从实验表格生成 SVG 图、结果小节草稿、实验 manifest 和 LaTeX table include 文件。
9. **Stage-6 package script** — `scripts/build_stage6_paper_package.sh` 一键把完整 run 目录转成论文材料包,并检查关键表格/图表是否生成完整。
10. **LLM response cache** — `benchmark/llm_cache.py` 按 system prompt、user prompt、model、temperature 生成 SHA256 key,避免重复付费。
11. **Resumable benchmark** — `--resume` 会跳过已完成 trial JSON,中断后可继续跑同一个 `run_id`。
12. **Run audit** — `benchmark/audit_run.py` 检查缺失 trial、未完成 trial、LLM error、cache hit rate 和 summary 行数。
13. **Stage-7 reliable pipeline** — `scripts/run_stage7_reliable_experiments.sh` 串起 benchmark、analysis、audit、paper assets。
14. **Qualitative casebook** — `benchmark/build_casebook.py` 自动选择代表性恢复案例、失败案例、refusal 案例,并生成代码 diff。
15. **Stage-8 package script** — `scripts/build_stage8_qualitative_package.sh` 一键生成 tables、audit、paper assets 和 casebook。
16. **Dual-arm robot + migration tasks** — 新增 `robots/dual_arm_robot.py`、`--robot dual_arm`、`mobility` / `bimanual` 任务族,默认 benchmark 对比 `mobile` 与 `dual_arm`。
17. **Stage-4 strict ablation runner** — `scripts/run_stage4_mobile_dual_ablation.sh` 一键运行 Mobile/Dual-arm 的 `api/fewshot/card/failure/card_failure` 对照实验。
18. **Stage-5 seeded validation** — `benchmark/validate_seeded_scenes.py` 在真实 LLM 实验前验证 seeded layouts 的物理可解性和预期能力边界。
19. **Optional robosuite complex-task backend** — 新增 `robosuite_backend/`、`examples/robosuite_migration_demo.py`、`benchmark/run_robosuite_migration.py`,用于 `TwoArmLift` / `TwoArmHandover` / `TwoArmPegInHole` 的源程序到目标机器人迁移。

---

## 项目结构变化

```
robots/
├── kuka_robot.py
├── franka_robot.py
├── mobile_robot.py          ⭐ Husky + KUKA 叠加,带导航 API
└── dual_arm_robot.py        ⭐ NEW: 2x KUKA 双臂原型,带左右臂 API

benchmark/
├── run_benchmark.py         ⭐ MODIFIED: strict ablation + task families
├── experiment_logging.py    ⭐ NEW: trial/prompt/code/failure logging
├── analyze_results.py       ⭐ NEW: paper tables + failure/code analysis
├── build_paper_assets.py    ⭐ NEW: SVG figures + results draft
├── audit_run.py             ⭐ NEW: run completeness/reproducibility audit
├── llm_cache.py             ⭐ NEW: prompt-response cache
├── build_casebook.py        ⭐ NEW: qualitative failure/recovery casebook
└── validate_seeded_scenes.py ⭐ NEW: Stage-5 seeded layout oracle validation

scripts/
├── run_stage4_mobile_dual_ablation.sh ⭐ NEW: Mobile/Dual-arm strict ablation
├── run_stage5_experiments.sh       ⭐ NEW: one-command seeded full experiment
├── build_stage6_paper_package.sh   ⭐ NEW: one-command paper package builder
├── run_stage7_reliable_experiments.sh ⭐ NEW: cache+resume+audit+paper pipeline
└── build_stage8_qualitative_package.sh ⭐ NEW: qualitative analysis package

paper/
├── experiment_section_template.md
├── main_paper_outline.md
├── qualitative_analysis_template.md
└── submission_readiness_checklist.md

capabilities/
└── capability_card.py       ⭐ MODIFIED: +mobile-aware 字段和 implications

prompts/
└── cap_prompt.py            ⭐ MODIFIED: 给 mobile 注入额外 API hint
```

---

## 快速运行

```bash
# 1. 进入环境
cd ~/Downloads/embodied_migration && conda activate em

# 2. 烟测(确保 mobile / dual_arm 机器人能加载)
python -m examples.smoke_test --robot mobile
python -m examples.smoke_test --robot dual_arm

# 3. 阶段 3: Mobile + Dual-arm 迁移任务
python -m benchmark.run_benchmark \
    --robots mobile dual_arm \
    --modes api fewshot card failure card_failure \
    --tasks migration \
    --trials 1 \
    --run-id stage3_mobile_dual

# 4. 生成论文表格
python -m benchmark.analyze_results results/runs/stage3_mobile_dual

# 5. 阶段 4: strict baseline / ablation 一键运行
bash scripts/run_stage4_mobile_dual_ablation.sh stage4_mobile_dual_ablation

# 6. 阶段 5: seeded full experiment
bash scripts/run_stage5_experiments.sh stage5_mobile_dual_seeded

# 7. 阶段 6: 生成论文图表包
bash scripts/build_stage6_paper_package.sh stage5_mobile_dual_seeded

# 8. 推荐正式主实验: 可恢复、带缓存、带审计
bash scripts/run_stage7_reliable_experiments.sh stage7_mobile_dual_seeded

# 9. 阶段 8: 生成质性案例分析包
bash scripts/build_stage8_qualitative_package.sh stage7_mobile_dual_seeded

# 10. 可选: robosuite 复杂双臂任务迁移 demo
python -m examples.robosuite_migration_demo \
    --task two_arm_lift \
    --target rs_dual_iiwa \
    --planner oracle
```

阶段 5 的显式命令等价于:

```bash
python -m benchmark.run_benchmark \
    --robots mobile dual_arm \
    --modes api fewshot card failure card_failure \
    --tasks migration \
    --trials 5 \
    --scene-variant seeded \
    --seed-base 0 \
    --run-id stage5_mobile_dual_seeded

python -m benchmark.analyze_results results/runs/stage5_mobile_dual_seeded

python -m benchmark.build_paper_assets results/runs/stage5_mobile_dual_seeded
```

阶段 7 推荐命令:

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_mobile_dual_seeded
```

等价核心命令:

```bash
python -m benchmark.run_benchmark \
    --robots mobile dual_arm \
    --modes api fewshot card failure card_failure \
    --tasks migration \
    --trials 5 \
    --scene-variant seeded \
    --seed-base 0 \
    --model "${EM_MODEL:-anthropic/claude-sonnet-4.5}" \
    --temperature 0.0 \
    --cache-dir results/llm_cache \
    --resume \
    --run-id stage7_mobile_dual_seeded

python -m benchmark.analyze_results results/runs/stage7_mobile_dual_seeded
python -m benchmark.audit_run results/runs/stage7_mobile_dual_seeded --fail-on-missing
python -m benchmark.build_paper_assets results/runs/stage7_mobile_dual_seeded
python -m benchmark.build_casebook results/runs/stage7_mobile_dual_seeded
```

如果只想验证缓存中已有结果,不允许 live API 调用:

```bash
OFFLINE_CACHE_ONLY=1 bash scripts/run_stage7_reliable_experiments.sh stage7_mobile_dual_seeded
```

阶段 5/6 输出:

```
results/runs/stage5_mobile_dual_seeded/tables/
├── method_summary.csv/.tex
├── robot_method_summary.csv/.tex
├── task_family_method_summary.csv/.tex
├── migration_score.csv/.tex
├── paired_method_deltas.csv/.tex
├── failure_breakdown.csv/.tex
├── failure_cases.csv
├── generated_code_features.csv
├── code_changes_after_feedback.csv
├── code_change_summary.csv
└── analysis_report.md

results/runs/stage5_mobile_dual_seeded/paper_assets/
├── experiment_manifest.json
├── paper_results_section.md
├── figure_index.md
├── fig_method_success.svg
├── fig_robot_method_success.svg
├── fig_task_family_success.svg
├── fig_migration_score.svg
└── table_includes.tex

results/runs/stage7_mobile_dual_seeded/audit/
├── audit_summary.json
└── audit_report.md

results/runs/stage7_mobile_dual_seeded/casebook/
├── qualitative_cases.csv
├── qualitative_casebook.md
└── qualitative_casebook.tex
```

---

## 关键观察点

跑完数据后,关注以下几件事:

1. **Mobile 在 baseline 模式下大量失败** —— 因为它不知道有 navigate_to。这是方法 A 价值的最强证明。
2. **Dual-arm 在 bimanual 任务中应明显优于 mobile** —— 因为它可以最终同时保持两个物体悬空。
3. **Capability Card 是否诱导正确 API** —— `card` / `card_failure` 应更容易输出 `navigate_to` 或 `pick_with_arm` 等 embodiment-aware 代码。
4. **Failure Report 是否真的改变代码** —— 看 `code_changes_after_feedback.csv` 和 `code_change_summary.csv`,尤其是是否新增 `navigate_to`,双臂 API,低释放高度或返回值检查。
5. **+B+A 应该把所有任务推回到 70%+** —— 如果没推上去,说明 capability card 或 failure report 还需要打磨。
6. **多 seed 是否稳定** —— 看 `seed_method_summary.csv`,确认结论不是某个初始布局偶然造成的。
7. **迁移分数是否提升** —— 看 `migration_score.csv`,这是论文里比单机器人 success rate 更有说服力的指标。

---

## 下一批做什么

- 正式跑 `stage7_mobile_dual_seeded`,检查 `analysis_report.md`、`audit/audit_report.md` 和 `paper_assets/paper_results_section.md`。
- 运行阶段 8,从 `casebook/qualitative_casebook.md` 选 3-5 个代表性案例写进论文。
- 根据失败案例修 prompt/card/report,然后冻结实验设置。
- 把 `paper/experiment_section_template.md` 和 `paper/main_paper_outline.md` 整合进论文草稿。
- 若时间允许,继续扩充到 4-6 个 say-no/refusal scenarios。
