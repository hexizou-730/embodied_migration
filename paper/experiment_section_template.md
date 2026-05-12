# Experiment Section Template

## Experimental Setup

We evaluate capability-conditioned failure-driven adaptation for LLM-generated
robot programs across two contrasting embodiments: a Husky-mounted mobile
manipulator with a KUKA arm and a fixed dual-arm KUKA prototype. Each
trial provides the LLM with a natural-language instruction, a scene description,
and a robot API. The generated Python program is executed in PyBullet and is
scored by task-specific state checkers.

We use 15 tasks grouped into five families: basic manipulation, geometric
layout construction, refusal, mobility, and bimanual tasks. The Stage-3
Mobile/Dual-arm slice focuses on `mobility` and `bimanual`: mobility tasks
require base-pose selection and reachability-aware tabletop manipulation,
whereas bimanual tasks require simultaneous holding or hold-while-place behavior
that a single-arm mobile manipulator cannot realize in the final state.

To avoid overfitting to a single initial arrangement, Stage 5 evaluates seeded
scene layouts. With `--scene-variant seeded`, trial `i` uses seed
`seed_base + i`, and the same seed is reused across methods and embodiments.
This creates matched trials for paired comparisons while keeping layouts
reproducible.

Before launching LLM calls, we validate the seeded layouts with scripted oracle
policies. The validator checks that mobility tasks are physically solvable for
both embodiments, bimanual tasks are solvable for the dual-arm robot, and the
single-arm mobile manipulator fails bimanual tasks for the intended capability
reason.

## Methods

We compare five prompt/adaptation conditions:

- `api`: API-only prompt, no examples, no Capability Card, no feedback.
- `fewshot`: API plus few-shot examples.
- `card`: few-shot prompt plus an embodiment Capability Card.
- `failure`: few-shot prompt plus Failure Report retry.
- `card_failure`: Capability Card plus Failure Report retry.

The `failure` and `card_failure` methods allow up to three attempts. Other
methods use one attempt.

## Metrics

We report task success rate with Wilson 95% confidence intervals, mean attempts,
recovery-after-feedback rate, failure-type breakdown, and a migration score. The
migration score counts a task as transferred when all evaluated embodiments
achieve more than 50% success on that task under a method.

We also analyze generated-code adaptation by comparing the first and final
attempt in retry trials. The analysis records whether the corrected code adds
navigation, reachability checks, explicit dual-arm APIs, low release height,
NumPy geometry, loops, conditionals, or refusal return values.

## Reproduction Commands

```bash
conda activate em

bash scripts/run_stage5_experiments.sh stage5_mobile_dual_seeded
bash scripts/build_stage6_paper_package.sh stage5_mobile_dual_seeded

# Stage-3 Mobile/Dual-arm run:
python -m benchmark.run_benchmark \
  --robots mobile dual_arm \
  --modes api fewshot card failure card_failure \
  --tasks migration \
  --trials 1 \
  --run-id stage3_mobile_dual
```

Equivalent explicit command:

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

python -m benchmark.audit_run results/runs/stage7_mobile_dual_seeded --fail-on-missing
python -m benchmark.build_casebook results/runs/stage7_mobile_dual_seeded
```

## Tables To Insert

- `results/runs/stage5_mobile_dual_seeded/tables/method_summary.tex`
- `results/runs/stage5_mobile_dual_seeded/tables/robot_method_summary.tex`
- `results/runs/stage5_mobile_dual_seeded/tables/task_family_method_summary.tex`
- `results/runs/stage5_mobile_dual_seeded/tables/migration_score.tex`
- `results/runs/stage5_mobile_dual_seeded/tables/paired_method_deltas.tex`
- `results/runs/stage5_mobile_dual_seeded/tables/failure_breakdown.tex`

## Figures To Insert

- `results/runs/stage5_mobile_dual_seeded/paper_assets/fig_method_success.svg`
- `results/runs/stage5_mobile_dual_seeded/paper_assets/fig_robot_method_success.svg`
- `results/runs/stage5_mobile_dual_seeded/paper_assets/fig_task_family_success.svg`
- `results/runs/stage5_mobile_dual_seeded/paper_assets/fig_migration_score.svg`

## Auto-Drafted Results Text

- `results/runs/stage5_mobile_dual_seeded/paper_assets/paper_results_section.md`
- `results/runs/stage5_mobile_dual_seeded/paper_assets/experiment_manifest.json`
- `results/runs/stage5_mobile_dual_seeded/paper_assets/table_includes.tex`

For the final reliable run, replace `stage5_mobile_dual_seeded` with
`stage7_mobile_dual_seeded`. Also inspect:

- `results/runs/stage7_mobile_dual_seeded/audit/audit_report.md`
- `results/runs/stage7_mobile_dual_seeded/audit/audit_summary.json`
- `results/runs/stage7_mobile_dual_seeded/casebook/qualitative_casebook.md`
- `results/runs/stage7_mobile_dual_seeded/casebook/qualitative_cases.csv`

## Analysis Claims To Check After Running

- Whether Capability Cards improve mobile manipulation by inducing navigation
  and reachability-aware code.
- Whether Failure Reports recover from physical execution failures rather than
  merely repairing syntax or API errors.
- Whether `card_failure` improves migration score across mobile and dual-arm embodiments.
- Whether `card_failure` induces mobile navigation and dual-arm API use more
  reliably than `fewshot`.
- Whether bimanual failures on mobile are capability-bound rather than syntax
  or API mistakes.
