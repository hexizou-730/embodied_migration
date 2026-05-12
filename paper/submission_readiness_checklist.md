# Submission Readiness Checklist

Use this after `stage5_mobile_dual_seeded` finishes and Stage-6 assets are generated.

## Data Integrity

- [ ] `results/runs/stage5_mobile_dual_seeded/metadata.json` records robots, modes,
  task names, scene variant, seed base, and scene seeds.
- [ ] For the final run, prefer `stage7_mobile_dual_seeded` and confirm
  `audit/audit_report.md` reports zero missing and zero incomplete trials.
- [ ] Every expected trial has a JSON file under `trials/`.
- [ ] `summary.csv` has no empty `success`, `robot`, `task`, or `canonical_mode`
  fields.
- [ ] `failure_cases.csv` has compact but interpretable failure excerpts.
- [ ] Seeded layouts are reused across methods for paired comparison.
- [ ] `llm_model`, `llm_temperature`, cache settings, and cache hit counts are
  recorded in `metadata.json` / `summary.csv`.

## Main Tables

- [ ] `method_summary.tex` supports the main ablation claim.
- [ ] `robot_method_summary.tex` shows whether gains are embodiment-specific.
- [ ] `task_family_method_summary.tex` separates mobility and bimanual performance.
- [ ] `migration_score.tex` supports the cross-embodiment migration claim.
- [ ] `paired_method_deltas.tex` shows matched improvement relative to few-shot.
- [ ] `failure_breakdown.tex` has meaningful failure categories.

## Main Figures

- [ ] `fig_method_success.svg` is the primary method comparison.
- [ ] `fig_robot_method_success.svg` is readable and not overloaded.
- [ ] `fig_task_family_success.svg` highlights where the method helps.
- [ ] `fig_migration_score.svg` matches the narrative about migration.

## Qualitative Examples

- [ ] Pick 2-3 representative failure-recovery trials.
- [ ] Run `python -m benchmark.build_casebook results/runs/stage7_mobile_dual_seeded`.
- [ ] Inspect `casebook/qualitative_cases.csv` and `casebook/qualitative_casebook.md`.
- [ ] For each, include first-attempt failure, Failure Report signal, and final
  code change.
- [ ] Avoid long code listings; show only the minimal lines that reveal the
  adaptation.
- [ ] Include at least one mobile navigation example if available.
- [ ] Include at least one dual-arm API example if available.
- [ ] Include at least one refusal example if available.
- [ ] Include one persistent failure example as a limitation if available.

## Claims To Freeze

- [ ] Capability Cards improve first-attempt embodiment awareness.
- [ ] Failure Reports improve retry recovery.
- [ ] Card + Failure improves migration score.
- [ ] Generated-code differences are interpretable, not just textual churn.
- [ ] Refusal behavior is evaluated separately from physical task success.

## Before Submission

- [ ] Freeze prompt templates and code version.
- [ ] Archive the exact run directory.
- [ ] Record LLM model name, temperature, and API provider.
- [ ] Archive `results/llm_cache` or document that raw LLM responses are already
  stored under the run directory.
- [ ] Add limitations on simulation, task scale, and real-robot transfer.
- [ ] Prepare a reproducibility appendix with commands.
