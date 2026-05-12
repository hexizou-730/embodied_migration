# Main Paper Outline

Working title:
**Capability-Conditioned Failure-Driven Adaptation of LLM-Generated Robot Programs**

## Abstract

- Problem: LLM-generated robot programs often fail to transfer across robot
  embodiments because the same instruction requires different embodiment-aware
  code.
- Core idea: combine static embodiment priors through Capability Cards with
  structured execution feedback through Failure Reports.
- Evaluation: mobile vs dual-arm embodiments, 15 tasks, strict ablations, seeded scene
  layouts, failure taxonomy, generated-code adaptation analysis.
- Main claim to substantiate after the full run: Capability Cards improve
  embodiment-aware first attempts, Failure Reports recover from concrete
  execution failures, and their combination improves cross-embodiment migration.

## 1. Introduction

- Motivation: program synthesis is attractive for open-ended robot control, but
  generated programs are brittle under embodiment shift.
- Gap: prior LLM robot-programming work often evaluates a single robot or
  focuses on task success without measuring how code changes across embodiments.
- Proposed framing: code migration is not "same prompt, different text"; it is
  adaptation of executable policies to embodiment-specific capabilities and
  failure modes.
- Contributions:
  - Capability Cards as structured embodiment priors.
  - Failure Reports as structured execution feedback for code rewriting.
  - A cross-embodiment benchmark with strict baselines, seeded layouts, refusal
    tasks, and generated-code adaptation analysis.

## 2. Related Work

- LLM-generated robot programs / Code as Policies.
- Embodied agents and robot foundation models.
- Failure-driven refinement and self-correction.
- Robot capability representation and affordance-aware planning.

## 3. Method

- Problem setup: instruction, scene, embodiment, generated program, execution.
- Capability Card:
  - fields: gripper, release height, workspace radius, mobile base, dual arms,
    bimanual holding, rotation ability, IK accuracy.
  - prompt insertion strategy.
- Failure Report:
  - expected vs actual state.
  - execution errors, action failures, task checker failures, ret_val failures.
  - retry loop and max attempts.
- Strict ablation modes:
  - `api`, `fewshot`, `card`, `failure`, `card_failure`.

## 4. Benchmark

- Robots:
  - Husky + KUKA mobile manipulator.
  - Dual-arm fixed manipulator (2x KUKA).
- Tasks:
  - basic manipulation.
  - geometric layout generation.
  - refusal tasks.
  - mobility tasks.
  - bimanual tasks.
- Scene variation:
  - fixed layout for debugging.
  - seeded layouts for main experiments.
- Metrics:
  - success rate with Wilson 95% CI.
  - mean attempts.
  - recovery after feedback.
  - migration score.
  - failure taxonomy.
  - generated-code feature deltas.

## 5. Experiments

Use `paper/experiment_section_template.md` and generated files under:

```text
results/runs/<run_id>/tables/
results/runs/<run_id>/paper_assets/
results/runs/<run_id>/audit/
results/runs/<run_id>/casebook/
```

Primary tables:

- `method_summary.tex`
- `robot_method_summary.tex`
- `task_family_method_summary.tex`
- `migration_score.tex`
- `paired_method_deltas.tex`
- `failure_breakdown.tex`

For the final paper run, use the Stage-7 pipeline:

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_mobile_dual_seeded
```

This records model/temperature/cache metadata, resumes interrupted trials, and
writes an audit report before generating paper assets.

Then build qualitative examples:

```bash
bash scripts/build_stage8_qualitative_package.sh stage7_mobile_dual_seeded
```

Use `casebook/qualitative_casebook.md` for the qualitative failure and
generated-code adaptation analysis.

Primary figures:

- `fig_method_success.svg`
- `fig_robot_method_success.svg`
- `fig_task_family_success.svg`
- `fig_migration_score.svg`

## 6. Analysis

- Does Capability Card mainly help mobile navigation and low-release behavior?
- Does Failure Report mainly repair action failures, checker failures, or ret_val
  failures?
- Are gains concentrated in geometric tasks, refusal tasks, or mobile tasks?
- Do corrected programs add interpretable code features?

## 7. Limitations

- PyBullet simplifications and limited embodiment diversity.
- Small object set and controlled tabletop scenes.
- LLM dependence and possible prompt sensitivity.
- Failure reports are hand-engineered rather than learned.

## 8. Conclusion

- Restate that executable code migration across embodiments needs both
  capability priors and execution-grounded adaptation.
- Point to future work on real robots, learned capability cards, richer scenes,
  and closed-loop perception.
