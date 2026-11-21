# Repair-granularity ablation analysis

Derived offline from raw cycle records under `/users/eleves-b/2025/hexi.zou/Embodied/embodied_migration/results/repair_granularity/granularity_ablation_n30`; original summaries and trials are never modified.

## Per-arm results

| arm | n | success | rate (Wilson 95%) | median cycles | actions | tokens |
|---|---:|---:|---|---:|---:|---:|
| method_patch | 30 | 28 | 0.93 [0.79, 0.98] | 1 | 1249 | 157726 |
| module_rewrite | 30 | 28 | 0.93 [0.79, 0.98] | 1 | 1413 | 253055 |

## Paired comparison

- pairs: 30; discordant method-only / module-only: 1 / 1; two-sided exact sign test p = 1.0
- mean action difference (module - method): 5.466666666666667 (bootstrap 95% [-14.83, 30.03])
- mean token difference (module - method): 3177.633333333333 (bootstrap 95% [2002.9, 4458.3])

- paired action difference is not separated from zero at this sample size
- module_rewrite consumed more tokens per run (paired CI above zero)

## Evidence checks

- 181 recomputation checks, 0 failed

## Limits

- n=30 paired repair runs from one shared failed candidate in one fixed scene
- resume-based repair evidence; never merge with from-scratch generation comparisons
- descriptive statistics only; no held-out-scene or cross-task generalization claim
