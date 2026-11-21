# Repair-granularity ablation analysis

Derived offline from raw cycle records under `/users/eleves-b/2025/hexi.zou/Embodied/embodied_migration/results/repair_granularity/granularity_ablation_n10`; original summaries and trials are never modified.

## Per-arm results

| arm | n | success | rate (Wilson 95%) | median cycles | actions | tokens |
|---|---:|---:|---|---:|---:|---:|
| method_patch | 10 | 9 | 0.90 [0.60, 0.98] | 1 | 377 | 55184 |
| module_rewrite | 10 | 10 | 1.00 [0.72, 1.00] | 1 | 349 | 79874 |

## Paired comparison

- pairs: 10; discordant method-only / module-only: 0 / 1; two-sided exact sign test p = 1.0
- mean action difference (module - method): -2.8 (bootstrap 95% [-17.3, 8.3])
- mean token difference (module - method): 2469.0 (bootstrap 95% [-42.1, 4740.5])

- paired action difference is not separated from zero at this sample size
- paired token difference is not separated from zero at this sample size

## Evidence checks

- 61 recomputation checks, 0 failed

## Limits

- n=10 paired repair runs from one shared failed candidate in one fixed scene
- resume-based repair evidence; never merge with from-scratch generation comparisons
- descriptive statistics only; no held-out-scene or cross-task generalization claim
