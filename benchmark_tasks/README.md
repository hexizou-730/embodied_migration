# Task Catalog

This directory defines the source-task side of the migration benchmark. A task
is identified only by its ManiSkill environment and source embodiment. Target
robots are supplied at run time, so this catalog never contains source-to-target
case names or prewritten target adapters.

Each `task.json` records the official environment, source robot, required
capabilities, official success signal, episode limit, and six difficulty
variants. Shared perturbation semantics live in `difficulty_profiles.json`.

Validate the catalog without importing ManiSkill:

```bash
python -m maniskill_backend.task_catalog
```

Difficulty levels 2-6 describe benchmark perturbations that will be applied by
the variant runtime in a later implementation step. Level 1 describes the
official scene with an expanded action budget. The current source/target
commands use their own episode budget, not these profile multipliers.

Passing the catalog check means that 20 task definitions and their six variant
descriptions are structurally valid. It does not mean that 120 environments
were executed, or that any source or target program succeeded. The active
experimental subset is T01, T02, T04, T06 and T07; the other candidates remain
available for later expansion.
