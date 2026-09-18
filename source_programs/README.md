# Source Programs

This directory contains the source side of the benchmark. Each task has one
candidate path and, after validation, at most one frozen source program.

- `candidates/`: editable source implementations.
- `frozen/`: immutable copies created only after every one of the ten development
  seed reaches ManiSkill's official success signal.
- `programs.json`: source controller, provenance, and freeze policy.
- `vendor/`: the small subset of ManiSkill 3.0.1 reference motion-planning
  code used by the official Panda candidates.

Eleven tasks start from ManiSkill's official motion-planning references. The
remaining nine have conservative state-driven bootstrap programs because
ManiSkill 3.0.1 does not package reference solvers for them. Any failed
bootstrap is generated and repaired on the remote simulator before it can be
frozen. A source candidate is not evidence of success; the adjacent
`frozen/*.json` record is.

## Fourth Step: One Remote Command

In the remote repository, with the existing OpenRouter API key available:

```bash
conda activate em-ms
git pull --ff-only
python source.py prepare --background
```

The default run is `phase4`. It uses the frozen public LLM settings in
`experiment_config.json`; reconnecting SSH no longer requires exporting the
model and token limit again. API keys are not changed or printed. Use
`--keep-llm-settings` to check, rather than replace, your current request settings.
Python, ManiSkill, SAPIEN and CUDA checks remain strict.

Inspect the current state:

```bash
python source.py status
```

The queue performs the following work:

1. Check the runtime contract and prepare missing official task/robot assets.
2. Validate the 11 reference candidates first, then the nine bootstrap candidates.
3. Test each candidate on seeds 0 and 1, then all seeds 0-9 if the smoke check passes.
4. Repair failures using the actual failed code, seed, sampled trajectory,
   object velocities, robot joints, task state and official evaluation.
   Joint-position reference solvers retain `run(env)` and their planner API;
   delta-EE sources retain the measured-state adapter API.
5. Freeze only the same code passing all ten seeds with real simulator actions.
   Record code/dependency/config hashes and reference or LLM provenance.

Each task runs in an isolated process. Missing assets, download timeouts, API
failures and physical task failures are reported separately. No task can spend
more than five repair cycles or one hour by default; asset setup has fifteen
minutes. Downloaded assets can be large. `--no-download` checks availability
without downloading. Generated code is statically checked and time-bounded;
this is not an OS security sandbox for adversarial code.

## Resume And Results

The background coordinator survives SSH disconnection. Repeat the same command
after an interrupted run to resume its saved cycles. Valid frozen tasks are
skipped. Completed unsuccessful tasks are not silently started over:

```bash
python source.py prepare --background --retry-failed
```

This rechecks failures but does not reset already recorded LLM cycles. Use a
new run name to allocate a new repair budget or after changing the runtime/code:

```bash
python source.py prepare --background --run-name phase4_v2
python source.py status --run-name phase4_v2
```

`results/source_preparation/phase4/` contains `summary.json`, `run.log`,
the pinned plan/runtime and `tasks/<task_id>/worker.log`. Source code, prompts,
per-seed trials and checkpoints are saved under each task's `repair/` directory.
No need to print full JSON logs to check progress.

An older v1 frozen record is kept but is revalidated once before a v2 freeze is
written. A two-seed smoke pass is **not** a freeze. Until remote simulation
reports full passes, these remain candidates, not successful source programs.

## Individual Tasks

Validate the official candidates over all development seeds and freeze only
the full passes:

```bash
python source.py validate \
  --tasks t01_push_cube,t02_pull_cube,t04_pick_cube,t05_place_sphere,t06_lift_peg_upright,t07_stack_cube,t10_pull_cube_tool,t15_peg_insertion_side,t16_plug_charger,t18_stack_pyramid,t19_draw_triangle \
  --freeze
```

Generate, repair, validate, and freeze a task without an official source
solver:

```bash
python source.py synthesize --tasks t03_roll_ball --max-cycles 5 --freeze
```

The default seed set is `0-9`. Results stay below `results/`; only the frozen
program and its checksum record are tracked.
