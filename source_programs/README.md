# Source Programs

This directory contains the source side of the benchmark. Each task has one
candidate path and, after validation, at most one frozen source program.

- `candidates/`: editable source implementations.
- `frozen/`: immutable copies created only after every requested development
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

Inspect the current state:

```bash
python source.py status
```

Prepare all 20 programs end to end on the remote simulator. Existing
candidates are tried first; only failures spend LLM calls:

```bash
python source.py prepare --tasks all --max-cycles 5
```

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
