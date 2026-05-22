# Embodied Migration

Real ManiSkill simulation experiments for full cross-embodiment robot migration.

The migration target is not only the LLM-written high-level LMP program. It is
the whole executable target stack needed for another embodiment to complete the
same task: task program, target skill adapter, control mode, controller
primitive, and embodiment-specific physical parameters.

## Current Scope

This repository is now focused on real simulation only:

- Real ManiSkill environments.
- Real `env.step(action)` skill wrappers.
- Real execution logs and Failure Reports.
- Program-level LLM repair with Capability Card + Failure Report.
- Target skill-wrapper and controller-primitive migration when source
  primitives do not physically transfer.

The old static fake runner, static benchmark, and text-only code-migration
experiments have been removed.

## Research Question

When a task stack works on one embodiment, what must be migrated so another
embodiment completes the same task in real simulation?

The project separates and then recombines two migration layers:

1. **Program migration**: adapt the LMP task code, API ordering, and tunable
   target parameters.
2. **Embodied execution migration**: port the target skill wrapper, planner
   path, controller mode, contact primitive, and grasp/tool geometry that turn
   the high-level program into `env.step(action)` execution.

The current working loop is:

- Panda source code succeeds.
- Panda source wrapper and target xarm6 wrapper are both tested in ManiSkill.
- Source-copy exposes program and execution-layer portability failures.
- The LLM revises target LMP code when the failure is program-level.
- `skill_adapter.py` and target planner/control primitives are migrated when
  the same high-level semantics fail physically on the target embodiment.
- We record success, failed layer, attempts, generated code changes, and
  adapter/controller changes.

Capability Cards and Failure Reports are now supporting context, not the main
research object.

## Case 01: PullCubeTool Panda to xarm6

`pull_cube_tool` is now the first fixed complete migration case:

| Field | Value |
|---|---|
| Case id | `case01_pull_cube_tool_panda_to_xarm6` |
| ManiSkill task | `pull_cube_tool` / `PullCubeTool-v1` |
| Source embodiment | `panda` |
| Target embodiment | `xarm6_robotiq` |
| Controller route | source and target use `pd_joint_pos` planner control |
| Fixed first seed | `0` |
| Episode budget | `300` steps |

This case is intentionally more than a high-level program rewrite. It must
record Panda source success, xarm6 source-copy failure, generated target LMP
attempts, target `skill_adapter.py` / controller changes, and final real
ManiSkill success evidence.

## Migration Layers

| Layer | What migrates | Current code location |
|---|---|---|
| Task program | LMP sequence, API choices, target-side parameters | `maniskill_backend/tasks.py`, generated LLM code |
| Embodiment profile | Robot limits and planner/control assumptions | `maniskill_backend/profiles.py` |
| Skill adapter | Grasp, place, hook, pull, align, insert execution | `maniskill_backend/skill_adapter.py` |
| Controller primitive | Control mode, planner route, TCP/tool compensation | `real_runner.py`, `skill_adapter.py` |
| Evaluation/reporting | Success, failed layer, physical failure evidence | `evaluation.py`, `reporting.py`, results logs |

## Full Pipeline

```text
source LMP program + source skill adapter
        ->
Panda source success in ManiSkill
        ->
source-copy on target LMP + target execution stack
        ->
program-level or execution-layer failure evidence
        ->
LLM adapts target LMP code when code can fix it
        +
skill adapter / controller primitive is migrated when physics cannot transfer
        ->
target xarm6 re-execution in real ManiSkill simulation
```

## Program-Level Iterative Runner

`iterative_runner` automates the program migration part of the full loop. It
does not replace target adapter migration: if repeated logs show a target grasp,
tool contact, or controller primitive mismatch, port that execution layer and
rerun the iterative trial.

```bash
python -m maniskill_backend.iterative_runner \
  --task pull_cube_tool \
  --source-robot panda \
  --target-robot xarm6_robotiq \
  --max-attempts 3 \
  --seed 0 \
  --target-control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300
```

Outputs:

```text
results/iterative_trials.jsonl
results/iterative_trials.md
results/iterative_summary.csv
```

## Current Real Tasks

| Task name | 中文任务 | ManiSkill env | Status |
|---|---|---|---|
| `pick_cube` | 抓取方块 | `PickCube-v1` | smoke and controller-portability support task |
| `stack_cube` | 堆叠方块 | `StackCube-v1` | supporting stacking task, official Panda solver succeeds at seed 0 |
| `pull_cube_tool` | 用工具拉方块 | `PullCubeTool-v1` | **Case 01** full-stack Panda to xarm6 migration |
| `peg_insertion` | 侧向插 peg | `PegInsertionSide-v1` | parked: official solver failed at seed 0 |

## Current Robots

| Robot | Status |
|---|---|
| `panda` | source and smoke-test robot |
| `xarm6_robotiq` | validated on `pick_cube` through ManiSkill planner |

Other profiles can stay in `profiles.py`, but they are not counted as validated
real experiments until their ManiSkill runs are working.

## Methods

| Method | Meaning |
|---|---|
| `source-copy` | execute the source LMP program directly on the target robot |
| `llm_card_report` | program-level LLM repair from target Capability Card plus real Failure Report |
| `oracle` | hand-written real-simulation upper bound for the task |

## Install

```bash
conda create -n em-ms python=3.10 -y
conda activate em-ms
pip install -r requirements.txt
pip install -r requirements-maniskill.txt
pip install "numpy>=1.24,<2" --force-reinstall
```

On the Polytechnique GPU machine, use the NVIDIA Vulkan ICD before running:

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

## Run

The benchmark CLI defaults to Case 01. The explicit equivalent command is:

```bash
python -m maniskill_backend.real_benchmark \
  --task pull_cube_tool \
  --robot xarm6_robotiq \
  --methods source-copy,llm_card_report,oracle \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300
```

Outputs:

```text
results/real_trials.jsonl
results/real_trials.md
results/real_summary.csv
```

See `docs/RUN.md` for the step-by-step workflow. For the current Mac +
Polytechnique remote handoff, read `docs/MAC_GUIDE_CN.md`.

## Current Validated Result

```text
pick_cube + panda + pd_ee_delta_pos -> success
pick_cube + xarm6_robotiq + pd_ee_delta_pos -> controller/skill-wrapper failure
pick_cube + xarm6_robotiq + pd_joint_pos planner -> success
stack_cube + official Panda solver -> success at seed 0
pull_cube_tool + official Panda solver -> success at seed 0
pull_cube_tool + iterative LLM xarm6 -> exposes and tests target tool-wrapper migration
peg_insertion + official Panda solver -> failure at seed 0, not used yet
```
