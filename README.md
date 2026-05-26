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
- Direct LLM generation of target adapter modules when source primitives do not
  physically transfer.

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
- The LLM sees the failed layer and generates a complete target adapter module
  for xarm6.
- The generated module is validated by unit tests and rerun in real ManiSkill
  simulation until success or attempt budget exhaustion.
- We record success, failed layer, generated target module code, execution logs,
  and source-vs-target migration analysis.

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

This case is intentionally more than a high-level program rewrite. The main
runner asks the LLM to generate the target xarm6 adapter module:
`maniskill_backend/generated_adapters/case01_xarm6_pull_tool.py`. The unchanged
LMP program then calls the generated adapter in real simulation. The required
evidence is Panda source success, xarm6 source-copy failure, generated target
adapter attempts, migration analysis, and final real ManiSkill success evidence.

## Migration Layers

| Layer | What migrates | Current code location |
|---|---|---|
| Task program | LMP sequence, API choices, target-side parameters | `maniskill_backend/tasks.py`, generated LLM code |
| Embodiment profile | Robot limits and planner/control assumptions | `maniskill_backend/profiles.py` |
| Skill adapter | Grasp, place, hook, pull, align, insert execution | `skill_adapter.py`, `generated_adapters/*.py` |
| Controller primitive | Control mode, planner route, TCP/tool compensation | `real_runner.py`, generated adapter modules |
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
LLM generates target-specific adapter module
        ->
tests + target xarm6 real simulation
        ->
LLM source-vs-target migration analysis
```

## Target Adapter Module Generation

This is now the main Case 01 workflow. The LLM returns a complete Python target
adapter module, not a patch diff.

```bash
python -m maniskill_backend.module_generation_runner \
  --case case01_pull_cube_tool_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

The runner:

1. verifies Panda source execution;
2. runs the unchanged target LMP program on xarm6;
3. gives the failure log and adapter context to the LLM;
4. writes `maniskill_backend/generated_adapters/case01_xarm6_pull_tool.py`;
5. runs unit tests;
6. reruns real xarm6 simulation;
7. saves an LLM migration analysis comparing source and target code.

Outputs:

```text
results/module_generation_trials.jsonl
results/module_generation_trials.md
```

A single target-module trial can also be run directly:

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube_tool \
  --robot xarm6_robotiq \
  --method target-module-generation \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300 \
  --code-file maniskill_backend/case_programs/case01_pull_cube_tool.py \
  --adapter-module maniskill_backend.generated_adapters.case01_xarm6_pull_tool
```

## Optional Patch Runner

`full_stack_runner` remains as a parked comparison route for patch-loop
experiments. It asks for unified diffs and writes:

```text
results/full_stack_trials.jsonl
results/full_stack_trials.md
```

## Program-Only Baseline

`iterative_runner` automates the program migration part of the full loop. It
is kept as a weaker baseline: it can rewrite the target LMP code feedback loop,
but it cannot patch the target adapter or controller route.

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
| `pull_cube_tool` | 用工具拉方块 | `PullCubeTool-v1` | **Case 01** direct target-adapter generation |
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
| `target-module-generation` | execute a fixed target LMP file through a generated adapter module |

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
