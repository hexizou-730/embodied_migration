# Embodied Migration

Real ManiSkill simulation experiments for LLM-generated robot program migration
across robot embodiments.

## Current Scope

This repository is now focused on real simulation only:

- Real ManiSkill environments.
- Real `env.step(action)` skill wrappers.
- Real execution logs and Failure Reports.
- LLM repair with Capability Card + Failure Report.

The old static fake runner, static benchmark, and text-only code-migration
experiments have been removed.

## Research Question

When a robot program works on one embodiment, what fails when it is executed on
another embodiment, and can an LLM repair the program using:

- the target robot's Capability Card, and
- a structured Failure Report generated from a real simulator attempt?

## Core Pipeline

```text
source LMP program
        ->
target robot in ManiSkill
        ->
real source-copy execution
        ->
real Failure Report
        ->
LLM generates corrected LMP code
        ->
real simulator re-execution
```

## Current Real Tasks

| Task name | 中文任务 | ManiSkill env | Status |
|---|---|---|---|
| `pick_cube` | 抓取方块 | `PickCube-v1` | validated smoke and controller-portability task |
| `stack_cube` | 堆叠方块 | `StackCube-v1` | second real task, official Panda solver succeeds at seed 0 |
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
| `llm_card_report` | give the LLM the target Capability Card plus a real Failure Report |
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

```bash
python -m maniskill_backend.real_benchmark \
  --task pick_cube \
  --robot xarm6_robotiq \
  --methods source-copy,llm_card_report,oracle \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu
```

Outputs:

```text
results/real_trials.jsonl
results/real_trials.md
results/real_summary.csv
```

See `docs/RUN.md` for the step-by-step workflow.

## Current Validated Result

```text
pick_cube + panda + pd_ee_delta_pos -> success
pick_cube + xarm6_robotiq + pd_ee_delta_pos -> controller/skill-wrapper failure
pick_cube + xarm6_robotiq + pd_joint_pos planner -> success
stack_cube + official Panda solver -> success at seed 0
peg_insertion + official Panda solver -> failure at seed 0, not used yet
```
