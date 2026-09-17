# Embodied Migration

Constraint-guided migration of executable robot code across ManiSkill robot
embodiments.

## Scope

The input is one working source-action Python file and a target ManiSkill robot
UID. The harness verifies the source actions, discovers the target runtime
interface, asks an LLM to synthesize a fresh target adapter, executes real
`env.step(action)` calls, and feeds measured failures into the next repair
cycle.

The migration process does not modify the simulator, controller, task success
criterion, or robot model. No task/robot migration case must be registered in
this repository.

## Run

```bash
python migrate.py examples/panda_pull.py xarm6_robotiq
```

Preview the run without ManiSkill or an LLM call:

```bash
python migrate.py examples/panda_pull.py xarm6_robotiq --dry-run
```

Generated adapters, prompts, state traces, and results are written below
`results/migrations/` and are intentionally not tracked by Git.

## Fixed Experiment Environment

[`experiment_config.json`](experiment_config.json) is the single tracked
experiment contract. It fixes Python 3.10.20, ManiSkill 3.0.1, SAPIEN 3.0.3,
CUDA 13.2, the exact OpenRouter/GPT model and generation parameters, the
controller defaults, and the simulation seed split.

Every invocation writes `runtime_environment.json` before calling the LLM or
creating a simulator. A real run stops on a contract mismatch. The record
contains the repository commit and dirty state, exact installed package
versions, ManiSkill source commit when available, CUDA driver/GPU information,
LLM parameters, robot UIDs, control modes, and the shared source/target seed.

Migration and repair use development seeds `0-9`. Seeds `100-109` are reserved
for later held-out evaluation and are never exposed to the repair loop.

Create the pinned Python environment with:

```bash
conda env create -f environment.yml
conda activate em-ms
```

CUDA is supplied by the remote NVIDIA driver rather than Conda; the harness
requires the driver to report CUDA API 13.2 and records the exact driver and GPU.

## Source File Contract

A source file declares the environment, source robot, controller, and existing
actions:

```python
ENV_ID = "SomeTask-v1"
SOURCE_ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"

def run(env):
    # Existing source-robot actions that call env.step(action).
    ...
```

Alternatively it may define a fixed `TASK_PROGRAM` and `build_robot(...)`.

## Active Code

```text
migrate.py                            command-line entrypoint
llm_client.py                         provider-neutral LLM client
maniskill_backend/dynamic_harness.py  generate/execute/observe/repair loop
maniskill_backend/dynamic_adapter.py  task-neutral runtime exposed to adapters
maniskill_backend/env_adapter.py      lazy ManiSkill environment wrapper
maniskill_backend/code_validation.py  generated-code safety checks
maniskill_backend/experiment_environment.py runtime capture and contract check
maniskill_backend/sim_check.py        environment smoke test
tests/test_dynamic_harness.py         generic harness regression tests
```

Historical case-specific adapters, prompts, paper runs, probe snapshots, and
reported success rates were removed on 2026-09-17. New evidence must be
generated from this clean baseline.

## Benchmark Tasks

[`benchmark_tasks/catalog.json`](benchmark_tasks/catalog.json) indexes 20
target-independent ManiSkill source tasks. Every task directory records its
source robot, required capabilities, official `evaluate()["success"]` signal,
official episode limit, and six shared difficulty variants. Target robot UIDs
are supplied only at migration time; there are no Panda-to-xArm6 case entries.

```bash
python -m maniskill_backend.task_catalog
```

## Source Programs

[`source_programs/programs.json`](source_programs/programs.json) assigns one
source-program path to every task. T01-T11 and T14-T20 use a compatible Panda
family robot; T12-T13 use Fetch. Eleven candidates wrap the ManiSkill 3.0.1
reference motion planners. The other nine start from conservative executable
bootstraps and are synthesized/repaired in the remote simulator when needed.

Candidates and successful sources are intentionally separate. A program moves
to `source_programs/frozen/` only when the exact same code reaches the official
success signal on every requested development seed; the frozen record includes
the code hash, seeds, controller, episode budget, and environment-contract hash.

```bash
python source.py status
python source.py prepare --tasks all --max-cycles 5
python source.py validate --tasks t01_push_cube,t02_pull_cube --freeze
python source.py synthesize --tasks t03_roll_ball --max-cycles 5 --freeze
```
