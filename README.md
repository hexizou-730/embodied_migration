# Embodied Migration

This project studies cross-embodiment migration for robot programs in real
ManiSkill simulation.

The current repository is intentionally narrow and clean: one task, two
robots, and one migration question.

## Current Scope

| Item | Current choice |
|---|---|
| Simulator | ManiSkill |
| Active task | `pull_cube` / `PullCube-v1` |
| Source robot | `panda` |
| Target robot | `fetch` |
| Primary case | `case01_pull_cube_panda_to_fetch` |
| Main migration object | generated target adapter module |

Older exploratory demos, parked tasks, and patch-based repair experiments are
no longer part of the active path.

## What The Project Does

The source program is deliberately simple:

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")

ret_val = robot.pull(cube, goal)
```

The research question is not whether an LLM can rewrite this short program.
The question is what must change when the same task is moved from one robot
embodiment to another:

- high-level LMP program choices;
- target robot capability description;
- skill adapter behavior;
- controller/contact primitive;
- simulator execution evidence and failure analysis.

The current main route asks the LLM to generate a complete Fetch target adapter
module for `PullCube-v1`, then validates that module with unit tests and real
ManiSkill execution.

## Active Files

| Purpose | File |
|---|---|
| Task definition | `maniskill_backend/tasks.py` |
| Fixed migration case | `maniskill_backend/cases.py` |
| Robot capability profiles | `maniskill_backend/profiles.py` |
| Shared PullCube skill wrapper | `maniskill_backend/skill_adapter.py` |
| Source LMP program | `maniskill_backend/case_programs/case01_pull_cube.py` |
| Generated target adapter | `maniskill_backend/generated_adapters/case01_fetch_pull_cube.py` |
| Real simulation runner | `maniskill_backend/real_runner.py` |
| Target-module generation runner | `maniskill_backend/module_generation_runner.py` |

## Install

```bash
conda create -n em-ms python=3.10 -y
conda activate em-ms
pip install -r requirements.txt
pip install -r requirements-maniskill.txt
pip install "numpy>=1.24,<2" --force-reinstall
```

On a Linux/NVIDIA machine, set the Vulkan ICD before running GPU-rendered
ManiSkill experiments:

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

Expected output:

```text
...
OK
```

## Run The Main Migration Case

```bash
python -m maniskill_backend.module_generation_runner \
  --case case01_pull_cube_panda_to_fetch \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

Expected behavior:

1. Panda source execution is checked.
2. Fetch source-copy execution is checked.
3. If target execution fails, the LLM receives the failure log.
4. The LLM writes a complete target adapter module.
5. Unit tests are run.
6. Fetch simulation is rerun with the generated adapter.
7. Results and migration analysis are saved.

Outputs:

```text
results/module_generation_trials.jsonl
results/module_generation_trials.md
```

## Run A Single Fetch Trial

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube \
  --robot fetch \
  --method target-module-generation \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 100 \
  --code-file maniskill_backend/case_programs/case01_pull_cube.py \
  --adapter-module maniskill_backend.generated_adapters.case01_fetch_pull_cube
```

Expected output contains a JSON-like result with fields such as:

```text
"task_id": "pull_cube"
"robot_uid": "fetch"
"method": "target-module-generation"
"success": true/false
"failure_type": ...
"failure_layer": ...
```

## Baseline: Program-Only LLM Adaptation

```bash
python -m maniskill_backend.iterative_runner \
  --task pull_cube \
  --source-robot panda \
  --target-robot fetch \
  --max-attempts 3 \
  --seed 0 \
  --target-control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 100
```

This baseline only rewrites the LMP program. It is weaker than target-module
generation because it cannot change the embodied execution layer.

## Current Status

The repository is now prepared for a cleaner PullCube migration study:

- active task list contains only `pull_cube`;
- active robot profiles contain only `panda` and `fetch`;
- Case 01 is fixed as `Panda -> Fetch` on `PullCube-v1`;
- old exploratory and patch-loop files are removed from the active path;
- tests check that removed tasks and robots are no longer accepted.

Next research work should focus on collecting repeated real simulation results,
comparing program-only adaptation against target-adapter generation, and
analyzing what the generated Fetch adapter changes at the skill/contact layer.
