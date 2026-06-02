# Embodied Migration

This project studies cross-embodiment migration for robot programs in real
ManiSkill simulation.

The current repository is intentionally narrow and clean: two manipulation
tasks, one source robot, two target embodiments, and one migration question.

## Current Scope

| Item | Current choice |
|---|---|
| Simulator | ManiSkill |
| Active tasks | `pull_cube` / `PullCube-v1`; `pick_cube` / `PickCube-v1` |
| Source robot | `panda` |
| Primary target robot | `xarm6_robotiq` |
| Preserved failure target | `fetch` |
| Primary case | `case03_pick_cube_panda_to_xarm6` |
| Main migration object | generated target adapter module |

Older exploratory demos, parked tasks, and patch-based repair experiments are
no longer part of the active path.

## What The Project Does

The completed PullCube source program is deliberately simple:

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

The current main route asks the LLM to generate a complete xarm6 target adapter
module for `PickCube-v1`, then validates that module with unit tests and real
ManiSkill execution. Unlike PullCube, PickCube requires a verified gripper
grasp, lift, and 3D transport:

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")

grasp_ok = robot.grasp(cube)
ret_val = robot.place(cube, goal) if grasp_ok else False
```

## Active Files

| Purpose | File |
|---|---|
| Task definition | `maniskill_backend/tasks.py` |
| Fixed migration case | `maniskill_backend/cases.py` |
| Robot capability profiles | `maniskill_backend/profiles.py` |
| Shared skill wrappers | `maniskill_backend/skill_adapter.py` |
| PullCube source LMP | `maniskill_backend/case_programs/case01_pull_cube.py` |
| PickCube source LMP | `maniskill_backend/case_programs/case03_pick_cube.py` |
| Current generated target adapter | `maniskill_backend/generated_adapters/case03_xarm6_pick_cube.py` |
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

## Configure LLM API

The LLM calls use the OpenAI Python SDK with OpenAI-compatible endpoints.

For DeepSeek direct API, create `.env`:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
EM_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
EM_MODEL=deepseek-v4-pro
EM_MAX_TOKENS=8192
EM_DEEPSEEK_THINKING=disabled
```

Quick check:

```bash
python - <<'PY'
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path.cwd() / ".env")
from maniskill_backend.llm import has_llm_key
from llm_client import current_provider, default_model
print("provider =", current_provider())
print("model =", default_model())
print("has_llm_key =", has_llm_key())
from llm_client import completion_token_limit, deepseek_thinking_mode
print("max_tokens =", completion_token_limit())
print("thinking =", deepseek_thinking_mode())
PY
```

Expected output:

```text
provider = deepseek
model = deepseek-v4-pro
has_llm_key = True
max_tokens = 8192
thinking = disabled
```

## Run The Main Migration Case

```bash
python -m maniskill_backend.module_generation_runner \
  --case case03_pick_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

Expected behavior:

1. Panda source execution is checked.
2. xarm6_robotiq target execution is checked.
3. If target execution fails, the LLM receives the failure log.
4. The LLM writes a complete target adapter module.
5. Unit tests are run.
6. xarm6_robotiq simulation is rerun with the generated adapter.
7. Results and migration analysis are saved.

Outputs:

```text
results/module_generation_trials.jsonl
results/module_generation_trials.md
```

## Run A Single xarm6 PickCube Trial

```bash
python -m maniskill_backend.real_runner \
  --task pick_cube \
  --robot xarm6_robotiq \
  --method target-module-generation \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500 \
  --code-file maniskill_backend/case_programs/case03_pick_cube.py \
  --adapter-module maniskill_backend.generated_adapters.case03_xarm6_pick_cube
```

Expected output contains a JSON-like result with fields such as:

```text
"task_id": "pick_cube"
"robot_uid": "xarm6_robotiq"
"method": "target-module-generation"
"success": true/false
"failure_type": ...
"failure_layer": ...
```

## Completed PullCube Baseline

The first completed migration remains:

```text
case02_pull_cube_panda_to_xarm6
```

It proves that an LLM-generated xarm6 adapter can migrate contact-based object
displacement. Case 03 extends the study to real grasping.

## Current Status

The repository is now prepared for a grasp-migration study:

- active task list contains `pull_cube` and `pick_cube`;
- active robot profiles contain `panda`, `fetch`, and `xarm6_robotiq`;
- Case 02 is the completed `Panda -> xarm6_robotiq` PullCube success;
- Case 03 is the main `Panda -> xarm6_robotiq` PickCube grasp-migration case;
- Case 01 preserves `Panda -> Fetch` as a diagnosed failure case;
- old exploratory and patch-loop files are removed from the active path;
- tests check that removed tasks and robots are no longer accepted.

Next research work should run Case 03 remotely, inspect grasp/lift/transport
failure evidence, and evaluate whether DeepSeek can generate a successful
xarm6 grasp adapter in one target-module generation run.
