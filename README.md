# Embodied Migration

LLM-driven robot code migration in real ManiSkill simulation.

The project studies whether a high-level robot program written for a source
robot can be migrated to a target robot by generating a target-side execution
adapter. The high-level LMP program is kept fixed; the generated adapter must
translate skills such as `robot.pull(cube, goal)` or `robot.grasp(cube)` into
real `env.step(action)` execution on the target embodiment.

## Current Scope

| Item | Current setting |
|---|---|
| Simulator | ManiSkill |
| Source robot | Panda |
| Main target robot | xarm6_robotiq |
| Secondary diagnosed target | Fetch |
| Active tasks | `PullCube-v1`, `PickCube-v1` |
| LLM model used in current runs | DeepSeek V4-Pro |
| Main method | direct target adapter module generation |

The project no longer focuses on simple PyBullet block-placement demos or
patch-loop repair. The active research path is full target-adapter generation
and real simulator verification.

## Current Results

| Case | Task | Source -> Target | Result | Main evidence |
|---|---|---|---|---|
| Case 02 | `PullCube-v1` | Panda -> xarm6_robotiq | Success | LLM-generated adapter reached `ret_val=True`, `elapsed_steps=460` |
| Case 03 | `PickCube-v1` | Panda -> xarm6_robotiq | Hard case / not solved | LLM reaches structured grasp logic, but force-closure grasp remains unstable |
| Case 01 | `PullCube-v1` | Panda -> Fetch | Diagnosed failure | Mobile-base/contact-side reachability and action-space mismatch |

The strongest current result is:

```text
PullCube-v1 can be migrated from Panda to xarm6_robotiq by an LLM-generated
target adapter, verified through real ManiSkill execution.
```

The main negative result is:

```text
PickCube-v1 exposes the limit of prompt-only adapter synthesis. The LLM can
generate structured grasp adapters and use probe feedback, but robust
force-closure grasping still fails due to descent, gripper-envelope, and
contact-force issues.
```

## Why PullCube Succeeds And PickCube Fails

| Dimension | PullCube | PickCube |
|---|---|---|
| Required physical interaction | Contact drag/push | Real two-finger grasp |
| Success requirement | Move cube to target region | Grasp, lift, and transport to 3D goal |
| Main adapter change | Contact side, drag pulses, action scaling | Grasp height, close timing, gripper envelope, lift preservation |
| Current outcome | Solved for xarm6 | Hard case |

The PickCube failure is not a high-level program error. The program remains:

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")

grasp_ok = robot.grasp(cube)
ret_val = robot.place(cube, goal) if grasp_ok else False
```

The failure occurs inside the target adapter, where xarm6 must create a real
Robotiq grasp under frozen ManiSkill controller semantics.

## Structured Probe

To avoid endless prompt tweaking, the project now includes an automatic
xarm6 PickCube grasp probe:

```bash
python scripts/xarm6_pick_grasp_probe.py \
  --sim-backend auto \
  --render-backend gpu
```

It runs a small fixed-XY sweep over:

```text
grasp_z_offset
close_steps
close_command
settle_steps
```

and records:

```text
is_grasping_after_close
is_grasping_after_lift
cube_disp_xy
tcp_grasp_xy
tcp_grasp_z
cube_lift_delta_z
```

Output files:

```text
results/xarm6_pick_grasp_probe.json
results/xarm6_pick_grasp_probe.md
results/xarm6_pick_grasp_probe_prompt.txt
```

`module_generation_runner` automatically reads
`results/xarm6_pick_grasp_probe_prompt.txt` and injects the structured probe
summary into the next LLM prompt.

Current probe conclusion:

```text
32 fixed-XY close-envelope cases tested.
0 cases achieved is_grasping=True.
Best case had low displacement and millimeter-level alignment, but still no
grasp.
```

This means the probe is not giving the answer to the LLM. It provides physical
evidence that simple close-envelope parameter tuning is insufficient.

## Run

Install:

```bash
conda create -n em-ms python=3.10 -y
conda activate em-ms
pip install -r requirements.txt
pip install -r requirements-maniskill.txt
pip install "numpy>=1.24,<2" --force-reinstall
```

Configure DeepSeek in `.env`:

```text
EM_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key
EM_MODEL=deepseek-v4-pro
EM_MAX_TOKENS=8192
EM_DEEPSEEK_THINKING=disabled
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run the successful PullCube migration case:

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

Run the PickCube hard case:

```bash
python scripts/xarm6_pick_grasp_probe.py \
  --sim-backend auto \
  --render-backend gpu

python -m maniskill_backend.module_generation_runner \
  --case case03_pick_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

Main outputs:

```text
results/module_generation_trials.jsonl
results/module_generation_trials.md
results/generated_modules/
results/xarm6_pick_grasp_probe.md
```

## Main Files

| Purpose | File |
|---|---|
| Migration cases | `maniskill_backend/cases.py` |
| Task specs | `maniskill_backend/tasks.py` |
| Shared skill adapters | `maniskill_backend/skill_adapter.py` |
| Module generation runner | `maniskill_backend/module_generation_runner.py` |
| PickCube probe | `scripts/xarm6_pick_grasp_probe.py` |
| Generated xarm6 PickCube adapter | `maniskill_backend/generated_adapters/case03_xarm6_pick_cube.py` |
| Chinese experiment report | `docs/EXPERIMENT_REPORT_CN.md` |
| Workshop framing notes | `docs/WORKSHOP_FRAMING_CN.md` |

## Research Framing

This project supports the following workshop-style claim:

```text
LLMs can migrate high-level robot programs across embodiments for contact-based
manipulation when the target adapter exposes the right control/contact
abstractions. However, force-closure grasp migration remains a hard case:
structured physical probing improves diagnosis, but robust transfer requires
constraint-aware repair and deeper contact/controller modeling.
```
