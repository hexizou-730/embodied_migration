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
| Case 03 | `PickCube-v1` | Panda -> xarm6_robotiq | Source succeeds; target hard case | Panda baseline reached `ret_val=True`, `elapsed_steps=40`; xarm6 force-closure grasp remains unstable |
| Case 01 | `PullCube-v1` | Panda -> Fetch | Diagnosed failure | Mobile-base/contact-side reachability and action-space mismatch |

The strongest current result is:

```text
PullCube-v1 can be migrated from Panda to xarm6_robotiq by an LLM-generated
target adapter, verified through real ManiSkill execution.
```

The main negative result is:

```text
PickCube-v1 is validated on the Panda source stack, but exposes the limit of
prompt-only target-adapter synthesis for xarm6_robotiq. The LLM can generate
structured grasp adapters and use probe feedback, but robust force-closure
grasping still fails due to descent, gripper-envelope, and contact-force issues.
```

## Why PullCube Succeeds And PickCube Fails

| Dimension | PullCube | PickCube |
|---|---|---|
| Required physical interaction | Contact drag/push | Real two-finger grasp |
| Success requirement | Move cube to target region | Grasp, lift, and transport to 3D goal |
| Main adapter change | Contact side, drag pulses, action scaling | Grasp height, close timing, gripper envelope, lift preservation |
| Current outcome | Solved for xarm6 | Panda source succeeds; xarm6 target remains hard case |

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

To avoid endless prompt tweaking, the project now includes a generic structured
probe entrypoint. It selects a bounded probe from the migration case, runs the
task-specific backend, ranks measured cases, and writes a compact prompt
feedback artifact:

```bash
python scripts/structured_probe_runner.py \
  --case case03_pick_cube_panda_to_xarm6 \
  --sim-backend auto \
  --render-backend gpu
```

For xarm6 PickCube, the selected backend runs a small fixed-XY sweep over:

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
results/structured_probes/case03_pick_cube_panda_to_xarm6/pick_cube_xarm6_close_envelope.json
results/structured_probes/case03_pick_cube_panda_to_xarm6/pick_cube_xarm6_close_envelope.md
results/structured_probes/case03_pick_cube_panda_to_xarm6/pick_cube_xarm6_close_envelope_prompt.txt
```

For xarm6 PullCube, the same entrypoint selects a contact-geometry backend:

```bash
python scripts/structured_probe_runner.py \
  --case case02_pull_cube_panda_to_xarm6 \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

The default PullCube probe runs 32 bounded contact/drag cases over:

```text
contact_x_offset
contact_z_offset
approach_height
drag_strength
down_bias
stages
```

and records:

```text
task_success
cube_goal_xy
cube_goal_improvement
cube_delta_x
tcp_contact_xy
tcp_contact_z
tcp_cube_xy
```

Output files:

```text
results/structured_probes/case02_pull_cube_panda_to_xarm6/pull_cube_xarm6_contact_geometry.json
results/structured_probes/case02_pull_cube_panda_to_xarm6/pull_cube_xarm6_contact_geometry.md
results/structured_probes/case02_pull_cube_panda_to_xarm6/pull_cube_xarm6_contact_geometry_prompt.txt
```

The legacy task-specific script remains available:

```bash
python scripts/xarm6_pick_grasp_probe.py \
  --sim-backend auto \
  --render-backend gpu
```

`module_generation_runner` automatically reads the measured structured probe
prompt file for the active case:

```text
results/structured_probes/<case_id>/*_prompt.txt
```

Dry-run probe files are ignored. If no new-format probe exists, the runner
falls back to the legacy `results/xarm6_pick_grasp_probe_prompt.txt` file.
The selected probe summary is injected into the next LLM prompt.

The same structured result can also drive a lightweight score-guided next
probe plan:

```bash
python scripts/structured_probe_runner.py \
  --case case03_pick_cube_panda_to_xarm6 \
  --adaptive-from results/structured_probes/case03_pick_cube_panda_to_xarm6/pick_cube_xarm6_close_envelope.json \
  --suggest-only
```

This reads the previous probe scores, perturbs the best measured candidates by
small local steps, skips already-tried parameter tuples, and writes
`next_probe_suggestions`. Removing `--suggest-only` runs those suggested cases
in ManiSkill instead of repeating the full Cartesian grid.

Current probe conclusion:

```text
32 fixed-XY close-envelope cases tested.
0 cases achieved is_grasping=True.
Best case had low displacement and millimeter-level alignment, but still no
grasp.
```

This means the probe is not giving the answer to the LLM. It provides physical
evidence that simple close-envelope parameter tuning is insufficient.

## One-Command Auto Run

For the user-facing migration request shape:

```bash
python migrate.py --task pull_cube --source panda --target xarm6_robotiq
```

This resolves the registered case:

```text
case02_pull_cube_panda_to_xarm6
```

and evaluates the current migrated adapter once. This is the minimal successful
case for the current project:

```text
PullCube-v1 + Panda -> xArm6
```

Dry-run without ManiSkill:

```bash
python migrate.py --task PullCube-v1 --source panda --target xarm6 --dry-run
```

List registered migration requests:

```bash
python migrate.py --list-cases
```

For the main PullCube Panda -> xarm6 workflow, use the short entrypoint:

```bash
python auto.py pull
```

This runs the autonomous loop. The LLM planner receives only
`agent_observation.json`, chooses the next tool, and the harness executes it:

```text
agent_observation -> LLM planner action -> harness executes tool -> new observation -> repeat
```

Useful shorter variants:

```bash
python auto.py pull --dry-run
python auto.py pull --seeds 0-9 --max-cycles 3
python auto.py pull --seeds 0-4 --max-cycles 1
```

All loop outputs are grouped under one run folder:

```text
results/auto_runs/<run_name>/
  summary.json
  summary.md
  commands.log
  cycle_01/
    agent_plan.json
    harness/
      case02_pull_cube_panda_to_xarm6/
        agent_observation.json
        human_report.md
    multiseed.jsonl              # if planner chose run_multi_seed
    structured_probe/            # if planner chose run_structured_probe
    module_generation.jsonl       # if planner chose run_llm_repair
```

The most recent run path is also written to:

```text
results/auto_runs/latest.txt
```

## Multi-Seed Generalization

After a seed-0 success, robustness is checked with multi-seed evaluation:

```bash
python scripts/pullcube_multiseed_eval.py \
  --seeds 0-9 \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

The report now includes automatic strategy selection:

```text
generalization_strategy.v1
status: accepted | needs_repair
selected_strategy: accept_current_adapter | reachability_aware_contact_selection | ...
failure_seed_clusters: grouped failed seeds by diagnosis/stage
```

For an existing JSONL result, recompute the strategy without rerunning
simulation:

```bash
python scripts/select_generalization_strategy.py \
  --input results/pullcube_xarm6_multiseed.jsonl \
  --markdown results/pullcube_xarm6_multiseed.md
```

## Autonomous Harness

The project now includes a lightweight harness layer that exposes ManiSkill to
an LLM agent through bounded tools instead of raw simulator access. The harness
separates the machine-facing observation from the human-facing report:

- `agent_observation.json`: facts, constraints, tool commands, and simulator outputs only.
- `human_report.md`: researcher summary and optional suggested next action.

```bash
python scripts/autonomous_harness_runner.py \
  --case case02_pull_cube_panda_to_xarm6 \
  --multiseed-jsonl results/pullcube_xarm6_multiseed.jsonl \
  --print-agent-observation
```

If no multi-seed file is provided, the agent observation still exposes the
single-seed and multi-seed tools. Human-facing suggestions are written
separately and should not be used as the LLM agent prompt.

Outputs:

```text
results/autonomous_harness/<case_id>/agent_observation.json
results/autonomous_harness/<case_id>/human_report.md
results/autonomous_harness/<case_id>/harness_bundle.json
```

See [`docs/HARNESS_ENGINEERING_CN.md`](docs/HARNESS_ENGINEERING_CN.md) for the
Chinese explanation and reporting framing.

## Simple Harness Demo

For presentations or quick checks, use the minimal demo folder:

```bash
python demos/simple_harness/demo.py
```

This dry run writes:

```text
results/simple_demo/<run_name>/
  agent_observation.json
  agent_plan.json
  selected_tool_command.txt
  tool_result.json
  README.md
```

On a remote GPU machine, run one real selected tool:

```bash
python demos/simple_harness/demo.py --run
```

This is not a full repair loop. It demonstrates that the project can expose a
bounded simulator tool to an Agent and record the result. Use `auto.py pull` for
the full autonomous loop.

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

For a Chinese overview of the repository layout, see
[`docs/PROJECT_STRUCTURE_CN.md`](docs/PROJECT_STRUCTURE_CN.md).

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
