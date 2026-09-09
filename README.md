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
| Target robots | xarm6_robotiq, Fetch |
| Active tasks | `PullCube-v1`, `PickCube-v1`, `PushCube-v1` |
| Configured LLM model | DeepSeek V4-Pro |
| Main method | counterexample-guided target adapter synthesis |

The active research path is full target-adapter generation, real simulator
counterexamples, active physical probing, and held-out verification.

## Direct Migration From Existing Actions

The simplest harness input is an existing source action file and a target robot:

```bash
python migrate.py examples/panda_pull.py xarm6
```

The source file declares `ENV_ID`, `SOURCE_ROBOT`, and either `run(env)` or an
existing `TASK_PROGRAM` plus `build_robot(...)`. The harness verifies those
source actions in ManiSkill, observes both robot interfaces, asks the LLM to
write a fresh target adapter, runs it, and feeds measured state and failure
results into the next generation round. No registered case or prewritten target
adapter is used. The terminal prints only progress and the final result; complete
prompts, traces, and generated code are saved under `results/migrations/`.

## Evidence Status

| Case | Task | Source -> Target | ManiSkill support | Current committed adapter | Tracked reproducible result |
|---|---|---|---|---|---|
| Case 01 | `PullCube-v1` | Panda -> Fetch | Officially supported | Hand-written oracle | Oracle upper bound, not LLM success |
| Case 02 | `PullCube-v1` | Panda -> xarm6_robotiq | Stress-test override | Neutral seed adapter | Not yet tracked in Git |
| Case 03 | `PickCube-v1` | Panda -> xarm6_robotiq | Officially supported | Neutral seed adapter | Not yet tracked in Git |
| Case 04 | `PickCube-v1` | Panda -> Fetch | Officially supported | Neutral seed adapter | Not yet tracked in Git |
| Case 05 | `PushCube-v1` | Panda -> Fetch | Officially supported | Neutral seed adapter | Awaiting remote pilot |
| Case 06 | `StackPyramid-v1` | Panda -> Fetch | Officially supported | Neutral source scaffold | Exploratory integration; no GPU result yet |

Earlier remote runs reported seed-0 successes, but the current repository does
not track the matching raw logs and generated adapter snapshots. They are
therefore historical observations, not reproducible repository evidence.
The machine-generated evidence ledger is:

[`docs/EVIDENCE_LEDGER_CN.md`](docs/EVIDENCE_LEDGER_CN.md)

Regenerate it after adding experiment artifacts:

```bash
python scripts/build_evidence_ledger.py
```

## Counterexample-Guided Migration

### Complex Task: StackPyramid (2026-09-08)

Build a three-cube pyramid: place the red base beside the green base, stack the
blue cube on both, release and wait for stability. The fixed program calls
`prepare_base(red, green)` then `stack_on(blue, red, green)` only if the base succeeds.

On the remote GPU host with ManiSkill and the LLM key configured:

```bash
python migrate.py --task stack --target fetch --mode agent
```

Add `--dry-run` to preview without an API call or simulation. The real command
checks the Panda baseline first, backs up and restores a neutral Fetch seed,
then requests a new LLM module and repairs within a bounded cycle budget. Source
and target use the same explicit 800-step limit (official task default: 250).
Source failure stops migration and is reported; this is not a guaranteed success.

Results include `stage_trace`, live poses/grasp/static states for all three cubes,
and an independent official final evaluation. **143 local tests pass; no new GPU
simulation result is claimed.** The task supports `agent`, `generate`, and
`evaluate` only. Its online LLM tools and structured probes are not implemented.
It stays outside the existing five-case paper matrix until the source baseline
and matched evaluation backends are validated.

Chinese guide: [StackPyramid integration](docs/STACK_PYRAMID_CN.md).

The main research path is now counterexample-guided embodiment adapter
synthesis:

```text
generate adapter
-> validate on development seeds
-> select a physical counterexample
-> identify violated embodiment constraints
-> actively probe relevant parameters
-> generate guarded adapter repair
-> evaluate once on held-out seeds
```

Run the complete loop:

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis
```

Inspect the plan without ManiSkill or an LLM call:

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis \
  --dry-run
```

Method details: [`docs/METHOD_CEGIS_ADAPTER_SYNTHESIS_CN.md`](docs/METHOD_CEGIS_ADAPTER_SYNTHESIS_CN.md)

Experiment protocol: [`docs/EXPERIMENT_PROTOCOL_CN.md`](docs/EXPERIMENT_PROTOCOL_CN.md)

## Paper Benchmark

### Local Validation (2026-09-07)

- 143 unit tests passed. New regressions cover candidate application, guard
  rejection, verification rollback, and dry-run without API or simulator calls.
- CEGIS explicitly requests a new generated candidate even if the starting
  adapter already succeeds. This prevents reuse from masquerading as generation.
- Full pilot smoke: 37/37 runnable plans completed, including 10/10 nested
  repair previews and 5/5 B5/Ours controlled comparisons. Three Oracle entries
  remain unavailable. These are **dry-run infrastructure checks, not task success**.
- The conservative pilot cap is 715 simulator episodes, including initial-target
  checks. A real GPU pilot and held-out result collection are still pending.

Current meeting slides: [`PAPER_STRUCTURE_PROGRESS_CURRENT.pptx`](docs/PAPER_STRUCTURE_PROGRESS_CURRENT.pptx).

The paper comparison is now controlled by one executable matrix rather than
hand-written per-run commands. It freezes method definitions, development and
held-out seeds, independent repetitions, adapter provenance, and evidence
paths.

Inspect the pilot matrix without running simulation:

```bash
python paper.py plan --tier pilot
```

Audit prompt leakage, seed separation, held-out no-repair enforcement, neutral
starting modules, oracle separation, probe availability, equal B5/Ours probe
budgets, the shared B5/Ours outer loop, and equal B3/B4/B5/Ours iterative
LLM-call budgets:

```bash
python paper.py audit
```

Exercise every executable command in the matrix without ManiSkill or an LLM:

```bash
python paper.py smoke --tier pilot --plan-name pilot_smoke
```

The smoke run expands the full internal command chain. For the controlled B5
and Ours comparison it previews initial generation, development evaluation,
counterexample selection, structured probing, repair generation, and held-out
evaluation. It also checks that B5 uses `fixed_grid`, Ours uses `active`, and
both retain the same outer loop, prompt policy, and budgets. These previews are
plans only: they do not create synthetic measurements or enter the evidence set.

Build a checksummed remote package and execute its exact frozen plan on the GPU
host:

```bash
export EM_LLM_PROVIDER=<provider>
export EM_MODEL=<exact-model-id>
export EM_MAX_TOKENS=<limit>
export EM_TEMPERATURE=<temperature>
python paper.py package --tier pilot --plan-name pilot_remote
# after unpacking on the remote host, only provide the provider API key
python paper.py start
```

`start` validates the package and key, then starts a detached `execute` process,
so an SSH disconnect does not stop the benchmark. `execute` verifies every
packaged source/plan SHA, runs protocol and simulator
preflight, checks the Panda source baseline, executes/resumes the frozen matrix,
verifies the final evidence, and automatically writes a checksummed return archive.
It never rebuilds the packaged plan. LLM provider/model/token/temperature/thinking
settings must be explicit before packaging; `execute` restores those non-secret
values from the checksummed plan, so the remote shell only supplies the API key.
API keys are never written into the archive.

The benchmark now streams each run to the terminal and
`results/paper/<plan>/benchmark_commands.log`. A reconnecting shell can inspect
the atomically updated progress ledger without interrupting execution:

```bash
python paper.py status
```

Use `python paper.py execute` only when a foreground run is preferable.

After a complete remote run, export one checksummed return archive:

```bash
python paper.py export --tier pilot --plan-name pilot_v1
```

This catches broken method/case wiring and output paths before using the remote
GPU or API budget.

Before spending LLM or benchmark budget, run the case preflight on the GPU
machine. It checks environment creation, reset, action layout, adapter loading,
one simulator step, and the task success signal:

```bash
python paper.py preflight
```

On a development machine without ManiSkill, only check repository interfaces:

```bash
python paper.py preflight --static-only
```

Collect the two real-simulation artifacts used by the paper's motivating
examples (PullCube multi-seed and PickCube structured probing):

```bash
python paper.py evidence
```

Use `python paper.py evidence --static-only` locally to inspect the two bounded
commands without starting ManiSkill. The real command does not call an LLM or
modify an adapter.

Run the key PullCube xArm6 pilot comparison on the GPU machine:

```bash
python paper.py run \
  --tier pilot \
  --methods B0,B1,B2,B3,B4,B5,Ours,Oracle \
  --cases case02_pull_cube_panda_to_xarm6
```

From a normal Git checkout, the shortest development entrypoint is:

```bash
python paper.py pilot --plan-name pilot_v1
```

It runs the protocol audit, real ManiSkill preflight, and the matching Panda
source-task multi-seed gate first. The benchmark starts only if all checks pass;
repeating the same command resumes completed runs.

For a paper run transferred as a checksummed package, prefer
`python paper.py execute`; that path executes the exact frozen plan instead of
rebuilding it from the current shell.

The source gate can also be inspected independently:

```bash
python paper.py source --tier pilot --plan-name pilot_v1
```

After a run, verify every seed, hash, frozen interface, and evidence artifact:

```bash
python paper.py verify --tier pilot --plan-name pilot_v1
```

The full paper tier uses development seeds `0-9`, held-out seeds `100-129`,
and three independent LLM-generation repetitions:

```bash
python paper.py run --tier paper
```

For an interruptible remote run, keep a stable plan name. Re-running the same
command resumes from matching `summary.json` files instead of spending API and
simulation budget again:

```bash
python paper.py run --tier paper --plan-name paper_v1
```

The aggregate output includes Markdown, JSON, CSV, and a LaTeX table. Held-out
results report repetition mean/std and a pooled Wilson 95% confidence interval.
It also writes `summary_support_subgroups.csv`, separating officially supported
task/robot combinations from the deliberate stress-test override.
`paired_method_comparisons.csv` aligns Ours and each baseline on the same case,
repetition, and held-out seed. It reports exact McNemar and Holm-adjusted tests,
plus a hierarchical bootstrap interval that resamples case, generation
repetition, and seed instead of treating all repeated seed observations as
independent.

`B2` through `B4` use explicit prompt policies for the information ablations.
`B5` and Ours intentionally share the full prompt, counterexample selection,
development evaluation, repair schedule, and held-out gate; only the probe
selector changes from deterministic fixed grid to active constraint-guided
selection.
Unavailable case/method combinations are marked `blocked` in the plan instead
of being reported as failed experiments. Details:
[`docs/PAPER_BENCHMARK_CN.md`](docs/PAPER_BENCHMARK_CN.md).

Case 02 is deliberately retained as a stress test because ManiSkill's PullCube
task does not list xarm6_robotiq as an officially supported robot. The other
three cases use task/robot combinations declared by ManiSkill; benchmark tables
preserve this distinction.

## Why PullCube And PickCube Are Different

| Dimension | PullCube | PickCube |
|---|---|---|
| Required physical interaction | Contact drag/push | Real two-finger grasp |
| Success requirement | Move cube to target region | Grasp, lift, and transport to 3D goal |
| Main adapter change | Contact side, drag pulses, action scaling | Grasp height, close timing, gripper envelope, lift preservation |
| Current evidence target | Robust contact migration across held-out seeds | Force-closure and grasp-preservation hard case |

The key point is that these are not high-level program changes. For PickCube,
the program remains:

```python
cube = scene.get_object("cube")
goal = scene.get_region("goal")

grasp_ok = robot.grasp(cube)
ret_val = robot.place(cube, goal) if grasp_ok else False
```

The embodiment-specific work happens inside the target adapter, where xarm6 must
create a real Robotiq grasp under frozen ManiSkill controller semantics.

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

Fetch PullCube uses the same interface, but its probe separates mobile-base
reachability from arm contact:

```bash
python scripts/structured_probe_runner.py \
  --case case01_pull_cube_panda_to_fetch \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500 \
  --max-cases 8
```

It varies a bounded base speed/duration, contact offsets, and drag strength,
then records `base_reach_improvement`, contact residuals, and cube progress.
The probe does not import or call the hand-written Fetch oracle. In paper runs,
B5 and Ours both receive eight explicit probe trials in total for the entire
run. They use the same counterexample-guided outer loop and split this budget
across the same remaining repair opportunities. B5 consumes non-overlapping
slices of one deterministic space-filling design; Ours selects parameters tied
to the violated constraint and updates later batches from earlier measurements.
Neither method probes after the final generation cycle.

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

Inside the full CEGIS method, subsequent active-probe batches use a stronger
deterministic kernel-UCB selector. It estimates local score from measured rows,
adds an uncertainty and batch-diversity bonus, excludes tried tuples, and only
varies parameters connected to the diagnosed violated constraint. The fixed
grid baseline keeps the same eight-case explicit probe budget and the same
outer loop without active selection. B3, B4, B5, and Ours also receive the same
per-run iterative LLM repair-call limit. B5 and Ours therefore have the same
maximum simulator-call budget; actual calls are measured because either method
may stop early.

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

### Discovery-first migration for unregistered tasks

An unregistered ManiSkill task no longer has to be added to `cases.py` before
the agent can start. Give the environment id and two installed robots:

```bash
python migrate.py \
  --env-id TurnFaucet-v1 \
  --source panda \
  --target fetch \
  --mode agent \
  --max-cycles 3
```

The dynamic harness initializes both environments, records action/controller
layouts, public pose-bearing task entities, environment source, and official
`evaluate()` fields. The LLM first writes a source task program and source
adapter. After official source success, the program is frozen and only the
target adapter may change. Each target attempt is accepted only when both the
program return value and `env.unwrapped.evaluate()['success']` are true.

Generated artifacts stay inside the run directory:

```text
case_manifest.json
source_observation.json
target_observation.json
dynamic_artifacts/task_program.py
dynamic_artifacts/source_adapter.py
dynamic_artifacts/target_adapter.py
source_cycle_*/system_prompt.txt
source_cycle_*/user_prompt.txt
source_cycle_*/source_adapter.py
source_cycle_*/trial_result.json
source_cycle_*/cycle_record.json
target_cycle_*/system_prompt.txt
target_cycle_*/user_prompt.txt
target_cycle_*/target_adapter.py
target_cycle_*/trial_result.json
target_cycle_*/cycle_record.json
dynamic_summary.json
dynamic_summary.md
```

The cycle record stores prompt and adapter SHA-256 values, while the final
manifest stores the hashes of the frozen program and accepted adapters. This
makes a successful run traceable to the exact code and simulator evidence that
produced it.

After a run, print the short result table with:

```bash
cat "$(cat results/migrations/latest.txt)/dynamic_summary.md"
```

This removes manual case registration, but it does not guarantee success. The
environment and robot models must already be installed and compatible. If the
source policy cannot reach official success, target migration stops. Registered
cases remain the frozen path for paper benchmarks and exact reproduction.

Dry-run without ManiSkill or an API call:

```bash
python migrate.py --task TurnFaucet-v1 --source panda --target fetch --mode agent --dry-run
```

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

For a from-zero autonomous migration run, use agent mode:

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode agent \
  --max-cycles 5
```

Agent mode restores the stable seed adapter, then repeats:

```text
LLM generates target adapter
-> real ManiSkill verification
-> structured probe if needed
-> next LLM repair
```

This is the closest current interface to the intended final workflow:

```text
task + source robot + target robot -> autonomous migration loop
```

For GUAVA-style online observe-decide-act control inside a single episode:

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode online
```

The same entrypoint also supports the grasp/place task:

```bash
python migrate.py \
  --task pick_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode online
```

Online mode runs a shorter loop inside the simulator episode:

```text
observe current TCP/cube/goal/grasp state
-> let the LLM choose one bounded semantic primitive
-> execute a few env.step(action) calls
-> observe again
```

This is the “边做边看” task harness, inspired by the iterative
perception-reasoning-action structure in
[GUAVA](https://guava-harness.github.io/). The current implementation reads
structured simulator state; RGB/multimodal observations are not connected yet.
It is different from agent mode:

| Mode | Feedback timing | Main use |
|---|---|---|
| `agent` | after a full trial fails or succeeds | rewrite/repair the adapter module |
| `online` | after every short action segment | adjust the next task primitive using live simulator state |

Online mode uses the LLM planner by default. To debug the loop without an API
call, select the deterministic fallback planner:

```bash
python migrate.py \
  --task pick_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode online \
  --online-planner fallback
```

Dry-run without ManiSkill:

```bash
python migrate.py --task PullCube-v1 --source panda --target xarm6 --dry-run
python migrate.py --task PullCube-v1 --source panda --target xarm6 --mode agent --dry-run
python migrate.py --task PullCube-v1 --source panda --target xarm6 --mode online --dry-run
```

List registered migration requests:

```bash
python migrate.py --list-cases
```

For the older PullCube-only harness workflow, use the short entrypoint:

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

Run the main PullCube migration case:

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

Run the PickCube grasp/place case:

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
| Embodiment contracts | `maniskill_backend/embodiment_contracts.py` |
| Counterexample extraction | `maniskill_backend/counterexamples.py` |
| Active probe selection | `maniskill_backend/active_probe.py` |
| CEGIS loop | `maniskill_backend/counterexample_loop.py` |
| Task specs | `maniskill_backend/tasks.py` |
| Shared skill adapters | `maniskill_backend/skill_adapter.py` |
| Module generation runner | `maniskill_backend/module_generation_runner.py` |
| PickCube probe | `scripts/xarm6_pick_grasp_probe.py` |
| Generated xarm6 PickCube adapter | `maniskill_backend/generated_adapters/case03_xarm6_pick_cube.py` |
| CEGIS method | `docs/METHOD_CEGIS_ADAPTER_SYNTHESIS_CN.md` |
| Experiment protocol | `docs/EXPERIMENT_PROTOCOL_CN.md` |
| Evidence ledger | `docs/EVIDENCE_LEDGER_CN.md` |
| Workshop framing | `docs/WORKSHOP_FRAMING_CN.md` |
| Current harness explanation | `docs/HARNESS_ENGINEERING_CN.md` |
| Current project structure | `docs/PROJECT_STRUCTURE_CN.md` |
| New collaborator onboarding | `docs/COLLABORATOR_ONBOARDING_CN.md` |
| Literature review | `docs/LITERATURE_REVIEW_CN.md` |
| Related-work positioning | `docs/RELATED_WORK_POSITIONING_CN.md` |
| Frozen paper benchmark and RQs | `docs/PAPER_BENCHMARK_CN.md` |
| Archived experiment report | `archive/legacy_docs/EXPERIMENT_REPORT_CN.md` |
| Archived workshop notes | `archive/legacy_docs/WORKSHOP_FRAMING_CN.md` |

## Research Framing

The intended research claim is:

```text
Fixed high-level robot programs can be migrated across embodiments by
synthesizing only target-side adapters. Machine-readable embodiment contracts,
simulation counterexamples, and active physical probes guide guarded adapter
repair under frozen controller and simulator semantics.
```
