# Run Real ManiSkill Experiments

This project now keeps only the real ManiSkill simulation path. The old static
fake runner and text-only benchmark have been removed.

The research target is full cross-embodiment migration, not only rewriting the
high-level LMP snippet. A complete migration trial must distinguish:

- program migration: target LMP ordering, API choices, and exposed parameters;
- execution migration: target skill wrapper, planner/control mode, grasp/contact
  primitive, TCP/tool compensation, and other `env.step(action)` details.

## Remote GPU Setup

On the Polytechnique GPU machine:

```bash
cd ~/Embodied/embodied_migration
git pull
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate em-ms
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

The xarm6 motion planner uses `mplib`/`toppra`, whose current wheels require
NumPy 1.x:

```bash
pip install "numpy>=1.24,<2" --force-reinstall
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Check ManiSkill

`sim_check` uses the official ManiSkill env id because it talks directly to
ManiSkill. The experiment runners below use the clearer task names.

```bash
python -m maniskill_backend.sim_check \
  --env PickCube-v1 \
  --robot panda \
  --obs-mode state \
  --control-mode pd_ee_delta_pos
```

## One Real Trial

Current task names:

```text
pick_cube      抓取方块       ManiSkill env: PickCube-v1
stack_cube     堆叠方块       ManiSkill env: StackCube-v1
pull_cube_tool 用工具拉方块   ManiSkill env: PullCubeTool-v1
peg_insertion  侧向插 peg     ManiSkill env: PegInsertionSide-v1
```

## Case 01: Pull Cube With Tool

The first fixed complete migration case is:

```text
case id: case01_pull_cube_tool_panda_to_xarm6
task: pull_cube_tool / PullCubeTool-v1
source robot: panda
target robot: xarm6_robotiq
source control: pd_joint_pos
target control: pd_joint_pos
first seed: 0
episode budget: 300
```

`pick_cube` and `stack_cube` stay as support tasks. Case 01 is the one that
must demonstrate both target LMP migration and target skill/controller
migration.

The intended research loop is automatic: the LLM may patch the target LMP file,
target skill wrapper, robot profile, or target controller route. It should keep
repairing the failing layer until xarm6 succeeds or the repair budget is used.

Panda smoke test:

```bash
python -m maniskill_backend.real_runner \
  --task pick_cube \
  --robot panda \
  --method source-copy \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu
```

xarm6 with ManiSkill's official planner path:

```bash
python -m maniskill_backend.real_runner \
  --task pick_cube \
  --robot xarm6_robotiq \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu
```

For `pick_cube`, omitting `--control-mode` automatically selects:

```text
panda -> pd_ee_delta_pos
xarm6_robotiq -> pd_joint_pos
```

Passing `--control-mode pd_ee_delta_pos` for xarm6 forces the raw delta-EE path.
That path is useful for diagnosing controller portability, but it may fail even
when the same high-level program succeeds through the official planner.

## LLM Repair Trial

`llm_card_report` is the only LLM adaptation method kept for real simulation.
It first runs a real `source-copy` attempt with the same task, robot, seed, and
sim settings. If that attempt fails, the execution log becomes the Failure
Report shown to the LLM.

```bash
python -m maniskill_backend.real_runner \
  --task pick_cube \
  --robot xarm6_robotiq \
  --method llm_card_report \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu
```

## Full-Stack LLM Migration

This is the main Case 01 runner:

```bash
python -m maniskill_backend.full_stack_runner \
  --case case01_pull_cube_tool_panda_to_xarm6 \
  --max-repair-rounds 3 \
  --sim-backend auto \
  --render-backend gpu
```

The runner:

1. requires a clean tracked worktree;
2. verifies the Panda source side;
3. runs the xarm6 target-side program file;
4. gives the real failure log and in-scope code context to the LLM;
5. applies one LLM unified diff to the bounded target migration surface;
6. runs unit tests;
7. reruns real xarm6 simulation;
8. repeats until success or budget exhaustion.

Allowed Case 01 patch files:

```text
maniskill_backend/case_programs/case01_pull_cube_tool.py
maniskill_backend/profiles.py
maniskill_backend/real_runner.py
maniskill_backend/skill_adapter.py
```

Outputs:

```text
results/full_stack_trials.jsonl
results/full_stack_trials.md
```

Successful LLM patches remain as local tracked diffs so they can be inspected,
benchmarked, and committed. A patch that fails unit tests is reverted before
the next repair round.

## Program-Only Iterative Baseline

This runner automates the program-migration layer. It verifies that the source
robot succeeds, asks the LLM to write target-robot code, runs the target code,
feeds the simulator failure log back to the LLM, and repeats up to
`--max-attempts`.

Do not treat repeated LMP failure as the end of the migration experiment. If
the source sequence is sound but the target robot cannot physically realize the
same grasp, tool contact, planner route, or controller path, migrate the target
execution layer in `maniskill_backend/skill_adapter.py` and rerun the same
trial.

For `pull_cube_tool`, the iterative runner exposes tunable target-code
parameters:

```python
robot.hook_object(tool, cube, hook_y_offset=-0.067, behind_margin=0.0, tool_grasp_x_offset=0.08)
robot.pull_with_tool(tool, cube, workspace, distance=0.35, stages=1, pull_frame="toward_base")
```

`pull_frame` can be `"tool"`, `"world"`, or `"toward_base"`. Panda keeps the
official tool-local pull by default; xarm6 defaults to pulling toward its base.
The wrapper also compensates for the held tool's actual offset from the TCP, so
tool-contact targets are not treated as gripper-center targets.
`tool_grasp_x_offset` lets xarm6 grasp deeper along the L-shaped tool handle
when the Robotiq gripper cannot stably carry the official Panda grasp point.

Run:

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

## Case 01 Full-Stack Migration Workflow

Use this order for `case01_pull_cube_tool_panda_to_xarm6`:

1. Run the Panda source side and require real simulation success.
2. Run xarm6 `source-copy` to expose the first portability failure.
3. Run `full_stack_runner` so the LLM chooses whether the next patch belongs in
   the target program, skill adapter, profile, or controller route.
4. Inspect the kept LLM patch and the rerun result after each repair round.
5. Rerun source-copy, program-only baselines, and benchmark commands around the
   full-stack result.
6. Record both code levels: target LMP diffs and adapter / controller diffs.

For Case 01, wrapper migration is already part of the experiment:
xarm6 needs a target-specific tool grasp depth, held-tool pose compensation,
contact correction, and a pull frame that is physically meaningful for its
planner path.

## Case 01 Real Benchmark

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

Running `python -m maniskill_backend.real_benchmark` with no CLI overrides now
uses this Case 01 task/target/controller/seed budget.

## Support Task: Stack Cube

`stack_cube` is a support task. It is more demanding than `pick_cube` because
cube A must remain stably on cube B after release.

Panda source run:

```bash
python -m maniskill_backend.real_runner \
  --task stack_cube \
  --robot panda \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 200
```

xarm6 target run:

```bash
python -m maniskill_backend.real_runner \
  --task stack_cube \
  --robot xarm6_robotiq \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 200
```

Benchmark:

```bash
python -m maniskill_backend.real_benchmark \
  --task stack_cube \
  --robot xarm6_robotiq \
  --methods source-copy,llm_card_report,oracle \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 200
```

## Case 01 Commands: Pull Cube With Tool

`pull_cube_tool` is a tool-use task. The source program must hook the cube with
the L-shaped tool before pulling it back into the robot workspace.

Panda source run:

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube_tool \
  --robot panda \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300
```

xarm6 target run:

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube_tool \
  --robot xarm6_robotiq \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300
```

Benchmark:

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

## Parked Task: Peg Insertion

`peg_insertion` is currently parked because ManiSkill's official Panda solver
also failed at seed 0 in this environment. Do not use it as a migration task
until the official source side is reliable.

Historical source-side check:

```bash
python -m maniskill_backend.real_runner \
  --task peg_insertion \
  --robot panda_wristcam \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

Interpretation:

```text
panda_wristcam failed -> do not evaluate xarm6 yet
```

Outputs:

```text
results/real_trials.jsonl
results/real_trials.md
results/real_summary.csv
```

Open `results/real_trials.md` to inspect the Capability Card, Failure Report,
generated code, raw LLM output, and real execution log.

## Current Validated Result

The current validated real simulation slice is:

```text
pick_cube + panda + pd_ee_delta_pos -> success
pick_cube + xarm6_robotiq + pd_ee_delta_pos -> controller/skill-wrapper failure
pick_cube + xarm6_robotiq + pd_joint_pos planner -> success
stack_cube + official Panda solver -> success at seed 0
pull_cube_tool + official Panda solver -> success at seed 0
peg_insertion + official Panda solver -> failure at seed 0, parked
```
