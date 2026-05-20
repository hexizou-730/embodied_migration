# Run Real ManiSkill Experiments

This project now keeps only the real ManiSkill simulation path. The old static
fake runner and text-only benchmark have been removed.

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
peg_insertion  侧向插 peg     ManiSkill env: PegInsertionSide-v1
```

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

## Real Benchmark

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

## Second Task: Peg Insertion

Use `peg_insertion` to decide whether the next research direction should focus
on high-level LMP migration, skill-wrapper migration, or controller/planner
selection.

First verify the official Panda source task. `PegInsertionSide-v1` officially
supports `panda_wristcam`, and this path uses ManiSkill's official motion
planner.

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

Then check whether ManiSkill can even create the same task with xarm6:

```bash
python -m maniskill_backend.sim_check \
  --env PegInsertionSide-v1 \
  --robot xarm6_robotiq \
  --obs-mode state \
  --control-mode pd_ee_pose
```

If xarm6 creation works, try the real target run:

```bash
python -m maniskill_backend.real_runner \
  --task peg_insertion \
  --robot xarm6_robotiq \
  --method source-copy \
  --seed 0 \
  --control-mode pd_ee_pose \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

Interpretation:

```text
panda_wristcam succeeds, xarm6 fails in align/insert -> high-level parameters or target skill wrapper matter
panda_wristcam fails -> fix the Panda peg_insertion wrapper before using it for migration
xarm6 env creation fails -> task is not yet a cross-robot ManiSkill experiment
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
```
