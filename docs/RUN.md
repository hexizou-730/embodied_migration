# Run

Current WSL2 workflow for the static migration prototype.

## Setup

```bash
cd ~/Embodied/embodied_migration
conda activate em-ms
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Check ManiSkill

Use this before writing a real simulation runner:

```bash
python -m maniskill_backend.sim_check \
  --env PegInsertionSide-v1 \
  --obs-mode state
```

Add `--render` only after reset/step works.

## Real ManiSkill Smoke Task

`PickCube-v1` is the first task wired toward the real ManiSkill action API.
It uses `ManiSkillPickCubeRobot` to translate high-level LMP calls such as
`robot.grasp(cube)` and `robot.place(cube, goal)` into `env.step(action)` calls.

```bash
python -m maniskill_backend.real_runner \
  --task PickCube-v1 \
  --robot panda \
  --method source-copy \
  --control-mode pd_ee_delta_pos
```

On the current WSL setup this may still fail at Vulkan/SAPIEN environment
creation. On native Ubuntu with working Vulkan, this is the first command to try.

For report-based real trials, `real_runner` first runs a real `source-copy`
attempt with the same task, robot, seed, and simulator settings. If that prior
attempt fails, its execution log is converted into the Failure Report that is
shown to the LLM and written into the result.

## Run One Trial

```bash
python -m maniskill_backend.static_runner \
  --task PegInsertionSide-v1 \
  --target so100 \
  --method llm_card_report
```

Common methods:

```text
source-copy
llm_no_card
llm_card_only
llm_report_only
llm_card_report
oracle
```

Old names still work as aliases:

```text
llm_failure_only -> llm_report_only
llm_card_failure -> llm_card_report
```

## Run Current Comparison

```bash
python -m maniskill_backend.run \
  --tasks PegInsertionSide-v1,PlugCharger-v1,PlugMulti-v1,PullCubeTool-v1,PegMulti-v1 \
  --targets so100 \
  --methods source-copy,llm_no_card,llm_card_only,llm_report_only,llm_card_report,oracle
```

Outputs:

```text
results/trials.jsonl   # full append-only trial log
results/trials.md      # readable report, overwritten each run
results/summary.csv    # summary table, overwritten each run
```

Open `results/trials.md` to see each trial's Capability Card, Failure Report,
generated code, and raw LLM output.

Capability Cards separate nominal specs from migration priors. In particular,
`recommended_alignment_tolerance_m` is a recommended control tolerance for the
target robot; it is not the task's physical success requirement.

For `llm_report_only` and `llm_card_report`, the Failure Report is generated
from the previous failed source-copy execution log. The readable markdown shows:

```text
Report Source Log   # failed attempt used to generate the report
Execution Log       # current generated code run
```

## Dry Run

Use this when you do not want to spend API calls:

```bash
python -m maniskill_backend.run \
  --tasks PegInsertionSide-v1,PlugCharger-v1,PlugMulti-v1,PullCubeTool-v1,PegMulti-v1 \
  --targets so100 \
  --methods source-copy,llm_no_card,llm_card_only,llm_report_only,llm_card_report,oracle \
  --dry-run
```
