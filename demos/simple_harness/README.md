# Simple Harness Demo

This is a minimal demo of the project idea:

```text
agent_observation.json
-> agent_plan.json
-> selected simulator tool
-> tool_result.json
```

The LLM/Agent does not directly edit simulator state. It sees a structured
observation, chooses one bounded tool, and the harness exposes the corresponding
command.

## Run

From the repository root:

```bash
python demos/simple_harness/demo.py
```

This is a dry run. It does not launch ManiSkill. It writes:

```text
results/simple_demo/<run_name>/
  agent_observation.json
  agent_plan.json
  selected_tool_command.txt
  tool_result.json
  README.md
```

## Run One Real Tool

On a remote GPU machine with ManiSkill configured:

```bash
python demos/simple_harness/demo.py --run
```

By default this runs only seed 0 for the PullCube Panda -> xArm6 case.

## Sample Outputs

`sample_outputs/` contains one committed dry-run output so the demo can be
understood without running anything.

## Relation To The Full System

| File | Purpose |
|---|---|
| `demos/simple_harness/demo.py` | Minimal demonstration |
| `auto.py` | Full autonomous loop |
| `maniskill_backend/autonomous_harness.py` | Builds agent observations and tool inventory |
| `maniskill_backend/agent_planner.py` | Lets an LLM or fallback policy choose tools |

Use this folder for presentation. Use `auto.py` for real experiments.
