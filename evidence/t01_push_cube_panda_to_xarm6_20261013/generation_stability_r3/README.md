# Generation Stability Pilot

Phase 7 repeats the same frozen-source migration three times from an empty target
candidate state. Repetitions do not share adapters or incumbents.

- Task: `t01_push_cube`
- Migration: `panda -> xarm6_robotiq`
- Model: `deepseek/deepseek-v4-pro`
- Fixed simulation seed: `0`
- Maximum target cycles per repetition: `8`
- Officially successful repetitions: `2/3`
- Total target actions: `1,659`
- Total tokens: `347,790`
- Recorded LLM cost: `$0.09325787`

Repetitions 1 and 2 reached official success in cycle 2. Repetition 3 exhausted
all eight cycles and ended with `contact_not_established`. The three target
adapter files are retained so their hashes can be checked against `summary.json`.

This is a pilot result, not the final paper-scale estimate of generation success.
Full execution traces remain in the remote `results/generation_stability/` run.
