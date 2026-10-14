# Phase 8 Method-Comparison Pilot

Run on 2026-10-14 with the frozen `t01_push_cube` Panda source, target
`xarm6_robotiq`, simulation seed 0, and `deepseek/deepseek-v4-pro` through
OpenRouter. Each method was run once. Repair methods had at most three target
generation cycles; one-shot had exactly one. ManiSkill's official `evaluate()`
result remained the success authority.

| method | success | cycles | target actions | tokens | cost USD |
|---|---:|---:|---:|---:|---:|
| one_shot | 1/1 | 1 | 20 | 17,514 | 0.008836 |
| raw_log | 1/1 | 1 | 29 | 18,458 | 0.010108 |
| diagnosis | 0/1 | 3 | 571 | 63,078 | 0.029901 |
| ours | 0/1 | 3 | 0 | 52,428 | 0.029674 |

This is an infrastructure pilot, not a method-ranking result. The two successful
methods happened to succeed on their first generation and therefore did not test
repair. The diagnosis run produced one executable but unsuccessful candidate. All
three candidates from `ours` failed static contract checks before action execution,
so incumbent preservation was never activated. More independent repetitions and
failure-triggering cases are required for a defensible comparison.

`cycle_evidence.json` keeps bounded per-cycle validation, execution and usage fields.
`adapters/` keeps the accepted adapters and the three invalid `ours` candidates.
The longer prompt, response and state-trace logs remain in the remote run directory
and are intentionally not committed. An earlier pilot was interrupted after an
OpenRouter request stalled; it is excluded from this evidence. The client now uses
a recorded 180-second request timeout and zero hidden retries.
