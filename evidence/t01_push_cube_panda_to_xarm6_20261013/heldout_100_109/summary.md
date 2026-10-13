# Fixed Adapter Multi-Seed Evaluation

- state: `complete`
- adapter SHA-256: `1c0500b983a5c85a182634830268fd64390c0c0fd9c40f9778788565a9dd6892`
- success: `5/10`
- success rate: `0.500`
- LLM calls: `0`
- code repairs: `0`

| seed | success | official | actions | failure reason | message |
|---:|---|---|---:|---|---|
| 100 | False | False | 661 | official_task_outcome_not_reached | task program returned failure |
| 101 | False | False | 632 | official_task_outcome_not_reached | task program returned failure |
| 102 | False | False | 320 | contact_not_established | task program returned failure |
| 103 | True | True | 310 | success | adapter returned success and official evaluate() returned success |
| 104 | True | True | 320 | success | adapter returned success and official evaluate() returned success |
| 105 | True | True | 331 | success | adapter returned success and official evaluate() returned success |
| 106 | False | False | 711 | official_task_outcome_not_reached | task program returned failure |
| 107 | False | False | 566 | official_task_outcome_not_reached | task program returned failure |
| 108 | True | True | 301 | success | adapter returned success and official evaluate() returned success |
| 109 | True | True | 292 | success | adapter returned success and official evaluate() returned success |
