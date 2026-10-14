# Generation Stability

- state: `complete`
- task: `t01_push_cube`
- migration: `panda -> xarm6_robotiq`
- fixed simulation seed: `0`
- success: `2/3`
- generation success rate: `0.667`
- total tokens: `347790`
- total LLM cost: `$0.093258`

| repetition | success | official | status | cycles | repairs | actions | tokens | cost USD |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | True | True | success | 2 | 1 | 10 | 36493 | 0.016532 |
| 2 | True | True | success | 2 | 1 | 28 | 38950 | 0.012801 |
| 3 | False | False | target_budget_exhausted | 8 | 7 | 1621 | 272347 | 0.063925 |
