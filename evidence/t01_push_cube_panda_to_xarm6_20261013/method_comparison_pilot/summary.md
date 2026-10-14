# Method Comparison

- state: `complete`
- task: `t01_push_cube`
- migration: `panda -> xarm6_robotiq`
- fixed seed: `0`
- completed: `4/4`

| method | success | rate | mean cycles | mean actions | tokens | cost USD |
|---|---:|---:|---:|---:|---:|---:|
| one_shot | 1/1 | 1.000 | 1 | 20 | 17514 | 0.008836 |
| raw_log | 1/1 | 1.000 | 1 | 29 | 18458 | 0.010108 |
| diagnosis | 0/1 | 0.000 | 3 | 571 | 63078 | 0.029901 |
| ours | 0/1 | 0.000 | 3 | 0 | 52428 | 0.029674 |

## Runs

| method | repetition | success | official | cycles | actions | tokens | cost USD |
|---|---:|---|---|---:|---:|---:|---:|
| one_shot | 1 | True | True | 1 | 20 | 17514 | 0.008836 |
| raw_log | 1 | True | True | 1 | 29 | 18458 | 0.010108 |
| diagnosis | 1 | False | False | 3 | 571 | 63078 | 0.029901 |
| ours | 1 | False | False | 3 | 0 | 52428 | 0.029674 |
