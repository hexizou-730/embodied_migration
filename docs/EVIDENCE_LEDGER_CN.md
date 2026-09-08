# 实验证据账本

本文件只记录当前 Git 仓库中可追溯的证据，不把口头运行结果、未提交的远程日志或 oracle 当成 LLM 迁移成功。

## 证据规则

一次可复现成功必须同时包含：非 dry-run 仿真日志、与日志一致的 adapter SHA、seed/环境配置，以及 ManiSkill 真实 success signal。

## 当前状态

| Case | 迁移 | 支持范围 | 当前 adapter 来源 | 已跟踪结果 | 可复现成功证据 | 状态 |
|---|---|---|---|---:|---|---|
| `case01_pull_cube_panda_to_fetch` | panda -> fetch (pull_cube) | `official_supported` | `hand_written_oracle` | 0 | `False` | `oracle_upper_bound_without_tracked_run` |
| `case02_pull_cube_panda_to_xarm6` | panda -> xarm6_robotiq (pull_cube) | `stress_test_override` | `neutral_seed` | 0 | `False` | `no_tracked_success_evidence` |
| `case03_pick_cube_panda_to_xarm6` | panda -> xarm6_robotiq (pick_cube) | `official_supported` | `neutral_seed` | 0 | `False` | `no_tracked_success_evidence` |
| `case04_pick_cube_panda_to_fetch` | panda -> fetch (pick_cube) | `official_supported` | `neutral_seed` | 0 | `False` | `no_tracked_success_evidence` |
| `case05_push_cube_panda_to_fetch` | panda -> fetch (push_cube) | `official_supported` | `neutral_seed` | 0 | `False` | `no_tracked_success_evidence` |

## 解释

- Case 01 的 Fetch PullCube adapter 明确标注为 hand-written oracle，只能作为性能上界。
- Case 04 的 Fetch PickCube adapter 是 neutral seed，不是 oracle，也不代表已迁移成功。
- B0、B1 和 Oracle 即使 held-out 成功，也不会计为 LLM adapter 迁移成功。
- xArm6 当前提交文件若与 seed adapter 相同，只代表中性起点，不代表 LLM 成功代码。
- 远程产生的新成功结果必须把运行 summary、adapter snapshot 和 SHA 一起带回仓库，才能升级本账本状态。
