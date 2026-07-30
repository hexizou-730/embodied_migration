# 实验证据账本

本文件只记录当前 Git 仓库中可追溯的证据，不把口头运行结果、未提交的远程日志或 oracle 当成 LLM 迁移成功。

## 证据规则

一次可复现成功必须同时包含：非 dry-run 仿真日志、与日志一致的 adapter SHA、seed/环境配置，以及 ManiSkill 真实 success signal。

## 当前状态

| Case | 迁移 | 当前 adapter 来源 | 已跟踪结果 | 可复现成功证据 | 状态 |
|---|---|---|---:|---|---|
| `case01_pull_cube_panda_to_fetch` | panda -> fetch (pull_cube) | `hand_written_oracle` | 0 | `False` | `oracle_upper_bound_without_tracked_run` |
| `case02_pull_cube_panda_to_xarm6` | panda -> xarm6_robotiq (pull_cube) | `neutral_seed` | 0 | `False` | `no_tracked_success_evidence` |
| `case03_pick_cube_panda_to_xarm6` | panda -> xarm6_robotiq (pick_cube) | `neutral_seed` | 0 | `False` | `no_tracked_success_evidence` |

## 解释

- Fetch 当前 adapter 明确标注为 hand-written oracle，只能作为性能上界。
- xArm6 当前提交文件若与 seed adapter 相同，只代表中性起点，不代表 LLM 成功代码。
- 远程产生的新成功结果必须把运行 summary、adapter snapshot 和 SHA 一起带回仓库，才能升级本账本状态。
