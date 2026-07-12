# Harness Engineering：把仿真环境暴露给 LLM Agent

## 一句话定义

这里的 Harness Engineering 不是让 LLM 直接操作 ManiSkill 的内部对象，而是把仿真环境封装成一组安全、可复现、可记录的工具接口：

```text
LLM Agent -> harness tools -> real ManiSkill env.step(action) -> structured result -> LLM Agent
```

这样 Agent 可以自主判断“当前代码是否符合物理约束”，但不能绕过任务成功信号，也不能直接修改 simulator、controller 或物体状态。

## 两种 harness 粒度

现在项目里有两种粒度：

| 粒度 | 什么时候看环境 | 解决什么问题 |
|---|---|---|
| Episode-level harness | 一整次 trial 结束后看 success/failure 和诊断 | 让 LLM 重写或修复 adapter |
| Online harness | 同一个 episode 中每执行几步就重新观察 TCP/cube/goal | 边做边看，下一小段动作按当前物理状态调整 |

你说的“实时感知环境”对应第二种 online harness。

它的核心循环是：

```text
observe current simulator state
-> choose bounded primitive
-> execute a short env.step(action) segment
-> observe again
-> choose the next primitive
```

当前 online harness 先落在 `PullCube` 上，状态包括：

```text
cube_pos
goal_pos
tcp_pos
tcp_contact_error
cube_goal_xy
allowed_primitives
```

可选 primitive 包括：

```text
move_to_pre_contact
move_to_contact
drag_toward_goal
hold
stop
```

运行方式：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode online
```

如果想让 LLM 在每个小段之前选择 primitive，可以加：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode online \
  --online-planner llm
```

## 现有项目中的 harness 工具

| 工具 | 作用 | 回传给 Agent 的信息 |
|---|---|---|
| `run_single_seed` | 跑一次真实 ManiSkill trial | success、failure_layer、failure_diagnosis、runtime_diagnostics |
| `run_multi_seed` | 多 seed 评估当前 adapter 泛化能力 | success_rate、失败 seed 分布、generalization_strategy |
| `run_structured_probe` | 针对失败物理机制做参数扫描 | probe table、best_probe_case、prompt_feedback |
| `run_llm_repair` | 基于诊断和 probe 结果重写 adapter | generated adapter、target_result、migration_analysis |
| `inspect_results` | 读取 compact Markdown/JSON 结果 | 最新实验摘要、失败聚类、runtime diagnostics |

## 双通道输出

Harness 的输出分两份：

| 文件 | 给谁看 | 内容 |
|---|---|---|
| `agent_observation.json` | LLM Agent | 事实状态、约束、工具列表、仿真输出 |
| `human_report.md` | 人/老师/组会 | 中文解释、总结、可选下一步建议 |

Agent 只应该拿 `agent_observation.json`，不拿 human report。

## Agent 看到的 observation

Harness 会把当前状态整理成一个稳定 JSON，但不写长篇报告，也不直接告诉它下一步该怎么做：

```text
case_id
task_id
source_robot / target_robot
current_adapter
high_level_program
constraints
latest_results
latest_results.multiseed.failure_rows
allowed_tools
```

这里面只有状态和工具。比如失败 seed 的 `runtime_diagnostics` 会暴露 TCP、cube、stage target、error norm 等数值，但不会写一段“你应该这样修”的中文报告。

## 安全边界

固定不变：

- 高层程序不变：例如 `robot.pull(cube, goal)` 或 `robot.grasp(cube); robot.place(cube, goal)`
- 只能改 target-side adapter
- 不能改 ManiSkill controller
- 不能改 simulator
- 不能改 success signal
- 所有验证必须通过真实 `env.step(action)`

这保证实验结论仍然是“目标机器人能否通过 adapter 在真实仿真中完成任务”，而不是代码绕过。

## 当前推荐使用方式

最简单的完整自动闭环：

```bash
python auto.py pull
```

它不是固定写死“失败后必须 probe”。现在每一轮都是：

```text
agent_observation.json -> LLM planner 选择工具 -> harness 校验并执行 -> 新 observation
```

LLM 可以在允许工具中自己选择：

```text
run_single_seed
run_multi_seed
run_structured_probe
run_llm_repair
inspect_results
stop
```

先只检查命令和输出结构，不真正跑仿真：

```bash
python auto.py pull --dry-run
```

多 seed 跑完后，用 harness 生成 Agent observation 和 human report：

```bash
python scripts/autonomous_harness_runner.py \
  --case case02_pull_cube_panda_to_xarm6 \
  --multiseed-jsonl results/pullcube_xarm6_multiseed.jsonl \
  --print-agent-observation
```

如果本地还没有 multi-seed 文件，也可以先生成初始 observation：

```bash
python scripts/autonomous_harness_runner.py \
  --case case02_pull_cube_panda_to_xarm6 \
  --print-agent-observation
```

输出文件：

```text
results/autonomous_harness/<case_id>/agent_observation.json
results/autonomous_harness/<case_id>/human_report.md
results/autonomous_harness/<case_id>/harness_bundle.json
```

## 和之前相比的提升

之前的流程更像：

```text
失败 -> 人手动读日志 -> 人总结原因 -> 人改 prompt -> LLM 再生成
```

现在的流程变成：

```text
失败 -> 自动诊断 -> 自动生成 Agent observation -> Agent 自己选择工具 -> LLM 修复 adapter
```

核心提升是：LLM 不再只看自然语言失败描述，而是看到真实仿真测得的物理约束证据；同时它不会被人类总结报告直接“喂答案”。

## 汇报时可以这样讲

我把 ManiSkill 仿真环境做成了一个 autonomous harness。LLM Agent 不直接访问 simulator，而是只能调用几个安全工具：跑单 seed、跑多 seed、跑 structured probe、读取结果、重写 adapter。关键是我区分了 human report 和 agent observation：给老师看的是中文总结，给 Agent 的只有 JSON 状态和工具输出。这样可以让 Agent 自己根据仿真反馈判断当前代码是否满足物理约束，而不是照着人工报告改代码。
