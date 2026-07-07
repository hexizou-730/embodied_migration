# 简单版 Harness Demo

这个 demo 用最少文件展示项目核心思路：

```text
LLM/Agent 看到 agent_observation.json
-> 选择一个安全工具
-> harness 执行这个工具
-> 写回 tool_result.json
```

## 一条命令看结构

```bash
python demo.py
```

这不会跑仿真，只生成 demo 文件：

```text
results/simple_demo/latest.txt
results/simple_demo/<run_name>/
  agent_observation.json
  agent_plan.json
  selected_tool_command.txt
  tool_result.json
  README.md
```

## 远程 GPU 上跑一次真实工具

```bash
python demo.py --run
```

默认只跑 seed 0，所以比完整自动循环更短。

## 这个 demo 讲什么

- `agent_observation.json`：给 LLM/Agent 的状态，不包含中文 human report。
- `agent_plan.json`：Agent 决定下一步调用什么工具。
- `selected_tool_command.txt`：harness 暴露出来的真实命令。
- `tool_result.json`：工具执行结果，后续可以再变成新的 observation。

## 和正式版的区别

| 版本 | 命令 | 用途 |
|---|---|---|
| 简单 demo | `python demo.py` | 汇报/教学，展示 harness 思路 |
| 跑一次工具 | `python demo.py --run` | 远程 GPU 上快速验证 |
| 完整闭环 | `python auto.py pull --seeds 0-9 --max-cycles 3` | 多轮 agent repair |

一句话：`demo.py` 是最小展示版，`auto.py` 是正式实验版。
