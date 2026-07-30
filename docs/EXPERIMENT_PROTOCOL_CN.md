# 实验协议

## 1. 核心原则

- 只修改 target-side adapter。
- 不修改高层 Program、controller、simulator 和 success signal。
- hand-written oracle 与 LLM-generated adapter 分开统计。
- development seeds 可用于诊断和修复。
- held-out seeds 只能用于最终评估，不能反馈给 LLM。
- 每个结果必须绑定 adapter SHA、Git commit、seed 和仿真配置。

## 2. Seed 划分

快速开发：

```text
development: 0-4
held-out:    100-109
```

论文实验：

```text
development: 0-9
held-out:    100-129
```

正式结果至少重复三次独立 LLM generation run，避免一次随机生成被当成稳定方法。

## 3. 对照方法

| ID | 方法 |
|---|---|
| B0 | Source adapter 直接复制 |
| B1 | Shared generic adapter |
| B2 | LLM one-shot |
| B3 | LLM + raw failure log |
| B4 | LLM + five-layer failure diagnosis |
| B5 | LLM + fixed-grid structured probe |
| Ours | contract + counterexample + active probe + guarded adapter |
| Oracle | hand-written adapter，只作为性能上界 |

## 4. 消融实验

- 去掉 embodiment contract；
- 去掉 counterexample selection；
- active probe 改为 fixed-grid；
- guarded adapter 改为固定轨迹；
- 去掉 held-out split；
- 去掉 explicit infeasibility branch。

## 5. 指标

- held-out success rate；
- development success rate；
- LLM 修复轮数；
- 仿真调用次数；
- probe case 数；
- environment steps；
- token/API 成本；
- 人工修改次数；
- 不可行判断准确率；
- 最终 adapter 代码长度和分支数。

## 6. 接受标准

快速开发默认：

```text
development success rate >= 0.8，且至少 5 个 seed
held-out success rate >= 0.8，且至少 5 个 seed
```

只有 held-out 达标才记为方法成功。Development 达标而 held-out 失败记为
`held_out_rejected`，系统不会继续用 held-out 反例调参。

## 7. 产物目录

每次 CEGIS run 应包含：

```text
results/cegis/<run>/
  adapter_before_run.py
  commands.log
  summary.json
  summary.md
  cycle_01/
    module_generation.jsonl
    module_generation.md
    development.jsonl
    development.md
    counterexamples.json
    structured_probe/
  cycle_02/
    held_out.jsonl
    held_out.md
```

当 held-out 达标时，系统会自动把精简证据复制到：

```text
evidence/runs/<run>/
  manifest.json
  final_adapter.py
  development_jsonl.jsonl
  held_out_jsonl.jsonl
  module_generation_jsonl.jsonl
  commands_log.log
```

`results/` 是本地完整日志，默认忽略；`evidence/runs/` 是准备提交到 Git 的论文证据。

成功结果提交时至少保留：

- `summary.json`；
- 最终 adapter snapshot；
- development 与 held-out JSONL；
- adapter SHA；
- Git commit；
- 模型/provider/max tokens；
- ManiSkill 和运行环境版本。

## 8. 推荐运行命令

快速实验：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis \
  --development-seeds 0-4 \
  --held-out-seeds 100-109 \
  --max-cycles 3
```

论文实验：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis \
  --development-seeds 0-9 \
  --held-out-seeds 100-129 \
  --max-cycles 5
```
