# 实验协议

## 1. 核心原则

- 只修改 target-side adapter。
- 不修改高层 Program、controller、simulator 和 success signal。
- hand-written oracle 与 LLM-generated adapter 分开统计。
- development seeds 可用于诊断和修复。
- held-out seeds 只能用于最终评估，不能反馈给 LLM。
- 每个结果必须绑定 adapter SHA、Git commit、seed 和仿真配置。
- 每次运行前后必须校验冻结接口 SHA；发生变化的 run 不计入成功结果。
- 同一 seed 协议下的 Panda 源任务必须先达标，否则目标迁移矩阵不启动。

## 2. Seed 划分

快速开发：

```text
development: 0-4
held-out:    100-104
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
| B5 | CEGIS + fixed-grid structured probe |
| Ours | contract + counterexample + active probe + guarded adapter |
| Oracle | hand-written adapter，只作为性能上界 |

B5 与 Ours 使用完全相同的外层流程：生成 adapter、跑 development seeds、选择反例、
probe、修复，并在通过 development 门槛后只做一次 held-out 验收。两者也共享完整 prompt
信息和相同的 **整次 run 总 probe budget**（8 次），唯一差别是 probe selector：B5 按一个
冻结的均匀网格分批取互不重复的候选，Ours 根据当前 violated constraint 主动选点。
最后一个 cycle 都不再做无法用于修复的 probe，两者都不得读取 held-out seed。
每批输出会保留累计 `all_probe_cases` 供下一轮使用，同时用 `num_new_cases` 单独计费，
防止历史测量既被遗忘又被重复计入预算。

迭代式方法 B3、B4、B5、Ours 的 LLM 代码生成上限也保持一致：pilot 每个 run 最多 3 次，
paper 每个 run 最多 5 次；B2 one-shot 固定为 1 次。B5 与 Ours 还具有相同的外层仿真
调用上限；实际调用数可能因任一方法提前成功或提前失败而不同，因此仍作为成本指标报告。

## 4. 消融实验

以下为待配置、待运行的消融，不能与已完成结果混用。所有消融保留同一 held-out 集。

- 去掉 embodiment contract；
- 去掉 counterexample selection；
- active probe 改为 fixed-grid；
- guarded adapter 改为固定轨迹；
- development 由多 seed 反例改为单 seed 修复，检验对未见 seed 的影响；
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
- counterexample reason/seed 多样性、CEGIS cycle 数和 kernel-UCB 批次。

## 6. 接受标准

快速开发默认：

```text
development success rate >= 0.8，且至少 5 个 seed
held-out success rate >= 0.8，且至少 5 个 seed
```

只有 held-out 达标才记为方法成功。Development 达标而 held-out 失败记为
`held_out_rejected`，系统立即停止，不会继续用 held-out 反例调参。每个 CEGIS cycle
都会记录 `repair_feedback.source_split`，held-out 记录固定为 `used_for_repair=false`；
`python paper.py audit` 会机器检查该策略。

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

先在 GPU 服务器执行一次基础设施预检；预检不调用 LLM，也不运行完整迁移：

```bash
python paper.py preflight
```

老师汇报所需的两个 motivating example 证据用一条命令采集；它不调用 LLM，也不修改
adapter：

```bash
python paper.py evidence
```

该命令运行 PullCube 多 seed 验证和 PickCube 结构化 probe，并把真实 JSON/JSONL、日志及
SHA256 汇总到 `results/paper/motivating_examples/`。

可单独检查源端 Panda 的同 seed 多场景基线：

```bash
python paper.py source --tier pilot --plan-name pilot_v1
```

`paper.py run` 会自动执行并缓存该门控。只有源端 development 和 held-out 都达到接受
阈值，才开始比较目标 adapter，避免把源任务本身的失败误记为迁移失败。

论文对照实验统一使用：

```bash
python paper.py plan --tier pilot
python paper.py run --tier pilot --methods B0,B1,B2,B3,B4,B5,Ours,Oracle
```

正式矩阵使用 `python paper.py run --tier paper`。详细的可执行状态、输出目录和
证据规则见 [`PAPER_BENCHMARK_CN.md`](PAPER_BENCHMARK_CN.md)。

下面的 `migrate.py` 命令用于单独调试 Ours，而不是替代完整 baseline 矩阵。

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
