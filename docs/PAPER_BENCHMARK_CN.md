# 论文实验入口

## 目的

论文实验不再手工拼接命令。`paper.py` 固定方法、case、seed 划分和重复次数，
并为每次运行保存 adapter、SHA、模型配置、开发集结果和 held-out 结果。

## 两档实验

| 档位 | Development | Held-out | LLM 独立重复 |
|---|---|---|---:|
| `pilot` | 0-4 | 100-104 | 1 |
| `paper` | 0-9 | 100-129 | 3 |

Held-out seed 只用于最终评估，不进入 LLM 修复提示。

## 研究问题

| RQ | 问题 | 主要指标 | 必需证据 |
|---|---|---|---|
| RQ1 | 反例驱动合成是否优于 LLM one-shot？ | 配对 held-out success、分层 bootstrap 95% CI | held-out JSONL、paired comparison CSV |
| RQ2 | 相同外层循环、prompt、probe 与 LLM 预算下，active probe 是否优于 fixed grid？ | success、simulator calls、probe cases、cycle 数 | probe plan/result、paired comparison CSV |
| RQ3 | 五层诊断能否正确定位真实失败？ | layer/reason accuracy、confusion matrix、Cohen's kappa | 盲标 JSONL、diagnosis evaluation |
| RQ4 | adapter 能否泛化到未见 seed 和 stress test？ | pooled success、跨重复方差、support subgroup success | held-out JSONL、subgroup CSV |
| RQ5 | 修复成本是多少？ | 仿真调用、环境步数、LLM tokens、wall-clock time、代码分支数 | run summary、summary CSV |

这五个问题及其指标、证据文件会写入每次冻结的 `plan.json`。协议审计会拒绝没有
指标或没有证据来源的 RQ。

## 预注册假设

- **H1（主假设）**：在 `official_supported` case 上，Ours 的配对 held-out 成功率高于 B2 one-shot。
- **H2（主假设）**：在相同外层循环、prompt、显式 probe 与 LLM 调用预算下，Ours 的配对 held-out 成功率高于 B5 fixed-grid。
- **H3（补充结果）**：`stress_test_override` 单独报告 Ours 与 B5，不并入主假设总体。

这些假设写入每个冻结 `plan.json`，统计前已经确定，避免看到结果后选择比较对象。

## 方法

| ID | 方法 | 当前执行状态 |
|---|---|---|
| B0 | Source adapter copy | 五个 case 可执行 |
| B1 | Shared generic adapter | 五个 case 可执行 |
| B2 | LLM one-shot | 五个 case 可执行 |
| B3 | LLM + raw failure | 五个 case 可执行 |
| B4 | LLM + five-layer diagnosis | 五个 case 可执行 |
| B5 | CEGIS + fixed-grid probe | 五个 case 均可执行；与 Ours 共用外层循环，固定预算为 8 次 probe |
| Ours | contract + counterexample + active probe + guarded repair | 五个 case 均可执行；整次 run 最多主动选择 8 次 probe |
| Oracle | hand-written upper bound | 两个 PullCube case 可执行；PickCube 与 PushCube 尚无 oracle |

B2-B4 的信息差异由 `prompt_policy` 强制控制。B5 是 selector 消融，因此与 Ours 同用
`full` prompt、development counterexample、repair schedule 和 held-out gate；两者唯一的
方法差异是 `probe_strategy=fixed_grid` 与 `probe_strategy=active`。
B3、B4、B5、Ours 的单次 run 采用相同 LLM 代码生成上限：pilot 为 3 次，paper 为 5 次；
B2 one-shot 始终只有 1 次。生成命令统一带 `--no-analysis`，因此人类报告的额外分析调用
不进入实验预算。
B5 与 Ours 使用相同的 **per-run total** probe 预算；Ours 不能在每个 cycle 重新获得 8 次调用。
B5 在完整参数网格中做确定性的均匀覆盖，
Ours 只改变与当前 violated constraint 相关的参数，因此比较的是 probe 选择策略，
而不是谁调用了更多次仿真。
B5 与 Ours 都将总预算按剩余 repair cycle 分批分配；例如 pilot 的 3 个 cycle 最多形成两批 `4+4`，
paper tier 的 5 个 cycle 最多形成四批 `2+2+2+2`。最后一个 cycle 不再运行无法进入下一轮
生成的 probe。B5 每批从同一个冻结设计中按 offset 取样，因此不会重复首批候选。
两种 selector 的下一轮 prompt 都能看到截至当前的累计 probe 证据；成本统计只累加
`num_new_cases`，不会把历史结果重复算作新的 simulator call。
从第二批 probe 开始，Ours 使用测量分数驱动的 kernel-UCB 采集函数，在高预测收益、
参数不确定性和批内多样性之间取舍；选择计划保存预测分数、不确定性与 selection rank，
便于复核，不由 LLM 随意挑选实验。
参数子空间不再只由 failure reason 文本决定：系统先把 counterexample 中机器可读的
`violated_constraints` 映射到可控参数，再根据已有 probe 的分数差估计参数敏感度，
优先探索实测影响更大的维度。未见过的新 failure reason 只要携带已知 constraint，
仍可选择对应 probe，而不是退回全网格。
Ours 生成的候选还要通过机器可检查的 guarded-adapter 结构契约；仅在 prompt 中声称
“有 guard”但没有读取运行时状态、终止检查、数值分支和真实失败路径的模块会被拒绝。

## 命令

查看完整协议：

```bash
python paper.py list
```

先运行静态协议审计：

```bash
python paper.py audit
```

它自动检查 prompt 信息边界、development/held-out seed 不重叠、held-out 仅用于
最终验收且不会进入 repair、sim/render backend 在每条命令中保持冻结、B5/Ours 整次
run 的 probe 预算相同、B3/B4/B5/Ours 的 LLM 修复上限相同、所有生成从 neutral seed 开始、oracle 单独标记，以及每个 Ours
case 都有 probe backend。审计结果写入 `results/paper/protocol_audit.json` 和 Markdown。

在本地预演整张实验矩阵，不启动 ManiSkill、不调用 LLM：

```bash
python paper.py smoke --tier pilot --plan-name pilot_smoke
```

每个可执行 method/case 都会进入各自的 dry-run 路径。对 B5 和 Ours，smoke 还会展开检查
“初次生成 → development 验证 → 反例选择 → probe → 下一轮修复 → held-out 验收”的内部命令链，
并确认 B5 使用 `fixed_grid`、Ours 使用 `active`，其余 prompt、外层循环和预算保持一致。
预演只构造命令，不启动 ManiSkill、不调用 LLM，也不会伪造 probe 数值或写入正式证据。
任一命令、adapter 隔离、selector 或输出路径出错时，smoke 返回非零状态。它检查的是实验基础设施，
不是任务成功率。

正式运行前先做预检（不调用 LLM）：

```bash
python paper.py preflight
```

该命令检查源端和目标端 ManiSkill 环境能否创建与 reset、action space、adapter
工厂、单步 `env.step` 和 success signal，并写入 `results/paper/preflight.json` 与
`preflight.md`。没有 ManiSkill 的开发机可用 `--static-only` 只检查代码接口。

在目标迁移方法运行前，系统还会用完全相同的 seed 划分验证源端 Panda 任务：

```bash
python paper.py source --tier pilot --plan-name pilot_v1
```

五个 case 会折叠为三个独立源任务，只执行 PullCube/Panda、PickCube/Panda 和 PushCube/Panda。
development 与 held-out 两组都达到 0.8（且至少 5 个 seed）才通过。结果绑定源 Program
和 source adapter 的 SHA；配置不变时自动复用。`paper.py run` 和 `paper.py pilot` 会自动
执行这一步，因此通常无需单独运行该命令。

只生成 pilot 计划，不运行 ManiSkill：

```bash
python paper.py plan --tier pilot
```

先跑最关键的 pilot 对照：

```bash
python paper.py run \
  --tier pilot \
  --methods B0,B1,B2,B3,B4,B5,Ours,Oracle \
  --cases case02_pull_cube_panda_to_xarm6
```

每份 `plan.json/plan.md` 都会自动给出保守资源上界。当前完整 Pilot 在所有生成与修复都
跑满的情况下，最多约为 **715 个 simulator episode、65 次 LLM 调用、80 个 probe case**；
正式矩阵对应上界约为 **6585 episodes、315 次 LLM 调用、240 个 probe case**。上界包括源端检查、每轮初始目标测试和生成后测试。这些是
预算保护值，不是预期消耗；因此先根据 Pilot 的失败率、运行时间和 API 花费决定是否扩到
正式矩阵。按 `--methods` 或 `--cases` 缩小矩阵后会自动重新计算。

远程服务器最简安全入口：

```bash
python paper.py pilot --plan-name pilot_v1
```

它依次执行 protocol audit、真实 ManiSkill preflight、Panda 源端多 seed 门控，再启动完整 pilot。任一检查失败
都会在调用 LLM 前停止；同名命令重跑会自动续跑已完成项目。可用 `--cases` 或
`--methods` 先缩小范围。

当所选方法包含 B2-B5 或 Ours 时，启动门还会检查 LLM provider、model、max tokens、
temperature、DeepSeek thinking 和对应 API key。检查不发送网络请求、不消耗额度，也不会
把 key 写入日志；key 缺失、首尾空格或非 ASCII 字符会在仿真前被拒绝。

正式论文矩阵：

```bash
export EM_TEMPERATURE=0.2
python paper.py run --tier paper
```

`paper` 档含三次 LLM 重复，因此启动门要求 `EM_TEMPERATURE > 0`。汇总表同时报告
`num_unique_adapter_sha256` 和重复代码警告；三次完全相同的 adapter 不会被悄悄解释成
三个独立生成结果。pilot 只有一次重复，不受这个限制。

把同一版本交给远程 GPU 服务器前，可生成不含 `.env` 和 API key 的运行包：

```bash
export EM_LLM_PROVIDER=<provider>
export EM_MODEL=<exact-model-id>
export EM_MAX_TOKENS=<limit>
export EM_TEMPERATURE=<temperature>
# DeepSeek 直连时还需显式设置 EM_DEEPSEEK_THINKING
python paper.py package --tier pilot --plan-name pilot_remote
```

归档和 SHA 清单写入 `results/paper/packages/`，包内含冻结 plan、运行说明、源码与测试，
不含历史结果。服务器解压后只需提供对应 provider 的 API key，再运行：

```bash
python paper.py start
```

打包时必须显式声明上述非秘密配置；系统不会把 provider/model 默认值悄悄写进实验计划。
API key 仍然只在远程 shell 中设置，不进入包。provider、model、token 上限、temperature 和
thinking 模式由 `execute` 从经过 SHA 校验的冻结 plan 自动恢复，不需要在远程重复填写，也不会
修改服务器的全局配置。`start` 完成包、模型配置和 key 检查后，会启动脱离 SSH 会话的
`execute` 后台进程，因此断开终端不会终止 Pilot。`execute` 随后运行 audit、真实 ManiSkill preflight、Panda
源端门控和**包内原始冻结矩阵**，最后自动执行 evidence verification 并生成回传证据压缩包。任何源码、probe
预算、seed 划分或命令参数被修改都会在仿真前拒绝；不会在远程重新生成一份看似相同的 plan。

执行时，每个 run 的输出会实时显示并同步写入 `benchmark_commands.log`；当前 run、完成数量、
失败数量和更新时间会原子写入 `progress.json`。SSH 重连后可直接查看，不会干扰正在运行的进程：

```bash
python paper.py status
```

需要在前台直接观察完整过程时，仍可使用 `python paper.py execute`。

长实验建议固定名称；同一命令中断后再次执行会自动跳过已经完成且配置匹配的 run：

```bash
python paper.py run --tier paper --plan-name paper_v1
```

这里的“配置匹配”不仅检查 run id 和 seed split，还会检查 sim/render backend 与 frozen
LLM provider/model/max tokens/temperature/thinking。任何一项变化，旧结果都不会被静默复用，
从而避免同一汇总表混入不同实验条件。

跑完后自动验收全部证据：

```bash
python paper.py verify --tier paper --plan-name paper_v1
```

该命令检查 seed 完整性、嵌套 CEGIS feedback ledger、held-out 后是否继续修复、B5/Ours
实际 probe 次数是否超出整次 run 预算、各方法实际 LLM 调用是否超出冻结预算、冻结接口、运行环境版本、adapter SHA 和 evidence
bundle；LLM run 的 provider/model/max tokens/temperature/thinking 也必须与 frozen plan
逐项一致。缺少日志会显示为 `missing`，不会被混成算法失败。

若需要对已经存在的完整结果重新导出，仍可单独运行：

```bash
python paper.py export --tier pilot --plan-name pilot_v1
```

`execute` 成功时已经自动完成这一步；这里的独立命令用于补导出。只有完整且有效的计划才能导出。归档包含 `results/paper/<plan>` 和对应的
`evidence/paper_runs`，并附文件 SHA 清单；本地在仓库根目录解压后再次运行同一条
`paper.py verify` 即可复核，省去人工挑选日志。

只有显式加入 `--rerun-completed` 才会重跑已有结果。每个 case 的可变 adapter
带文件锁，避免两个实验进程同时覆盖同一模块。

正式矩阵会很耗时和 API 费用，应先确认 pilot 的日志、模型配置和证据归档正确。
当前 120 项正式计划中 111 项可直接执行；两个 PickCube case 和一个 PushCube case
的 9 次 Oracle 重复被标记为 blocked，因为尚未注册人工上界。这个 blocked 状态不会
被混入方法失败率。

五个 case 中，Case 01、03、04、05 是 ManiSkill 文档声明支持的机器人组合；Case 02
（PullCube Panda -> xarm6）明确标为 `stress_test_override`。它用于检验跨出任务官方
支持集合后的迁移能力，论文主结果应同时报告“官方支持子集”和“压力测试”结果。

Case 05（PushCube Panda -> Fetch）用于检验接触侧选择是否能从 PullCube 泛化到
方向相反的推任务。当前已接入 neutral seed、B0/B1、LLM baselines、active probe、
online observation 和 held-out 验证协议；尚未在远程 GPU 上产生可追溯成功证据。

## 输出

完整本地日志：

```text
results/paper/<plan>/
  plan.json
  plan.md
  commands.sh
  summary.json
  summary.md
  summary.csv
  summary_support_subgroups.csv
  paired_method_comparisons.csv
  summary.tex
  source_baselines/
    summary.json
    summary.md
    pull_cube/{development,held_out}.{jsonl,md}
    pick_cube/{development,held_out}.{jsonl,md}
    push_cube/{development,held_out}.{jsonl,md}
  runs/<run_id>/
```

可提交的精简证据：

```text
evidence/paper_runs/<run_id>/
  manifest.json
  final_adapter.py
  development_jsonl.jsonl
  held_out_jsonl.jsonl
  module_generation_jsonl.jsonl
  commands_log.log
```

一次 LLM 方法只有在确实产生并保留了 LLM adapter，且 held-out 成功率达到阈值时，
才会被记为成功。Oracle 永远单独标记为 `hand_written_oracle`。

每个 run 还保存 `frozen_interface_integrity`：执行前后比较高层 Program、neutral
seed、controller bridge、任务定义、success evaluation 和环境适配层的 SHA。任一
冻结接口发生变化，本次实验自动判为无效，即使仿真 success signal 为真也不计成功。
运行记录同时保存 ManiSkill、Gymnasium、SAPIEN 和 NumPy 版本。

汇总表同时记录 simulator calls、environment steps、probe cases、LLM API calls、
实际 prompt/completion tokens、wall-clock time、adapter 行数和状态分支数。B0、B1 和 Oracle 的成功
不会被证据账本计为 LLM 迁移成功。
Ours 还记录 CEGIS cycle 数、不同 counterexample reason/seed 数、active-probe 批次
和 kernel-UCB 批次；这些指标用于比较“是否更快找到有效修复”，而不仅是最终成功率。
统计结果同时给出跨重复的 mean/std、合并 held-out seed 成功率和 Wilson 95% 置信区间。
`summary_support_subgroups.csv` 另外把 `official_supported` 与
`stress_test_override` 分开汇总，避免压力测试结果改变官方支持子集的主结论。
`paired_method_comparisons.csv` 把 Ours 与每个 baseline 按 case、repetition、held-out seed
一一配对，报告成功率差、exact McNemar p-value 和 case-level Holm 多重比较校正；同时按
case、独立 LLM repetition 和 seed 做分层 bootstrap，避免把同一个 seed 在三次生成中的观测
简单当成完全独立样本；
official、stress-test 和全体 pooled 行作为描述性聚合单列。缺失的 run 或 seed不会被当作失败补齐。

## 诊断器独立评测

五层诊断不能只用自己生成的标签证明正确。先从真实失败日志生成不含系统预测的盲标文件：

```bash
python paper.py diagnosis-template \
  --trial-jsonl results/paper/pilot_v1/runs/<run_id>/held_out.jsonl
```

由独立标注者填写 `label_layer` 和可选的 `label_reason`，再运行：

```bash
python paper.py diagnosis-eval
```

输出 `diagnosis_evaluation.json` 和 `diagnosis_evaluation.md`，报告标注覆盖率、layer/reason
accuracy 与混淆矩阵。`message`、execution log、runtime diagnostics 和 `final_info` 会保留为
物理证据，但系统原有 `failure_layer`、预测结果和 repair hint 不进入盲标模板。
若有第二位独立标注者，可另存一份相同 `trial_id` 的文件并运行
`python paper.py diagnosis-eval --annotations-secondary <第二份.jsonl>`；报告会同时给出
layer/reason 的 Cohen's kappa。

## 自动生成 Motivating Examples

论文中的典型失败不能从聊天记录或人工抄写的表格恢复。在 GPU 远程机上一条命令采集两份
所需证据：

```bash
python paper.py evidence
```

它顺序运行 PullCube 的 `0-9` 多 seed 验证和 PickCube 的 32 组结构化 probe，保存完整日志，
随后自动生成 motivating examples。这个入口不调用 LLM，也不修改 adapter。在无 ManiSkill 的
本地机器上，可先查看将要执行的命令：

```bash
python paper.py evidence --static-only
```

若证据已经从远程同步到默认路径，只重新汇总而不运行仿真：

```bash
python paper.py motivation
```

默认读取：

```text
results/pullcube_xarm6_multiseed.jsonl
results/structured_probes/case03_pick_cube_panda_to_xarm6/
  pick_cube_xarm6_close_envelope.json
```

输出 `results/paper/motivating_examples/` 下的 JSON 和 Markdown，并记录每个输入文件的
SHA256。若文件不存在、只有 dry-run，或 PullCube 不足五个真实 seed，命令返回非零状态，
报告明确标记为 not paper ready，不生成貌似完整的结论。
