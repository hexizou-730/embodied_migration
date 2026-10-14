# Embodied Migration

面向 ManiSkill 的机器人代码迁移：输入一份源机器人动作程序和目标机器人 UID，由 LLM 生成目标 Adapter，在仿真中执行，并根据实际失败信息进行有限轮次的修复。

不修改机器人模型、底层控制器、仿真器或官方成功标准。迁移入口不要求预先注册“源机器人 → 目标机器人”组合，但目标机器人仍须能在所选环境中初始化，并具备完成任务所需的能力。

## 当前进度

截至 2026-10-13，框架已经完成一次可复核的端到端验收：冻结的 Panda
`PushCube-v1` 程序迁移到 `xarm6_robotiq`，LLM 生成的 Adapter 在修复后的
Harness 中执行 294 个动作，返回成功，且 ManiSkill 官方 `success=True`。

| 内容 | 状态 |
|---|---|
| 环境配置、接口读取、生成/执行/诊断/修复流程 | 已实现 |
| 20 个任务目录与 20 份候选源程序 | 已建立，候选不等于成功 |
| 冻结源程序 | `t01_push_cube` 已同步，Panda seeds 0–9 为 10/10；其余候选尚未纳入当前证据 |
| 目标迁移 | Panda → xArm6、PushCube、seed 0 已完成一次严格回放验收 |
| 固定 Adapter 留出评估 | xArm6 seeds 100–109 为 5/10；LLM 调用 0、代码修复 0 |
| 从零生成稳定性 pilot | 相同配置独立运行 3 次，2 次官方成功；总成本约 $0.0933 |
| 六档难度 | 配置已定义；难度扰动执行器及正式评估待完成 |

当前只把 PushCube 作为已验收结果。证据见
[evidence/t01_push_cube_panda_to_xarm6_20261013](evidence/t01_push_cube_panda_to_xarm6_20261013)。
一次成功不代表生成过程稳定：随后一次全新 8 轮生成共执行 122 个目标动作，
仍因未建立有效接触而失败。固定成功 Adapter 在未见 seeds 100–109 上取得
5/10，说明当前策略可以泛化到部分初始状态，但鲁棒性仍不足。三次独立从零
生成有两次成功，说明生成闭环可以重复成功，但 3 次仍只是 pilot 样本量。

完整的进度、证据边界和下一步见 [PROGRESS.md](PROGRESS.md)。旧实验不作为当前成果引用。

## 三个入口

| 命令 | 用途 |
|---|---|
| `source.py` | 准备源程序、多 seed 验证、冻结及复验 |
| `target.py` | 只使用有效冻结源程序，批量迁移到一个目标机器人 |
| `migrate.py` | 输入自己的源代码文件，进行单任务迁移 |
| `accept.py` | 运行一个真实迁移并自动检查端到端证据 |
| `evaluate.py` | 固定成功 Adapter，在未见 seed 上评估，不调用 LLM 或修复代码 |
| `stability.py` | 多次从零生成 Adapter，统计生成成功率、动作、token 和费用 |
| `benchmark.py` | 在同一任务和预算下比较四种生成/修复方法 |

单文件迁移，不需要在任务目录中注册：

```bash
python migrate.py examples/panda_pull.py xarm6
```

只查看计划，不调用 LLM 或仿真：

```bash
python migrate.py examples/panda_pull.py xarm6 --dry-run
```

真实迁移会调用付费 API 和仿真；`--dry-run` 不代表源端或目标端成功。

## 接着上次的实验做

在已有 ManiSkill 环境的远程服务器上运行，先检查再启动，不要直接重跑全部 20 个任务：

```bash
cd ~/Embodied/embodied_migration
conda activate em-ms
git status --short
git pull --ff-only
python source.py status --run-name phase4_refresh
```

有未提交的更改时先确认并保留，不用强制覆盖。检查输出中的 `Currently ready` 和各任务状态：

- `ready`：冻结证据有效，可以迁移。
- `needs_revalidation`：依赖或配置与旧证据不一致，不等于过去的仿真失败。
- `not_frozen` / `record_missing`：当前机器缺少冻结代码或记录。先找回服务器上的文件，不要手工把状态改成 True。

若没有仍在运行的复验，且已有冻结文件仅需更新验证：

```bash
python source.py refresh --tasks t01_push_cube
```

此命令前台运行，只复验原冻结代码，不调用 LLM、不修改动作策略。确认 `t01_push_cube` 为 `ready` 后，先跑一个目标迁移：

```bash
python accept.py xarm6 --task t01_push_cube
```

验收只有在目标代码经过至少一轮生成、执行了真实 action、官方成功为
True，且 Adapter、运行环境、manifest 和哈希证据齐全时才会输出 `PASS`。
失败也会生成 `acceptance_report.json`，列出缺失条件。默认使用 seed 0、
最多 5 轮目标代码生成；通过后再扩展到其余四个已验证源任务。批量迁移
仍使用 `target.py ... --background`。更多源端操作见
[source_programs/README.md](source_programs/README.md)。

固定 Adapter 的第六步评估使用：

```bash
python evaluate.py
```

默认运行 held-out seeds 100–109，输出写入 `results/adapter_evaluation/`。
该命令校验 Adapter 哈希并记录官方结果，但不会调用 LLM，也不会根据测试失败
修改代码。可用 `--seeds development` 做开发集回放，正式泛化结果使用默认值。

第七步生成稳定性实验使用：

```bash
python stability.py run xarm6 --background
python stability.py status --run-name <命令输出的名称>
```

默认只运行 3 次独立生成，每次最多 8 轮；各次之间不共享 Adapter 或 incumbent。

第八步方法对照实验使用：

```bash
python benchmark.py run xarm6 --task t01_push_cube --repetitions 1 --background
python benchmark.py status --run-name <命令输出的名称>
```

默认依次比较 `one_shot`、`raw_log`、`diagnosis` 和 `ours`。四种方法共用冻结源程序、
模型、seed、环境步数和修复轮数上限，只改变反馈内容与候选管理策略。先用每种方法
1 次检查流程；正式结果再提高重复次数，不能把单次 pilot 当作稳定结论。单次 LLM
请求固定为 180 秒超时且不做隐藏重试，避免供应商请求无限拖住整组实验。

## 运行机制

```text
源代码 + 目标机器人
→ 校验实验环境与源端成功
→ 读取目标 action 布局、TCP 和场景状态
→ LLM 生成目标 Adapter
→ Adapter 读取状态并执行 env.step(action)
→ 官方 evaluate() + 执行轨迹 + 异常
→ 候选按物理结果评分，只有优于 incumbent 才替换当前最佳 Adapter
→ 失败定位到局部修复范围，再生成并验证，直到成功或预算耗尽
```

TCP 是工具中心点，即用于描述末端位置的参考点。Adapter 在动作循环内读取状态并决定下一步动作；LLM 在外层根据一次执行结果修改代码，不是每个仿真步都调用一次 LLM。

目标修复采用保守接受规则：官方成功、真实动作、约束安全和失败阶段优先，
动作数仅用于同级候选的成本比较。接触、越界、预算耗尽和代码异常分别映射到
不同修复范围；非接口故障不得修改公共技能签名或控制器/action 映射。每轮记录
`accepted_as_incumbent`、候选评分和修复范围，失败候选不会覆盖较好的历史版本。

源文件需声明 `ENV_ID`、`SOURCE_ROBOT`、`CONTROL_MODE`，并提供 `run(env)`；也支持 `TASK_PROGRAM` 加 `build_robot(...)`。程序返回 True 不足以算成功，还必须有实际仿真动作和官方成功信号。静态检查及超时限制不等同于操作系统安全沙箱。

## 环境与数据

[experiment_config.json](experiment_config.json) 固定关键版本、LLM 请求参数、控制模式及 seed 规则。每次实际运行记录 `runtime_environment.json`，不匹配则停止。新服务器可用 `conda env create -f environment.yml` 安装依赖；GPU 驱动和任务资产需另外准备。

- 固定版本：Python 3.10.20、ManiSkill 3.0.1、SAPIEN 3.0.3；要求驱动报告 CUDA API 13.2。
- API 配置见 [.env.example](.env.example)，密钥只放本机环境或 `.env`，不要提交 Git。
- 开发/修复 seeds：0–9。留出集 100–109 已完成一次固定 Adapter 评估，结果为 5/10；
  留出结果未反馈给 LLM，也未用于修改该 Adapter。
- `results/` 保存执行记录、生成代码、prompt 和成本；默认不提交。`git push` 不会自动同步服务器实验。
- `source_programs/frozen/` 保存冻结代码及校验记录；当前已跟踪 `t01_push_cube`。

## 文件导航

| 位置 | 作用 |
|---|---|
| `maniskill_backend/dynamic_harness.py` | 单任务生成、执行、反馈修复 |
| `maniskill_backend/dynamic_adapter.py` / `env_adapter.py` | 通用状态/动作接口与 ManiSkill 封装 |
| `maniskill_backend/source_programs.py` / `source_preparation.py` | 源程序验证、冻结与批量准备 |
| `maniskill_backend/target_preparation.py` | 目标接口检查、迁移队列和结果汇总 |
| `llm_client.py` / `maniskill_backend/llm.py` | 模型调用、返回文本及用量 |
| `lmp/` / `maniskill_backend/code_validation.py` | 代码提取、执行及静态检查 |
| `benchmark_tasks/` | 20 个任务和六档难度的定义，非 120 个已验证实验 |
| `source_programs/candidates/` / `vendor/` | 候选源程序及所需参考规划器，保留上游许可 |
| `tests/` | 不依赖真实机器人成功的框架回归测试 |

安装基础依赖后可运行本地检查，不消耗 LLM 额度：

```bash
python -m unittest discover -s tests -v
python -m maniskill_backend.task_catalog
```

测试中的模拟环境与模拟 API 返回不能当作真实 ManiSkill 实验结果。
