# 新同学上手文档：Embodied Migration

这份文档是给新加入项目的同学看的。目标是让你不用先读完所有代码，也能在第一天理解项目、跑通最小实验，并知道接下来可以推进什么。

## 1. 项目一句话

本项目研究：

> 同一段高层机器人程序，换成不同机器人后，能不能让 LLM 自动生成目标机器人侧的 adapter，并通过真实 ManiSkill 仿真反馈持续修复，直到任务成功？

高层程序尽量不变，例如：

```python
ret_val = robot.pull(cube, goal)
```

或：

```python
grasp_ok = robot.grasp(cube)
ret_val = robot.place(cube, goal) if grasp_ok else False
```

LLM 主要生成的是 target-side adapter。adapter 负责把 `robot.pull(...)` / `robot.grasp(...)` 翻译成目标机器人的 `env.step(action)` 序列。

## 2. 当前已有成果

目前已经注册的核心迁移 case：

| Case | 任务 | 源机器人 | 目标机器人 | 当前状态 |
|---|---|---|---|---|
| case01 | PullCube-v1 | Panda | Fetch | 当前提交为 hand-written oracle；LLM 成功证据待补 |
| case02 | PullCube-v1 | Panda | xarm6_robotiq | 当前提交为 neutral seed；主 CEGIS 实验 case |
| case03 | PickCube-v1 | Panda | xarm6_robotiq | 当前提交为 neutral seed；grasp/contact hard case |

远程历史运行曾报告 seed-0 成功，但对应原始日志和 adapter snapshot 未提交，因此不能作为当前仓库的可复现结论。以 `docs/EVIDENCE_LEDGER_CN.md` 为准。

当前最重要的研究进展不是“某个 seed 成功”，而是已经形成了一个迁移闭环：

```text
高层程序不变
-> LLM 生成 target adapter
-> ManiSkill env.step(action) 真实执行
-> development seeds 找出反例
-> embodiment contract + 五层诊断
-> active structured probe
-> guarded adapter 修复
-> held-out seeds 最终验证
```

## 3. 你第一天应该先看哪些文件

按这个顺序看就够了：

1. `README.md`：项目总览、主要命令和当前结果。
2. `docs/PROJECT_STRUCTURE_CN.md`：目录结构，哪些文件重要。
3. `docs/METHOD_CEGIS_ADAPTER_SYNTHESIS_CN.md`：当前统一方法。
4. `docs/EXPERIMENT_PROTOCOL_CN.md`：seed 划分、baseline 和证据要求。
5. `docs/EVIDENCE_LEDGER_CN.md`：哪些结论在仓库中真正可复现。
6. `docs/HARNESS_ENGINEERING_CN.md`：harness 是什么，Agent 怎么和仿真交互。
7. `maniskill_backend/case_programs/case01_pull_cube.py`：高层任务程序。
8. `maniskill_backend/generated_adapters/case02_xarm6_pull_cube.py`：目标 adapter 起点。
9. `migrate.py`：统一入口。

## 4. 环境准备

推荐在远程有 GPU / ManiSkill 的机器上跑真实仿真。本地电脑可以做 dry-run、读代码、写文档。

```bash
git clone https://github.com/hexizou-730/embodied_migration.git
cd embodied_migration
conda activate em-ms
```

如果是新环境：

```bash
conda create -n em-ms python=3.10 -y
conda activate em-ms
pip install -r requirements.txt
pip install -r requirements-maniskill.txt
pip install "numpy>=1.24,<2" --force-reinstall
```

检查 ManiSkill 能不能跑：

```bash
python -m maniskill_backend.sim_check \
  --env PullCube-v1 \
  --robot xarm6_robotiq \
  --control-mode pd_ee_delta_pos
```

## 5. LLM API 配置

项目支持 OpenRouter 或 DeepSeek 直连。

DeepSeek 直连：

```text
EM_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key
EM_MODEL=deepseek-v4-pro
EM_MAX_TOKENS=8192
EM_DEEPSEEK_THINKING=disabled
```

OpenRouter：

```text
EM_LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
EM_MODEL=deepseek/deepseek-chat-v3.1
EM_MAX_TOKENS=8192
```

不要把 `.env` 提交到 GitHub。

## 6. 最小命令

列出当前注册的迁移 case：

```bash
python migrate.py --list-cases
```

不跑仿真，只检查命令结构：

```bash
python migrate.py --task pull_cube --source panda --target xarm6 --dry-run
```

验证当前 adapter，一次 seed：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode evaluate \
  --seed 0
```

从零让 LLM 生成/修复 adapter：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode agent \
  --max-cycles 5
```

在线 harness：同一个 episode 中边观察边执行：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode online \
  --max-online-steps 240
```

PullCube 多 seed 泛化评估：

```bash
python scripts/pullcube_multiseed_eval.py \
  --seeds 0-9 \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

## 7. 输出在哪里

重要输出默认都在 `results/`，但 `results/` 默认不进 git。

常见路径：

```text
results/migrations/
results/auto_runs/
results/module_generation_trials.jsonl
results/module_generation_trials.md
results/generated_modules/
results/structured_probes/
results/online_harness/
```

如果某次实验很重要，不要只留在 `results/`。请把结论整理成 `docs/` 下的 Markdown 文档，再提交。

## 8. 当前代码结构

| 路径 | 作用 |
|---|---|
| `migrate.py` | 用户级统一入口 |
| `auto.py` | PullCube 自动闭环短入口 |
| `maniskill_backend/tasks.py` | 任务定义 |
| `maniskill_backend/cases.py` | 迁移 case 注册 |
| `maniskill_backend/skill_adapter.py` | 基础技能实现 |
| `maniskill_backend/generated_adapters/` | 当前目标 adapter |
| `maniskill_backend/seed_adapters/` | 从零迁移的起始模板 |
| `maniskill_backend/module_generation_runner.py` | LLM 生成 adapter 的核心脚本 |
| `maniskill_backend/autonomous_harness.py` | episode-level Agent observation |
| `maniskill_backend/online_harness.py` | online observe-decide-act harness |
| `maniskill_backend/structured_probe.py` | probe 参数网格和反馈 |
| `scripts/` | 实验脚本 |
| `docs/` | 当前文档和汇报材料 |

## 9. 现在最需要推进的目标

### 目标 A：复现实验结果

请先在远程机器上复现三个 seed-0 case：

```bash
python migrate.py --task pull_cube --source panda --target xarm6 --mode evaluate --seed 0
python migrate.py --task pick_cube --source panda --target xarm6 --mode evaluate --seed 0
python migrate.py --task pull_cube --source panda --target fetch --mode evaluate --seed 0
```

把结果整理到一个新文件：

```text
docs/REPRODUCTION_LOG_CN.md
```

记录字段：

```text
date
machine
git commit
GPU
case
seed
success
elapsed_steps
message
```

### 目标 B：多 seed 泛化

单 seed 只能作为 smoke test。论文结论必须使用 development/held-out 分离的多 seed 结果。

优先跑：

```bash
python migrate.py --task pull_cube --source panda --target xarm6 --mode cegis
```

要回答：

```text
哪些 seed 成功？
哪些 seed 失败？
失败集中在 reachability、contact_geometry，还是 controller primitive？
提高 max_episode_steps 是否真的有帮助？
```

### 目标 C：把 online harness 做成真正实时闭环

当前 online harness 已经能在 PullCube 内部循环：

```text
observe -> primitive -> env.step segment -> observe again
```

下一步可以做：

1. 给 online trace 增加更清楚的 stage 成功/失败诊断。
2. 支持 `--online-planner llm` 的真实远程运行。
3. 把 PickCube 也接入 online primitives，例如 `align_xy`, `descend`, `close`, `lift`, `place`。

### 目标 D：新增迁移 case

以后加新任务/机器人，优先走这个流程：

1. 在 `tasks.py` 里确认任务定义。
2. 在 `cases.py` 注册新的 source-target case。
3. 添加 `case_programs/` 高层程序。
4. 添加 `seed_adapters/` 初始目标 adapter。
5. 跑：

```bash
python migrate.py --task <task> --source panda --target <robot> --mode agent
```

## 10. 代码修改规则

请尽量遵守：

1. 不改 ManiSkill simulator。
2. 不改 controller 的成功信号。
3. 不直接 teleport cube/goal/robot。
4. 高层程序尽量保持稳定。
5. 主要改 target-side adapter、harness、probe、诊断器。
6. 每次重要实验后写简短 Markdown 记录，不要只截图。

## 11. 推荐 Git 流程

每次开始：

```bash
git pull
git status --short --branch
```

新功能建议开分支：

```bash
git switch -c codex/your-feature-name
```

提交前：

```bash
python -m unittest tests.test_real_backend -q
git status --short
git diff --stat
```

不要提交：

```text
.env
__pycache__/
results/ 下的大量临时文件
```

可以提交：

```text
maniskill_backend/
scripts/
demos/
docs/ 下整理好的结论文档
小体积的展示图或 Word/PDF
```

## 12. 给老师汇报时的核心说法

可以这样概括：

> 这个项目研究 cross-embodiment robot code migration。高层程序、底层 controller、simulator 和 success signal 固定，只生成目标 adapter。系统在 development seeds 上找物理反例，把失败映射为违反的 embodiment constraints，再主动选择 probe 并生成带状态 guard 的修复代码，最后只在 held-out seeds 上评估。Fetch oracle 与 LLM 结果严格分开，所有成功结论必须绑定 adapter SHA 和原始仿真日志。
