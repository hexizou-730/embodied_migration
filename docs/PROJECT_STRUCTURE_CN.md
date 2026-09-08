# 项目结构说明

这份说明用于快速判断：哪些文件是主线，哪些文件只是实验记录或展示材料。

## 一句话主线

本项目研究：高层机器人程序不变时，LLM 能否为目标机器人生成新的 `adapter`，让任务在真实 ManiSkill 仿真中执行成功。

```text
高层程序 -> target adapter -> development seeds -> counterexample/probe
-> guarded repair -> held-out evaluation
```

## Adapter 是什么

通俗说，adapter 是“翻译器 + 执行方案”。

高层程序只表达任务意图：

```python
ret_val = robot.pull(cube, goal)
```

这句话只说明“我要把方块拉到目标区域”，但没有说明：

```text
机械臂先往哪走？
夹爪开还是关？
每一步 action 怎么填？
接触点选哪里？
失败后怎么处理？
```

这些都由 adapter 决定。

```text
高层程序 = 我要做什么
adapter = 这个机器人具体怎么做
controller = 底层怎么把 action 变成关节运动
```

本项目主要让 LLM 生成的是 adapter，而不是直接修改 ManiSkill 底层 controller。这样可以保持高层任务代码稳定，同时比较不同机器人在同一任务下需要怎样的执行策略。

## 当前主线

| 内容 | 当前选择 |
|---|---|
| Source robot | `panda` |
| Main target robot | `xarm6_robotiq` |
| Main positive task | `PullCube-v1` |
| Grasp/contact task | `PickCube-v1` |
| Secondary target | `fetch` |
| 当前主要方法 | 反例驱动的 target adapter synthesis |

## 目录怎么读

### 0. 最短入口

未注册任务可以直接给 ManiSkill 环境名，不再先修改 `cases.py`：

```bash
python migrate.py \
  --env-id TurnFaucet-v1 \
  --source panda \
  --target fetch \
  --mode agent
```

运行时会自动生成 `case_manifest.json`，发现源端和目标端动作空间，让 LLM
先写源程序和源 Adapter。源端通过官方成功判定后冻结程序，再循环生成和修复
目标 Adapter。这里的“未注册”不等于环境可以不存在，任务和机器人仍需已经安装
在 ManiSkill 中。

动态运行目录会保存两端 observation、每轮 prompt、候选代码、仿真结果和 SHA-256。
因此探索成功后可以准确复现“哪份输入产生了哪份 Adapter”，再把该组合注册为正式
benchmark case，而不是直接把一次临时成功写进论文主结果。

```text
注册 case：用于固定论文实验和复现
动态 manifest：用于新任务的自动发现和迁移
```

如果想按“任务 + 源机器人 + 目标机器人”的形式发起迁移请求，用：

```bash
python migrate.py --task pull_cube --source panda --target xarm6_robotiq
```

这是当前 adapter 单次验证入口：

```text
PullCube-v1 + Panda -> xArm6
```

不跑仿真时可以先检查：

```bash
python migrate.py --task PullCube-v1 --source panda --target xarm6 --dry-run
```

如果要从零开始运行当前主方法，用：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode cegis
```

这个命令会先恢复 neutral seed adapter，然后循环执行：

```text
LLM 生成 adapter
-> development seeds
-> 失败反例和违反约束
-> active probe
-> guarded adapter 修复
-> held-out seeds
```

如果要展示“边做边看边改”的在线 harness，用：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode online
```

它不是等整局结束后才看失败，而是在同一个 episode 里循环：

```text
观察 TCP / cube / goal
-> 选择一个安全 primitive
-> 执行几步 env.step(action)
-> 再观察
-> 再决定下一步
```

`PickCube` 使用同一条命令，只替换任务名：

```bash
python migrate.py \
  --task pick_cube \
  --source panda \
  --target xarm6_robotiq \
  --mode online
```

当前 online harness 已支持 `PullCube`、`PickCube` 和 `PushCube` 的任务专用 observation；
Pull/Push 共用目标方向驱动的短时接触 primitive，Pick 使用抓取专用 primitive。
默认由 LLM 在每个短动作段前选择下一步；`--online-planner fallback` 用于无
API 的确定性调试。当前观察来自结构化仿真状态，RGB/多模态视觉尚未接入。

如果只想用旧的 PullCube 专用自动实验闭环，用根目录短命令：

```bash
python auto.py pull
```

它会自动串起：

```text
Agent observation -> LLM planner 选择工具 -> harness 执行 -> 新 observation
```

底层脚本还在，但多数时候不用手动记。

### 1. `maniskill_backend/`

核心代码都在这里。

| 文件/目录 | 作用 |
|---|---|
| `tasks.py` | 定义任务，例如 `pull_cube`, `pick_cube` |
| `cases.py` | 定义迁移实验，例如 Panda -> xArm6 |
| `case_programs/` | 高层 LMP 程序，基本不变 |
| `skill_adapter.py` | Panda/source 默认技能实现，也是目标 adapter 继承的基础 |
| `generated_adapters/` | LLM 或人工生成的目标机器人 adapter |
| `seed_adapters/` | 从零迁移前恢复用的 neutral seed adapter |
| `embodiment_contracts.py` | action layout、机器人能力边界和冻结接口 |
| `counterexamples.py` | 把失败 trial 转成可排序的物理反例 |
| `active_probe.py` | 根据违反约束选择少量 probe 参数 |
| `counterexample_loop.py` | development/held-out 分离的 CEGIS 主循环 |
| `autonomous_harness.py` | 把仿真结果整理成 Agent observation / human report |
| `online_harness.py` | 在线 observe-decide-act harness：边执行边读状态边选下一段动作 |
| `structured_probe.py` | 定义 structured probe 的参数网格、打分和反馈格式 |
| `generalization.py` | 多 seed 成功率和失败聚类策略选择 |
| `module_generation_runner.py` | LLM 生成 adapter 的主入口 |
| `real_runner.py` | 真正创建 ManiSkill 环境并执行 `env.step(action)` |
| `env_adapter.py` | ManiSkill 环境包装 |
| `llm.py` | 调用 LLM 的薄封装 |

### 2. `maniskill_backend/case_programs/`

这是“要做什么”的高层代码。

| 文件 | 任务 |
|---|---|
| `case01_pull_cube.py` | `robot.pull(cube, goal)` |
| `case03_pick_cube.py` | `robot.grasp(cube)` 后 `robot.place(cube, goal)` |

这些文件应该尽量保持简单稳定。迁移重点不是改这里。

### 3. `maniskill_backend/generated_adapters/`

这是“目标机器人具体怎么做”的代码。

| 文件 | 当前用途 |
|---|---|
| `case02_xarm6_pull_cube.py` | xArm6 PullCube 当前 neutral seed 起点 |
| `case02_xarm6_pull_cube_adaptive.py` | 多 seed 自适应 PullCube 实验 adapter |
| `case03_xarm6_pick_cube.py` | xArm6 PickCube 当前 neutral seed 起点 |
| `case01_fetch_pull_cube.py` | Fetch hand-written oracle 上界 |

当前证据来源以 `docs/EVIDENCE_LEDGER_CN.md` 为准。

### 3.1 `maniskill_backend/seed_adapters/`

这是“从零迁移”的起点模板。agent 模式默认会先把对应 seed adapter 复制回 `generated_adapters/`，再让 LLM 重新生成目标 adapter。

这样做的意义是：

```text
不是从已有答案继续改
而是从一个固定、可复现、未解决目标问题的初始 adapter 开始迁移
```

### 4. `scripts/`

这是底层实验辅助脚本。一般先用 `python auto.py pull`，只有调试某一步时才直接运行这些脚本。

| 文件 | 作用 |
|---|---|
| `autonomous_loop_runner.py` | 自动闭环主流程，被 `auto.py pull` 调用 |
| `cegis_migration_runner.py` | 当前统一反例驱动迁移入口 |
| `multiseed_eval.py` | 通用任务多 seed 评估入口 |
| `autonomous_harness_runner.py` | 只生成 agent observation / human report，不执行完整闭环 |
| `online_harness_runner.py` | 在线 harness 调试入口，会输出 online trace |
| `pullcube_multiseed_eval.py` | PullCube 多 seed 成功率评估 |
| `structured_probe_runner.py` | 通用 structured probe 入口 |
| `xarm6_pull_contact_probe.py` | xArm6 PullCube 接触参数探针 |
| `xarm6_pull_diagnostics.py` | xArm6 PullCube 接触行为诊断 |
| `xarm6_pick_grasp_probe.py` | xArm6 PickCube 抓取参数探针 |

### 5. `demos/`

这是展示用的最小 demo，不替代正式实验脚本。

| 文件/目录 | 作用 |
|---|---|
| `demos/simple_harness/demo.py` | 最小 harness demo：生成 observation、plan、命令和 result |
| `demos/simple_harness/sample_outputs/` | 已提交的 dry-run 示例输出 |

推荐现场展示：

```bash
python demos/simple_harness/demo.py
python demos/simple_harness/demo.py --run
```

它展示的是：

```text
agent_observation.json -> agent_plan.json -> selected simulator tool -> tool_result.json
```

### 6. `docs/`

这是当前还在使用的说明和汇报材料。历史报告、旧 PPT/Word、安装备忘录已经移到 `archive/`。

| 文件 | 作用 |
|---|---|
| `PROJECT_STRUCTURE_CN.md` | 当前项目结构说明 |
| `METHOD_CEGIS_ADAPTER_SYNTHESIS_CN.md` | 当前方法和算法闭环 |
| `EXPERIMENT_PROTOCOL_CN.md` | development/held-out、baseline 和指标 |
| `EVIDENCE_LEDGER_CN.md` | 当前 Git 仓库可追溯的实验结论 |
| `HARNESS_ENGINEERING_CN.md` | harness engineering 中文解释 |
| `COLLABORATOR_ONBOARDING_CN.md` | 新同学上手文档、复现实验命令和下一阶段目标 |
| `LITERATURE_REVIEW_CN.md` | 文献综述和论文定位 |
| `GROUP_MEETING_UPDATE_2026_07_08_CN.md` | 组会前两周进度速览 |
| `GROUP_MEETING_PPT_ADDENDUM_2026_07_08_CN.md` | 可复制到 PPT 的补充页文案 |
| `LLM_ROBOT_MIGRATION_SPEAKER_SCRIPT_CN.md` | 10-15 分钟汇报演讲稿 |

### 7. `results/`

实验输出目录。这里的文件默认不进 git。

常见输出：

```text
results/auto_runs/
results/module_generation_trials.jsonl
results/module_generation_trials.md
results/generated_modules/
results/pullcube_xarm6_multiseed.md
results/xarm6_pick_grasp_probe.md
```

现在推荐优先看自动闭环输出：

```text
results/auto_runs/<run_name>/
  summary.md                 # 一次自动实验的总览
  commands.log               # 自动执行过的命令
  cycle_01/multiseed.md       # 多 seed 结果
  cycle_01/harness/...        # agent_observation / human_report
  cycle_01/structured_probe/  # probe 表格
  cycle_01/module_generation.md
```

如果要保留某次重要实验，建议把关键结论整理进 `docs/` 下的新 Markdown 文件，不要直接依赖 `results/` 里的临时文件。

### 8. `archive/`

这是历史材料归档区，不作为当前开发主线。

| 文件/目录 | 作用 |
|---|---|
| `archive/legacy_docs/` | 旧实验报告、旧运行命令、旧 setup 文档、旧 workshop framing |
| `archive/setup_scripts/` | 旧环境配置脚本 |
| `archive/untracked_materials/` | 本地未跟踪材料：旧 PPT/Word、patch、临时文件、历史 results |

`archive/untracked_materials/` 被 git 忽略，只作为本地整理区。

## 当前推荐展示顺序

1. `README.md`：项目一句话和当前结果。
2. `auto.py`：一条命令自动实验入口。
3. `maniskill_backend/case_programs/case01_pull_cube.py`：高层程序没有变。
4. `maniskill_backend/generated_adapters/case02_xarm6_pull_cube.py`：xArm6 adapter。
5. `maniskill_backend/autonomous_harness.py`：Agent observation / human report 分离。
6. `demos/simple_harness/`：最小 harness demo。
7. `results/auto_runs/<run_name>/summary.md`：自动实验结果。
8. `docs/GROUP_MEETING_UPDATE_2026_07_08_CN.md`：人工汇报总结。

## 可以暂时忽略的内容

| 内容 | 说明 |
|---|---|
| `__pycache__/` | Python 缓存，不看 |
| `.DS_Store` | macOS 缓存，不看 |
| `results/docx_render_*` | Word 渲染检查图片，不是实验核心 |
| `capabilities/` | 机器人 profile 辅助，目前不是汇报重点 |
| `lmp/` | 高层代码执行器，稳定基础设施 |
| `archive/` | 历史材料，不是当前开发入口 |

## 后续整理建议

当前已经按这个规则整理：

```text
核心代码保留在 maniskill_backend/
实验脚本保留在 scripts/
当前说明保留在 docs/
历史材料保留在 archive/
临时输出留在 results/
```

后续继续开发时，只需要优先维护：

1. `migrate.py`：用户级入口。
2. `auto.py`：自动闭环入口。
3. `maniskill_backend/`：核心框架。
4. `scripts/`：实验工具。
5. `demos/simple_harness/`：最小演示。
