# Embodied Migration 项目中文使用文档

本文档面向第一次接触本项目的使用者，目标是讲清楚三件事：

1. 这个项目到底在研究什么。
2. 如何一步一步配置环境、启动 GUI demo、跑 benchmark、生成论文表格和案例分析。
3. 每个重要文件/目录应该怎么使用，输出结果应该去哪里看。

建议你把本文档当成项目主入口来用：先读第 1-3 节理解项目，再按第 4-8 节从小测试跑到完整实验，最后按第 9-12 节生成论文材料。

---

## 1. 项目一句话说明

本项目研究的是：

**LLM 生成的机器人控制代码，如何在不同机器人具身条件之间迁移，并在失败后根据结构化反馈自动修正。**

更具体地说，用户给同一个自然语言任务，例如：

```text
Put the red block into the yellow tray.
```

项目会让大语言模型生成 Python 机器人控制代码，然后分别在不同机器人上执行：

- 固定底座 KUKA iiwa
- 固定底座 Franka Panda
- Husky + KUKA 移动操作机器人

不同机器人能力不一样：

- KUKA 是吸盘夹具，放置高度太高容易弹跳。
- Franka 是平行夹爪，稳定性和释放高度不同。
- Mobile Manipulator 必须先 `navigate_to()` 到桌边，再抓取和放置。

因此，“代码迁移”不是简单地说同一个 prompt 输出了不同文本，而是：

> 同一个任务在不同机器人能力、工作空间、夹具、移动底盘等约束下，需要生成不同的可执行机器人程序。

项目核心方法是：

- **Capability Card**：在 prompt 中显式告诉 LLM 机器人能力和限制。
- **Failure Report**：当代码执行失败后，把失败原因、期望状态、实际状态结构化反馈给 LLM，让它重写代码。

论文方向对应：

```text
Capability-Conditioned Failure-Driven Adaptation of LLM-Generated Robot Programs
```

---

## 2. 项目主要能力

项目目前支持以下能力：

### 2.1 交互式 GUI demo

你可以打开 PyBullet GUI，输入自然语言任务，让 LLM 生成机器人代码并执行。

典型用途：

- 给老师展示项目效果。
- 观察机器人是否真的抓取/放置。
- 看移动机器人是否先导航再操作。

### 2.2 手写 smoke test

不调用 LLM，直接执行一段手写 LMP 代码，验证物理仿真、机器人模型、抓取/放置是否正常。

典型用途：

- 检查环境是否装好。
- 检查 PyBullet 是否能正常打开。
- 检查 mobile 机器人是否还能完成导航和抓取。

### 2.3 严格 benchmark

支持多机器人、多任务、多方法对比。

机器人：

```text
kuka
franka
mobile
```

任务类别：

```text
basic       基础操作任务
geometric   空间几何任务
refusal     拒绝/不可执行任务
all         全部任务
```

方法模式：

```text
api            只有 API 描述，没有 few-shot，没有 Capability Card，没有失败反馈
fewshot        API + few-shot 示例
card           few-shot + Capability Card
failure        few-shot + Failure Report retry
card_failure   Capability Card + Failure Report retry
```

### 2.4 实验日志

每次 benchmark 会保存：

```text
metadata.json
summary.csv
trials/*.json
prompts/*.txt
raw_responses/*.txt
generated_code/*.py
```

这些文件用于：

- 复现实验。
- 分析失败类型。
- 对比不同方法生成代码的差异。
- 写论文表格和案例分析。

### 2.5 论文表格和图表

项目可以自动生成：

- 成功率表格
- 机器人分组表格
- 任务类别分组表格
- failure breakdown
- migration score
- paired method deltas
- SVG 图
- LaTeX 表格
- Markdown 结果小节草稿

### 2.6 质性案例分析

项目可以自动挑选代表性案例：

- 成功通过失败反馈修复的案例
- mobile 机器人新增导航代码的案例
- 新增低释放高度的案例
- 新增 NumPy 几何计算的案例
- refusal 成功案例
- persistent failure 失败案例

这些会被写入：

```text
casebook/qualitative_casebook.md
casebook/qualitative_cases.csv
casebook/qualitative_casebook.tex
```

---

## 3. 目录结构说明

项目根目录：

```text
embodied_migration/
├── main.py
├── llm_client.py
├── requirements.txt
├── README.md
├── docs/
│   └── PROJECT_USAGE_CN.md
├── robots/
├── perception/
├── prompts/
├── lmp/
├── capabilities/
├── examples/
├── benchmark/
├── scripts/
├── paper/
└── results/
```

### 3.1 `main.py`

交互式 demo 主入口。

用途：

- 打开 PyBullet GUI。
- 输入自然语言。
- 调用 LLM 生成代码。
- 自动执行代码。
- 如果失败，生成 Failure Report 并重试。

常用命令：

```bash
python main.py --robot kuka
python main.py --robot franka
python main.py --robot mobile
```

### 3.2 `llm_client.py`

LLM 客户端封装。

默认使用 OpenRouter：

```python
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
```

可以用环境变量覆盖：

```bash
export EM_MODEL=anthropic/claude-sonnet-4.5
```

或在 `.env` 中写：

```text
EM_MODEL=anthropic/claude-sonnet-4.5
```

### 3.3 `robots/`

机器人实现。

```text
robots/
├── base_robot.py
├── kuka_robot.py
├── franka_robot.py
└── mobile_robot.py
```

重要接口：

```python
robot.pick(position_3d)
robot.place(target_position_3d)
robot.pick_and_place(src_pos_3d, target_pos_3d)
robot.move_ee_to(position_3d)
```

Mobile 机器人额外有：

```python
mobile.navigate_to(x, y, theta=None)
mobile.is_reachable(target_position_3d)
mobile.get_base_position()
```

### 3.4 `perception/`

场景和感知 API。

核心文件：

```text
perception/scene.py
```

LLM 生成代码时可以调用：

```python
scene.get_object_names()
scene.get_object_position("red block")
```

### 3.5 `prompts/`

prompt 构造。

核心文件：

```text
prompts/cap_prompt.py
```

里面定义：

- 系统提示 `SYSTEM_PROMPT`
- few-shot 示例
- 是否注入 Capability Card
- 是否注入 Failure Report

### 3.6 `capabilities/`

Capability Card 定义。

核心文件：

```text
capabilities/capability_card.py
```

Capability Card 会描述：

- 夹具类型
- 是否需要低释放高度
- 推荐释放高度
- 工作空间半径
- 是否有移动底盘
- 是否能旋转物体
- IK 精度

### 3.7 `lmp/`

LLM 生成代码的提取、执行和失败报告。

```text
lmp/
├── executor.py
├── extractor.py
└── failure_report.py
```

其中：

- `executor.py`：安全执行 LLM 生成的 Python 代码。
- `extractor.py`：从 LLM 回复中提取代码块。
- `failure_report.py`：构造 Failure Report。

### 3.8 `examples/`

快速测试脚本。

```text
examples/
├── smoke_test.py
└── test_capability_card.py
```

### 3.9 `benchmark/`

实验核心。

```text
benchmark/
├── run_benchmark.py
├── experiment_logging.py
├── analyze_results.py
├── audit_run.py
├── build_paper_assets.py
├── build_casebook.py
└── llm_cache.py
```

用途分别是：

- `run_benchmark.py`：跑实验。
- `experiment_logging.py`：记录 trial、prompt、代码、summary。
- `analyze_results.py`：生成 CSV/LaTeX 表格。
- `audit_run.py`：检查实验是否完整。
- `build_paper_assets.py`：生成论文 SVG 图和结果小节草稿。
- `build_casebook.py`：生成质性案例分析包。
- `llm_cache.py`：缓存 LLM 回复，避免重复 API 成本。

### 3.10 `scripts/`

一键脚本。

```text
scripts/
├── run_stage5_experiments.sh
├── build_stage6_paper_package.sh
├── run_stage7_reliable_experiments.sh
└── build_stage8_qualitative_package.sh
```

推荐正式实验使用：

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
bash scripts/build_stage8_qualitative_package.sh stage7_seeded_full
```

### 3.11 `paper/`

论文写作辅助材料。

```text
paper/
├── experiment_section_template.md
├── main_paper_outline.md
├── qualitative_analysis_template.md
└── submission_readiness_checklist.md
```

---

## 4. 从零开始配置环境

以下假设你使用 conda，并且环境名叫 `em`。

### 4.1 进入项目目录

```bash
cd /Users/xifan/Downloads/embodied_migration
```

预期：

```text
当前工作目录切换到 embodied_migration
```

可以用下面命令确认：

```bash
pwd
```

预期类似：

```text
/Users/xifan/Downloads/embodied_migration
```

### 4.2 创建 conda 环境

如果还没有 `em` 环境：

```bash
conda create -n em python=3.10 -y
```

预期输出中会出现类似：

```text
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
```

如果已经有 `em` 环境，可以跳过这一步。

### 4.3 激活环境

```bash
conda activate em
```

预期：

命令行前面出现：

```text
(em)
```

例如：

```text
(em) xifan@Mac embodied_migration %
```

### 4.4 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：

```text
pybullet
numpy
openai
python-dotenv
```

预期输出类似：

```text
Successfully installed ...
```

如果已经安装，会看到：

```text
Requirement already satisfied
```

### 4.5 配置 `.env`

项目需要 OpenRouter API key 才能调用 LLM。

复制模板：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```text
OPENROUTER_API_KEY=你的_OpenRouter_Key
EM_MODEL=anthropic/claude-sonnet-4.5
```

`EM_MODEL` 可选，不写时默认使用：

```text
anthropic/claude-sonnet-4.5
```

注意：

- `.env` 不要提交给别人。
- 完整 benchmark 会调用大量 LLM API，会产生费用。

---

## 5. 最小验证：不调用 LLM 的 smoke test

第一次使用项目时，不要直接跑大实验。先跑 smoke test。

### 5.1 测 KUKA

```bash
python -m examples.smoke_test --robot kuka
```

预期输出包含：

```text
Scene ready. Embodiment: KUKA iiwa
LMP executed successfully
ret_val = 'success'
physical_success = True
```

### 5.2 测 Franka

```bash
python -m examples.smoke_test --robot franka
```

预期输出包含：

```text
Scene ready. Embodiment: Franka Panda
ret_val = 'success'
physical_success = True
```

### 5.3 测 Mobile Manipulator

```bash
python -m examples.smoke_test --robot mobile
```

预期输出包含：

```text
Scene ready. Embodiment: Mobile Manipulator
[Mobile] Navigated base to ...
Attached 'red block'
Released
ret_val = 'success'
physical_success = True
```

如果 mobile 输出：

```text
failed to move above object
```

说明机器人没有停在合适的桌边位置，或者导航/可达性逻辑出问题。

### 5.4 打开 GUI 看 smoke test

如果你想看到 GUI：

```bash
python -m examples.smoke_test --robot mobile --gui
```

预期：

- 弹出 PyBullet GUI。
- 看到 Husky + KUKA 移动机器人。
- 看到桌子、红色方块、绿色方块、黄色托盘。
- 机器人先导航到桌边，再抓红色方块，放到托盘附近。

关闭方式：

```text
在终端按 Ctrl+C
```

---

## 6. 检查 Capability Card 和 Failure Report

运行：

```bash
python -m examples.test_capability_card
```

预期输出包含：

```text
TEST 1: Every robot class declares a capability_card
TEST 2: CapabilityCard.to_prompt_section() produces readable text
TEST 3: FailureReport auto-diagnoses position deviation
TEST 4: build_user_prompt wires card + report together
All smoke tests passed
```

这个测试说明：

- 每个机器人都有 Capability Card。
- Capability Card 可以写入 prompt。
- Failure Report 可以根据 expected/actual 自动诊断。
- prompt 能同时注入 Capability Card 和 Failure Report。

---

## 7. 交互式 GUI demo 怎么用

这是给人看的 demo，也是最直观的部分。

### 7.1 启动 KUKA GUI demo

```bash
python main.py --robot kuka
```

预期：

```text
Launching PyBullet (robot=kuka)...
Scene ready. Embodiment: KUKA iiwa
Enter a natural-language instruction
```

同时会弹出 PyBullet GUI。

你可以输入：

```text
put the red block into the yellow tray
```

预期流程：

1. 程序把自然语言任务发给 LLM。
2. LLM 生成 Python 代码。
3. 项目执行代码。
4. GUI 中机器人移动、抓取、放置。
5. 终端显示执行结果。

### 7.2 启动 Franka GUI demo

```bash
python main.py --robot franka
```

输入示例：

```text
move the green block 10 cm to the right
```

### 7.3 启动 Mobile GUI demo

```bash
python main.py --robot mobile
```

输入示例：

```text
put the red block into the yellow tray
```

预期正确行为：

- Mobile 机器人不能直接在原地抓。
- 它应该先调用 `mobile.navigate_to(...)`。
- 停到桌边合适位置。
- 再执行 `pick` 和 `place`。

如果你看到小车穿进桌子或机器人够不到物体，说明 LLM 生成的导航位置不好。正确代码通常应该类似：

```python
table_x, table_y = float(scene.table_position[0]), float(scene.table_position[1])
mobile.navigate_to(table_x - 0.4, table_y + 0.65)
```

而不是：

```python
mobile.navigate_to(red_block_pos[0], red_block_pos[1])
```

后者会把底盘停到物体正下方，容易穿模或导致机械臂不可达。

### 7.4 一次性输入 instruction

不想进入 REPL，可以直接运行：

```bash
python main.py \
  --robot mobile \
  --instruction "put the red block into the yellow tray"
```

执行结束后终端会提示：

```text
[Press Enter to close]
```

### 7.5 不同 demo 模式

`main.py` 支持三种模式：

```bash
python main.py --robot kuka --mode baseline
python main.py --robot kuka --mode b
python main.py --robot kuka --mode ba
```

含义：

```text
baseline  不用 Capability Card，不重试，只尝试 1 次
b         不用 Capability Card，但失败后用 Failure Report 重试
ba        使用 Capability Card，也使用 Failure Report 重试，默认模式
```

也可以单独关闭：

```bash
python main.py --robot kuka --no-card
python main.py --robot kuka --no-retry
```

---

## 8. benchmark 任务和方法说明

benchmark 入口：

```bash
python -m benchmark.run_benchmark
```

### 8.1 机器人

```bash
--robots kuka franka mobile
```

三个机器人代表三种具身条件：

```text
kuka     固定底座 + 吸盘
franka   固定底座 + 平行夹爪
mobile   Husky 移动底盘 + KUKA 机械臂
```

### 8.2 任务类别

```bash
--tasks basic
--tasks geometric
--tasks refusal
--tasks all
```

`basic` 包括：

```text
pick_red_to_tray
move_green_right
stack_two
report_leftmost
```

`geometric` 包括：

```text
arrange_line
arrange_triangle
arrange_circle
mirror_layout
sort_left_to_right
```

`refusal` 包括：

```text
refuse_rotate_object
refuse_missing_object
```

`all` 会跑全部 11 个任务。

### 8.3 方法模式

推荐论文实验使用严格模式：

```bash
--modes api fewshot card failure card_failure
```

含义：

```text
api
  只给 API 描述。
  没有 few-shot。
  没有 Capability Card。
  没有 Failure Report。
  只尝试 1 次。

fewshot
  API + few-shot 示例。
  没有 Capability Card。
  没有 Failure Report。
  只尝试 1 次。

card
  API + few-shot + Capability Card。
  没有 Failure Report。
  只尝试 1 次。

failure
  API + few-shot + Failure Report retry。
  没有 Capability Card。
  最多 3 次尝试。

card_failure
  API + few-shot + Capability Card + Failure Report retry。
  最多 3 次尝试。
```

旧别名仍然可用：

```text
baseline -> fewshot
b        -> failure
ba       -> card_failure
```

### 8.4 场景布局

固定布局：

```bash
--scene-variant fixed
```

随机但可复现布局：

```bash
--scene-variant seeded --seed-base 0
```

如果 `--trials 5 --seed-base 0`，则使用 seed：

```text
0, 1, 2, 3, 4
```

这样同一个 seed 会在不同方法/机器人中复用，方便做 paired comparison。

---

## 9. 逐步跑 benchmark

### 9.1 最小 benchmark

先只跑一个机器人、一个任务类别、一个方法：

```bash
python -m benchmark.run_benchmark \
  --robots kuka \
  --modes api \
  --tasks refusal \
  --trials 1 \
  --run-id quick_test
```

预期输出：

```text
Logging run artifacts to: results/runs/quick_test
MODE: API
Robot: kuka
[refuse_rotate_object] ...
[refuse_missing_object] ...
Cross-Embodiment Migration Ablation Results
Wrote summary.csv and trial artifacts to: results/runs/quick_test
```

结果目录：

```text
results/runs/quick_test/
```

### 9.2 几何任务快速对比

```bash
python -m benchmark.run_benchmark \
  --robots kuka franka mobile \
  --modes api fewshot card failure card_failure \
  --tasks geometric \
  --trials 1 \
  --run-id geometric_smoke
```

这个命令会跑：

```text
3 robots × 5 methods × 5 geometric tasks × 1 trial = 75 trials
```

注意：

- 会调用 LLM。
- 会产生 API 成本。
- 如果只是测试流程，可以先把 `--robots` 或 `--modes` 缩小。

### 9.3 完整正式实验推荐命令

推荐使用阶段 7 脚本：

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
```

这个脚本会执行：

1. benchmark
2. analyze results
3. audit run
4. build paper assets

默认设置：

```text
robots: kuka franka mobile
modes: api fewshot card failure card_failure
tasks: all
trials: 5
scene_variant: seeded
seed_base: 0
cache: enabled
resume: enabled
```

总 trial 数：

```text
3 robots × 5 modes × 11 tasks × 5 trials = 825 trials
```

这是正式实验量级，会调用很多次 LLM。请确认 API 额度。

### 9.4 如果中断了怎么办

阶段 7 脚本默认使用 `--resume`。

如果运行中断，直接重新运行：

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
```

它会跳过已经完成的 trial。

### 9.5 只使用缓存，不允许 live API

如果你只想验证已有缓存，不想产生新 API 调用：

```bash
OFFLINE_CACHE_ONLY=1 bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
```

如果 cache miss，会报错：

```text
LLM cache miss and no live client is available
```

这说明该 prompt 之前没有缓存，必须允许 live API 或先补齐缓存。

---

## 10. benchmark 输出怎么看

一次 run 的目录一般是：

```text
results/runs/<run_id>/
```

例如：

```text
results/runs/stage7_seeded_full/
```

里面主要有：

```text
metadata.json
summary.csv
trials/
prompts/
raw_responses/
generated_code/
tables/
audit/
paper_assets/
casebook/
```

### 10.1 `metadata.json`

记录实验设置：

```text
robots
modes
mode_configs
n_trials
scene_variant
seed_base
scene_seeds
llm_model
llm_temperature
llm_cache_enabled
task_names
task_families
```

论文复现时一定要保留。

### 10.2 `summary.csv`

每个 trial 一行，适合快速查看。

重要字段：

```text
mode
canonical_mode
robot
task
task_family
scene_seed
success
attempts
llm_model
llm_temperature
llm_cache_hits
failure_type
failure_subtype
used_mobile_navigate_to
used_low_release_height
used_numpy
used_refusal_ret_val
lines_of_code
```

### 10.3 `trials/*.json`

每个 trial 的完整记录。

包含：

- 初始场景
- 最终场景
- 每次 attempt 的 prompt
- LLM raw response
- generated code
- ret_val
- action failures
- checker expected/actual
- failure report 信息

这是最重要的可复现文件。

### 10.4 `prompts/*.txt`

每次发给 LLM 的完整 user prompt。

用来分析：

- Capability Card 是否注入。
- Failure Report 是否注入。
- mobile API hint 是否出现。

### 10.5 `raw_responses/*.txt`

LLM 原始回复。

### 10.6 `generated_code/*.py`

从 LLM 回复中提取出来的 Python 代码。

用于写 generated-code difference analysis。

---

## 11. 生成统计表格

如果 benchmark 已经跑完：

```bash
python -m benchmark.analyze_results results/runs/stage7_seeded_full
```

输出目录：

```text
results/runs/stage7_seeded_full/tables/
```

重要文件：

```text
method_summary.csv/.tex
robot_method_summary.csv/.tex
task_family_method_summary.csv/.tex
task_method_summary.csv
scene_variant_method_summary.csv
seed_method_summary.csv
migration_score.csv/.tex
paired_method_deltas.csv/.tex
failure_breakdown.csv/.tex
failure_cases.csv
generated_code_features.csv
code_changes_after_feedback.csv
code_change_summary.csv
analysis_report.md
```

### 11.1 `method_summary`

总体方法对比。

看：

```text
success_rate
success_ci95
mean_attempts
recovered_after_feedback_rate
```

这张表回答：

> card_failure 是否比 api/fewshot/card/failure 更好？

### 11.2 `robot_method_summary`

按机器人分组的方法对比。

这张表回答：

> 方法是否主要帮助 mobile？是否对 KUKA/Franka 也有效？

### 11.3 `task_family_method_summary`

按任务类别分组。

这张表回答：

> 方法是帮助 basic、geometric，还是 refusal？

### 11.4 `migration_score`

迁移分数。

定义：

> 一个任务只有在所有机器人上都超过成功阈值，才算 successfully migrated。

这比单纯 success rate 更贴近“跨具身迁移”主题。

### 11.5 `paired_method_deltas`

与 few-shot baseline 做 matched comparison。

匹配条件：

```text
robot
task
scene_variant
scene_seed
trial_index
```

这能减少随机布局带来的噪声。

### 11.6 `failure_breakdown`

失败类型统计。

粗粒度：

```text
llm_error
exec_error
action_failure
check_failure
ret_val_failure
```

细粒度：

```text
missing_mobile_navigation
grasp_failure
pick_failure
place_failure
ik_or_workspace_failure
geometric_layout_mismatch
incorrect_refusal_decision
wrong_ret_val
...
```

---

## 12. 审计实验完整性

正式实验跑完后，必须审计。

```bash
python -m benchmark.audit_run results/runs/stage7_seeded_full --fail-on-missing
```

输出：

```text
results/runs/stage7_seeded_full/audit/
├── audit_summary.json
└── audit_report.md
```

重点看：

```text
Expected trials
Found trial JSON files
Missing Trials
Incomplete Trials
LLM Error Trials
Cache hit attempts
```

理想状态：

```text
missing=0
incomplete=0
```

如果 `--fail-on-missing` 检测到缺失，会返回非零退出码。

---

## 13. 生成论文图表和结果草稿

运行：

```bash
python -m benchmark.build_paper_assets results/runs/stage7_seeded_full
```

输出：

```text
results/runs/stage7_seeded_full/paper_assets/
```

包含：

```text
experiment_manifest.json
paper_results_section.md
figure_index.md
fig_method_success.svg
fig_robot_method_success.svg
fig_task_family_success.svg
fig_migration_score.svg
table_includes.tex
```

### 13.1 `fig_method_success.svg`

总体方法成功率图。

适合放主实验结果。

### 13.2 `fig_robot_method_success.svg`

按机器人分组的成功率图。

适合说明 mobile 是最强的具身迁移测试。

### 13.3 `fig_task_family_success.svg`

按任务类别分组。

适合说明：

- geometric 是否更难
- refusal 是否被正确处理

### 13.4 `fig_migration_score.svg`

跨具身迁移分数图。

适合支撑论文主张：

> 方法提升了跨机器人可迁移性。

### 13.5 `paper_results_section.md`

自动生成的结果小节草稿。

你不能原封不动投稿，但可以作为写作起点。

### 13.6 `table_includes.tex`

LaTeX 表格 include 文件。

里面类似：

```tex
\input{results/runs/stage7_seeded_full/tables/method_summary.tex}
\input{results/runs/stage7_seeded_full/tables/robot_method_summary.tex}
...
```

---

## 14. 生成质性案例分析包

运行：

```bash
python -m benchmark.build_casebook results/runs/stage7_seeded_full
```

或者使用阶段 8 脚本：

```bash
bash scripts/build_stage8_qualitative_package.sh stage7_seeded_full
```

输出：

```text
results/runs/stage7_seeded_full/casebook/
├── qualitative_cases.csv
├── qualitative_casebook.md
└── qualitative_casebook.tex
```

### 14.1 `qualitative_cases.csv`

案例索引。

字段包括：

```text
case_id
category
trial_id
canonical_mode
robot
task_family
task
success
attempts
failure_type
failure_subtype
adaptation_summary
first_code_excerpt
final_code_excerpt
trial_path
```

### 14.2 `qualitative_casebook.md`

最适合阅读。

每个案例包含：

- trial 信息
- 方法/机器人/任务
- 是否成功
- 失败类型
- adaptation summary
- Failure Report excerpt
- first attempt code
- final attempt code
- code diff

### 14.3 `qualitative_casebook.tex`

LaTeX 片段。

适合放 appendix 或 supplementary。

### 14.4 论文里怎么用

建议选 3-5 个案例：

1. mobile 导航修复案例
2. 低释放高度修复案例
3. 几何任务新增 NumPy 计算案例
4. refusal 成功案例
5. persistent failure 限制案例

写作结构：

```text
Case: <case_id>
Robot / task: <robot> / <task>
Failure: <failure_subtype>
Adaptation: <final code added what>
Outcome: success/failure
```

---

## 15. 推荐完整工作流

### 15.1 第一次安装后

```bash
conda activate em
python -m examples.smoke_test --robot kuka
python -m examples.smoke_test --robot franka
python -m examples.smoke_test --robot mobile
python -m examples.test_capability_card
```

全部通过后，再进入 GUI demo 或 benchmark。

### 15.2 展示 GUI demo

```bash
python main.py --robot mobile
```

输入：

```text
put the red block into the yellow tray
```

观察：

- 是否先导航到桌边。
- 是否抓取红色方块。
- 是否放到黄色托盘。
- 失败后是否 retry。

### 15.3 快速实验

```bash
python -m benchmark.run_benchmark \
  --robots kuka \
  --modes api card_failure \
  --tasks refusal \
  --trials 1 \
  --run-id quick_test

python -m benchmark.analyze_results results/runs/quick_test
```

### 15.4 正式实验

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
```

### 15.5 正式实验后生成质性分析

```bash
bash scripts/build_stage8_qualitative_package.sh stage7_seeded_full
```

### 15.6 检查论文材料

看这些文件：

```text
results/runs/stage7_seeded_full/tables/analysis_report.md
results/runs/stage7_seeded_full/audit/audit_report.md
results/runs/stage7_seeded_full/paper_assets/paper_results_section.md
results/runs/stage7_seeded_full/casebook/qualitative_casebook.md
```

---

## 16. 常见问题

### 16.1 没有 GUI

如果运行：

```bash
python -m examples.smoke_test --robot mobile
```

默认不会显示 GUI，因为 smoke test 默认 headless。

要显示 GUI：

```bash
python -m examples.smoke_test --robot mobile --gui
```

交互式 demo `main.py` 默认会显示 GUI：

```bash
python main.py --robot mobile
```

### 16.2 LLM API key 报错

错误类似：

```text
Please set OPENROUTER_API_KEY in your .env file or environment.
```

解决：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```text
OPENROUTER_API_KEY=你的key
```

### 16.3 Mobile 机器人抓不到

常见失败：

```text
move_ee_to failed
pick: failed to move above object
```

原因可能是：

- LLM 没有先导航。
- LLM 导航到物体正下方。
- 停车点离桌子太远或太近。

比较合理的导航：

```python
table_x, table_y = float(scene.table_position[0]), float(scene.table_position[1])
mobile.navigate_to(table_x - 0.4, table_y + 0.65)
```

### 16.4 为什么 benchmark 很慢

完整实验规模：

```text
3 robots × 5 modes × 11 tasks × 5 trials = 825 trials
```

其中 `failure` 和 `card_failure` 最多会尝试 3 次，因此 LLM 调用次数可能超过 825。

建议：

先跑小实验：

```bash
python -m benchmark.run_benchmark \
  --robots kuka \
  --modes api card_failure \
  --tasks basic \
  --trials 1 \
  --run-id debug_small
```

确认没问题后再跑完整实验。

### 16.5 中断后能不能继续

可以。

使用：

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
```

或者手动加：

```bash
--resume
```

它会跳过已经完成的 trial JSON。

### 16.6 如何避免重复 API 成本

默认启用 LLM cache。

cache 目录：

```text
results/llm_cache/
```

如果同一个 prompt、model、temperature 再次出现，会直接读取缓存。

只允许用缓存，不允许 live API：

```bash
OFFLINE_CACHE_ONLY=1 bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
```

### 16.7 生成的表格为空

先确认 run 目录里有 trial：

```bash
find results/runs/<run_id>/trials -name "*.json" | wc -l
```

如果是 0，说明 benchmark 没有成功跑完。

再运行：

```bash
python -m benchmark.audit_run results/runs/<run_id>
```

看 missing/incomplete。

---

## 17. 论文写作建议

### 17.1 主实验表格

优先使用：

```text
method_summary.tex
robot_method_summary.tex
task_family_method_summary.tex
migration_score.tex
paired_method_deltas.tex
failure_breakdown.tex
```

### 17.2 主图

优先使用：

```text
fig_method_success.svg
fig_robot_method_success.svg
fig_task_family_success.svg
fig_migration_score.svg
```

### 17.3 质性案例

从：

```text
casebook/qualitative_casebook.md
```

选案例。

不要在正文中贴很长代码。正文中建议只贴关键 diff，例如：

```diff
+ mobile.navigate_to(table_x - 0.4, table_y + 0.65)
+ robot.place(target, pre_release_height=0.005)
```

完整代码可放 appendix。

### 17.4 论文主张

本项目比较适合支撑以下论点：

1. Capability Card 能提升第一轮生成代码的具身感知能力。
2. Failure Report 能把执行失败转化为可操作的代码修复信号。
3. 两者结合可以提高跨机器人迁移成功率。
4. 代码迁移可以通过生成代码特征变化来度量，而不只是看自然语言回答。
5. refusal 任务应该单独评估，因为正确拒绝也是机器人能力边界的一部分。

---

## 18. 最推荐的最终命令清单

从干净环境开始：

```bash
cd /Users/xifan/Downloads/embodied_migration
conda activate em
pip install -r requirements.txt
```

检查：

```bash
python -m examples.smoke_test --robot kuka
python -m examples.smoke_test --robot franka
python -m examples.smoke_test --robot mobile
python -m examples.test_capability_card
```

GUI demo：

```bash
python main.py --robot mobile
```

正式实验：

```bash
bash scripts/run_stage7_reliable_experiments.sh stage7_seeded_full
```

质性分析包：

```bash
bash scripts/build_stage8_qualitative_package.sh stage7_seeded_full
```

最终你主要查看：

```text
results/runs/stage7_seeded_full/tables/analysis_report.md
results/runs/stage7_seeded_full/audit/audit_report.md
results/runs/stage7_seeded_full/paper_assets/paper_results_section.md
results/runs/stage7_seeded_full/casebook/qualitative_casebook.md
```

---

## 19. 一句话总结

这个项目不是单纯做“让 LLM 控机器人”，而是研究：

> 当机器人身体、夹具、工作空间和移动能力发生变化时，LLM 生成的机器人程序如何根据能力先验和执行失败反馈进行代码迁移。

最终产物不仅包括可运行 demo，还包括：

- benchmark
- 失败类型统计
- 代码差异分析
- 迁移分数
- 多 seed 可靠性评估
- 论文表格
- 论文图表
- 质性案例分析包

这也是它可以往 ICRA/IROS/CoRL 主会论文方向继续推进的基础。
