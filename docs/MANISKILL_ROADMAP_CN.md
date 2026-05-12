# ManiSkill 后续研究路线

## 为什么转向 ManiSkill

之前的 PyBullet 和 robosuite/MuJoCo 原型可以说明代码迁移的基本想法，但任务仍然偏简单：

- 成功/失败不够丰富；
- 任务主要是搬运或简化双臂 skill；
- 很多判断依赖手写状态机；
- 很难支撑更严格的论文实验。

后续建议把主平台切换到 ManiSkill 3，在 Ubuntu 原生系统上做正式实验。

## 推荐平台

```text
Ubuntu 22.04 / 24.04
NVIDIA driver
RTX 3060 Laptop GPU
Conda Python 3.10
ManiSkill 3
```

不建议把 WSL 作为主实验平台，因为 GUI 渲染和 GPU 仿真更容易出问题。

## 第一批任务

优先做：

```text
PegInsertionSide-v1
PlugCharger-v1
PullCubeTool-v1
```

备选任务：

```text
PickSingleYCB-v1
OpenCabinetDrawer-v1
PickCube-v1  # 只做 smoke test
```

## 第一批机器人

建议先用：

```text
panda
fetch
xarm6_robotiq
so100
widowxai
```

研究重点不是机器人越多越好，而是每个机器人之间的能力差异要能导致明确的代码迁移需求。

## 代码结构规划

```text
maniskill_backend/
├── profiles.py
├── tasks.py
├── skills.py
├── env_adapter.py
├── migration.py
└── evaluation.py
```

每个文件职责：

```text
profiles.py     定义机器人 Capability Card
tasks.py        定义源任务、源程序、目标任务
skills.py       给 LLM 使用的 high-level skill API
env_adapter.py  封装 ManiSkill reset / step / render
migration.py    source-copy / oracle / llm 迁移流程
evaluation.py   从 info / observation 中提取成功和失败原因
```

## 实验方法

需要比较：

```text
source-copy
llm_no_card
llm_card_only
llm_failure_only
llm_card_failure
oracle
```

核心指标：

```text
success rate
attempts to success
failure type distribution
invalid code rate
refusal correctness
API-call difference
code edit distance
```

## 失败类型

需要明确统计：

```text
API mismatch
reachability failure
gripper/force failure
alignment failure
insertion speed failure
tool-use ordering failure
unsafe or impossible task
invalid generated code
```

## 最近一步

先不要急着写论文。下一步应该是：

```text
1. 在 Ubuntu 上跑通 ManiSkill GUI。
2. 用 PickCube-v1 做 smoke test。
3. 用 PegInsertionSide-v1 做第一个正式任务。
4. 实现 maniskill_backend/env_adapter.py。
5. 实现一个 oracle program。
6. 再接 LLM。
```

