# 反例驱动的机器人 Adapter 合成

## 1. 问题定义

给定一个在源机器人上成立的高层程序：

```python
ret_val = robot.pull(cube, goal)
```

迁移时固定以下四项：

- 高层程序；
- ManiSkill 底层 controller；
- simulator；
- 任务 success signal。

系统只允许生成目标机器人的 adapter。Adapter 负责把 `robot.pull(...)` 或
`robot.grasp(...)` 翻译成目标机器人的 `env.step(action)` 序列。

## 2. 方法

当前方法称为：

> Counterexample-Guided Embodiment Adapter Synthesis

完整闭环为：

```text
中性 seed adapter
→ LLM 生成候选 adapter
→ development seeds 仿真
→ 选择信息最完整的失败反例
→ 定位违反的 embodiment constraint
→ 主动选择少量 probe
→ probe 测量进入下一轮 prompt
→ LLM 生成带状态 guard 的新 adapter
→ development 达标后只在 held-out seeds 上最终验证
```

Held-out seeds 只用于最终评估，不会反馈给 LLM。

## 3. Embodiment Contract

`maniskill_backend/embodiment_contracts.py` 给每个机器人定义机器可读契约：

- action 维度和通道布局；
- fixed/mobile base；
- TCP frame；
- gripper action 语义；
- controller、simulator、success signal 等冻结边界；
- 必须通过仿真测量的 reachability/contact 约束。

例如：

```text
xArm6: action[0:3]=TCP xyz, action[3]=gripper, fixed base
Fetch: action[0:3]=TCP xyz, action[3]=gripper,
       action[4:7]=body, action[7:9]=base
```

Contract 约束 LLM 的可修改范围，但不提供成功轨迹。

## 4. Counterexample

一次失败不再只是：

```text
success=False
```

而会转成结构化反例：

```json
{
  "seed": 5,
  "failure_layer": "contact_geometry",
  "failure_reason": "contact_side_reachability_failure",
  "violated_constraints": [
    "contact_pose_reachable_before_descent",
    "contact_side_selected_from_runtime_geometry"
  ],
  "evidence": {
    "stage": "descent",
    "tcp_stage_error_norm": 0.09094,
    "tcp_cube_xy": 0.02912
  }
}
```

系统按诊断可信度、数值完整度和物理阶段给反例打信息分，选出一个最有解释力的主反例；
同时从不同 `failure_reason + stage` 中保留最多三个支持反例。LLM 因而需要兼顾跨 seed
失败模式并生成 guarded branch，而不是只拟合一个 seed 的固定常数。跨 cycle 还维护
反例历史：新 failure reason 加 2 分、新 seed 加 1 分，完全相同的
`seed + reason + stage` 重复项减 4 分。这样高信息反例仍可再次被选中，但循环会优先
覆盖尚未处理的失败机制，减少连续几轮对同一个现象生成近似修复。

## 5. Active Probe

Probe 不再默认完整扫描所有参数。系统根据失败约束选择参数：

| 失败 | 主动 probe 参数 |
|---|---|
| far-side 不可达 | contact x/z offset、approach height |
| 已接触但拖动不足 | drag strength、down bias、stages |
| 抓取侧推 | grasp height、close command、close steps |
| 对齐但无法形成抓取 | close envelope 和 settle timing |

第一次没有测量时，系统使用小规模 one-factor-at-a-time 设计。获得真实 probe
分数后，下一轮使用确定性的 kernel-UCB 选择器：核回归估计候选参数的局部收益，
到最近观测点的归一化距离表示不确定性，采集分数同时奖励高预测值与未探索区域，
批内再加入多样性项。它只在诊断关联参数上搜索，并排除已经试过的组合。
这里的“不确定性”是距离启发式，不是经过校准的概率或高斯过程后验。
是否比固定网格更有效，需要等预算的真实仿真对照，当前不宣称已经证明。

```text
acquisition = normalized predicted score
            + 0.75 * uncertainty
            + 0.20 * batch diversity
```

因此 B5 与 Ours 都使用 8 次显式 probe 预算，并共用生成、development 验证、反例选择、
修复和 held-out 验收的外层循环。B5 从一个冻结的均匀 fixed-grid 设计中分批取样，Ours
根据反例选择参数，并用历史测量更新下一批候选。两者的 prompt、迭代 LLM 修复上限和
最大 simulator-call 预算相同；实际调用数可能因提前停止而不同，会如实报告。
Probe 文件同时保存累计历史和本批新增行，前者用于后续选择与修复，后者用于严格预算核算。
Probe 提供的是有限实验数据，不是
成功轨迹，也不是把 oracle 答案直接告诉 LLM。

## 6. Guarded Adapter

目标不是生成一个只适配 seed 0 的固定轨迹，而是生成带状态分支的 adapter：

```python
if far_side_is_reachable:
    use_far_side_contact()
elif near_side_is_reachable:
    use_alternative_contact()
else:
    return measured_infeasible_failure()
```

每个 fallback 必须产生可测量进展，或者返回有证据的失败；不能靠增加 episode
步数反复执行同一无效动作。

Active probe 先依据反例中的 `violated_constraints` 缩小参数空间，再用已有测量结果
估计各参数的分数敏感度，并以 deterministic kernel-UCB 平衡预测收益、不确定性和
批内多样性。这样 probe 选择由约束和数据共同决定，不由 LLM 自由猜参数。

B5 与 Ours 的候选在进入仿真前都要通过结构验证：必须覆写当前任务技能，读取 TCP、物体
和任务成功/抓取状态，调用 episode 终止检查，包含基于数值阈值的条件分支，并通过
`_fail` 或 `_log` 返回真实结果。两者使用相同的 guarded contract，使对照只改变
probe 选点方式。B2-B4 不应用该结构约束。结构检查不能证明物理可行性，仍须仿真验证。

## 7. 一条命令

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis
```

无 ManiSkill/GPU 时先检查流程：

```bash
python migrate.py \
  --task pull_cube \
  --source panda \
  --target xarm6 \
  --mode cegis \
  --dry-run
```

## 8. 当前边界

当前有两条不同用途的执行路径：

- **注册 case**：已经接入五层诊断、structured probe、active probe、development
  counterexample 和 held-out gate，用于论文中的冻结对照实验。
- **未注册任务**：通过 `migrate.py --task <ManiSkillEnv-v1> --mode agent` 动态发现环境，
  生成源 Program 和源/目标 Adapter，并进行 episode-level 修复；当前尚未自动接入每个
  新任务专属的 structured probe 参数空间。

两条路径都不保证任意任务和任意机器人自动成功。环境和机器人必须已安装且能够由
ManiSkill 初始化。一次超时或高残差只能说明当前策略未到达目标，不能据此证明机器人
在几何上绝对不可行。未注册任务在形成真实成功证据后，仍应冻结为注册 case，才能进入
公平的多 seed 论文评测。
