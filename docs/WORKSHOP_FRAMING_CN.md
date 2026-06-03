# Workshop Framing：LLM 机器人程序迁移

## 1. 研究问题

高层机器人程序看起来可以跨机器人复用，例如：

```python
robot.pull(cube, goal)
robot.grasp(cube)
robot.place(cube, goal)
```

但真正迁移到不同 embodiment 时，失败往往不在高层程序，而在：

- target robot action space
- skill adapter
- TCP/contact geometry
- gripper close timing
- reachability and controller response
- real simulator success evaluation

本项目研究：**LLM 能否根据目标机器人约束和真实失败日志，自动生成 target-side adapter，使源程序在目标机器人上成功执行。**

## 2. 方法

当前方法是 direct target-module generation：

```text
source LMP program
  + robot profile
  + target capability constraints
  + real ManiSkill failure logs
  -> LLM-generated target adapter
  -> unit tests
  -> real ManiSkill env.step(action)
  -> success/failure analysis
```

高层程序保持不变，LLM 主要修改 target adapter。

## 3. 当前结果

| 任务 | 迁移方向 | 结果 | 说明 |
|---|---|---|---|
| `PullCube-v1` | Panda -> xArm6 | 成功 | LLM adapter 成功迁移接触拖拽任务 |
| `PickCube-v1` | Panda -> xArm6 | source succeeds; target hard case | Panda baseline 成功，xArm6 真实 force-closure grasp 仍失败 |
| `PullCube-v1` | Panda -> Fetch | diagnosed failure | 移动底盘与接触侧可达性问题 |

## 4. 正结果：PullCube

`PullCube-v1` 是接触拖拽任务。成功说明：

- LLM 可以生成 target-specific adapter；
- xArm6 action mapping、接触侧、拖拽脉冲和执行参数可以被迁移；
- 最终成功来自真实 ManiSkill `env.step(action)` 和 `evaluate()`，不是 fake success。

该结果可以作为 workshop paper 的主正例。

## 5. 负结果：PickCube

`PickCube-v1` 要求真实抓取、抬升和三维搬运。

当前发现：

- Panda source-copy baseline 已成功：`ret_val=True`, `elapsed_steps=40`；
- LLM 可以生成结构化 grasp adapter；
- LLM 能利用 probe feedback；
- 但稳定 grasp 仍失败；
- 失败集中在 descent depth、gripper envelope、side push、transient `is_grasping=True` preservation。

这不是坏结果。它说明：

```text
contact dragging migration is feasible with LLM adapter generation;
force-closure grasp migration requires stronger constraint handling.
```

## 6. 自动探针

为避免无限 prompt，新增 xArm6 PickCube grasp probe：

```bash
python scripts/xarm6_pick_grasp_probe.py \
  --sim-backend auto \
  --render-backend gpu
```

探针枚举：

```text
grasp_z_offset
close_steps
close_command
settle_steps
```

并记录：

```text
is_grasping_after_close
is_grasping_after_lift
cube_disp_xy
tcp_grasp_xy
tcp_grasp_z
```

当前结果：32 组 fixed-XY close-envelope 参数中没有成功 grasp。

意义：probe 不给 LLM 成功答案，而是给它结构化物理证据，说明简单 close 参数调节不足。

## 7. Workshop Claim

英文版本：

```text
LLM-generated target adapters can migrate contact-based manipulation programs
across embodiments, but force-closure grasping exposes the limits of
prompt-only repair. Structured physical probing turns simulator failures into
constraints for more reliable adapter generation.
```

中文版本：

```text
LLM 生成目标 adapter 可以完成接触拖拽类跨机器人迁移；
但真实抓取任务暴露了 prompt-only repair 的局限。
结构化物理探针可以把失败日志转化为约束，
为后续 constraint-aware / learning-guided repair 提供基础。
```

## 8. 适合投 workshop 的贡献点

1. 一个真实 ManiSkill 中的 LLM target-adapter generation pipeline。
2. 一个成功迁移案例：Panda -> xArm6 `PullCube-v1`。
3. 一个 hard-case failure analysis：Panda -> xArm6 `PickCube-v1`。
4. 一个结构化物理探针，用于把失败从自然语言日志变成约束证据。
5. 对比说明：接触拖拽与 force-closure grasp 的迁移难度差异。

## 9. 明天组会建议讲法

开头：

```text
我现在已经不把这个项目当作简单代码生成，而是当作 target adapter migration。
目前 PullCube 已经成功，PickCube 作为 hard case 暴露出真实抓取迁移瓶颈。
```

中间：

```text
PullCube 成功说明 LLM 可以迁移接触式 manipulation adapter。
PickCube 失败说明 force-closure grasp 不能只靠 prompt 和参数搜索。
```

结尾：

```text
下一步我计划停止无限 prompt，固定当前 pipeline，
跑 PullCube 多 seed 作为主结果，
把 PickCube 整理为 hard-case failure analysis，
并把 structured probing 作为后续改进方向。
```
