# 诊断驱动的机器人代码迁移：组会演讲稿

对应 PPT：`docs/LLM_ROBOT_MIGRATION_PROGRESS_REPORT_CN_15SLIDES.pptx`  
建议时长：10-15 分钟

## 0. 组会前 30 秒更新

今天我想先补充这两周的新进展。之前项目主要是在做“LLM 生成一个 target adapter，然后我手动读日志、手动改 prompt”。这两周我把流程推进成了一个更接近 Guava-style 的 autonomous harness：系统会生成 `agent_observation.json`，Agent 选择一个安全工具，比如跑 single seed、multi-seed、structured probe 或 LLM repair，然后 harness 在真实 ManiSkill 环境里执行并保存结果。

另外我刚补了一个更接近最终目标的用户级入口：

```bash
python migrate.py --task pull_cube --source panda --target xarm6_robotiq
```

它会自动找到当前至少一个成功案例：`PullCube-v1 + Panda -> xArm6`，也就是 `case02_pull_cube_panda_to_xarm6`。这样汇报时就不用说“我手动指定 case id”，而是可以说系统已经开始支持“任务 + 源机器人 + 目标机器人”的迁移请求形式。

现在已经有一个最小 demo：

```bash
python demos/simple_harness/demo.py
python demos/simple_harness/demo.py --run
```

其中 `--run` 已经在远程跑通，返回：

```text
selected_tool = run_multi_seed
executed = true
returncode = 0
```

所以这次汇报我想强调的不是又调了几个 prompt，而是项目从“手动迁移实验”往“仿真闭环里的自动诊断和修复”推进了一步。

## 1. 诊断驱动的机器人代码迁移

大家好，我今天汇报的是目前机器人代码迁移项目的进展。这个项目的核心问题是：同一个高层机器人程序，换一个机器人之后，为什么会失败，以及能不能让系统自动定位失败并修复 adapter。这里的失败很多时候不是 Python 语法错误，而是机器人 embodiment 的约束，比如动作空间不同、末端几何不同、接触点不可达等。当前我已经完成了 PullCube 的一个成功迁移案例，并且把失败诊断、structured probing 和多 seed 泛化评估接进了实验流程。

## 2. 一句话总结

这个项目可以用三句话概括。第一，问题定义是：同一段高层机器人代码，换机器人之后为什么失败。第二，方法上我不直接重写高层任务程序，而是生成目标机器人一侧的 adapter。第三，当前实验发现是：PullCube 这个接触拖拽任务可以从 Panda 迁移到 xArm6，但 PickCube 这种真实抓取任务暴露了更强的物理约束问题。因此项目现在已经不只是“让 LLM 写代码”，而是在做一个仿真闭环里的诊断和修复系统。

## 3. 研究问题：不是换一句代码那么简单

这里想强调的是，高层程序看起来可以完全一样。比如 PullCube 里就是 `robot.pull(cube, goal)`，PickCube 里就是先 `robot.grasp(cube)` 再 `robot.place(cube, goal)`。但是换机器人以后，底层自由度、动作空间、TCP、夹爪形状和工作空间都会变。所以失败原因可能不在高层 program，而在更底层的执行策略。我的研究目标就是让 LLM 不只是生成一段新代码，而是能根据真实 ManiSkill 仿真反馈，修复目标机器人对应的 adapter。

## 4. 系统分层：我到底改哪里？

这一页是我现在项目结构里最重要的一页。最上层是 Program，它表达任务意图，比如“把 cube 拉到 goal”。中间是 Adapter，它负责把这个高层技能翻译成目标机器人能执行的动作序列。再往下是 action、`env.step(action)` 和最终 success signal。我的实验里，主要修改对象是 target-side adapter；controller、simulator 和 success signal 都尽量冻结。这样问题会更干净：同一个高层程序能否通过 adapter 适配不同 embodiment。

## 5. Adapter 通俗理解

Adapter 可以理解成“翻译器加执行方案”。高层程序只说我要做什么，但没有说明机械臂每一步往哪里走、夹爪开还是关、接触点选哪里、失败后怎么重试。这些都由 adapter 决定。比如 Panda 的 `pull` 可以用比较固定的接触点直接拖拽，但 xArm6 的 adapter 就需要重新考虑接触侧、下降高度、拖拽脉冲和失败诊断。所以迁移的重点不是改 `robot.pull(cube, goal)` 这一句，而是改这句背后的执行翻译器。

## 6. 五层失败分类

为了避免只看 success/fail，我把失败分成五层。第一层是 program，比如对象名或高层 API 调用错。第二层是 skill adapter，也就是技能执行策略错，比如 approach 或 descent 用尽 episode。第三层是 controller primitive，比如 action 维度或控制接口不匹配。第四层是 contact geometry，比如 TCP 到了但接触点或夹爪包络不对。第五层是 infeasibility，比如固定基座根本到不了某个接触侧。这个分类的作用是告诉 LLM 应该改哪一层，而不是泛泛地说“任务失败了”。

## 7. 闭环方法：simulation-in-the-loop repair

目前的方法是一个仿真闭环。第一步，在真实 ManiSkill 环境里运行 adapter。第二步，失败后自动诊断 layer、reason 和 repair hint。第三步，如果失败信息还不够，就运行 structured probe，在一个小范围参数空间里做真实仿真实验。第四步，把这些结构化证据放进下一轮 LLM prompt，让 LLM 生成新的 adapter。这里 probe 不是把答案告诉 LLM，而是把失败空间变成可以测量的数据，让下一轮修复少猜一点。

## 8. 当前实验设置与结果总览

目前主要有三个 case。Case 02 是 PullCube 从 Panda 到 xArm6，这是目前的正结果，seed 0 已经成功。Case 03 是 PickCube 从 Panda 到 xArm6，Panda baseline 成功，但 xArm6 target 目前是 hard case。Case 01 是 Panda 到 Fetch 的 PullCube，保留为诊断失败案例，因为它涉及移动底盘、9D action space 和接触侧可达性问题。我的主线是先巩固 PullCube 的正结果，同时用 PickCube 展示真实抓取迁移的瓶颈。

## 9. PullCube：已有正结果，但泛化不足

PullCube 的高层代码保持不变，就是获取 cube 和 goal，然后调用 `robot.pull(cube, goal)`。在 xArm6 上，LLM 生成的 adapter 在 seed 0 下可以成功，`ret_val=True`，大概用了 460 个环境步。这说明方法是可行的，至少对接触拖拽类任务，LLM 能生成一个可执行 adapter。但多 seed 评估发现只有 seeds 0、4、9 成功，其他 seed 多数在 approach 或 descent 阶段失败。这说明 seed 0 成功不是终点，adapter 对不同初始几何还不够鲁棒。

## 10. PullCube structured contact probing

为了解决多 seed 失败时 LLM 只能猜的问题，我把 PullCube 也接入了 structured contact probing。它默认扫 32 组参数，包括接触点距离、接触高度、接近高度、拖拽力度、下压力和分段数。每组都真实调用 `env.step(action)`，然后记录 task success、cube 是否向 goal 变近、TCP 是否到达接触点、TCP 离 cube 多远等指标。它本身不会直接提高成功率，但它能判断失败卡在接触点不可达、接触高度不对、拖拽力度不够，还是方块被推错方向。下一轮就是把 probe 结果反馈给 LLM，让它生成新的 adapter，再跑 multi-seed 验证。

## 11. PickCube：hard case 说明物理约束关键

PickCube 是目前更困难的任务。Panda baseline 已经验证成功，所以源端程序是有效的。但迁移到 xArm6 后，即使 TCP 有时已经接近目标抓取点，也经常无法形成稳定的 force-closure grasp。之前还出现过闭爪时把 cube 侧向推走的问题。后来我做了 32 组 close-envelope probe，包括 z offset、闭爪步数、闭爪命令等，但没有一组能形成稳定抓取。这说明 PickCube 不是简单再调几个参数就能解决，它暴露了真实抓取迁移里的接触和夹爪包络问题。

## 12. 相比 prompt-only，有什么提升？

相比一开始的 prompt-only 方式，现在系统有几个提升。失败反馈不再只是一段 message，而是五层诊断加 runtime metrics 和 probe table。Prompt 也不再无限变长，而是固定结构：固定约束、当前失败、repair hint 和 probe 反馈。参数搜索从手动试变成 structured probing，并且后续可以根据 score 生成下一批局部搜索建议。泛化判断也不再只看 seed 0，而是用 multi-seed 统计成功率和失败 seed cluster。这让项目从“调 prompt”变成了一个更系统的 repair loop。

## 13. 论文 / workshop framing

如果往论文或 workshop 方向组织，我觉得可以这样讲。问题是 LLM 迁移机器人程序时，失败经常来自 embodiment constraint，而不是代码生成错误。方法是生成 target-side adapter，并结合真实仿真失败分类、structured probing 和 compact retry prompt 来驱动修复。实验发现是：接触拖拽任务可以迁移成功，但真实抓取任务仍然困难；probe 能提供物理证据，但不等同于直接给答案。后续可以自然连接到 constraint-aware prompting、simulation-in-the-loop repair 和 learning-guided optimization。

## 14. 下一步与组会讨论点

接下来我想做四件事。第一，用 PullCube probe 修复多 seed 泛化问题。第二，跑新 adapter 的 0 到 9 seed 成功率。第三，把 PickCube 保留为 hard case，系统整理 failure taxonomy。第四，减少人工 prompt，不再无限追日志。也想请老师帮我判断两个问题：当前贡献更像工程系统，还是已经足够形成论文问题；以及下一步应该优先做 learning-guided optimization，还是更深入地建模 grasp/contact primitive。我的目标是把项目从“能跑起来”推进到“有可解释失败、有自动修复、有泛化评估”。

## 15. 现场 demo 讲法

如果现场要展示，我会只打开 simple harness demo，不跑完整长实验。先运行：

```bash
RUN_DIR=$(cat results/simple_demo/latest.txt)
cat "$RUN_DIR/agent_plan.json"
cat "$RUN_DIR/selected_tool_command.txt"
cat "$RUN_DIR/tool_result.json"
```

我会解释这三个文件：

```text
agent_plan.json：Agent 决定下一步调用 run_multi_seed；
selected_tool_command.txt：harness 暴露出的真实可执行命令；
tool_result.json：远程实际执行结果，executed=true 且 returncode=0。
```

这页的重点不是证明任务最终泛化成功，而是证明新的实验框架已经跑通：LLM/Agent 不再直接乱改代码，而是在一个安全的工具接口里根据仿真反馈做下一步决策。
