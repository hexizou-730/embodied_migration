# StackPyramid：复杂迁移任务

更新：2026-09-08。状态：代码已接入，Panda 和 Fetch 的 GPU 仿真结果均待验证。

## 做什么

先把红积木摆在绿积木旁边，形成底座；再把蓝积木叠在两者上面，松手并等待稳定。
它比单次抓取多了前后依赖：底座没摆好就不能继续，放置时也不能把底座撞散。

固定程序：

```python
red = scene.get_object("cubeA")
green = scene.get_object("cubeB")
blue = scene.get_object("cubeC")
base_ok = robot.prepare_base(red, green)
ret_val = robot.stack_on(blue, red, green) if base_ok else False
```

## 怎么运行

在远程项目目录、`em-ms` 环境下，先确认代码已同步、ManiSkill 和模型 API 可用。

```bash
python migrate.py --task stack --target fetch --mode agent
```

默认源机器人 Panda，目标 Fetch；最多 3 个循环，每轮最多 1 次生成。源/目标统一上限 800 步。
系统先备份当前目标模块、恢复中性起点，再检查源端。Panda 失败就停止，不把源端失败算成目标迁移失败。
通过后才调用 LLM，保存生成代码、测试输出、仿真反馈，并在预算内修复。
运行会消耗 API 和 GPU 资源。加 `--dry-run` 仅预览命令。

只验证当前 Fetch adapter：

```bash
python migrate.py --task stack --target fetch --mode evaluate
```

单独检查 Panda 基线，不调用 LLM：

```bash
python -m maniskill_backend.real_runner --task stack_pyramid --robot panda --method source-copy --max-episode-steps 800
```

## 看哪些结果

每次运行保存在 `results/migrations/<运行名>/`，`latest.txt` 指向最近目录。
`agent_loop/summary.json` 汇总最终状态，`cycle_XX/module_generation.jsonl` 保存源端结果、每轮代码及目标运行。

| 字段 | 用途 |
|---|---|
| `source_result` | Panda 是否完成，失败在哪个阶段 |
| `stage_trace` | 各阶段结束时的三块积木状态与到点误差 |
| `runtime_diagnostics` | 最后停在哪个阶段、是否超时、底座是否仍满足条件 |
| `official_evaluation` | 运行器独立读取官方 `evaluate()`，不只相信 adapter 返回值 |
| `attempts[].used_llm` / `module_kept` | 是否真正调用模型，候选是否保留 |

阶段包括 `base.approach/descent/close/lift/transport/lower/release/retreat/settle/verify`，
以及对应的 `top.*`。底座几何检查是执行前置条件，不替代官方成功判定。
观测还包含位置、四元数、抓取状态、静止状态、底座距离及顶部目标残差。

## 当前边界

- `stack_pyramid.py` 是手写、尚未经过 GPU 验证的源策略和公共接口。它不是 LLM 的成功结果。
- `seed_adapters/case06_fetch_stack_pyramid.py` 是中性起点，`generated_adapters/` 中对应文件才是运行时由 LLM 替换的目标模块。
- 当前控制模式固定 `pd_ee_delta_pos`，源策略没有主动调整积木姿态；随机旋转可能导致失败，需在源端验证阶段记录。
- 800 步是本项目显式设置的对照预算，官方默认是 250 步。修改预算必须对源端与目标端一致。
- adapter 在移动过程中每步读取 TCP、检查抓取，LLM 在试验结束后看阶段反馈并修复。尚未接入本任务的逐段在线 LLM 决策和 structured probing。
- 本地 143 项测试通过，其中使用简化环境的测试只验证程序逻辑，不代表真实物理成功率。
- 该案例不进入现有五案例论文矩阵。先取得源端基线和 Fetch 的真实日志，再冻结策略、补多 seed 统计与对照。

官方来源：[StackPyramid 源码](https://maniskill.readthedocs.io/en/latest/_modules/mani_skill/envs/tasks/tabletop/stack_pyramid.html)。
当前官方文档列出 Panda / panda_wristcam / Fetch；远程实际安装版本仍需 preflight 核验。
