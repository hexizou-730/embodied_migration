# StackCube 抓取诊断与接口纠正

后续放置诊断见 `../stackcube_placement_20261023/README.md`：手工只修对齐阶段已获
官方完整任务成功；该结果与本文的 LLM 生成失败分开记录。

日期：2026-10-23。任务为 StackCube-v1，目标为 xarm6_robotiq，控制模式
pd_ee_delta_pose，所有运行使用 seed 0。环境、控制器和官方成功标准保持原样。

## 确认的两个问题

1. 旧 LLM 代码用 `_move_towards(self._tcp_pos(), gripper=close, ...)` 闭爪。
   该 helper 在到达位置时直接返回，因此原抓取阶段没有发送静止闭爪动作。
2. 原实验接口把开闭方向写反。自由空间各执行 20 步，命令 0 的指垫间距为
   92.36 mm，命令 0.81 的间距为 6.72 mm。因此本模式下 0 张开，0.81 闭合。

原始测量见 `results/grasp_verified/gripper_endpoints.json`；安装版本和机器人实现
快照见同目录的 `runtime_metadata.json` 与 `installed_xarm6_robotiq.py`。
间距是两指垫 link 原点距离，不是精确内表面净宽。

## 无 LLM 的诊断对照

`grasp_probe.py` 从旧候选生成诊断副本，每次重置到同一 seed，记录实际 action、
指垫间距、接触力、官方抓取信号和提升高度。修改来源是人工诊断，LLM 调用为 0。

| 条件 | 配置中的“close”动作数 | 官方抓取 | 提升后保持抓取 | 方块提升 |
|---|---:|---|---|---:|
| 旧代码、旧方向 | 0 | False | False | 0 |
| 仅改为实际执行静止命令、旧方向 | 72 | False | False | 0 |
| 静止命令和开闭方向均纠正 | 12 | True | True | 80.15 mm |

第二行的配置仍将 0 标为 close，但它实际是张开命令；这些次数不能解释为物理闭爪次数。
额外测试抓取高度偏移 +12 mm 和 +24 mm，均保持抓取并提升约 8 cm。
这些是同一 seed 上的参数诊断，不能写成独立场景成功率或完整堆叠成功。
每组 trial、生成的诊断副本和 action 列表位于 `results/grasp_verified/`。

## 纠正说明后，由 LLM 重新生成

仅向生成器纠正接口开闭方向和 helper 调用语义，没有把诊断副本当作迁移答案输入。
使用 DeepSeek v4.1 Flash，最多三轮；实际 Adapter 保存在各 cycle 目录。

| 轮次 | 校验 | action | 程序判定已完成阶段 | 停止位置 |
|---:|---|---:|---|---|
| 1 | 通过 | 27 | approach、grasp、lift | align |
| 2 | 通过 | 101 | approach、grasp、lift、align | release |
| 3 | 拒绝 | 0 | 沿用此前候选 | 只有注释变化，没有实质修复 |

第二轮只有 `_stage_align` 的 AST 哈希变化；其余阶段保持不变。三次 LLM 请求共计
$0.00495675。官方 StackCube success 仍为 False，释放后 cubeA 掉回桌面。
阶段通过由生成代码自己的判据给出，因此“align 返回成功”不代表其物理判据充分。
下一步需要检查持物 TCP 偏移、对齐容差和释放时碰撞，并验证失败是否源于上一阶段。

完整记录见 `results/selective_migration/corrected_gripper/summary.json`。
旧三方法 pilot 使用了错误的开闭方向，已标为不适合方法有效性对照。
纠正后的三方法比较尚未运行。
