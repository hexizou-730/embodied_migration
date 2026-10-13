# 项目进度

核对日期：2026-10-13。当前代码基线：`e615968`。

## 已完成

1. 建立通用迁移 Harness：读取源程序和目标接口，调用 LLM 生成 Adapter，在
   ManiSkill 中执行真实 action，并把官方结果、物理轨迹和异常反馈给下一轮。
2. 建立严格验收条件：必须实际生成目标代码、执行至少一个目标 action、获得
   ManiSkill 官方成功信号，并保存 Adapter、环境、manifest 和哈希证据。
3. 修复两个关键执行问题：无动作循环会被中止；官方成功终止信号不会再被
   LMP 执行器误记为失败。
4. 冻结 `t01_push_cube` 的 Panda 源程序，开发 seeds 0-9 全部通过。
5. 完成一例 Panda 到 xArm6 的目标迁移回归。LLM 生成的 Adapter 在 seed 0
   执行 294 个动作，返回成功，ManiSkill 官方 `success=True`。

验收证据位于
[`evidence/t01_push_cube_panda_to_xarm6_20261013`](evidence/t01_push_cube_panda_to_xarm6_20261013)。

## 当前边界

- 这证明生成、执行、反馈和官方验证能够端到端连通，不证明任意任务或机器人
  都能自动迁移成功。
- 最新一次从零重新生成运行了 8 轮、122 个目标动作，仍因没有建立有效接触而
  失败。说明 Harness 已能工作，但 LLM 生成策略的稳定性还不够。
- 当前目标端结果只有一个 seed；还没有 held-out seeds、重复运行成功率或跨任务
  统计，因此暂时只能作为系统可行性证据，不能作为完整论文实验结论。

## 下一阶段

1. 固定当前成功 Adapter，在目标端运行多 seed，区分代码鲁棒性与生成随机性。
2. 对同一迁移重复从零生成，报告成功率、轮数、动作数、token 和费用。
3. 再冻结并迁移 PullCube、PickCube 等任务，形成任务难度和机器人差异对照。
4. 加入 one-shot、raw-log repair 和人工 Adapter 等基线，再开始论文主实验。

当前不继续无限增加 LLM 修复轮数，也不把失败运行改写为成功。
