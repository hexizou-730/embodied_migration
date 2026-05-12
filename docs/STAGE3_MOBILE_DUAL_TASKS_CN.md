# 阶段 3: Mobile + Dual-arm 任务集

阶段 3 的目标是让任务本身体现具身差异, 而不只是重复 “put red block into tray”。
当前 benchmark 默认对比 `mobile` 和 `dual_arm`, 默认任务集为 `migration`。

## 新增任务族

### mobility

这类任务主要考察移动底盘带来的代码迁移:

- `wide_blue_to_tray`
  - 指令: 将 blue block 放进 yellow tray。
  - 难点: 物体与托盘在桌面上相隔较远, mobile 需要先选择合适停车点再抓取。
  - 成功判定: blue block 与 yellow tray 的水平距离小于 0.15m。

- `collect_red_and_blue_to_tray`
  - 指令: 将 red block 和 blue block 都放入 yellow tray。
  - 难点: 多物体顺序操作, 需要检查每一步动作返回值, 并使用低释放高度。
  - 成功判定: red/blue 两个方块都靠近 yellow tray。

### bimanual

这类任务主要考察双臂能力, mobile 单臂通常无法完成最终物理状态:

- `hold_red_while_place_green`
  - 指令: 一只手保持 red block 悬空, 另一只手将 green block 放入 yellow tray。
  - 难点: 需要同时使用左右臂, 而不是简单顺序 pick-place。
  - 成功判定: red block 仍悬空, green block 在 tray 附近。

- `lift_red_and_green_together`
  - 指令: 同时举起 red block 和 green block, 每只手一个物体。
  - 难点: 单臂机器人无法在最终状态同时保持两个物体悬空。
  - 成功判定: red/green 两个方块都高于桌面。

## 推荐运行命令

只跑阶段 3 的 Mobile + Dual-arm 任务:

```bash
conda activate em
python -m benchmark.run_benchmark \
  --robots mobile dual_arm \
  --tasks migration \
  --modes api fewshot card failure card_failure \
  --trials 1 \
  --run-id stage3_mobile_dual
```

只跑移动能力任务:

```bash
python -m benchmark.run_benchmark --robots mobile dual_arm --tasks mobility --modes card_failure --trials 1
```

只跑双臂任务:

```bash
python -m benchmark.run_benchmark --robots mobile dual_arm --tasks bimanual --modes card_failure --trials 1
```

生成统计表:

```bash
python -m benchmark.analyze_results results/runs/stage3_mobile_dual
```

## 预期观察

- `mobile` 在 mobility 任务中应该倾向生成 `navigate_to` / `is_reachable` 相关代码。
- `dual_arm` 在 bimanual 任务中应该倾向生成 `pick_with_arm`, `place_with_arm`, `choose_arm_for`, `robot.left`, `robot.right` 等代码。
- `mobile` 在 bimanual 任务中即使动作 API 返回成功, 最终 checker 也应失败, 因为单臂无法同时保持两个物体悬空。
- 分析表中的 generated-code features 会记录 mobile 和 dual-arm 相关 API 是否被使用。
