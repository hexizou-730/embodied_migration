# t04 PickCube 源端阶段提取与从零生成冒烟

日期:2026-11-21。任务 PickCube-v1,源机器人 Panda(冻结程序 seeds 0–9 为
10/10,2026-09-18 签发),目标 xarm6_robotiq,pd_ee_delta_pose。模型为
DeepSeek 直连 deepseek-flash。环境、控制器与官方成功标准不变。

## 源端阶段表示(零 LLM)

- seed 0 源端重放官方成功,70 actions;source_action_trace 70/70 归入源码行。
- `stages.py` 提取三阶段 approach / grasp / align,与 study 配置的人工预期一致
  (valid=True);实体 cube / goal_site。
- 通用化后的接口在此任务首次实际使用;t02_pull_cube 因预闭合夹爪产生的
  [grasp, approach, align] 乱序已被判定为不适用,未纳入本目录。

## 从零生成冒烟(1 次,约 $0.002)

首轮即官方成功:24 actions(approach 1 / grasp 15 / align 8),官方
is_obj_placed / is_robot_static / is_grasped 全 True;1 轮、8,158 tokens。
生成的 Adapter 从实测状态推导预抓取位,有界循环全部带 early_stop,完成判据
为实测几何条件。通用化后的三阶段接口 prompt 通过检查:无 five/cubeA 残留,
study 配置声明的 stage_semantics(is_obj_placed 等)正常进入 prompt。

## 边界

- 一次从零成功不构成 t04 的成功率结论;从零稳定性需 n≥5 再定。
- PickCube 目标端成功不代表 StackCube 式难任务可从零稳定生成;两任务的难度
  差异是论文难度分层叙事的一部分。
- xarm6_robotiq 不在 PickCube-v1 支持列表内,与 StackCube 研究同样作为
  非原生目标处理。

原始重放轨迹:`t04_pick_cube_seed_000.json`;阶段程序:`t04_stage_program.json`;
生成与执行证据:`t04_selective_smoke/`(prompt、LLM 响应、trial、cycle 记录)。
