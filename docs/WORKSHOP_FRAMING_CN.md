# Workshop Framing

## 暂定标题

**Counterexample-Guided Embodiment Adapter Synthesis for Fixed Robot Programs**

中文：**面向固定高层机器人程序的反例驱动 Embodiment Adapter 合成**

## 问题

同一个高层机器人程序在更换机器人后，往往不是语法错误，而是因为 action layout、
TCP/夹爪几何、可达空间、移动底盘和接触动力学发生变化。

本文研究一个受控问题：

```text
固定 Program
固定 Controller
固定 Simulator
固定 Success Criterion
只合成 Target Adapter
```

核心问题是：

> 能否利用仿真反例和显式 embodiment constraints，只修改 adapter，就把固定高层程序迁移到新机器人？

## 方法

系统包含四个核心组件：

1. Machine-readable embodiment contract；
2. 五层失败诊断和 structured counterexample；
3. Constraint-guided active physical probing；
4. Guarded target-adapter synthesis。

Development seeds 用于找反例和修复；held-out seeds 只用于最终验证。

## 与普通 LLM 修复的区别

普通流程：

```text
失败日志 -> 加长 prompt -> LLM 再猜一版代码
```

本文流程：

```text
失败 trial
-> 违反的机器人物理约束
-> 选择最有信息量的反例
-> 只测量相关参数
-> 生成带状态 guard 的 adapter
-> held-out 验证
```

## 预期贡献

- 提出固定 Program/Controller/Simulator/Success 下的 adapter-only 跨 embodiment 迁移协议；
- 提出结构化 embodiment contract 和物理反例表示；
- 用主动 probe 减少无关参数扫描和仿真次数；
- 合成可根据当前几何选择策略或声明不可行的 guarded adapter；
- 提供完整 provenance、development/held-out split 和 oracle 隔离协议。

## 当前实验范围

| 任务 | 源机器人 | 目标机器人 | 角色 |
|---|---|---|---|
| PullCube | Panda | xArm6 Robotiq | 主方法案例 |
| PickCube | Panda | xArm6 Robotiq | grasp/contact hard case |
| PullCube | Panda | Fetch | mobile-base/action-layout stress case |

Fetch hand-written oracle 只作为性能上界，不计为 LLM 成功。

## 关键对照

- source adapter copy；
- LLM one-shot；
- LLM + raw failure；
- LLM + failure taxonomy；
- LLM + fixed-grid probe；
- 完整 counterexample/contract/active-probe 方法；
- hand-written oracle upper bound。

## 关键指标

- held-out success rate；
- simulator calls；
- probe cases；
- repair cycles；
- environment steps；
- token/API cost；
- manual interventions；
- infeasibility accuracy。

## 可以安全声称的结论

在没有提交远程真实日志前，只能声称：

> 已实现一个完整、可测试、可 dry-run 的反例驱动 adapter synthesis 框架，并为三个注册迁移 case 提供了 contract、诊断和执行入口。

产生并提交 held-out 结果后，才可以声称具体机器人迁移成功率。

## 不应声称

- 任意任务和任意机器人都能自动迁移；
- simulator 成功等价于真实机器人成功；
- hand-written oracle 是 LLM-generated result；
- seed 0 成功代表跨 seed 泛化；
- fixed-grid probe 本身是新的优化算法。

## 下一版论文需要回答

1. Active probe 是否用更少仿真次数达到更高 held-out success？
2. Embodiment contract 是否减少 action/controller 接口错误？
3. Counterexample selection 是否优于使用最新失败或随机失败？
4. Guarded adapter 是否优于固定 seed 轨迹？
5. 系统能否正确识别当前 embodiment 下的不可行任务？
