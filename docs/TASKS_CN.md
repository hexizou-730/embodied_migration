# Task Design

目标不是只证明 source-copy 会失败，而是让不同信息条件产生可解释差异：

```text
source-copy       不做迁移，作为下界
llm_no_card       只给任务和源代码，不给目标机器人信息
llm_card_only     给 Capability Card，测试 embodiment prior 的作用
llm_report_only   给 Failure Report，测试失败反馈的作用
llm_card_report   同时给 Card + Report，测试组合效果
oracle            手写上界
```

## 当前有效信号

`PullCubeTool-v1 / so100` 已经能区分信息条件：

```text
source-copy      失败：没有先对齐 tool 和 cube
llm_no_card      失败：模型照抄了 source-copy 的顺序
llm_card_only    成功：从 Capability Card 读到 tool hook 需要 alignment
llm_report_only  成功：从 Failure Report 读到 hook_object 前必须 align
llm_card_report  成功：Card + Report 都可用
oracle           成功
```

这说明 Card 和 Report 已经开始有边际作用，不再只是 source-copy 一个组失败。

## 下一批复杂任务

优先做三类任务，每类都要让失败类型不同：

1. Tool-order task

   目标：测试动作顺序和工具使用。

   例子：`PullCubeTool-v1`。

   期望现象：

   ```text
   llm_no_card 失败
   llm_card_only 或 llm_report_only 成功
   ```

2. Contact-speed task

   目标：测试接触丰富任务里的速度和精度约束。

   例子：`PlugCharger-v1`。

   设计：

   ```text
   source code 先使用足够宽松的 alignment tolerance 完成粗对齐
   source code 再使用偏快 insertion speed
   target robot card 给出更慢 speed limit
   failure report 给出实际 contact-speed 失败原因
   ```

   期望现象：

   ```text
   llm_no_card 失败
   llm_card_only 成功或部分成功
   llm_report_only 成功
   ```

3. Multi-cause task

   目标：让一次迁移同时涉及 precision、speed、ordering 中至少两个问题。

   例子：`PegMulti-v1`、`PlugMulti-v1`。

   ```text
   coarse alignment tolerance 太紧
   fine alignment tolerance 也太紧
   insert speed 太快
   pre-alignment / seating alignment 两阶段都要满足经验余量
   ```

   期望现象：

   ```text
   llm_card_only 可能只修一个问题
   llm_card_report 更容易一次修全
   ```

## 判断标准

一个任务值得保留，至少要满足其中一个条件：

```text
llm_no_card < llm_card_only
llm_no_card < llm_report_only
llm_card_only < llm_card_report
```

其中 `<` 表示成功率更低，或者失败类型更严重。

如果所有 LLM 方法都成功，这个任务只能作为 pipeline demo，不适合作为主实验任务。

## Report Strength

Failure Report 不能直接给最终答案数值，否则 `llm_report_only` 会太强，体现不出 `Card + Report` 的互补关系。

Report 只提供：

```text
失败发生在哪一步
是 alignment / speed / ordering 哪类问题
是否所有同类调用都要一起修
是否需要 nominal card 之外的经验余量
经验规则，例如 nominal tolerance + 0.010 m，或 speed limit * 75%
```

Report 不直接提供：

```text
最终 tolerance 数值
最终 speed 数值
目标机器人的完整 Capability Card
```

这样预期关系更清楚：

```text
card_only     有 nominal 参数，但不知道任务需要经验余量
report_only   知道失败模式和经验规则，但缺少目标机器人的 nominal 参数
card_report   同时具备 nominal 参数和失败反馈，应该最稳
```
