"""
Failure Report: 方法 B (Failure-Driven Code Rewriting) 的核心数据结构。

当代码在目标机器人上执行失败时, 采集:
- 任务描述
- 期望的物理状态 (expected)
- 实际的物理状态 (actual)
- 偏差分析 (diagnosis)
- 可行的修正建议 (suggestions)

这些信息被结构化地喂回给 LLM, 让 LLM 基于「具体失败原因」而不是「语言模糊描述」
来修正代码 —— 这就是 failure-driven 的含义。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FailureReport:
    task_name: str
    instruction: str
    robot_name: str

    # 物理状态比较 (dict, 因为不同任务关心不同字段)
    expected: Dict[str, object] = field(default_factory=dict)
    actual: Dict[str, object] = field(default_factory=dict)

    # 代码执行情况
    code_raised: bool = False                 # 代码是否抛异常
    traceback: Optional[str] = None           # 如果抛了, 完整 traceback

    # 诊断与建议 (框架自动填 + LLM 自推理)
    diagnosis: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_prompt_section(self) -> str:
        """把失败报告格式化成 LLM 可读的 prompt 片段。"""
        lines = [
            "# Failure Report (previous attempt failed)",
            f"Task: {self.task_name}",
            f"Instruction: {self.instruction!r}",
            f"Robot: {self.robot_name}",
            "",
        ]

        if self.code_raised:
            lines.append("[Code Execution]")
            lines.append("  Code raised an exception:")
            tb = (self.traceback or "").strip()
            for tline in tb.splitlines()[-6:]:  # 只保留最后 6 行 traceback
                lines.append(f"    {tline}")
            lines.append("")
        else:
            lines.append("[Code Execution]  OK (no exception)")
            lines.append("")

        if self.expected or self.actual:
            lines.append("[Physical State Comparison]")
            keys = sorted(set(self.expected) | set(self.actual))
            for k in keys:
                exp = self.expected.get(k, "<n/a>")
                act = self.actual.get(k, "<n/a>")
                marker = "  ✓" if exp == act else "  ✗"
                lines.append(f"{marker} {k}:")
                lines.append(f"      expected = {exp}")
                lines.append(f"      actual   = {act}")
            lines.append("")

        if self.diagnosis:
            lines.append("[Diagnosis]")
            for d in self.diagnosis:
                lines.append(f"  - {d}")
            lines.append("")

        if self.suggestions:
            lines.append("[Suggestions to LLM]")
            for s in self.suggestions:
                lines.append(f"  - {s}")
            lines.append("")

        lines.append(
            "Please output a CORRECTED code snippet that fixes the above issues. "
            "Think about what physical parameter (release height, approach angle, "
            "hover time, etc.) needs to change."
        )
        return "\n".join(lines)


def build_failure_report(
    task_name: str,
    instruction: str,
    robot_name: str,
    expected: Dict[str, object],
    actual: Dict[str, object],
    code_raised: bool = False,
    traceback_str: Optional[str] = None,
) -> FailureReport:
    """便捷构造函数: 根据 expected/actual 自动填一些 diagnosis 和 suggestions。"""
    report = FailureReport(
        task_name=task_name,
        instruction=instruction,
        robot_name=robot_name,
        expected=dict(expected),
        actual=dict(actual),
        code_raised=code_raised,
        traceback=traceback_str,
    )

    # 自动诊断: 比较 position_xyz 这类常见字段
    for k in expected:
        if k not in actual:
            continue
        exp_v, act_v = expected[k], actual[k]
        # 如果都是 (x, y, z) 元组, 算 delta
        if (isinstance(exp_v, (tuple, list)) and isinstance(act_v, (tuple, list))
                and len(exp_v) == len(act_v) == 3):
            dx = act_v[0] - exp_v[0]
            dy = act_v[1] - exp_v[1]
            dz = act_v[2] - exp_v[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist > 0.02:
                report.diagnosis.append(
                    f"'{k}' deviates by {dist * 100:.1f}cm from expected "
                    f"(delta: Δx={dx:+.3f}, Δy={dy:+.3f}, Δz={dz:+.3f})."
                )
                # 基于偏差方向的启发式建议
                if abs(dz) > max(abs(dx), abs(dy)) and dz < -0.01:
                    report.suggestions.append(
                        "Object ended up lower than expected — likely fell during release. "
                        "Try reducing pre_release_height in place()."
                    )
                if max(abs(dx), abs(dy)) > 0.03:
                    report.suggestions.append(
                        "Significant horizontal drift — object may have rolled on landing. "
                        "Try lowering release height or holding longer before release."
                    )

    if code_raised:
        report.diagnosis.append("Code raised an exception during execution; check the traceback.")
        report.suggestions.append(
            "Ensure the code follows the provided API signatures and does not use forbidden imports."
        )

    return report
