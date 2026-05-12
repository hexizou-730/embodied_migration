"""Prompt builder for source-to-target robosuite program migration."""
from __future__ import annotations

from typing import Optional

from lmp.failure_report import FailureReport
from robosuite_backend.profiles import RobosuiteProfile
from robosuite_backend.symbolic import RobosuiteSkillRobot, RobosuiteSymbolicScene
from robosuite_backend.tasks import RobosuiteTask


SYSTEM_PROMPT = """You are a robot program migration assistant.
You receive a source robot program that succeeded on a source embodiment and a
target robot capability card. Your job is to output a short Python snippet that
uses ONLY the target robot APIs to accomplish the same task on the target robot.

Rules:
  - Output ONLY code inside ```python ... ``` fences. No prose.
  - Do NOT use import. numpy is already available as np.
  - Do not invent APIs. Use only APIs listed under Available Target APIs.
  - Check boolean return values from robot APIs.
  - Set ret_val = 'success' or True only if the task succeeds.
  - If the target capability card says the task is impossible, do not move;
    set ret_val to a short refusal string such as 'refuse_requires_dual_arm'.
  - If a Failure Report is present, fix the specific issue in the previous code.
"""


def build_migration_prompt(
    task: RobosuiteTask,
    source_profile: RobosuiteProfile,
    target_profile: RobosuiteProfile,
    target_robot: RobosuiteSkillRobot,
    target_scene: RobosuiteSymbolicScene,
    source_code: str,
    use_capability_card: bool = True,
    failure_report: Optional[FailureReport] = None,
) -> str:
    parts = [
        "## Migration Task",
        task.describe(),
        "",
        "## Source Robot",
        source_profile.describe(),
        "",
        "## Source Successful Program",
        "```python",
        source_code.strip(),
        "```",
        "",
        "## Target Robot",
        target_profile.describe(),
        "",
        "## Available Target APIs",
        target_robot.available_api_prompt(),
        "",
        "## Target Scene",
        target_scene.describe(),
    ]
    if use_capability_card:
        parts[5:5] = [
            "",
            "## Source Capability Card",
            source_profile.to_prompt_section(),
        ]
        insert_at = parts.index("## Available Target APIs")
        parts[insert_at:insert_at] = [
            "",
            "## Target Capability Card",
            target_profile.to_prompt_section(),
        ]
    if failure_report is not None:
        parts.extend(["", f"## {failure_report.to_prompt_section()}"])
    parts.extend(["", "Now output the migrated target program:"])
    return "\n".join(parts)
