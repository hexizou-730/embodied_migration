"""Task specifications and source programs for the current ManiSkill migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    display_name: str
    name_cn: str
    maniskill_env_id: str
    instruction: str
    instruction_cn: str
    source_robot: str
    target_robots: Tuple[str, ...]
    source_program: str
    expected_failure_modes: Tuple[str, ...]
    notes: str = ""

    def to_prompt_section(self) -> str:
        return "\n".join(
            [
                f"# Task: {self.task_id}",
                f"Task name: {self.display_name}",
                f"中文任务: {self.name_cn}",
                f"ManiSkill env: {self.maniskill_env_id}",
                f"Instruction: {self.instruction}",
                f"中文说明: {self.instruction_cn}",
                f"Source robot: {self.source_robot}",
                f"Target robots: {', '.join(self.target_robots)}",
                f"Expected failure modes: {', '.join(self.expected_failure_modes)}",
                "",
                "Source program:",
                "```python",
                self.source_program.strip(),
                "```",
            ]
        )


PULL_CUBE_SOURCE = """
cube = scene.get_object("cube")
goal = scene.get_region("goal")

ret_val = robot.pull(cube, goal)
"""


TASK_SPECS: Dict[str, TaskSpec] = {
    "pull_cube": TaskSpec(
        task_id="pull_cube",
        display_name="Pull cube to target",
        name_cn="拉方块到目标区域",
        maniskill_env_id="PullCube-v1",
        instruction="Pull the cube backward onto the target region.",
        instruction_cn="把方块向后拉到目标区域。",
        source_robot="panda",
        target_robots=("panda", "fetch"),
        source_program=PULL_CUBE_SOURCE,
        expected_failure_modes=(
            "contact execution failure",
            "reachability failure",
            "controller primitive failure",
            "task outcome failure",
        ),
        notes=(
            "Current primary migration task. PullCube-v1 is an official "
            "ManiSkill pulling/contact task supported by panda and fetch."
        ),
    ),
}


TASK_ALIASES: Dict[str, str] = {
    "PullCube-v1": "pull_cube",
    "pull-cube": "pull_cube",
    "pullcube": "pull_cube",
    "pull": "pull_cube",
}


def get_task_spec(task_id: str) -> TaskSpec:
    task_id = TASK_ALIASES.get(task_id, task_id)
    try:
        return TASK_SPECS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_SPECS))
        raise KeyError(f"Unknown task {task_id!r}. Available: {available}") from exc


def iter_task_specs() -> Iterable[TaskSpec]:
    return TASK_SPECS.values()
