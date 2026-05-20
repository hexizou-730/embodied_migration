"""Task specifications and source programs for real ManiSkill trials."""

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


PICK_CUBE_SOURCE = """
cube = scene.get_object("cube")
goal = scene.get_region("goal")

ok = robot.grasp(cube)
if ok:
    ret_val = robot.place(cube, goal)
else:
    ret_val = "failure: grasp"
"""


PEG_INSERTION_SOURCE = """
peg = scene.get_object("peg")
hole = scene.get_object("hole")

ok = robot.grasp(peg)
if ok:
    aligned = robot.align_to_target(peg, hole, tolerance=0.01)
    if aligned:
        ret_val = robot.insert(peg, hole, speed=0.015)
    else:
        ret_val = "failure: alignment"
else:
    ret_val = "failure: grasp"
"""


TASK_SPECS: Dict[str, TaskSpec] = {
    "pick_cube": TaskSpec(
        task_id="pick_cube",
        display_name="Pick and place cube",
        name_cn="抓取方块",
        maniskill_env_id="PickCube-v1",
        instruction="Pick up the cube and move it to the target goal position.",
        instruction_cn="抓起方块，并把它移动到目标位置。",
        source_robot="panda_wristcam",
        target_robots=("panda", "xarm6_robotiq"),
        source_program=PICK_CUBE_SOURCE,
        expected_failure_modes=(
            "gripper/force failure",
            "reachability failure",
            "execution failure",
        ),
        notes="Smoke and controller-portability task for real ManiSkill simulation.",
    ),
    "peg_insertion": TaskSpec(
        task_id="peg_insertion",
        display_name="Side peg insertion",
        name_cn="侧向插 peg",
        maniskill_env_id="PegInsertionSide-v1",
        instruction="Insert the peg into the side hole.",
        instruction_cn="抓住 peg，并从侧面插入孔中。",
        source_robot="panda",
        target_robots=("panda", "xarm6_robotiq"),
        source_program=PEG_INSERTION_SOURCE,
        expected_failure_modes=(
            "reachability failure",
            "alignment failure",
            "insertion speed failure",
            "precision failure",
        ),
        notes="First contact-rich real ManiSkill task after PickCube.",
    ),
}


TASK_ALIASES: Dict[str, str] = {
    "PickCube-v1": "pick_cube",
    "pick-cube": "pick_cube",
    "pickcube": "pick_cube",
    "PegInsertionSide-v1": "peg_insertion",
    "peg-insertion": "peg_insertion",
    "peginsertion": "peg_insertion",
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
