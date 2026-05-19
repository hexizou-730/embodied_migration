"""Task specifications and source programs for the first research slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    maniskill_env_id: str
    instruction: str
    source_robot: str
    target_robots: Tuple[str, ...]
    source_program: str
    expected_failure_modes: Tuple[str, ...]
    notes: str = ""

    def to_prompt_section(self) -> str:
        return "\n".join(
            [
                f"# Task: {self.task_id}",
                f"ManiSkill env: {self.maniskill_env_id}",
                f"Instruction: {self.instruction}",
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


PLUG_CHARGER_SOURCE = """
charger = scene.get_object("charger")
socket = scene.get_object("socket")

ok = robot.grasp(charger)
if ok:
    aligned = robot.align_to_target(charger, socket, tolerance=0.04)
    if aligned:
        ret_val = robot.insert(charger, socket, speed=0.012)
    else:
        ret_val = "failure: alignment"
else:
    ret_val = "failure: grasp"
"""


PLUG_MULTI_SOURCE = """
charger = scene.get_object("charger")
socket = scene.get_object("socket")

ok = robot.grasp(charger)
if ok:
    prealign = robot.align_to_target(charger, socket, tolerance=0.012)
    if prealign:
        seated = robot.align_to_target(charger, socket, tolerance=0.010)
        if seated:
            ret_val = robot.insert(charger, socket, speed=0.014)
        else:
            ret_val = "failure: seating alignment"
    else:
        ret_val = "failure: prealign"
else:
    ret_val = "failure: grasp"
"""


PULL_CUBE_TOOL_SOURCE = """
tool = scene.get_object("tool")
cube = scene.get_object("cube")
target = scene.get_region("goal")

ok = robot.grasp(tool)
if ok:
    hooked = robot.hook_object(tool, cube)
    if hooked:
        ret_val = robot.pull_with_tool(tool, cube, target)
    else:
        ret_val = "failure: tool alignment"
else:
    ret_val = "failure: grasp tool"
"""


PEG_MULTI_SOURCE = """
peg = scene.get_object("peg")
hole = scene.get_object("hole")

ok = robot.grasp(peg)
if ok:
    coarse = robot.align_to_target(peg, hole, tolerance=0.01)
    if coarse:
        fine = robot.align_to_target(peg, hole, tolerance=0.006)
        if fine:
            ret_val = robot.insert(peg, hole, speed=0.02)
        else:
            ret_val = "failure: fine alignment"
    else:
        ret_val = "failure: coarse alignment"
else:
    ret_val = "failure: grasp"
"""


TASK_SPECS: Dict[str, TaskSpec] = {
    "PickCube-v1": TaskSpec(
        task_id="PickCube-v1",
        maniskill_env_id="PickCube-v1",
        instruction="Pick up the cube and move it to the target goal position.",
        source_robot="panda",
        target_robots=("fetch", "xarm6_robotiq", "so100", "widowxai"),
        source_program=PICK_CUBE_SOURCE,
        expected_failure_modes=(
            "gripper/force failure",
            "reachability failure",
            "execution failure",
        ),
        notes="First task intended for the real ManiSkill skill adapter.",
    ),
    "PegInsertionSide-v1": TaskSpec(
        task_id="PegInsertionSide-v1",
        maniskill_env_id="PegInsertionSide-v1",
        instruction="Insert the peg into the side hole.",
        source_robot="panda",
        target_robots=("fetch", "xarm6_robotiq", "so100", "widowxai"),
        source_program=PEG_INSERTION_SOURCE,
        expected_failure_modes=(
            "reachability failure",
            "alignment failure",
            "insertion speed failure",
            "precision failure",
        ),
        notes="First formal task after PickCube smoke tests.",
    ),
    "PlugCharger-v1": TaskSpec(
        task_id="PlugCharger-v1",
        maniskill_env_id="PlugCharger-v1",
        instruction="Plug the charger into the socket.",
        source_robot="panda",
        target_robots=("fetch", "xarm6_robotiq", "so100", "widowxai"),
        source_program=PLUG_CHARGER_SOURCE,
        expected_failure_modes=(
            "contact-rich insertion failure",
            "insertion speed failure",
            "alignment failure",
        ),
    ),
    "PlugMulti-v1": TaskSpec(
        task_id="PlugMulti-v1",
        maniskill_env_id="PlugCharger-v1",
        instruction="Plug the charger after pre-alignment and seating alignment.",
        source_robot="panda",
        target_robots=("fetch", "xarm6_robotiq", "so100", "widowxai"),
        source_program=PLUG_MULTI_SOURCE,
        expected_failure_modes=(
            "multi-cause contact failure",
            "alignment failure",
            "insertion speed failure",
        ),
        notes="Static multi-cause variant of PlugCharger-v1.",
    ),
    "PullCubeTool-v1": TaskSpec(
        task_id="PullCubeTool-v1",
        maniskill_env_id="PullCubeTool-v1",
        instruction="Use the tool to pull the cube to the target region.",
        source_robot="panda",
        target_robots=("fetch", "xarm6_robotiq", "so100", "widowxai"),
        source_program=PULL_CUBE_TOOL_SOURCE,
        expected_failure_modes=("tool-use ordering failure", "reachability failure"),
    ),
    "PegMulti-v1": TaskSpec(
        task_id="PegMulti-v1",
        maniskill_env_id="PegInsertionSide-v1",
        instruction="Insert the peg after coarse and fine alignment checks.",
        source_robot="panda",
        target_robots=("fetch", "xarm6_robotiq", "so100", "widowxai"),
        source_program=PEG_MULTI_SOURCE,
        expected_failure_modes=(
            "multi-cause failure",
            "alignment failure",
            "insertion speed failure",
        ),
        notes="Static multi-cause variant of PegInsertionSide-v1.",
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_SPECS))
        raise KeyError(f"Unknown task {task_id!r}. Available: {available}") from exc


def iter_task_specs() -> Iterable[TaskSpec]:
    return TASK_SPECS.values()
