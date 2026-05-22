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


STACK_CUBE_SOURCE = """
cube_a = scene.get_object("cubeA")
cube_b = scene.get_object("cubeB")

ok = robot.grasp(cube_a)
if ok:
    ret_val = robot.place(cube_a, cube_b)
else:
    ret_val = "failure: grasp"
"""


PULL_CUBE_TOOL_SOURCE = """
tool = scene.get_object("l_shape_tool")
cube = scene.get_object("cube")
workspace = scene.get_region("workspace")

ok = robot.hook_object(tool, cube)
if ok:
    ret_val = robot.pull_with_tool(tool, cube, workspace)
else:
    ret_val = "failure: hook"
"""


TASK_SPECS: Dict[str, TaskSpec] = {
    "pick_cube": TaskSpec(
        task_id="pick_cube",
        display_name="Pick and place cube",
        name_cn="抓取方块",
        maniskill_env_id="PickCube-v1",
        instruction="Pick up the cube and move it to the target goal position.",
        instruction_cn="抓起方块，并把它移动到目标位置。",
        source_robot="panda",
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
        source_robot="panda_wristcam",
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
    "stack_cube": TaskSpec(
        task_id="stack_cube",
        display_name="Stack cube A on cube B",
        name_cn="堆叠方块",
        maniskill_env_id="StackCube-v1",
        instruction="Pick up cube A and stack it on top of cube B.",
        instruction_cn="抓起 cube A，并把它稳定放到 cube B 上方。",
        source_robot="panda",
        target_robots=("panda", "xarm6_robotiq"),
        source_program=STACK_CUBE_SOURCE,
        expected_failure_modes=(
            "gripper/force failure",
            "reachability failure",
            "placement stability failure",
            "execution failure",
        ),
        notes="Supporting stacking task: official Panda solver succeeds at seed 0.",
    ),
    "pull_cube_tool": TaskSpec(
        task_id="pull_cube_tool",
        display_name="Pull cube with L-shaped tool",
        name_cn="用工具拉方块",
        maniskill_env_id="PullCubeTool-v1",
        instruction="Use the L-shaped tool to pull the cube back into the robot workspace.",
        instruction_cn="使用 L 形工具钩住方块，并把方块拉回机器人可达区域。",
        source_robot="panda",
        target_robots=("panda", "xarm6_robotiq"),
        source_program=PULL_CUBE_TOOL_SOURCE,
        expected_failure_modes=(
            "tool-use ordering failure",
            "tool-use execution failure",
            "gripper/force failure",
            "reachability failure",
            "execution failure",
        ),
        notes="Case 01 full-stack migration task: official Panda solver succeeds at seed 0.",
    ),
}


TASK_ALIASES: Dict[str, str] = {
    "PickCube-v1": "pick_cube",
    "pick-cube": "pick_cube",
    "pickcube": "pick_cube",
    "PegInsertionSide-v1": "peg_insertion",
    "peg-insertion": "peg_insertion",
    "peginsertion": "peg_insertion",
    "StackCube-v1": "stack_cube",
    "stack-cube": "stack_cube",
    "stackcube": "stack_cube",
    "PullCubeTool-v1": "pull_cube_tool",
    "pull-cube-tool": "pull_cube_tool",
    "pullcubetool": "pull_cube_tool",
    "pull_cube_too": "pull_cube_tool",
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
