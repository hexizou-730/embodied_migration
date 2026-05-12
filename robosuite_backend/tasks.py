"""Complex robosuite task specs used for source-program migration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class RobosuiteTask:
    name: str
    robosuite_env: str
    instruction: str
    success_criteria: str
    source_robot: str
    source_program: str
    recommended_targets: List[str]

    def describe(self) -> str:
        return (
            f"Task: {self.name}\n"
            f"Robosuite env: {self.robosuite_env}\n"
            f"Instruction: {self.instruction}\n"
            f"Success criteria: {self.success_criteria}"
        )


TASKS: Dict[str, RobosuiteTask] = {
    "two_arm_lift": RobosuiteTask(
        name="two_arm_lift",
        robosuite_env="TwoArmLift",
        instruction=(
            "Use both arms to grasp the two pot handles and lift the pot above "
            "the table while keeping it level."
        ),
        success_criteria=(
            "Both handles are grasped by different arms, the pot is lifted at "
            "least 12 cm, and the pot remains level."
        ),
        source_robot="rs_dual_panda",
        source_program="""left_ok = robot.grasp_pot_handle('left', 'left_handle')
right_ok = robot.grasp_pot_handle('right', 'right_handle')
if left_ok and right_ok:
    ret_val = robot.lift_pot(lift_height=0.16, keep_level=True)
else:
    ret_val = 'failure'""",
        recommended_targets=["rs_dual_iiwa", "rs_baxter", "rs_mobile_tiago"],
    ),
    "two_arm_handover": RobosuiteTask(
        name="two_arm_handover",
        robosuite_env="TwoArmHandover",
        instruction=(
            "Pick up the hammer with the nearer arm, hand it over to the other "
            "arm, then place it on the target region."
        ),
        success_criteria=(
            "The hammer changes from the pickup arm to the receiving arm and "
            "is finally placed on the target region."
        ),
        source_robot="rs_dual_panda",
        source_program="""pick_arm = robot.choose_arm_for('hammer')
other_arm = robot.other_arm(pick_arm)
picked = robot.pick_hammer(pick_arm)
if picked:
    handed = robot.handover_object(pick_arm, other_arm, object_name='hammer')
    if handed:
        ret_val = robot.place_hammer_on_target(other_arm)
    else:
        ret_val = 'failure'
else:
    ret_val = 'failure'""",
        recommended_targets=["rs_baxter", "rs_dual_iiwa"],
    ),
    "two_arm_peg_in_hole": RobosuiteTask(
        name="two_arm_peg_in_hole",
        robosuite_env="TwoArmPegInHole",
        instruction=(
            "Hold the board with one arm, grasp the peg with the other arm, "
            "align the peg to the square hole, and insert it slowly."
        ),
        success_criteria=(
            "One arm holds the board, the other holds the peg, the peg is aligned "
            "within tolerance, and insertion speed stays below the embodiment limit."
        ),
        source_robot="rs_dual_panda",
        source_program="""board_arm = 'left'
peg_arm = robot.other_arm(board_arm)
board_ok = robot.hold_board(board_arm)
peg_ok = robot.grasp_peg(peg_arm)
if board_ok and peg_ok:
    aligned = robot.align_peg_to_hole(tolerance=0.02)
    if aligned:
        ret_val = robot.insert_peg(speed=0.02)
    else:
        ret_val = 'failure'
else:
    ret_val = 'failure'""",
        recommended_targets=["rs_dual_iiwa", "rs_baxter"],
    ),
}


def get_task(name: str) -> RobosuiteTask:
    key = name.lower()
    if key not in TASKS:
        raise ValueError(f"Unknown robosuite task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[key]


def task_names() -> List[str]:
    return sorted(TASKS)
