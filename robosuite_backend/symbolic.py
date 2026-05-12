"""High-level robosuite skill layer for LMP execution.

This module provides executable task-level APIs for complex robosuite tasks.
The APIs are intentionally high-level: the research question here is whether an
LLM can migrate robot programs across embodiment capability differences, not
whether it can synthesize continuous MuJoCo torques from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from robosuite_backend.profiles import RobosuiteProfile
from robosuite_backend.tasks import RobosuiteTask


@dataclass
class RobosuiteSymbolicScene:
    task: RobosuiteTask
    state: Dict[str, object] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.events.clear()
        self.state = {
            "navigated_station": None,
            "held_by": {},
            "grip_force": 0.5,
            "pot_height": 0.0,
            "pot_level": False,
            "pot_lifted": False,
            "handover_pose_ready": False,
            "hammer_handed_over": False,
            "hammer_on_target": False,
            "board_held": False,
            "peg_held": False,
            "peg_aligned": False,
            "peg_inserted": False,
        }

    def describe(self) -> str:
        return (
            f"{self.task.describe()}\n"
            "Objects:\n"
            "  - pot with left_handle and right_handle\n"
            "  - hammer\n"
            "  - board with square_hole\n"
            "  - peg\n"
            "  - target region"
        )

    def get_object_names(self) -> List[str]:
        return ["pot", "left_handle", "right_handle", "hammer", "board", "square_hole", "peg", "target region"]

    def get_object_position(self, name: str) -> np.ndarray:
        positions = {
            "pot": np.array([0.0, 0.0, 0.85]),
            "left_handle": np.array([0.0, -0.18, 0.85]),
            "right_handle": np.array([0.0, 0.18, 0.85]),
            "hammer": np.array([0.15, -0.12, 0.82]),
            "board": np.array([0.0, 0.10, 0.88]),
            "square_hole": np.array([0.0, 0.10, 0.92]),
            "peg": np.array([0.20, -0.12, 0.82]),
            "target region": np.array([0.25, 0.18, 0.82]),
        }
        return positions.get(name, np.zeros(3)).copy()

    def log(self, message: str) -> None:
        self.events.append(message)
        print(f"  [robosuite] {message}")

    def check_success(self) -> Tuple[bool, Dict[str, object], Dict[str, object]]:
        task_name = self.task.name
        if task_name == "two_arm_lift":
            success = bool(
                self.state["pot_lifted"]
                and self.state["pot_level"]
                and float(self.state["pot_height"]) >= 0.12
            )
            expected = {"pot_lifted": True, "pot_level": True, "min_height_m": 0.12}
            actual = {
                "pot_lifted": self.state["pot_lifted"],
                "pot_level": self.state["pot_level"],
                "pot_height_m": round(float(self.state["pot_height"]), 3),
                "held_by": dict(self.state["held_by"]),
            }
            if self.state.get("real_control_enabled"):
                actual.update({
                    "real_control_enabled": True,
                    "real_physical_success": bool(self.state.get("real_physical_success", False)),
                    "real_pot_height_m": round(float(self.state.get("real_pot_height_m", 0.0)), 3),
                    "real_controller_reached_handles": bool(
                        self.state.get("real_controller_reached_handles", False)
                    ),
                })
            return success, expected, actual

        if task_name == "two_arm_handover":
            success = bool(self.state["hammer_handed_over"] and self.state["hammer_on_target"])
            expected = {"hammer_handed_over": True, "hammer_on_target": True}
            actual = {
                "hammer_handed_over": self.state["hammer_handed_over"],
                "hammer_on_target": self.state["hammer_on_target"],
                "held_by": dict(self.state["held_by"]),
                "handover_pose_ready": self.state["handover_pose_ready"],
            }
            if self.state.get("real_control_enabled"):
                actual.update({
                    "real_control_enabled": True,
                    "real_hammer_picked": bool(self.state.get("real_hammer_picked", False)),
                    "real_hammer_handed_over": bool(self.state.get("real_hammer_handed_over", False)),
                    "real_hammer_on_target": bool(self.state.get("real_hammer_on_target", False)),
                    "real_hammer_height_m": round(float(self.state.get("real_hammer_height_m", 0.0)), 3),
                    "real_hammer_target_dist_m": round(float(self.state.get("real_hammer_target_dist_m", 0.0)), 3),
                    "real_handover_pose_ready": bool(self.state.get("real_handover_pose_ready", False)),
                })
            return success, expected, actual

        if task_name == "two_arm_peg_in_hole":
            success = bool(self.state["board_held"] and self.state["peg_held"] and self.state["peg_inserted"])
            expected = {"board_held": True, "peg_held": True, "peg_inserted": True}
            actual = {
                "board_held": self.state["board_held"],
                "peg_held": self.state["peg_held"],
                "peg_aligned": self.state["peg_aligned"],
                "peg_inserted": self.state["peg_inserted"],
            }
            if self.state.get("real_control_enabled"):
                actual.update({
                    "real_control_enabled": True,
                    "real_board_held": bool(self.state.get("real_board_held", False)),
                    "real_peg_held": bool(self.state.get("real_peg_held", False)),
                    "real_peg_aligned": bool(self.state.get("real_peg_aligned", False)),
                    "real_peg_inserted": bool(self.state.get("real_peg_inserted", False)),
                    "real_peg_align_d": round(float(self.state.get("real_peg_align_d", 1.0)), 4),
                    "real_peg_align_t": round(float(self.state.get("real_peg_align_t", 0.0)), 4),
                    "real_peg_align_cos": round(float(self.state.get("real_peg_align_cos", 0.0)), 4),
                })
            return success, expected, actual

        return False, {"known_task": True}, {"known_task": False, "task": task_name}


class RobosuiteSkillRobot:
    """Executable high-level skill interface for complex robosuite tasks."""

    def __init__(self, profile: RobosuiteProfile, scene: RobosuiteSymbolicScene):
        self.profile = profile
        self.scene = scene
        self.embodiment_name = profile.display_name
        self.dof = profile.dof
        self.gripper_type = profile.gripper_type
        self.capability_card = profile.capability_card()
        self.left = self
        self.right = self
        self.reset_action_log()

    def reset_action_log(self) -> None:
        self.action_failures = []

    def get_action_failures(self) -> list:
        return list(getattr(self, "action_failures", []))

    def _record_action_failure(self, message: str) -> bool:
        if not hasattr(self, "action_failures"):
            self.action_failures = []
        self.action_failures.append(message)
        return False

    def _fail_action(self, message: str) -> bool:
        print(f"  ❌ {message}")
        return self._record_action_failure(message)

    def describe(self) -> str:
        return self.profile.describe()

    # BaseRobot abstract methods. Complex robosuite migration should use the
    # high-level APIs below rather than raw end-effector control.
    def get_ee_pose(self):
        return np.zeros(3), np.array([0, 0, 0, 1])

    def move_ee_to(self, position, orientation=None, steps: int = 240) -> bool:
        return self._fail_action("move_ee_to is not part of the robosuite high-level task API")

    def activate_gripper(self) -> bool:
        return self._fail_action("activate_gripper is too low-level for this robosuite task; use task skills")

    def release_gripper(self) -> None:
        self.scene.log("release_gripper ignored; use task-level release/place APIs")

    def available_api_prompt(self) -> str:
        return """# Robosuite high-level migration APIs:
robot.navigate_to_station(station_name: str) -> bool
    # Mobile profiles should navigate before interacting. Fixed profiles should not call it.
robot.set_grip_force(force: float) -> bool
    # Some grippers require force >= capability_card.extra['required_grip_force'] for pot lifting.
robot.choose_arm_for(object_name: str) -> str
robot.other_arm(arm_name: str) -> str

# TwoArmLift skills:
robot.grasp_pot_handle(arm_name: str, handle_name: str) -> bool
robot.lift_pot(lift_height: float = 0.16, keep_level: bool = True) -> bool

# TwoArmHandover skills:
robot.pick_hammer(arm_name: str = None) -> bool
robot.move_to_handover_pose(clearance: float = None) -> bool
robot.handover_object(from_arm: str, to_arm: str, object_name: str = 'hammer') -> bool
robot.place_hammer_on_target(arm_name: str) -> bool
# Note: 'target_region' is defined as a fixed table-top location reachable by the
# hammer-holding arm. The placement skill puts the hammer onto that location and
# requires |hammer_pos - target| < 0.08 m for physical success when --real-control
# is on.

# TwoArmPegInHole skills:
robot.hold_board(arm_name: str) -> bool
robot.grasp_peg(arm_name: str) -> bool
robot.align_peg_to_hole(tolerance: float = 0.02) -> bool
robot.insert_peg(speed: float = 0.02) -> bool

# Success convention:
# Set ret_val = 'success' or ret_val = True only when all required API calls succeed.
"""

    def _valid_arm(self, arm_name: str) -> bool:
        if arm_name not in self.profile.arm_names:
            return self._fail_action(f"invalid arm '{arm_name}', available arms={list(self.profile.arm_names)}")
        return True

    def _require_dual_arm(self, skill_name: str) -> bool:
        if len(self.profile.arm_names) < 2:
            return self._fail_action(f"{skill_name}: requires two arms, but this embodiment has {len(self.profile.arm_names)} arm")
        return True

    def _require_navigation_if_needed(self, station: str) -> bool:
        if self.profile.requires_navigation and self.scene.state.get("navigated_station") != station:
            return self._fail_action(f"must call robot.navigate_to_station('{station}') before interacting")
        return True

    def navigate_to_station(self, station_name: str) -> bool:
        if not self.profile.has_mobile_base:
            return self._fail_action("navigate_to_station called on a fixed-base robot")
        self.scene.state["navigated_station"] = station_name
        self.scene.log(f"navigated mobile base to {station_name}")
        return True

    def set_grip_force(self, force: float) -> bool:
        self.scene.state["grip_force"] = float(force)
        self.scene.log(f"set grip force to {float(force):.2f}")
        return True

    def choose_arm_for(self, object_name: str) -> str:
        if len(self.profile.arm_names) == 1:
            return self.profile.arm_names[0]
        if object_name in {"left_handle", "board"}:
            return self.profile.arm_names[0]
        if object_name in {"right_handle", "hammer", "peg"}:
            return self.profile.arm_names[-1]
        return self.profile.arm_names[0]

    def other_arm(self, arm_name: str) -> str:
        for candidate in self.profile.arm_names:
            if candidate != arm_name:
                return candidate
        return arm_name

    def grasp_pot_handle(self, arm_name: str, handle_name: str) -> bool:
        if not self._require_dual_arm("grasp_pot_handle"):
            return False
        if not self._require_navigation_if_needed("pot_station"):
            return False
        if not self._valid_arm(arm_name):
            return False
        if handle_name not in {"left_handle", "right_handle"}:
            return self._fail_action(f"unknown pot handle '{handle_name}'")
        required = float(self.profile.required_grip_force)
        if required > 0 and float(self.scene.state["grip_force"]) < required:
            return self._fail_action(
                f"grasp_pot_handle: grip_force={float(self.scene.state['grip_force']):.2f} "
                f"is below required {required:.2f}"
            )
        held_by = self.scene.state["held_by"]
        held_by[handle_name] = arm_name
        self.scene.log(f"{arm_name} grasped {handle_name}")
        return True

    def lift_pot(self, lift_height: float = 0.16, keep_level: bool = True) -> bool:
        if not self._require_dual_arm("lift_pot"):
            return False
        held_by = self.scene.state["held_by"]
        left_arm = held_by.get("left_handle")
        right_arm = held_by.get("right_handle")
        if not left_arm or not right_arm:
            return self._fail_action("lift_pot: both pot handles must be grasped before lifting")
        if left_arm == right_arm:
            return self._fail_action("lift_pot: pot handles must be held by different arms")
        if not keep_level:
            return self._fail_action("lift_pot: keep_level=True is required for success")
        if float(lift_height) < 0.12:
            return self._fail_action("lift_pot: lift_height is below the 0.12m success threshold")
        self.scene.state["pot_height"] = float(lift_height)
        self.scene.state["pot_level"] = True
        self.scene.state["pot_lifted"] = True
        self.scene.log(f"lifted pot to {float(lift_height):.2f}m while level")
        return True

    def pick_hammer(self, arm_name: Optional[str] = None) -> bool:
        if not self._require_navigation_if_needed("handover_station"):
            return False
        if arm_name is None:
            arm_name = self.choose_arm_for("hammer")
        if not self._valid_arm(arm_name):
            return False
        self.scene.state["held_by"]["hammer"] = arm_name
        self.scene.log(f"{arm_name} picked hammer")
        return True

    def move_to_handover_pose(self, clearance: Optional[float] = None) -> bool:
        if not self._require_dual_arm("move_to_handover_pose"):
            return False
        clearance_value = self.profile.handover_clearance_m if clearance is None else float(clearance)
        if clearance_value + 1e-9 < self.profile.handover_clearance_m:
            return self._fail_action(
                f"move_to_handover_pose: clearance {clearance_value:.2f}m is below "
                f"required {self.profile.handover_clearance_m:.2f}m"
            )
        self.scene.state["handover_pose_ready"] = True
        self.scene.log(f"moved both arms to handover pose with {clearance_value:.2f}m clearance")
        return True

    def handover_object(self, from_arm: str, to_arm: str, object_name: str = "hammer") -> bool:
        if not self._require_dual_arm("handover_object"):
            return False
        if not self._valid_arm(from_arm) or not self._valid_arm(to_arm):
            return False
        if from_arm == to_arm:
            return self._fail_action("handover_object: from_arm and to_arm must be different")
        held_by = self.scene.state["held_by"]
        if held_by.get(object_name) != from_arm:
            return self._fail_action(f"handover_object: {from_arm} is not holding {object_name}")
        if self.profile.handover_clearance_m >= 0.10 and not self.scene.state["handover_pose_ready"]:
            return self._fail_action("handover_object: this embodiment requires move_to_handover_pose(clearance>=0.10)")
        held_by[object_name] = to_arm
        self.scene.state["hammer_handed_over"] = True
        self.scene.log(f"handed {object_name} from {from_arm} to {to_arm}")
        return True

    def place_hammer_on_target(self, arm_name: str) -> bool:
        if not self._valid_arm(arm_name):
            return False
        if self.scene.state["held_by"].get("hammer") != arm_name:
            return self._fail_action(f"place_hammer_on_target: {arm_name} is not holding hammer")
        self.scene.state["held_by"].pop("hammer", None)
        self.scene.state["hammer_on_target"] = True
        self.scene.log(f"{arm_name} placed hammer on target")
        return True

    def hold_board(self, arm_name: str) -> bool:
        if not self._require_dual_arm("hold_board"):
            return False
        if not self._require_navigation_if_needed("peg_station"):
            return False
        if not self._valid_arm(arm_name):
            return False
        self.scene.state["held_by"]["board"] = arm_name
        self.scene.state["board_held"] = True
        self.scene.log(f"{arm_name} held board")
        return True

    def grasp_peg(self, arm_name: str) -> bool:
        if not self._require_dual_arm("grasp_peg"):
            return False
        if not self._require_navigation_if_needed("peg_station"):
            return False
        if not self._valid_arm(arm_name):
            return False
        if self.scene.state["held_by"].get("board") == arm_name:
            return self._fail_action("grasp_peg: peg arm must differ from board-holding arm")
        self.scene.state["held_by"]["peg"] = arm_name
        self.scene.state["peg_held"] = True
        self.scene.log(f"{arm_name} grasped peg")
        return True

    def align_peg_to_hole(self, tolerance: float = 0.02) -> bool:
        if not self.scene.state["board_held"] or not self.scene.state["peg_held"]:
            return self._fail_action("align_peg_to_hole: board and peg must both be held first")
        tolerance_value = float(tolerance)
        if tolerance_value > self.profile.peg_alignment_tolerance_m:
            return self._fail_action(
                f"align_peg_to_hole: tolerance {tolerance_value:.3f}m is looser than "
                f"required {self.profile.peg_alignment_tolerance_m:.3f}m"
            )
        self.scene.state["peg_aligned"] = True
        self.scene.log(f"aligned peg to hole within {tolerance_value:.3f}m")
        return True

    def insert_peg(self, speed: float = 0.02) -> bool:
        speed_value = float(speed)
        if not self.scene.state["peg_aligned"]:
            return self._fail_action("insert_peg: peg must be aligned before insertion")
        if speed_value > self.profile.peg_insert_speed_limit:
            return self._fail_action(
                f"insert_peg: speed {speed_value:.3f} exceeds limit "
                f"{self.profile.peg_insert_speed_limit:.3f}"
            )
        self.scene.state["peg_inserted"] = True
        self.scene.log(f"inserted peg at speed {speed_value:.3f}")
        return True
