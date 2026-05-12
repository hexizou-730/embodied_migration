"""
DualArmRobot: two fixed KUKA iiwa arms sharing one tabletop scene.

This is the phase-2 PyBullet prototype for comparing a mobile manipulator
against a dual-arm fixed robot. It intentionally keeps the same high-level
pick/place contract as BaseRobot while exposing explicit left/right arm APIs.
"""
from typing import Optional, Tuple

import numpy as np
import pybullet as p

from capabilities import CapabilityCard
from robots.base_robot import BaseRobot


class DualArmHandle:
    """One arm inside DualArmRobot.

    The handle follows the BaseRobot pick/place protocol by duck-typing the
    required methods. This gives generated programs a natural `robot.left.pick`
    API without making each arm a separate top-level robot embodiment.
    """

    ATTACH_RADIUS = 0.15
    HOME_JOINTS = [0, 0.4, 0, -1.2, 0, 1.0, 0]

    def __init__(self, owner, arm_name: str, base_position, base_yaw: float = 0.0):
        self.owner = owner
        self.arm_name = arm_name
        self.base_position = np.asarray(base_position, dtype=float)
        self.base_yaw = float(base_yaw)

        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=self.base_position.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0, 0, self.base_yaw]),
            useFixedBase=True,
        )
        self.ee_link_index = 6
        self.num_joints = p.getNumJoints(self.robot_id)
        self.attached_constraint: Optional[int] = None
        self.attached_object_id: Optional[int] = None

        self._default_orn = p.getQuaternionFromEuler([0, np.pi, self.base_yaw])
        self._reset_home()

    @property
    def capability_card(self) -> CapabilityCard:
        return self.owner.capability_card

    @property
    def scene(self):
        return self.owner.scene

    def _reset_home(self):
        for j, q in enumerate(self.HOME_JOINTS):
            if j < self.num_joints:
                p.resetJointState(self.robot_id, j, q)

    def _step_physics(self, seconds: float):
        self.owner._step_physics(seconds)

    def _record_action_failure(self, message: str) -> bool:
        return self.owner._record_action_failure(f"{self.arm_name}: {message}")

    def _fail_action(self, message: str) -> bool:
        print(f"  ❌ [{self.arm_name}] {message}")
        return self._record_action_failure(message)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        return np.array(state[4]), np.array(state[5])

    def move_ee_to(self, position, orientation=None, steps=240) -> bool:
        target_pos = np.asarray(position, dtype=float)
        target_orn = list(orientation) if orientation is not None else list(self._default_orn)

        try:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                target_pos.tolist(),
                target_orn,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )
        except Exception as e:
            print(f"  [DualArm:{self.arm_name}] IK exception: {e}")
            self._record_action_failure(f"move_ee_to: IK exception: {e}")
            return False

        for i in range(min(self.num_joints, len(joint_poses))):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
            )
        self._step_physics(steps / 240.0)

        actual_pos, _ = self.get_ee_pose()
        err = float(np.linalg.norm(actual_pos - target_pos))
        if err > 0.12:
            print(
                f"  [DualArm:{self.arm_name}] move_ee_to failed: "
                f"err={err:.3f}m (target {target_pos.round(3).tolist()})"
            )
            self._record_action_failure(
                f"move_ee_to: target unreachable or inaccurate, err={err:.3f}m"
            )
            return False
        if err > 0.08:
            print(
                f"  [DualArm:{self.arm_name}] move_ee_to warn: "
                f"err={err:.3f}m (target {target_pos.round(3).tolist()})"
            )
        return True

    def activate_gripper(self) -> bool:
        if self.attached_constraint is not None:
            return True
        ee_pos, _ = self.get_ee_pose()
        if self.scene is None:
            print(f"  [DualArm:{self.arm_name}] activate_gripper: scene not set!")
            self._record_action_failure("activate_gripper: scene not set")
            return False

        held_ids = self.owner.held_object_ids()
        nearest_id, nearest_name, nearest_dist = None, None, self.ATTACH_RADIUS
        for obj_name, obj_id in self.scene.object_ids.items():
            if obj_id in held_ids:
                continue
            if "tray" in obj_name.lower() or "bowl" in obj_name.lower():
                continue
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            d = float(np.linalg.norm(np.array(pos) - ee_pos))
            if d < nearest_dist:
                nearest_id, nearest_name, nearest_dist = obj_id, obj_name, d

        if nearest_id is None:
            print(
                f"  [DualArm:{self.arm_name}] No object within "
                f"{self.ATTACH_RADIUS}m of EE (ee_pos={ee_pos.round(3).tolist()})"
            )
            self._record_action_failure(
                f"activate_gripper: no object within {self.ATTACH_RADIUS}m of end effector"
            )
            return False

        self.attached_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.ee_link_index,
            childBodyUniqueId=nearest_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.02],
            childFramePosition=[0, 0, 0],
        )
        self.attached_object_id = nearest_id
        print(f"  [DualArm:{self.arm_name}] Attached '{nearest_name}' ({nearest_dist:.3f}m)")
        return True

    def release_gripper(self) -> None:
        if self.attached_constraint is not None:
            p.removeConstraint(self.attached_constraint)
            self.attached_constraint = None
            self.attached_object_id = None
            print(f"  [DualArm:{self.arm_name}] Released")
        self._step_physics(0.3)

    def pick(self, object_position, hover_height: float = 0.15,
             pre_grasp_height: float = 0.02) -> bool:
        return BaseRobot.pick(self, object_position, hover_height, pre_grasp_height)

    def place(self, target_position, hover_height: float = 0.15,
              pre_release_height: Optional[float] = None) -> bool:
        return BaseRobot.place(self, target_position, hover_height, pre_release_height)

    def pick_and_place(self, source_pos, target_pos,
                       place_release_height: Optional[float] = None) -> bool:
        return BaseRobot.pick_and_place(self, source_pos, target_pos, place_release_height)


class DualArmRobot(BaseRobot):
    embodiment_name = "Dual-arm Fixed Manipulator (2x KUKA)"
    dof = 14
    gripper_type = "dual_suction"

    capability_card = CapabilityCard(
        grasp_mechanism="suction",
        stable_when_stacked=False,
        release_must_be_low=True,
        recommended_release_height_m=0.005,
        workspace_radius_m=0.75,
        can_rotate_object=False,
        max_payload_kg=3.0,
        ik_accuracy_m=0.04,
        has_mobile_base=False,
        global_reachable=False,
        has_dual_arms=True,
        can_bimanual=True,
        can_hold_object=True,
        can_coordinate_arms=True,
        left_workspace_radius_m=0.75,
        right_workspace_radius_m=0.75,
        extra={
            "notes": (
                "Dual-arm robot: choose left/right by reachability. "
                "Useful for hold-then-manipulate and handoff-style tasks, "
                "and can run coordinated two-arm lift phases, "
                "but it cannot navigate to distant tables."
            ),
        },
    )

    def __init__(
        self,
        left_base_position=(0.25, -0.55, 0.6),
        right_base_position=(0.25, 0.55, 0.6),
    ):
        self.scene = None
        self.action_failures = []
        self.left = DualArmHandle(self, "left", left_base_position)
        self.right = DualArmHandle(self, "right", right_base_position)
        self.active_arm_name = "left"
        self._last_two_object_assignment = None
        self._step_physics(0.5)

    def _step_physics(self, seconds: float):
        for _ in range(int(seconds * 240)):
            p.stepSimulation()

    def reset_action_log(self) -> None:
        self.action_failures = []

    def held_object_ids(self) -> set:
        ids = set()
        for arm in (self.left, self.right):
            if arm.attached_object_id is not None:
                ids.add(arm.attached_object_id)
        return ids

    def arm(self, arm_name: str) -> DualArmHandle:
        key = str(arm_name).lower()
        if key in {"left", "l"}:
            return self.left
        if key in {"right", "r"}:
            return self.right
        raise ValueError("arm_name must be 'left' or 'right'")

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.arm(self.active_arm_name).get_ee_pose()

    def move_ee_to(self, position, orientation=None, steps=240) -> bool:
        return self.arm(self.active_arm_name).move_ee_to(position, orientation, steps)

    def activate_gripper(self) -> bool:
        return self.arm(self.active_arm_name).activate_gripper()

    def release_gripper(self) -> None:
        self.arm(self.active_arm_name).release_gripper()

    def is_reachable_by(self, arm_name: str, target_position) -> bool:
        target = np.asarray(target_position, dtype=float)
        arm = self.arm(arm_name)
        radius = (
            self.capability_card.left_workspace_radius_m
            if arm.arm_name == "left"
            else self.capability_card.right_workspace_radius_m
        )
        horiz_dist = float(np.linalg.norm(target[:2] - arm.base_position[:2]))
        return 0.15 <= horiz_dist <= radius

    def choose_arm_for(self, target_position) -> str:
        target = np.asarray(target_position, dtype=float)
        preferred = ["left", "right"] if target[1] <= 0 else ["right", "left"]
        reachable = [name for name in preferred if self.is_reachable_by(name, target)]
        if reachable:
            return reachable[0]

        distances = {
            "left": float(np.linalg.norm(target[:2] - self.left.base_position[:2])),
            "right": float(np.linalg.norm(target[:2] - self.right.base_position[:2])),
        }
        return min(distances, key=distances.get)

    def _ik_for_arm(self, arm: DualArmHandle, target_position, orientation=None):
        target_pos = np.asarray(target_position, dtype=float)
        target_orn = list(orientation) if orientation is not None else list(arm._default_orn)
        try:
            return p.calculateInverseKinematics(
                arm.robot_id,
                arm.ee_link_index,
                target_pos.tolist(),
                target_orn,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )
        except Exception as e:
            self._record_action_failure(f"{arm.arm_name}: coordinated IK exception: {e}")
            return None

    def _command_arm_joints(self, arm: DualArmHandle, joint_poses) -> None:
        for i in range(min(arm.num_joints, len(joint_poses))):
            p.setJointMotorControl2(
                arm.robot_id,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
            )

    def move_both_ee_to(
        self,
        left_position,
        right_position,
        steps: int = 240,
        left_orientation=None,
        right_orientation=None,
        record_failure: bool = True,
    ) -> bool:
        """Move left and right end effectors in one synchronized control phase."""
        left_target = np.asarray(left_position, dtype=float)
        right_target = np.asarray(right_position, dtype=float)
        left_joints = self._ik_for_arm(self.left, left_target, left_orientation)
        right_joints = self._ik_for_arm(self.right, right_target, right_orientation)
        if left_joints is None or right_joints is None:
            return False

        self._command_arm_joints(self.left, left_joints)
        self._command_arm_joints(self.right, right_joints)
        self._step_physics(steps / 240.0)

        left_actual, _ = self.left.get_ee_pose()
        right_actual, _ = self.right.get_ee_pose()
        left_err = float(np.linalg.norm(left_actual - left_target))
        right_err = float(np.linalg.norm(right_actual - right_target))
        ok = True
        if left_err > 0.12:
            if record_failure:
                self._record_action_failure(
                    f"left: coordinated move inaccurate, err={left_err:.3f}m"
                )
            ok = False
        if right_err > 0.12:
            if record_failure:
                self._record_action_failure(
                    f"right: coordinated move inaccurate, err={right_err:.3f}m"
                )
            ok = False
        if not ok:
            level = "failed" if record_failure else "warn"
            print(
                f"  [DualArm:both] coordinated move {level}: "
                f"left_err={left_err:.3f}m right_err={right_err:.3f}m"
            )
        return ok

    def _assign_two_arms(self, first_position, second_position):
        first = np.asarray(first_position, dtype=float)
        second = np.asarray(second_position, dtype=float)
        preferred = ("left", "right") if first[1] <= second[1] else ("right", "left")
        first_arm, second_arm = preferred
        if (
            self.is_reachable_by(first_arm, first)
            and self.is_reachable_by(second_arm, second)
        ):
            return first_arm, second_arm
        swapped = (second_arm, first_arm)
        if (
            self.is_reachable_by(swapped[0], first)
            and self.is_reachable_by(swapped[1], second)
        ):
            return swapped
        return first_arm, second_arm

    def lift_two_objects(
        self,
        first_position,
        second_position,
        lift_height: float = 0.18,
        hover_height: float = 0.15,
        pre_grasp_height: float = 0.02,
    ) -> bool:
        """Coordinated bimanual lift.

        The two arms move through hover, grasp, and lift phases together. This
        is the preferred API for instructions such as "lift both blocks at the
        same time"; separate pick_with_arm calls are intentionally sequential.
        """
        first = np.asarray(first_position, dtype=float)
        second = np.asarray(second_position, dtype=float)
        first_arm_name, second_arm_name = self._assign_two_arms(first, second)
        if first_arm_name == second_arm_name:
            return self._fail_action("lift_two_objects: requires two distinct arms")
        if not self.is_reachable_by(first_arm_name, first):
            return self._fail_action(f"{first_arm_name}: first target outside arm workspace")
        if not self.is_reachable_by(second_arm_name, second):
            return self._fail_action(f"{second_arm_name}: second target outside arm workspace")

        first_is_left = first_arm_name == "left"
        left_pos = first if first_is_left else second
        right_pos = second if first_is_left else first

        left_hover = left_pos + np.array([0, 0, hover_height])
        right_hover = right_pos + np.array([0, 0, hover_height])
        left_grasp = left_pos + np.array([0, 0, pre_grasp_height])
        right_grasp = right_pos + np.array([0, 0, pre_grasp_height])
        left_lift = left_pos + np.array([0, 0, lift_height])
        right_lift = right_pos + np.array([0, 0, lift_height])

        print(
            "  [DualArm:both] Coordinated lift plan: "
            f"{first_arm_name}+{second_arm_name}"
        )
        if not self.move_both_ee_to(left_hover, right_hover):
            return self._fail_action("lift_two_objects: failed to move both arms above objects")
        if not self.move_both_ee_to(left_grasp, right_grasp, steps=180):
            return self._fail_action("lift_two_objects: failed to descend both arms")

        left_ok = self.left.activate_gripper()
        right_ok = self.right.activate_gripper()
        if not (left_ok and right_ok):
            return self._fail_action("lift_two_objects: failed to grasp both objects")

        self.move_both_ee_to(left_grasp, right_grasp, steps=30)
        if not self.move_both_ee_to(left_lift, right_lift, steps=180):
            return self._fail_action("lift_two_objects: failed to lift both objects")
        print("  [DualArm:both] Coordinated lift complete")
        self._last_two_object_assignment = {
            "first_arm": first_arm_name,
            "second_arm": second_arm_name,
        }
        return True

    def pick_two_objects(self, first_position, second_position, **kwargs) -> bool:
        """Alias for LLM readability."""
        return self.lift_two_objects(first_position, second_position, **kwargs)

    def place_two_objects(
        self,
        first_target_position,
        second_target_position,
        hover_height: float = 0.15,
        pre_release_height: Optional[float] = None,
    ) -> bool:
        """Coordinated bimanual place for two currently held objects.

        If this follows `lift_two_objects(first, second)`, `first_target_position`
        is assigned to the arm that lifted the first object and
        `second_target_position` to the arm that lifted the second object.
        """
        if pre_release_height is None:
            pre_release_height = self.capability_card.recommended_release_height_m
        held = [arm.arm_name for arm in (self.left, self.right)
                if arm.attached_constraint is not None]
        if set(held) != {"left", "right"}:
            return self._fail_action("place_two_objects: both arms must be holding objects")

        assignment = self._last_two_object_assignment or {
            "first_arm": "left",
            "second_arm": "right",
        }
        first_target = np.asarray(first_target_position, dtype=float)
        second_target = np.asarray(second_target_position, dtype=float)
        first_arm = assignment["first_arm"]
        second_arm = assignment["second_arm"]

        if not self.is_reachable_by(first_arm, first_target):
            return self._fail_action(f"{first_arm}: first place target outside arm workspace")
        if not self.is_reachable_by(second_arm, second_target):
            return self._fail_action(f"{second_arm}: second place target outside arm workspace")

        left_target = first_target if first_arm == "left" else second_target
        right_target = second_target if second_arm == "right" else first_target
        left_above = left_target + np.array([0, 0, hover_height])
        right_above = right_target + np.array([0, 0, hover_height])
        left_release = left_target + np.array([0, 0, pre_release_height])
        right_release = right_target + np.array([0, 0, pre_release_height])

        print(
            "  [DualArm:both] Coordinated place plan: "
            f"{first_arm}+{second_arm}"
        )
        if not self.move_both_ee_to(left_above, right_above):
            return self._fail_action("place_two_objects: failed to move both arms above targets")
        if not self.move_both_ee_to(left_release, right_release, steps=180):
            return self._fail_action("place_two_objects: failed to descend both arms")
        self.left.release_gripper()
        self.right.release_gripper()
        failures_before_retract = len(self.action_failures)
        if not self.move_both_ee_to(
            left_above,
            right_above,
            steps=180,
            record_failure=False,
        ):
            # The task-relevant part of place_two_objects is complete once both
            # objects have been released at the target. Retraction IK can be a
            # little noisy near workspace limits, so treat post-release retract
            # inaccuracy as a warning instead of forcing LLM retry on a mutated
            # scene where the objects have already moved.
            del self.action_failures[failures_before_retract:]
            print("  [DualArm:both] retract warn: objects already released; continuing")
        print("  [DualArm:both] Coordinated place complete")
        self._last_two_object_assignment = None
        return True

    def pick_and_place_two_objects(
        self,
        first_position,
        second_position,
        first_target_position,
        second_target_position,
        lift_height: float = 0.18,
        place_release_height: Optional[float] = None,
    ) -> bool:
        if not self.lift_two_objects(
            first_position,
            second_position,
            lift_height=lift_height,
        ):
            return False
        return self.place_two_objects(
            first_target_position,
            second_target_position,
            pre_release_height=place_release_height,
        )

    def pick_with_arm(self, arm_name: str, object_position, hover_height: float = 0.15,
                      pre_grasp_height: float = 0.02) -> bool:
        arm = self.arm(arm_name)
        self.active_arm_name = arm.arm_name
        if not self.is_reachable_by(arm.arm_name, object_position):
            return self._fail_action(f"{arm.arm_name}: pick target outside arm workspace")
        return arm.pick(object_position, hover_height, pre_grasp_height)

    def place_with_arm(self, arm_name: str, target_position, hover_height: float = 0.15,
                       pre_release_height: Optional[float] = None,
                       place_release_height: Optional[float] = None) -> bool:
        arm = self.arm(arm_name)
        self.active_arm_name = arm.arm_name
        if arm.attached_constraint is None:
            return self._fail_action(f"{arm.arm_name}: place: no object is currently held")
        if not self.is_reachable_by(arm.arm_name, target_position):
            return self._fail_action(f"{arm.arm_name}: place target outside arm workspace")
        release_height = (
            place_release_height
            if place_release_height is not None
            else pre_release_height
        )
        return arm.place(target_position, hover_height, release_height)

    def pick_and_place_with_arm(self, arm_name: str, source_pos, target_pos,
                                place_release_height: Optional[float] = None) -> bool:
        if not self.pick_with_arm(arm_name, source_pos):
            return False
        return self.place_with_arm(
            arm_name,
            target_pos,
            pre_release_height=place_release_height,
        )

    def hold_with_arm(self, arm_name: str, object_name_or_position) -> bool:
        if isinstance(object_name_or_position, str):
            if self.scene is None:
                return self._fail_action("hold_with_arm: scene not set")
            object_position = self.scene.get_object_position(object_name_or_position)
        else:
            object_position = object_name_or_position
        return self.pick_with_arm(arm_name, object_position)

    def release_arm(self, arm_name: str) -> None:
        arm = self.arm(arm_name)
        self.active_arm_name = arm.arm_name
        arm.release_gripper()

    def pick(self, object_position, hover_height: float = 0.15,
             pre_grasp_height: float = 0.02) -> bool:
        arm_name = self.choose_arm_for(object_position)
        return self.pick_with_arm(arm_name, object_position, hover_height, pre_grasp_height)

    def place(self, target_position, hover_height: float = 0.15,
              pre_release_height: Optional[float] = None) -> bool:
        held = [arm.arm_name for arm in (self.left, self.right)
                if arm.attached_constraint is not None]
        arm_name = self.active_arm_name if self.active_arm_name in held else None
        if arm_name is None:
            arm_name = held[0] if held else self.choose_arm_for(target_position)
        return self.place_with_arm(
            arm_name,
            target_position,
            hover_height=hover_height,
            pre_release_height=pre_release_height,
        )

    def pick_and_place(self, source_pos, target_pos,
                       place_release_height: Optional[float] = None) -> bool:
        arm_name = self.choose_arm_for(source_pos)
        return self.pick_and_place_with_arm(arm_name, source_pos, target_pos, place_release_height)

    def describe(self) -> str:
        return (
            f"Embodiment: {self.embodiment_name} | "
            f"DoF: {self.dof} | "
            f"Gripper: {self.gripper_type} | "
            "Arms: left/right; explicit APIs: pick_with_arm, place_with_arm, "
            "is_reachable_by, choose_arm_for, lift_two_objects, place_two_objects"
        )
