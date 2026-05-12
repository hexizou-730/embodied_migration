"""
DualFrankaRobot: two fixed Franka Panda arms sharing one tabletop scene.

This embodiment complements:

  - dual_arm: fixed dual KUKA arms with suction grasping
  - mobile_dual_arm: mobile base plus dual KUKA arms
  - dual_franka: fixed dual Franka arms with parallel-jaw grasping

It reuses the coordinated dual-arm API from DualArmRobot while swapping the
per-arm hardware model and capability card.
"""
from typing import Optional

import numpy as np
import pybullet as p

from capabilities import CapabilityCard
from robots.dual_arm_robot import DualArmHandle, DualArmRobot


class DualFrankaHandle(DualArmHandle):
    """One Franka arm inside DualFrankaRobot."""

    ARM_JOINTS = list(range(7))
    FINGER_JOINTS = [9, 10]
    EE_LINK_INDEX = 11
    FINGER_OPEN = 0.04
    FINGER_CLOSED = 0.01
    ATTACH_RADIUS = 0.15
    HOME_JOINTS = [0, 0.2, 0, -1.8, 0, 2.0, 0.785]

    def __init__(self, owner, arm_name: str, base_position, base_yaw: float = 0.0):
        self.owner = owner
        self.arm_name = arm_name
        self.base_position = np.asarray(base_position, dtype=float)
        self.base_yaw = float(base_yaw)

        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.base_position.tolist(),
            baseOrientation=p.getQuaternionFromEuler([0, 0, self.base_yaw]),
            useFixedBase=True,
        )
        self.ee_link_index = self.EE_LINK_INDEX
        self.num_joints = len(self.ARM_JOINTS)
        self.attached_constraint: Optional[int] = None
        self.attached_object_id: Optional[int] = None

        self._default_orn = p.getQuaternionFromEuler([np.pi, 0, self.base_yaw])
        self._reset_home()
        self._open_fingers()

    def _reset_home(self):
        for j, q in zip(self.ARM_JOINTS, self.HOME_JOINTS):
            p.resetJointState(self.robot_id, j, q)

    def _open_fingers(self):
        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=self.FINGER_OPEN,
                force=20,
            )
        self._step_physics(0.2)

    def _close_fingers(self):
        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=self.FINGER_CLOSED,
                force=50,
            )
        self._step_physics(0.2)

    def activate_gripper(self) -> bool:
        self._close_fingers()
        if self.attached_constraint is not None:
            return True
        ee_pos, _ = self.get_ee_pose()
        if self.scene is None:
            print(f"  [DualFranka:{self.arm_name}] activate_gripper: scene not set")
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
                f"  [DualFranka:{self.arm_name}] No object within "
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
            parentFramePosition=[0, 0, 0.03],
            childFramePosition=[0, 0, 0],
        )
        self.attached_object_id = nearest_id
        print(f"  [DualFranka:{self.arm_name}] Grasped '{nearest_name}' ({nearest_dist:.3f}m)")
        return True

    def release_gripper(self) -> None:
        if self.attached_constraint is not None:
            p.removeConstraint(self.attached_constraint)
            self.attached_constraint = None
            self.attached_object_id = None
            print(f"  [DualFranka:{self.arm_name}] Released")
        self._open_fingers()
        self._step_physics(0.3)


class DualFrankaRobot(DualArmRobot):
    embodiment_name = "Dual-arm Franka Panda (2x Franka)"
    dof = 14
    gripper_type = "dual_parallel_jaw"

    capability_card = CapabilityCard(
        grasp_mechanism="parallel_jaw",
        stable_when_stacked=True,
        release_must_be_low=False,
        recommended_release_height_m=0.03,
        workspace_radius_m=0.75,
        can_rotate_object=True,
        max_payload_kg=3.0,
        ik_accuracy_m=0.025,
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
                "Dual Franka robot: fixed-base bimanual manipulator with "
                "parallel-jaw grasping. Compared with dual KUKA suction, "
                "migration should preserve bimanual task structure while adapting "
                "grasp/release priors."
            ),
        },
    )

    def __init__(
        self,
        left_base_position=(0.20, -0.50, 0.6),
        right_base_position=(0.20, 0.50, 0.6),
    ):
        self.scene = None
        self.action_failures = []
        self.left = DualFrankaHandle(self, "left", left_base_position)
        self.right = DualFrankaHandle(self, "right", right_base_position)
        self.active_arm_name = "left"
        self._last_two_object_assignment = None
        self._step_physics(0.5)

    def _command_arm_joints(self, arm: DualFrankaHandle, joint_poses) -> None:
        for j in arm.ARM_JOINTS:
            p.setJointMotorControl2(
                arm.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=500,
            )

    def describe(self) -> str:
        return (
            f"Embodiment: {self.embodiment_name} | "
            f"DoF: {self.dof} | "
            f"Gripper: {self.gripper_type} | "
            "Arms: left/right Franka; APIs: pick_with_arm, place_with_arm, "
            "is_reachable_by, choose_arm_for, lift_two_objects, place_two_objects"
        )
