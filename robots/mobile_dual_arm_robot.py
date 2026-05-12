"""
MobileDualArmRobot: a mobile base carrying two coordinated KUKA arms.

This embodiment is intended for the clean migration comparison:

  - dual_arm: fixed-base bimanual robot
  - mobile_dual_arm: same bimanual manipulation APIs plus mobile navigation

The prototype keeps navigation abstracted as discrete base relocation, matching
the existing mobile manipulator model. The two arms are reloaded at table-side
mount points whenever the base navigates.
"""
from typing import Optional, Tuple

import numpy as np
import pybullet as p

from capabilities import CapabilityCard
from robots.dual_arm_robot import DualArmHandle, DualArmRobot


class MobileDualArmRobot(DualArmRobot):
    embodiment_name = "Mobile Dual-arm Manipulator (Husky + 2x KUKA)"
    dof = 14
    gripper_type = "mobile_dual_suction"

    capability_card = CapabilityCard(
        grasp_mechanism="suction",
        stable_when_stacked=False,
        release_must_be_low=True,
        recommended_release_height_m=0.005,
        workspace_radius_m=0.95,
        can_rotate_object=False,
        max_payload_kg=3.0,
        ik_accuracy_m=0.045,
        has_mobile_base=True,
        global_reachable=True,
        nav_min_clearance_m=0.90,
        has_dual_arms=True,
        can_bimanual=True,
        can_hold_object=True,
        can_coordinate_arms=True,
        left_workspace_radius_m=0.95,
        right_workspace_radius_m=0.95,
        extra={
            "notes": (
                "Mobile dual-arm robot: navigate to a safe table-side standoff, "
                "then use coordinated bimanual APIs. For same-goal migration "
                "from fixed dual-arm, keep lift/place logic but add navigation "
                "and reachability checks."
            ),
        },
    )

    def __init__(self, base_position=(0.0, 0.0, 0.0)):
        self.scene = None
        self.action_failures = []
        self.husky_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=list(base_position),
            useFixedBase=True,
        )
        self._disable_base_collisions()
        self.base_xy = np.asarray(base_position[:2], dtype=float)
        self.base_theta = 0.0
        left_pos, right_pos = self._arm_mount_positions(
            self.base_xy[0],
            self.base_xy[1],
            self.base_theta,
        )
        self.left = DualArmHandle(self, "left", left_pos, base_yaw=self.base_theta)
        self.right = DualArmHandle(self, "right", right_pos, base_yaw=self.base_theta)
        self.active_arm_name = "left"
        self._last_two_object_assignment = None
        self._step_physics(0.5)

    def _disable_base_collisions(self):
        for link in range(-1, p.getNumJoints(self.husky_id)):
            p.setCollisionFilterGroupMask(self.husky_id, link, 0, 0)

    def _arm_mount_positions(self, x: float, y: float, theta: float):
        """Return world positions for left/right arm bases.

        Arms are placed at a visible table-side crossbar in front of the mobile
        base. This keeps the Husky outside the table while both arms can cover
        the tray and two blocks.
        """
        forward = np.array([np.cos(theta), np.sin(theta)])
        lateral = np.array([-np.sin(theta), np.cos(theta)])
        base = np.array([float(x), float(y)])
        arm_center = base + 0.35 * forward
        left_xy = arm_center + 0.28 * lateral
        right_xy = arm_center - 0.28 * lateral
        z = 0.60
        return (
            (float(left_xy[0]), float(left_xy[1]), z),
            (float(right_xy[0]), float(right_xy[1]), z),
        )

    def _reload_arms_at(self, x: float, y: float, theta: float) -> bool:
        if self.held_object_ids():
            print("  [MobileDualArm] navigate_to failed: cannot navigate while holding objects")
            self._record_action_failure("navigate_to: cannot navigate while holding objects")
            return False
        p.removeBody(self.left.robot_id)
        p.removeBody(self.right.robot_id)
        left_pos, right_pos = self._arm_mount_positions(x, y, theta)
        self.left = DualArmHandle(self, "left", left_pos, base_yaw=theta)
        self.right = DualArmHandle(self, "right", right_pos, base_yaw=theta)
        self.active_arm_name = "left"
        self._step_physics(0.2)
        return True

    def _safe_table_standoff(self, x: float, y: float) -> Tuple[float, float]:
        if self.scene is None or not hasattr(self.scene, "table_position"):
            return float(x), float(y)
        tx, ty = float(self.scene.table_position[0]), float(self.scene.table_position[1])
        dx, dy = float(x) - tx, float(y) - ty
        if abs(dx) >= 0.85 or abs(dy) >= 0.85:
            return float(x), float(y)
        side = 1.0 if dy >= 0 else -1.0
        safe_x = tx
        safe_y = ty + side * self.capability_card.nav_min_clearance_m
        if abs(safe_x - float(x)) > 1e-3 or abs(safe_y - float(y)) > 1e-3:
            print(
                "  [MobileDualArm] adjusted table standoff "
                f"from ({float(x):.2f}, {float(y):.2f}) "
                f"to ({safe_x:.2f}, {safe_y:.2f}) to avoid table collision"
            )
        return safe_x, safe_y

    def get_base_position(self) -> np.ndarray:
        pos, orn = p.getBasePositionAndOrientation(self.husky_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return np.array([pos[0], pos[1], yaw])

    def navigate_to(self, x: float, y: float, theta: Optional[float] = None) -> bool:
        if self.held_object_ids():
            print("  [MobileDualArm] navigate_to failed: cannot navigate while holding objects")
            self._record_action_failure("navigate_to: cannot navigate while holding objects")
            return False
        x, y = self._safe_table_standoff(float(x), float(y))
        if theta is None:
            if self.scene is not None and hasattr(self.scene, "table_position"):
                tx, ty = float(self.scene.table_position[0]), float(self.scene.table_position[1])
                theta = float(np.arctan2(ty - y, tx - x))
            else:
                theta = float(np.arctan2(-y, -x))

        orn = p.getQuaternionFromEuler([0, 0, theta])
        p.resetBasePositionAndOrientation(self.husky_id, [float(x), float(y), 0.0], orn)
        p.resetBaseVelocity(self.husky_id, [0, 0, 0], [0, 0, 0])
        if not self._reload_arms_at(float(x), float(y), float(theta)):
            return False
        self.base_xy = np.array([float(x), float(y)])
        self.base_theta = float(theta)
        self._step_physics(0.5)
        print(f"  [MobileDualArm] 🚐 Navigated base to ({x:.2f}, {y:.2f}, θ={theta:.2f})")
        return True

    def is_reachable(self, target_position) -> bool:
        return (
            self.is_reachable_by("left", target_position)
            or self.is_reachable_by("right", target_position)
        )

    def _assign_two_arms(self, first_position, second_position):
        """Assign two objects to the nearer distinct mobile-mounted arms."""
        first = np.asarray(first_position, dtype=float)
        second = np.asarray(second_position, dtype=float)
        candidates = []
        for first_arm, second_arm in (("left", "right"), ("right", "left")):
            if (
                self.is_reachable_by(first_arm, first)
                and self.is_reachable_by(second_arm, second)
            ):
                d_first = np.linalg.norm(first[:2] - self.arm(first_arm).base_position[:2])
                d_second = np.linalg.norm(second[:2] - self.arm(second_arm).base_position[:2])
                candidates.append((float(d_first + d_second), first_arm, second_arm))
        if candidates:
            _, first_arm, second_arm = min(candidates, key=lambda item: item[0])
            return first_arm, second_arm
        return super()._assign_two_arms(first, second)

    def describe(self) -> str:
        return (
            f"Embodiment: {self.embodiment_name} | "
            f"DoF: {self.dof} | "
            f"Gripper: {self.gripper_type} | "
            "Mobile base + coordinated left/right arms; APIs: navigate_to, "
            "is_reachable, lift_two_objects, place_two_objects"
        )
