"""
MobileManipulatorRobot: Husky 底盘 (PyBullet 自带) + KUKA iiwa 7-DOF 臂叠加。

异构性来源 (相对 KUKA / Franka):
- 增加 mobile_base, 引入 navigate_to / get_base_position / is_reachable 三个新 API
- 单点 arm 可达半径 0.85m, 但可以在场地任何点 navigate, 全局 5x5m
- has_mobile_base=True 在 capability_card 里, LLM 必须显式处理 nav-then-manipulate 流程

物理实现:
- Husky 底盘用 husky/husky.urdf 作为可视化底盘和位姿载体
- KUKA 臂 spawn 在底盘上方
- navigate_to(x, y) 用 resetBasePositionAndOrientation 同步移动底盘和机械臂
- 简化版: 离散重定位导航 (中等真实度的 setting 1)
"""
import numpy as np
import pybullet as p
from typing import Optional, Tuple

from robots.base_robot import BaseRobot
from capabilities import CapabilityCard


class MobileManipulatorRobot(BaseRobot):
    embodiment_name = "Mobile Manipulator (Husky + KUKA)"
    dof = 7  # arm only; base is non-holonomic but we abstract it as holonomic for nav
    gripper_type = "suction"

    ATTACH_RADIUS = 0.15
    ARM_BASE_OFFSET_Z = 0.4   # KUKA 臂相对底盘原点的高度偏移
    HOME_JOINTS = [0, 0.4, 0, -1.2, 0, 1.0, 0]

    # === Capability Card: 标志性的 has_mobile_base=True ===
    capability_card = CapabilityCard(
        grasp_mechanism="suction",
        stable_when_stacked=False,
        release_must_be_low=True,
        recommended_release_height_m=0.005,
        workspace_radius_m=0.9,        # 移动底盘停在桌边时, 臂可覆盖桌面局部区域
        can_rotate_object=False,
        max_payload_kg=3.0,
        ik_accuracy_m=0.04,             # 移动平台 IK 误差稍大
        has_mobile_base=True,           # ★ 关键
        global_reachable=True,
        nav_min_clearance_m=0.70,
        extra={
            "notes": (
                "Mobile manipulator: navigate first, then manipulate. "
                "After navigation the arm workspace shifts; always re-check is_reachable."
            ),
        },
    )

    def __init__(self, base_position=(0.0, 0.0, 0.0)):
        # 1) 加载 Husky 底盘。这里把 Husky 当作可视化底盘和位姿载体;
        #    关闭碰撞可避免传送式导航时把桌子/物体撞飞。
        self.husky_id = p.loadURDF(
            "husky/husky.urdf",
            basePosition=list(base_position),
            useFixedBase=True,
        )
        self._disable_base_collisions()

        # 2) 在 Husky 上方加载 KUKA 臂
        # base_position 是底盘的位置, 臂 spawn 在底盘原点上方 ARM_BASE_OFFSET_Z
        arm_base_pos = (
            base_position[0],
            base_position[1],
            base_position[2] + self.ARM_BASE_OFFSET_Z,
        )
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=list(arm_base_pos),
            useFixedBase=True,
        )

        self.ee_link_index = 6
        self.num_joints = p.getNumJoints(self.robot_id)

        self.attached_constraint: Optional[int] = None
        self.attached_object_id: Optional[int] = None
        self.scene = None

        self._default_orn = p.getQuaternionFromEuler([0, np.pi, 0])

        self._reset_arm_home()

        self._step_physics(0.5)

    # ============================================================
    # 物理辅助
    # ============================================================
    def _step_physics(self, seconds: float):
        for _ in range(int(seconds * 240)):
            p.stepSimulation()

    def _disable_base_collisions(self):
        for link in range(-1, p.getNumJoints(self.husky_id)):
            p.setCollisionFilterGroupMask(self.husky_id, link, 0, 0)

    def _reset_arm_home(self):
        for j, q in enumerate(self.HOME_JOINTS):
            if j < self.num_joints:
                p.resetJointState(self.robot_id, j, q)

    def _reload_arm_at(self, x: float, y: float, theta: float):
        if self.attached_constraint is not None:
            print("  [Mobile] navigate_to failed: cannot reload arm while holding an object")
            self._record_action_failure(
                "navigate_to: cannot reload arm while holding an object"
            )
            return False

        p.removeBody(self.robot_id)
        arm_orn = p.getQuaternionFromEuler([0, 0, theta])
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[float(x), float(y), self.ARM_BASE_OFFSET_Z],
            baseOrientation=arm_orn,
            useFixedBase=True,
        )
        self.num_joints = p.getNumJoints(self.robot_id)
        self._reset_arm_home()
        self._step_physics(0.2)
        return True

    def _safe_table_standoff(self, x: float, y: float) -> Tuple[float, float]:
        """Nudge table-near navigation targets to a visible side standoff.

        Generated code sometimes asks the base to navigate to an object or tray
        center. That is physically meaningless for a tabletop task and looks
        like the Husky is parked under the table. We keep the command semantics
        as "go near the table" but project unsafe near-table targets to a side
        pose that remains reachable for the arm and visually clear of the table.
        """
        if self.scene is None or not hasattr(self.scene, "table_position"):
            return float(x), float(y)

        tx, ty = float(self.scene.table_position[0]), float(self.scene.table_position[1])
        dx, dy = float(x) - tx, float(y) - ty
        near_table_x = abs(dx) < 0.75
        near_table_y = abs(dy) < 0.75
        if not (near_table_x and near_table_y):
            return float(x), float(y)

        side = 1.0 if dy >= 0 else -1.0
        safe_x = tx - 0.05
        safe_y = ty + side * self.capability_card.nav_min_clearance_m
        if abs(safe_x - float(x)) > 1e-3 or abs(safe_y - float(y)) > 1e-3:
            print(
                "  [Mobile] adjusted table standoff "
                f"from ({float(x):.2f}, {float(y):.2f}) "
                f"to ({safe_x:.2f}, {safe_y:.2f}) to avoid table collision"
            )
        return safe_x, safe_y

    # ============================================================
    # ★ 新 API: 导航 / 基座查询 / 可达性
    # 这些是 MobileManipulator 独有的, KUKA / Franka 没有
    # ============================================================
    def get_base_position(self) -> np.ndarray:
        """Return (x, y, theta) of the husky base."""
        pos, orn = p.getBasePositionAndOrientation(self.husky_id)
        # 提取 yaw
        euler = p.getEulerFromQuaternion(orn)
        return np.array([pos[0], pos[1], euler[2]])

    def navigate_to(self, x: float, y: float, theta: Optional[float] = None) -> bool:
        """Navigate the base to (x, y) [, theta].

        默认朝向规则: 让底盘朝着「桌面中心或场地中心」方向, 这样 KUKA 才能向前伸臂操作。
        如果 scene 已设置, 朝桌面中心; 否则朝原点。

        中等真实度: 用 resetBasePositionAndOrientation 传送, 然后 step 几下让物理稳定。
        """
        if self.attached_constraint is not None:
            print("  [Mobile] navigate_to failed: cannot navigate while holding an object")
            self._record_action_failure("navigate_to: cannot navigate while holding an object")
            return False

        x, y = self._safe_table_standoff(float(x), float(y))

        if theta is None:
            # 默认朝向: 朝桌面中心 (而不是朝原点)
            if self.scene is not None and hasattr(self.scene, "table_position"):
                tx, ty = float(self.scene.table_position[0]), float(self.scene.table_position[1])
                theta = float(np.arctan2(ty - y, tx - x))
            else:
                # fallback: 朝原点
                theta = float(np.arctan2(-y, -x))

        new_orn = p.getQuaternionFromEuler([0, 0, theta])
        # 把底盘和机械臂一起"瞬移"到目标。这个原型把导航抽象成
        # 离散重定位; 同步 reset 两个 body 可以避免固定约束把底盘拉回旧位置。
        p.resetBasePositionAndOrientation(
            self.husky_id,
            [float(x), float(y), 0.0],   # z=0 因为 husky 在地面
            new_orn,
        )
        p.resetBaseVelocity(self.husky_id, [0, 0, 0], [0, 0, 0])
        if not self._reload_arm_at(x, y, theta):
            return False
        self._step_physics(0.5)
        print(f"  [Mobile] 🚐 Navigated base to ({x:.2f}, {y:.2f}, θ={theta:.2f})")
        return True

    def is_reachable(self, target_position) -> bool:
        """Check if target is reachable by the arm at the CURRENT base position.

        简化模型: 末端可达半径 = capability_card.workspace_radius_m
        计算 target 到当前 base 的水平距离, 在 [0.2, radius] 区间内才算可达
        (太近的话臂折叠不开)。
        """
        target = np.asarray(target_position, dtype=float)
        base = self.get_base_position()  # (x, y, theta)
        horiz_dist = float(np.linalg.norm(target[:2] - base[:2]))
        radius = self.capability_card.workspace_radius_m
        reachable = (0.2 <= horiz_dist <= radius)
        return reachable

    # ============================================================
    # Level-1 API (BaseRobot 抽象方法)
    # ============================================================
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.robot_id, self.ee_link_index)
        return np.array(state[4]), np.array(state[5])

    def move_ee_to(self, position, orientation=None, steps=240) -> bool:
        target_pos = list(np.asarray(position, dtype=float))
        target_orn = list(orientation) if orientation is not None else list(self._default_orn)

        try:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.ee_link_index,
                target_pos,
                target_orn,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )
        except Exception as e:
            print(f"  [Mobile] IK exception: {e}")
            self._record_action_failure(f"move_ee_to: IK exception: {e}")
            return False

        for i in range(min(self.num_joints, len(joint_poses))):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
            )
        self._step_physics(steps / 240.0)

        actual_pos, _ = self.get_ee_pose()
        err = float(np.linalg.norm(actual_pos - np.array(target_pos)))
        if err > 0.12:
            print(f"  [Mobile] move_ee_to failed: err={err:.3f}m "
                  f"(target {np.array(target_pos).round(3).tolist()})")
            self._record_action_failure(
                f"move_ee_to: target unreachable or inaccurate, err={err:.3f}m"
            )
            return False
        if err > 0.08:  # mobile 平台 IK 误差较大, 阈值放宽
            print(f"  [Mobile] move_ee_to warn: err={err:.3f}m "
                  f"(target {np.array(target_pos).round(3).tolist()})")
        return True

    def activate_gripper(self) -> bool:
        if self.attached_constraint is not None:
            return True
        ee_pos, _ = self.get_ee_pose()
        if self.scene is None:
            print("  [Mobile] activate_gripper: scene not set!")
            self._record_action_failure("activate_gripper: scene not set")
            return False

        nearest_id, nearest_name, nearest_dist = None, None, self.ATTACH_RADIUS
        for obj_name, obj_id in self.scene.object_ids.items():
            if "tray" in obj_name.lower() or "bowl" in obj_name.lower():
                continue
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            d = float(np.linalg.norm(np.array(pos) - ee_pos))
            if d < nearest_dist:
                nearest_id, nearest_name, nearest_dist = obj_id, obj_name, d

        if nearest_id is None:
            print(f"  [Mobile] ⚠️ No object within {self.ATTACH_RADIUS}m of EE "
                  f"(ee_pos={ee_pos.round(3).tolist()})")
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
        print(f"  [Mobile] 🧲 Attached '{nearest_name}' (was {nearest_dist:.3f}m away)")
        return True

    def release_gripper(self) -> None:
        if self.attached_constraint is not None:
            p.removeConstraint(self.attached_constraint)
            self.attached_constraint = None
            self.attached_object_id = None
            print("  [Mobile] 🧲 Released")
        self._step_physics(0.3)
