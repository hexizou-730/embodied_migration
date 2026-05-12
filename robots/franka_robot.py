"""
Franka Emika Panda: 7-DOF arm + 2-finger parallel jaw gripper.

v2 修复:
- Home 位姿调整,末端初始就在桌面上方中央,IK 更容易解
- attach 阈值放宽到 15cm
- 跳过托盘/碗类物体,避免被错误抓取
- 失败时打印详细日志
"""
import numpy as np
import pybullet as p
from typing import Optional, Tuple

from robots.base_robot import BaseRobot
from capabilities import CapabilityCard


class FrankaRobot(BaseRobot):
    embodiment_name = "Franka Panda"
    dof = 7
    gripper_type = "parallel_jaw"

    ARM_JOINTS = list(range(7))
    FINGER_JOINTS = [9, 10]
    EE_LINK_INDEX = 11
    FINGER_OPEN = 0.04
    FINGER_CLOSED = 0.01
    ATTACH_RADIUS = 0.15

    # === 方法 A: Franka 的能力卡 ===
    capability_card = CapabilityCard(
        grasp_mechanism="parallel_jaw",
        stable_when_stacked=True,         # 夹爪张开时有下压效果, 堆叠稳定
        release_must_be_low=False,         # 释放高度容忍度较高
        recommended_release_height_m=0.03,  # 默认 3cm
        workspace_radius_m=0.75,
        can_rotate_object=True,
        max_payload_kg=3.0,
        ik_accuracy_m=0.02,
        extra={
            "notes": "Parallel jaws grip objects from the sides; objects must fit the finger gap.",
        },
    )

    def __init__(self, base_position=(0, 0, 0.6)):
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=list(base_position),
            useFixedBase=True,
        )
        self.num_joints = p.getNumJoints(self.robot_id)
        self.scene = None
        self.attached_constraint: Optional[int] = None
        self.attached_object_id: Optional[int] = None

        self._default_orn = p.getQuaternionFromEuler([np.pi, 0, 0])

        # Home 位姿: 末端朝下,位于桌面中央上方 ~30cm
        # 比原来的 [0,-0.3,0,-2.0,0,1.7,0.785] 更靠前
        home = [0, 0.2, 0, -1.8, 0, 2.0, 0.785]
        for j, q in zip(self.ARM_JOINTS, home):
            p.resetJointState(self.robot_id, j, q)
        self._open_fingers()
        self._step_physics(0.5)

    def _step_physics(self, seconds: float):
        for _ in range(int(seconds * 240)):
            p.stepSimulation()

    def _open_fingers(self):
        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL,
                targetPosition=self.FINGER_OPEN, force=20,
            )
        self._step_physics(0.3)

    def _close_fingers(self):
        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL,
                targetPosition=self.FINGER_CLOSED, force=50,
            )
        self._step_physics(0.3)

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.robot_id, self.EE_LINK_INDEX)
        return np.array(state[4]), np.array(state[5])

    def move_ee_to(self, position, orientation=None, steps=240) -> bool:
        target_pos = list(np.asarray(position, dtype=float))
        target_orn = list(orientation) if orientation is not None else list(self._default_orn)

        try:
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.EE_LINK_INDEX,
                target_pos,
                target_orn,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )
        except Exception as e:
            print(f"  [Franka] IK exception: {e}")
            self._record_action_failure(f"move_ee_to: IK exception: {e}")
            return False

        for j in self.ARM_JOINTS:
            p.setJointMotorControl2(
                self.robot_id, j, p.POSITION_CONTROL,
                targetPosition=joint_poses[j],
                force=500,
            )
        self._step_physics(steps / 240.0)

        actual_pos, _ = self.get_ee_pose()
        err = float(np.linalg.norm(actual_pos - np.array(target_pos)))
        if err > 0.08:
            print(f"  [Franka] move_ee_to failed: target={np.array(target_pos).round(3).tolist()}, "
                  f"actual={actual_pos.round(3).tolist()}, err={err:.3f}m")
            self._record_action_failure(
                f"move_ee_to: target unreachable or inaccurate, err={err:.3f}m"
            )
            return False
        if err > 0.05:
            print(f"  [Franka] move_ee_to warn: target={np.array(target_pos).round(3).tolist()}, "
                  f"actual={actual_pos.round(3).tolist()}, err={err:.3f}m")
        return True

    def activate_gripper(self) -> bool:
        self._close_fingers()
        if self.attached_constraint is not None:
            return True

        ee_pos, _ = self.get_ee_pose()
        if self.scene is None:
            print("  [Franka] activate_gripper: scene not set!")
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
            print(f"  [Franka] ⚠️ No object within {self.ATTACH_RADIUS}m of EE "
                  f"(ee_pos={ee_pos.round(3).tolist()})")
            self._record_action_failure(
                f"activate_gripper: no object within {self.ATTACH_RADIUS}m of end effector"
            )
            return False

        self.attached_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.EE_LINK_INDEX,
            childBodyUniqueId=nearest_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.03],
            childFramePosition=[0, 0, 0],
        )
        self.attached_object_id = nearest_id
        print(f"  [Franka] 🤖 Grasped '{nearest_name}' (was {nearest_dist:.3f}m away)")
        return True

    def release_gripper(self) -> None:
        if self.attached_constraint is not None:
            p.removeConstraint(self.attached_constraint)
            self.attached_constraint = None
            self.attached_object_id = None
            print("  [Franka] 🤖 Released")
        self._open_fingers()
        self._step_physics(0.3)
