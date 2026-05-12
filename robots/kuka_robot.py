"""
KUKA iiwa 7-DOF with simulated suction gripper.

v2 修复:
- 用 createConstraint 代替手动 resetBasePositionAndOrientation (不再抖动)
- attach 阈值从 10cm 放宽到 15cm (容忍 IK 误差)
- Home 位姿让末端初始就在桌面上方
- 失败时打印详细日志
"""
import numpy as np
import pybullet as p
from typing import Optional, Tuple

from robots.base_robot import BaseRobot
from capabilities import CapabilityCard


class KukaRobot(BaseRobot):
    embodiment_name = "KUKA iiwa"
    dof = 7
    gripper_type = "suction"

    ATTACH_RADIUS = 0.15

    # === 方法 A: KUKA 的能力卡 ===
    capability_card = CapabilityCard(
        grasp_mechanism="suction",
        stable_when_stacked=False,        # 吸盘释放时物体会自由下落, 堆叠不稳
        release_must_be_low=True,          # 必须贴近目标点释放
        recommended_release_height_m=0.005,  # 建议 5mm 释放
        workspace_radius_m=0.85,
        can_rotate_object=False,
        max_payload_kg=3.0,
        ik_accuracy_m=0.03,
        extra={
            "notes": "Suction cup has no lateral grip; objects rotate freely during transport.",
        },
    )

    def __init__(self, base_position=(0, 0, 0.6)):
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=list(base_position),
            useFixedBase=True,
        )
        self.ee_link_index = 6
        self.num_joints = p.getNumJoints(self.robot_id)

        self.attached_constraint: Optional[int] = None
        self.attached_object_id: Optional[int] = None
        self.scene = None

        self._default_orn = p.getQuaternionFromEuler([0, np.pi, 0])

        # Home 位姿 - 让末端初始在桌面中上方而不是笔直朝上
        home = [0, 0.4, 0, -1.2, 0, 1.0, 0]
        for j, q in enumerate(home):
            if j < self.num_joints:
                p.resetJointState(self.robot_id, j, q)

        self._step_physics(0.5)

    def _step_physics(self, seconds: float):
        for _ in range(int(seconds * 240)):
            p.stepSimulation()

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
            print(f"  [KUKA] IK exception: {e}")
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
        if err > 0.08:
            print(f"  [KUKA] move_ee_to failed: target={np.array(target_pos).round(3).tolist()}, "
                  f"actual={actual_pos.round(3).tolist()}, err={err:.3f}m")
            self._record_action_failure(
                f"move_ee_to: target unreachable or inaccurate, err={err:.3f}m"
            )
            return False
        if err > 0.05:
            print(f"  [KUKA] move_ee_to warn: target={np.array(target_pos).round(3).tolist()}, "
                  f"actual={actual_pos.round(3).tolist()}, err={err:.3f}m")
        return True

    def activate_gripper(self) -> bool:
        if self.attached_constraint is not None:
            return True

        ee_pos, _ = self.get_ee_pose()
        if self.scene is None:
            print("  [KUKA] activate_gripper: scene not set!")
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
            print(f"  [KUKA] ⚠️ No object within {self.ATTACH_RADIUS}m of EE "
                  f"(ee_pos={ee_pos.round(3).tolist()})")
            self._record_action_failure(
                f"activate_gripper: no object within {self.ATTACH_RADIUS}m of end effector"
            )
            return False

        # 让 constraint 把物体固定到末端下方 2cm
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
        print(f"  [KUKA] 🧲 Attached '{nearest_name}' (was {nearest_dist:.3f}m away)")
        return True

    def release_gripper(self) -> None:
        if self.attached_constraint is not None:
            p.removeConstraint(self.attached_constraint)
            self.attached_constraint = None
            self.attached_object_id = None
            print("  [KUKA] 🧲 Released")
        self._step_physics(0.3)
