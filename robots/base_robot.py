"""
BaseRobot: 所有机器人必须实现的统一接口。

v3 改进 (B+A 支持):
- 新增 capability_card: 显式描述 embodiment-specific 能力 (方法 A)
- pick/place 支持 release_height 参数, 让 LLM 可根据能力卡调整
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from capabilities import CapabilityCard


class BaseRobot(ABC):
    embodiment_name: str = "Unknown"
    dof: int = 0
    gripper_type: str = "none"

    # ---- 方法 A: capability_card 作为静态先验 ----
    # 子类覆盖这个属性来声明自己的 embodiment-specific 能力
    capability_card: CapabilityCard = CapabilityCard()

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

    # ---------- Level-1 API ----------
    @abstractmethod
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def move_ee_to(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        steps: int = 240,
    ) -> bool: ...

    @abstractmethod
    def activate_gripper(self) -> bool: ...

    @abstractmethod
    def release_gripper(self) -> None: ...

    # ---------- Level-2 API ----------
    def pick(self, object_position, hover_height: float = 0.15,
             pre_grasp_height: float = 0.02) -> bool:
        """Pick sequence:
        1) 移到 hover_height 正上方
        2) 下降到物体上方 pre_grasp_height 处 (末端和物体中心距离更近)
        3) 吸盘/夹爪激活
        4) 抬回 hover
        """
        pos = np.asarray(object_position, dtype=float)
        above = pos + np.array([0, 0, hover_height])
        grasp = pos + np.array([0, 0, pre_grasp_height])

        if not self.move_ee_to(above):
            return self._fail_action("pick: failed to move above object")
        if not self.move_ee_to(grasp, steps=180):
            return self._fail_action("pick: failed to descend to grasp")
        if not self.activate_gripper():
            return self._fail_action("pick: failed to grasp object")
        # 给 attach 一点时间稳定
        self.move_ee_to(grasp, steps=30)
        if not self.move_ee_to(above, steps=180):
            return self._fail_action("pick: failed to lift")
        return True

    def place(self, target_position, hover_height: float = 0.15,
              pre_release_height: Optional[float] = None) -> bool:
        """Place held object at target_position.

        pre_release_height: 如果为 None, 默认从 capability_card 读取
        (recommended_release_height_m). LLM 在明确任务需要更稳定时
        (例如 stacking), 应该传一个更小的值, 比如 0.005 (5mm).
        """
        pos = np.asarray(target_position, dtype=float)
        if (hasattr(self, "attached_constraint")
                and getattr(self, "attached_constraint") is None):
            return self._fail_action("place: no object is currently held")
        if pre_release_height is None:
            pre_release_height = self.capability_card.recommended_release_height_m
        above = pos + np.array([0, 0, hover_height])
        release_pos = pos + np.array([0, 0, pre_release_height])

        if not self.move_ee_to(above):
            return self._fail_action("place: failed to move above target")
        if not self.move_ee_to(release_pos, steps=180):
            return self._fail_action("place: failed to descend to release")
        self.release_gripper()
        if not self.move_ee_to(above, steps=180):
            return self._fail_action("place: failed to retract")
        return True

    def pick_and_place(self, source_pos, target_pos,
                       place_release_height: Optional[float] = None) -> bool:
        """Pick-and-place with optional fine-grained release height override."""
        if not self.pick(source_pos):
            return False
        return self.place(target_pos, pre_release_height=place_release_height)

    def describe(self) -> str:
        return (
            f"Embodiment: {self.embodiment_name} | "
            f"DoF: {self.dof} | "
            f"Gripper: {self.gripper_type}"
        )
