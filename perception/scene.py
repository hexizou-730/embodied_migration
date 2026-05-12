"""
Scene: 管理桌面物体、提供感知 API。

v4 改动: 桌子位置可配置, 支持远距离场景 (mobile 机器人需要先 navigate)。
"""
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, List, Tuple


class TabletopScene:
    """桌面场景: 桌子 + 若干物体 + 机器人。"""

    def __init__(self, table_position=(0.0, 0.0, 0.0)):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", basePosition=list(table_position))
        self.table_position = np.array(table_position)
        # 桌面高度大约 0.63m (table.urdf 默认值)
        self.table_top_z = 0.63

        # object_name -> body_id
        self.object_ids: Dict[str, int] = {}
        # 初始位置缓存 (供 task checker 用相对判定)
        self.object_initial_positions: Dict[str, np.ndarray] = {}

    def add_cube(self, name: str, position, color=(1, 0, 0, 1), size=None) -> int:
        """加一个小立方体 (用 cube_small.urdf, 约 5cm)。"""
        obj_id = p.loadURDF("cube_small.urdf", list(position))
        p.changeVisualShape(obj_id, -1, rgbaColor=list(color))
        self.object_ids[name] = obj_id
        self.object_initial_positions[name] = np.array(position, dtype=float)
        return obj_id

    def add_tray(self, name: str, position, color=(0.8, 0.8, 0.8, 1)) -> int:
        """加一个托盘 (用 tray/tray.urdf)。"""
        obj_id = p.loadURDF("tray/tray.urdf", list(position), globalScaling=0.5)
        p.changeVisualShape(obj_id, -1, rgbaColor=list(color))
        self.object_ids[name] = obj_id
        self.object_initial_positions[name] = np.array(position, dtype=float)
        return obj_id

    # ============================================================
    # 感知 API (LLM 可以在生成的代码里调用)
    # ============================================================
    def get_object_names(self) -> List[str]:
        return list(self.object_ids.keys())

    def get_object_position(self, name: str) -> np.ndarray:
        if name not in self.object_ids:
            raise KeyError(f"Object '{name}' not in scene. Available: {self.get_object_names()}")
        pos, _ = p.getBasePositionAndOrientation(self.object_ids[name])
        return np.array(pos)

    def get_object_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        obj_id = self.object_ids[name]
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        return np.array(pos), np.array(orn)

    def describe(self) -> str:
        """给 LLM 看的场景描述。"""
        lines = ["Objects in the scene:"]
        for name in self.get_object_names():
            pos = self.get_object_position(name)
            lines.append(f"  - {name}: position={pos.round(3).tolist()}")
        return "\n".join(lines)
