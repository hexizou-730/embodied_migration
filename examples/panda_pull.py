"""Panda source actions for PullCube-v1.

The file is input behavior, not a target adapter. Run:

    python migrate.py examples/panda_pull.py xarm6_robotiq
"""

from __future__ import annotations

from typing import Any

import numpy as np

from maniskill_backend.dynamic_adapter import ManiSkillDynamicRobot, SkillTarget


ENV_ID = "PullCube-v1"
SOURCE_ROBOT = "panda"
CONTROL_MODE = "pd_ee_delta_pos"

TASK_PROGRAM = '''
cube = scene.get_object("cube")
goal = scene.get_region("goal")
ret_val = robot.pull(cube, goal)
'''


class PandaPullSource(ManiSkillDynamicRobot):
    """Existing Panda behavior used as the migration source."""

    def pull(self, obj: SkillTarget, target: SkillTarget) -> bool:
        cube_pos = self._entity_pos(obj.name)
        goal_pos = self._entity_pos(target.name)
        contact = cube_pos + np.array([0.07, 0.0, 0.02], dtype=np.float32)
        pre_contact = contact + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        drag_end = np.array([goal_pos[0] - 0.02, goal_pos[1], contact[2]], dtype=np.float32)

        self._move_towards(pre_contact, gripper=-1.0, steps=14)
        self._move_towards(contact, gripper=-1.0, steps=14)
        for stage in range(1, 5):
            waypoint = contact + (drag_end - contact) * (stage / 4.0)
            self._move_towards(waypoint, gripper=-1.0, steps=11)
            if self._official_success():
                return self._log("pull", {"obj": obj.name, "target": target.name}, True, True)

        self._repeat_action(np.zeros(3, dtype=np.float32), gripper=-1.0, steps=10)
        success = self._official_success()
        return self._log(
            "pull",
            {"obj": obj.name, "target": target.name},
            success,
            success,
            "" if success else "official task success was not reached",
        )


def build_robot(env: Any, *, control_mode: str, robot_uid: str) -> PandaPullSource:
    return PandaPullSource(env, control_mode=control_mode, robot_uid=robot_uid)
