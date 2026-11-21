def _stage_release(self) -> bool:
    # Open the gripper in place, then verify the official released/static/on-cube relation.
    for _ in range(8):
        if self._early_stop():
            return False
        self._repeat_action(np.zeros(3, dtype=np.float32),
                            gripper=self.gripper_open, steps=self.settle_steps)
        ev = self._official_evaluation()
        if bool(ev.get("success", False)):
            return True
        if (not bool(ev.get("is_cubeA_grasped", True))
                and bool(ev.get("is_cubeA_static", False))
                and bool(ev.get("is_cubeA_on_cubeB", False))):
            return True

    # If the cube slipped off during opening, re-place it above cubeB and release again.
    for _ in range(4):
        if self._early_stop():
            return False
        ev = self._official_evaluation()
        if bool(ev.get("success", False)):
            return True
        if bool(ev.get("is_cubeA_grasped", False)):
            break
        pos_a = self._entity_pos("cubeA")
        pos_b = self._entity_pos("cubeB")
        target = np.array([pos_b[0], pos_b[1], pos_b[2] + 0.06], dtype=np.float32)
        self._move_towards(target, gripper=self.gripper_open, steps=self.move_steps)
        if self._early_stop():
            return False
        self._repeat_action(np.zeros(3, dtype=np.float32),
                            gripper=self.gripper_open, steps=self.settle_steps)
        ev = self._official_evaluation()
        if bool(ev.get("success", False)):
            return True

    ev = self._official_evaluation()
    return bool(ev.get("success", False))
