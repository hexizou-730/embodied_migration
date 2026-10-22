def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Source stack geometry: cubeA center should sit one cube height above cubeB center.
    stack_z = float(cubeB[2]) + 0.04
    hover = np.array([cubeB[0], cubeB[1], stack_z + 0.08], dtype=np.float32)
    # First bring the held cube over cubeB in XY while hovering above stack height.
    for _ in range(6):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        if np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.008:
            break
        self._move_towards(hover, gripper=self.gripper_close, steps=self.move_steps)
    # Descend until the held cube's center reaches stack height above cubeB.
    target = np.array([cubeB[0], cubeB[1], stack_z], dtype=np.float32)
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2]) - stack_z)
        if xy_err <= 0.006 and z_err <= 0.006:
            break
        # Correct XY drift toward cubeB while descending toward stack height.
        goal = np.array([cubeB[0], cubeB[1], stack_z], dtype=np.float32)
        self._move_towards(goal, gripper=self.gripper_close, steps=self.move_steps)
    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.008)
    z_ok = bool(abs(float(cubeA[2]) - stack_z) <= 0.008)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
