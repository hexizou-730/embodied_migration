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
    # Descend in small increments, correcting XY each time, until cubeA rests at stack height.
    target = np.array([cubeB[0], cubeB[1], stack_z], dtype=np.float32)
    for _ in range(10):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2]) - stack_z)
        if xy_err <= 0.006 and z_err <= 0.008:
            break
        # Aim TCP so that the held cube lands on target, compensating for current offset.
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        aim = target - offset
        self._move_towards(aim, gripper=self.gripper_close, steps=self.move_steps)
    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.008)
    z_ok = bool(abs(float(cubeA[2]) - stack_z) <= 0.01)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
