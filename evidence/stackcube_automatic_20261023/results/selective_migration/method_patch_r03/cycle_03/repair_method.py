def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    stack_z = float(cubeB[2]) + 0.04

    # First bring the held cube's XY over cubeB while staying above stack height.
    for _ in range(12):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        if xy_err <= 0.003:
            break
        # Aim TCP so the held cube's center lands over cubeB; compensate for the
        # TCP-to-cube offset measured from the current grasp.
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        goal = np.array([cubeB[0] - offset[0], cubeB[1] - offset[1],
                         max(float(cubeA[2]), stack_z + 0.02)], dtype=np.float32)
        self._move_towards(goal, gripper=self.gripper_close, steps=self.move_steps)

    # Then descend so the held cube's center sits at stack height above cubeB.
    for _ in range(14):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2]) - stack_z)
        if xy_err <= 0.003 and z_err <= 0.003:
            break
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        goal = np.array([cubeB[0] - offset[0], cubeB[1] - offset[1],
                         stack_z - offset[2]], dtype=np.float32)
        self._move_towards(goal, gripper=self.gripper_close, steps=self.move_steps)

    # Measure completion against the official stack geometry, not TCP arrival.
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    offset = cubeA - cubeB
    xy_ok = bool(np.linalg.norm(offset[:2]) <= 0.025)
    z_ok = bool(abs(float(offset[2]) - 0.04) <= 0.005)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
