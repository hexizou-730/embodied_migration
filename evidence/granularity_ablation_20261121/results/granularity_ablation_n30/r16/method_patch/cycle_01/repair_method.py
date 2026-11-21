def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Target cubeA center: directly above cubeB at one cube height (2 * half_size).
    half = 0.02
    target_z = float(cubeB[2]) + 2.0 * half
    target_xy = np.array([cubeB[0], cubeB[1]], dtype=np.float32)

    # Compute TCP-to-held-cube offset so we can drive the cube, not just the TCP.
    offset = np.array(cubeA, dtype=np.float32) - np.array(self._tcp_pos(), dtype=np.float32)

    # Phase 1: raise to a safe hover height above the stack target.
    hover_z = target_z + 0.08
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        desired_cube = np.array([target_xy[0], target_xy[1], hover_z], dtype=np.float32)
        desired_tcp = desired_cube - offset
        delta = desired_tcp - np.array(tcp, dtype=np.float32)
        if np.linalg.norm(delta) <= 0.01:
            break
        self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)

    # Phase 2: align XY precisely over cubeB while staying at hover height.
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        desired_cube = np.array([target_xy[0], target_xy[1], hover_z], dtype=np.float32)
        desired_tcp = desired_cube - offset
        delta = desired_tcp - np.array(tcp, dtype=np.float32)
        if np.linalg.norm(delta[:2]) <= 0.005:
            break
        self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)

    # Phase 3: descend slowly to final stack height, re-reading cube pose each step.
    for _ in range(10):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        desired_cube = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
        desired_tcp = desired_cube - offset
        delta = desired_tcp - np.array(tcp, dtype=np.float32)
        if np.linalg.norm(delta) <= 0.008:
            break
        self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)

    # Settle in place to let the held cube stabilize.
    for _ in range(3):
        if self._early_stop():
            return False
        self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.settle_steps)

    # Measure completion against the official object-relative predicate.
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_err = float(np.linalg.norm(np.array(cubeA[:2]) - np.array(cubeB[:2])))
    z_err = float(abs((float(cubeA[2]) - float(cubeB[2])) - 2.0 * half))
    xy_ok = xy_err <= 0.01
    z_ok = z_err <= 0.01
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
