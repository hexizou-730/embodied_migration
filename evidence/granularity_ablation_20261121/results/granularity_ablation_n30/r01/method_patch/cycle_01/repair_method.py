def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Target cubeA center: directly above cubeB at one full cube height (2 * half_size).
    half = 0.02
    target_z = float(cubeB[2]) + 2.0 * half
    target = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32)

    # First move TCP so the held cubeA hovers above cubeB, using the measured
    # TCP-to-cubeA offset to command the TCP rather than the object directly.
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        hover = np.array([target[0], target[1], target[2] + 0.06], dtype=np.float32)
        tcp_hover = hover - offset
        self._move_towards(tcp_hover, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        if np.linalg.norm(cubeA[:2] - target[:2]) <= 0.01 and abs(float(cubeA[2]) - float(hover[2])) <= 0.02:
            break

    # Descend to the exact stack height, again compensating for the held offset.
    for _ in range(10):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        tcp_target = target - offset
        self._move_towards(tcp_target, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        xy_err = float(np.linalg.norm(cubeA[:2] - target[:2]))
        z_err = abs(float(cubeA[2]) - float(target_z))
        if xy_err <= 0.008 and z_err <= 0.008:
            break

    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - target[:2]) <= 0.012)
    z_ok = bool(abs(float(cubeA[2]) - float(target_z)) <= 0.012)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
