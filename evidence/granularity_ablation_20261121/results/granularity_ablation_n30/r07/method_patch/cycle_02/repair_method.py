def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    half = 0.02
    target_z = float(cubeB[2]) + 2.0 * half

    # Measure the TCP-to-held-cube offset so we can command the TCP to place cubeA.
    tcp = self._tcp_pos()
    offset = cubeA - tcp

    # Hover above cubeB first, keeping the held cube clear of the target.
    hover_cube = np.array([cubeB[0], cubeB[1], target_z + 0.10], dtype=np.float32)
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        hover_tcp = hover_cube - offset
        self._move_towards(hover_tcp, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        if np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.012:
            break

    # Descend and correct using measured cubeA error, commanding TCP via offset.
    for _ in range(14):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        err = np.array([cubeB[0] - cubeA[0],
                        cubeB[1] - cubeA[1],
                        target_z - cubeA[2]], dtype=np.float32)
        norm = float(np.linalg.norm(err))
        if norm > 0.03:
            err = err * (0.03 / norm)
        cmd = err
        self._repeat_action(cmd, gripper=self.gripper_close, steps=1)
        cubeA = self._entity_pos("cubeA")
        if (np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.006
                and abs(float(cubeA[2]) - target_z) <= 0.006):
            break

    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.008)
    z_ok = bool(abs(float(cubeA[2]) - target_z) <= 0.008)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
