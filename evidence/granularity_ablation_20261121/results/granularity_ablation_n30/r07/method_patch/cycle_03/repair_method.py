def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    half = 0.02
    target_z = float(cubeB[2]) + 2.0 * half

    # Hover above cubeB, keeping the held cube clear of the target.
    for _ in range(10):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        hover_cube = np.array([cubeB[0], cubeB[1], target_z + 0.08], dtype=np.float32)
        hover_tcp = hover_cube - offset
        self._move_towards(hover_tcp, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        if np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.010:
            break

    # Descend onto cubeB, correcting using measured cubeA error via TCP offset.
    for _ in range(20):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        err = np.array([cubeB[0] - cubeA[0],
                        cubeB[1] - cubeA[1],
                        target_z - cubeA[2]], dtype=np.float32)
        norm = float(np.linalg.norm(err))
        if norm > 0.025:
            err = err * (0.025 / norm)
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
