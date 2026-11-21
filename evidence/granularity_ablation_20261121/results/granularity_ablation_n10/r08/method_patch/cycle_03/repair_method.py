def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    half = 0.02
    target_z = float(cubeB[2]) + 2.0 * half

    # Re-read held-cube offset relative to TCP each iteration so the commanded
    # TCP target places cubeA (not the TCP) at the stack pose above cubeB.
    def _offset():
        return self._entity_pos("cubeA") - self._tcp_pos()

    # Phase 1: move above the stack, correcting XY from measured held-cube offset.
    hover_z = target_z + 0.08
    for _ in range(20):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        desired_tcp = np.array([cubeB[0], cubeB[1], hover_z], dtype=np.float32) - offset
        err = desired_tcp - tcp
        norm = float(np.linalg.norm(err))
        if norm > 0.04:
            err = err * (0.04 / norm)
        self._repeat_action(err, gripper=self.gripper_close, steps=1)
        cubeA = self._entity_pos("cubeA")
        if (np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.008
                and abs(float(cubeA[2]) - hover_z) <= 0.012):
            break

    # Phase 2: descend to exact stack height, correcting XY and Z from offset.
    for _ in range(24):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        desired_tcp = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32) - offset
        err = desired_tcp - tcp
        norm = float(np.linalg.norm(err))
        if norm > 0.04:
            err = err * (0.04 / norm)
        self._repeat_action(err, gripper=self.gripper_close, steps=1)
        cubeA = self._entity_pos("cubeA")
        if (np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.005
                and abs(float(cubeA[2]) - target_z) <= 0.005):
            break

    # Phase 3: settle in place while holding, re-correcting drift.
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        desired_tcp = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32) - offset
        err = desired_tcp - tcp
        norm = float(np.linalg.norm(err))
        if norm > 0.02:
            err = err * (0.02 / norm)
        self._repeat_action(err, gripper=self.gripper_close, steps=1)

    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.006)
    z_ok = bool(abs(float(cubeA[2]) - target_z) <= 0.006)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
