def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Target the held cube's center so it sits directly above cubeB at stack height.
    target_z = cubeB[2] + 0.06
    target = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32)
    for _ in range(10):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        # Correct the TCP by the measured cube-to-TCP offset so the cube, not the TCP, lands on target.
        offset = cubeA - tcp
        cmd_target = target - offset
        self._move_towards(cmd_target, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = float(abs(cubeA[2] - target_z))
        if xy_err < 0.012 and z_err < 0.012:
            break
    cubeA = self._entity_pos("cubeA")
    xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
    z_err = float(abs(cubeA[2] - target_z))
    return bool(xy_err < 0.02 and z_err < 0.02)
