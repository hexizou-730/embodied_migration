def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    # Target: cubeA center directly above cubeB center at stack height (2 * half_size).
    target_z = float(cubeB[2]) + 0.04
    for _ in range(12):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        # Compensate for TCP-to-held-object offset so cubeA lands over cubeB.
        offset = cubeA - tcp
        desired_tcp = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32) - offset
        self._move_towards(desired_tcp, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04)
        if xy_err < 0.008 and z_err < 0.008:
            break
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_ok = float(np.linalg.norm(cubeA[:2] - cubeB[:2])) <= 0.01
    z_ok = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04) <= 0.01
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
