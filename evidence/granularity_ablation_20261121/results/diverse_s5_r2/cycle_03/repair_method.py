def _stage_align(self) -> bool:
    half = 0.02
    for _ in range(40):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        target_z = float(cubeB[2]) + 2.0 * half + float(offset[2])
        align_target = np.array(
            [cubeB[0] - offset[0], cubeB[1] - offset[1], target_z],
            dtype=np.float32,
        )
        self._move_towards(align_target, gripper=self.gripper_close, steps=3)
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2] - cubeB[2]) - 2.0 * half)
        if xy_err <= 0.002 and z_err <= 0.002:
            break
    for _ in range(6):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        target_z = float(cubeB[2]) + 2.0 * half + float(offset[2])
        align_target = np.array(
            [cubeB[0] - offset[0], cubeB[1] - offset[1], target_z],
            dtype=np.float32,
        )
        self._move_towards(align_target, gripper=self.gripper_close, steps=2)
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2] - cubeB[2]) - 2.0 * half)
        if xy_err <= 0.0015 and z_err <= 0.0015:
            break
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_ok = float(np.linalg.norm(cubeA[:2] - cubeB[:2])) <= 0.004
    z_ok = abs(float(cubeA[2] - cubeB[2]) - 2.0 * half) <= 0.004
    grasped = bool(self._is_grasping_entity("cubeA"))
    return bool(xy_ok and z_ok and grasped)
