def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Use the measured TCP-to-held-object offset so the cube, not the TCP,
    # lands centered above cubeB at the official stack height.
    tcp = self._tcp_pos()
    offset = cubeA - tcp
    target_z = float(cubeB[2]) + 0.04 + float(offset[2])
    align_target = np.array(
        [cubeB[0] - offset[0], cubeB[1] - offset[1], target_z],
        dtype=np.float32,
    )
    for _ in range(10):
        if self._early_stop():
            return False
        self._move_towards(align_target, gripper=self.gripper_close, steps=6)
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2] - cubeB[2]) - 0.04)
        if xy_err <= 0.008 and z_err <= 0.008:
            break
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_ok = float(np.linalg.norm(cubeA[:2] - cubeB[:2])) <= 0.012
    z_ok = abs(float(cubeA[2] - cubeB[2]) - 0.04) <= 0.012
    grasped = bool(self._is_grasping_entity("cubeA"))
    return bool(xy_ok and z_ok and grasped)
