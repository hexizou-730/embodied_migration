def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    tcp = self._tcp_pos()
    # Account for TCP-to-held-object offset so cubeA lands above cubeB.
    offset = cubeA - tcp
    stack_z = cubeB[2] + 0.04
    align_target = np.array([cubeB[0] - offset[0], cubeB[1] - offset[1], stack_z - offset[2]], dtype=np.float32)
    for _ in range(10):
        if self._early_stop():
            return False
        self._move_towards(align_target, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        align_target = np.array([cubeB[0] - offset[0], cubeB[1] - offset[1], stack_z - offset[2]], dtype=np.float32)
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04)
        if xy_err <= 0.008 and z_err <= 0.008:
            break
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
    z_err = abs(float(cubeA[2]) - float(cubeB[2]) - 0.04)
    grasped = bool(self._is_grasping_entity("cubeA"))
    return bool(xy_err <= 0.012 and z_err <= 0.012 and grasped)
