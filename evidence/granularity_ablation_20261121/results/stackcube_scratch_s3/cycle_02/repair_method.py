def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    tcp = self._tcp_pos()
    # Target held-object center above cubeB by one cube height (stack geometry).
    target_xy = cubeB[:2]
    target_z = cubeB[2] + 0.04
    # Account for TCP-to-held-object offset measured at runtime.
    offset = cubeA - tcp
    for _ in range(10):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        desired_cubeA = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
        tcp_target = desired_cubeA - offset
        self._move_towards(tcp_target, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        xy_err = float(np.linalg.norm(cubeA[:2] - target_xy))
        z_err = float(abs(cubeA[2] - target_z))
        if xy_err <= 0.008 and z_err <= 0.008:
            break
    cubeA = self._entity_pos("cubeA")
    xy_err = float(np.linalg.norm(cubeA[:2] - target_xy))
    z_err = float(abs(cubeA[2] - target_z))
    return bool(
        self._is_grasping_entity("cubeA")
        and xy_err <= 0.012
        and z_err <= 0.012
    )
