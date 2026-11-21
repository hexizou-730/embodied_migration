def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Target the held cube's center above cubeB, compensating for the
    # TCP-to-held-object offset measured at runtime.
    tcp = self._tcp_pos()
    offset = cubeA - tcp
    desired_cubeA = np.array(
        [cubeB[0], cubeB[1], cubeB[2] + 0.04 + 0.02], dtype=np.float32
    )
    align_target = desired_cubeA - offset
    for _ in range(10):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA_now = self._entity_pos("cubeA")
        tcp_now = self._tcp_pos()
        offset_now = cubeA_now - tcp_now
        target_now = desired_cubeA - offset_now
        err = target_now - tcp_now
        if float(np.linalg.norm(err)) <= 0.008:
            break
        self._move_towards(target_now, gripper=self.gripper_close, steps=self.move_steps)
    cubeA_now = self._entity_pos("cubeA")
    xy_err = float(np.linalg.norm(cubeA_now[:2] - cubeB[:2]))
    z_err = float(abs((cubeA_now[2] - cubeB[2]) - 0.04))
    return bool(
        self._is_grasping_entity("cubeA")
        and xy_err <= 0.012
        and z_err <= 0.012
    )
