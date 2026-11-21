def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Object-relative stack geometry: cubeA center must sit one cube height above cubeB center.
    cube_h = 0.04
    target = np.array([cubeB[0], cubeB[1], cubeB[2] + cube_h], dtype=np.float32)
    # Account for TCP-to-held-object offset measured at runtime.
    offset = cubeA - self._tcp_pos()
    hover = target + np.array([0.0, 0.0, 0.06], dtype=np.float32)
    for _ in range(6):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        offset = cubeA - self._tcp_pos()
        tcp_hover = hover - offset
        self._move_towards(tcp_hover, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        if np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.01:
            break
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        offset = cubeA - self._tcp_pos()
        tcp_target = target - offset
        self._move_towards(tcp_target, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2]) - float(target[2]))
        if xy_err <= 0.008 and z_err <= 0.008:
            break
    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.012)
    z_ok = bool(abs(float(cubeA[2]) - float(target[2])) <= 0.012)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
