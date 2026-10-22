def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Source stack geometry: cubeA center should sit one cube height above cubeB center.
    target_z = cubeB[2] + 0.04
    target = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32)
    # Move TCP so the held cubeA lands above cubeB; compensate for TCP-to-cubeA offset.
    for _ in range(8):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        desired_tcp = target - offset
        self._move_towards(desired_tcp, gripper=0.81, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        xy_err = np.linalg.norm(cubeA[:2] - cubeB[:2])
        z_err = abs(cubeA[2] - target_z)
        if xy_err < 0.008 and z_err < 0.008:
            break
    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) < 0.012)
    z_ok = bool(abs(cubeA[2] - target_z) < 0.012)
    return xy_ok and z_ok and bool(self._is_grasping_entity("cubeA"))
