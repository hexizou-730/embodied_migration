def _stage_align(self) -> bool:
    half = 0.02
    target_z = 2.0 * half
    for _ in range(40):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        desired_cubeA = np.array(
            [cubeB[0], cubeB[1], cubeB[2] + target_z], dtype=np.float32
        )
        align_target = desired_cubeA - offset
        delta = align_target - tcp
        norm = float(np.linalg.norm(delta))
        if norm > 0.04:
            delta = delta * (0.04 / norm)
        action = self._make_action(delta, gripper=self.gripper_close)
        self._step(action)
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2] - cubeB[2]) - target_z)
        if xy_err <= 0.004 and z_err <= 0.004:
            break
    for _ in range(20):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        desired_cubeA = np.array(
            [cubeB[0], cubeB[1], cubeB[2] + target_z], dtype=np.float32
        )
        align_target = desired_cubeA - offset
        delta = align_target - tcp
        norm = float(np.linalg.norm(delta))
        if norm > 0.02:
            delta = delta * (0.02 / norm)
        action = self._make_action(delta, gripper=self.gripper_close)
        self._step(action)
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = abs(float(cubeA[2] - cubeB[2]) - target_z)
        if xy_err <= 0.003 and z_err <= 0.003:
            break
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_ok = float(np.linalg.norm(cubeA[:2] - cubeB[:2])) <= 0.006
    z_ok = abs(float(cubeA[2] - cubeB[2]) - target_z) <= 0.006
    grasped = bool(self._is_grasping_entity("cubeA"))
    return bool(xy_ok and z_ok and grasped)
