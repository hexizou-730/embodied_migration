def _stage_align(self) -> bool:
    cube_h = 0.04
    for _ in range(30):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        desired_cubeA = np.array([cubeB[0], cubeB[1], cubeB[2] + cube_h], dtype=np.float32)
        obj_err = desired_cubeA - cubeA
        if float(np.linalg.norm(obj_err)) <= 0.004:
            break
        offset = cubeA - tcp
        target_tcp = desired_cubeA - offset
        delta = target_tcp - tcp
        norm = float(np.linalg.norm(delta))
        if norm > self.max_delta_m:
            delta = delta * (self.max_delta_m / norm)
        self._repeat_action(delta, gripper=self.gripper_close, steps=1)
    for _ in range(12):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        desired_cubeA = np.array([cubeB[0], cubeB[1], cubeB[2] + cube_h], dtype=np.float32)
        obj_err = desired_cubeA - cubeA
        if float(np.linalg.norm(obj_err)) <= 0.003:
            break
        delta = obj_err
        norm = float(np.linalg.norm(delta))
        if norm > self.max_delta_m:
            delta = delta * (self.max_delta_m / norm)
        self._repeat_action(delta, gripper=self.gripper_close, steps=1)
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
    z_err = float(abs((cubeA[2] - cubeB[2]) - cube_h))
    return bool(
        self._is_grasping_entity("cubeA")
        and xy_err <= 0.008
        and z_err <= 0.008
    )
