def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    tcp = self._tcp_pos()
    # Preserve the source object-relative stack geometry: cubeA center should
    # sit one cube height above cubeB center, with xy aligned to cubeB.
    cube_half = 0.02
    desired_cubeA = np.array(
        [cubeB[0], cubeB[1], cubeB[2] + 2.0 * cube_half],
        dtype=np.float32,
    )
    # Account for the target TCP-to-held-object transform by measuring the
    # current cubeA-to-TCP offset and commanding the TCP to the implied pose.
    for _ in range(18):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA_now = self._entity_pos("cubeA")
        tcp_now = self._tcp_pos()
        offset_now = cubeA_now - tcp_now
        target_tcp = desired_cubeA - offset_now
        err = target_tcp - tcp_now
        if float(np.linalg.norm(err)) <= 0.005:
            break
        self._move_towards(target_tcp, gripper=self.gripper_close, steps=self.move_steps)
    # Fine correction pass driven directly by measured object error.
    for _ in range(8):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA_now = self._entity_pos("cubeA")
        tcp_now = self._tcp_pos()
        obj_err = desired_cubeA - cubeA_now
        if float(np.linalg.norm(obj_err)) <= 0.003:
            break
        self._move_towards(tcp_now + obj_err, gripper=self.gripper_close, steps=self.move_steps)
    cubeA_now = self._entity_pos("cubeA")
    cubeB_now = self._entity_pos("cubeB")
    xy_err = float(np.linalg.norm(cubeA_now[:2] - cubeB_now[:2]))
    z_err = float(abs((cubeA_now[2] - cubeB_now[2]) - 2.0 * cube_half))
    return bool(
        self._is_grasping_entity("cubeA")
        and xy_err <= 0.008
        and z_err <= 0.008
    )
