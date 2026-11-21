def _stage_align(self) -> bool:
    cube_half = 0.02
    # Re-read measured state at runtime; do not trust the failed snapshot.
    for _ in range(40):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        tcp = self._tcp_pos()
        # Source object-relative stack geometry: cubeA center one cube height
        # above cubeB center, xy aligned to cubeB.
        desired_cubeA = np.array(
            [cubeB[0], cubeB[1], cubeB[2] + 2.0 * cube_half],
            dtype=np.float32,
        )
        # Account for target TCP-to-held-object transform by measuring the
        # current cubeA-to-TCP offset and commanding the implied TCP pose.
        offset_now = cubeA - tcp
        target_tcp = desired_cubeA - offset_now
        obj_err = desired_cubeA - cubeA
        if float(np.linalg.norm(obj_err)) <= 0.004:
            break
        # Bias toward correcting object error directly, bounded per step.
        cmd = target_tcp + 0.5 * obj_err
        self._move_towards(cmd, gripper=self.gripper_close, steps=self.move_steps)
    # Settle in place while holding, then verify measured completion.
    for _ in range(6):
        if self._early_stop():
            return False
        if not self._is_grasping_entity("cubeA"):
            return False
        cubeA = self._entity_pos("cubeA")
        cubeB = self._entity_pos("cubeB")
        desired_cubeA = np.array(
            [cubeB[0], cubeB[1], cubeB[2] + 2.0 * cube_half],
            dtype=np.float32,
        )
        obj_err = desired_cubeA - cubeA
        if float(np.linalg.norm(obj_err)) <= 0.003:
            break
        tcp = self._tcp_pos()
        self._move_towards(tcp + obj_err, gripper=self.gripper_close, steps=self.move_steps)
    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
    z_err = float(abs((cubeA[2] - cubeB[2]) - 2.0 * cube_half))
    return bool(
        self._is_grasping_entity("cubeA")
        and xy_err <= 0.008
        and z_err <= 0.008
    )
