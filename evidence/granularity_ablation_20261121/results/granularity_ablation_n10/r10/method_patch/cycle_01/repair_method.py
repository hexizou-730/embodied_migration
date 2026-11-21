def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    # Target cubeA center: directly above cubeB at one cube height (2 * half_size).
    half = 0.02
    target_z = float(cubeB[2]) + 2.0 * half
    target = np.array([cubeB[0], cubeB[1], target_z], dtype=np.float32)

    # Hover above the target first to avoid dragging cubeA sideways into cubeB.
    hover = np.array([cubeB[0], cubeB[1], target_z + 0.08], dtype=np.float32)
    for _ in range(6):
        if self._early_stop():
            return False
        self._move_towards(hover, gripper=self.gripper_close, steps=self.move_steps)
        cubeA = self._entity_pos("cubeA")
        if np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.01:
            break

    # Descend to stack height, correcting XY each iteration using measured cubeA.
    for _ in range(10):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        err = target - cubeA
        if np.linalg.norm(err[:2]) <= 0.006 and abs(float(err[2])) <= 0.008:
            break
        # Command TCP by the measured cubeA error (TCP-to-held-object offset is roughly constant).
        step_target = self._tcp_pos() + err
        self._move_towards(step_target, gripper=self.gripper_close, steps=self.move_steps)

    cubeA = self._entity_pos("cubeA")
    xy_ok = bool(np.linalg.norm(cubeA[:2] - cubeB[:2]) <= 0.008)
    z_ok = bool(abs(float(cubeA[2]) - target_z) <= 0.01)
    return bool(xy_ok and z_ok and self._is_grasping_entity("cubeA"))
