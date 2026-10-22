def _stage_align(self) -> bool:
    cubeB = np.asarray(self._entity_pos("cubeB"), dtype=np.float32).reshape(-1)
    cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
    tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
    cube_height = 0.04

    # Re-establish grasp if it was lost during lift/transit.
    if not self._is_grasping_entity("cubeA"):
        self._move_towards(cubeA, gripper=0.0, steps=8)
        self._repeat_action(np.zeros(3), gripper=0.81, steps=8)
        if self._early_stop():
            return False

    # Desired held-object center: directly above cubeB at one cube height.
    desired_A = cubeB.copy()
    desired_A[2] = cubeB[2] + cube_height

    for _ in range(20):
        if self._early_stop():
            return False
        cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
        tcp = np.asarray(self._tcp_pos(), dtype=np.float32).reshape(-1)
        xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
        z_err = float(abs(cubeA[2] - desired_A[2]))
        if xy_err <= 0.006 and z_err <= 0.006:
            break
        # Move TCP by the object's current offset from its desired pose.
        delta = desired_A - cubeA
        target = tcp + delta
        self._move_towards(target, gripper=0.81, steps=5)

    cubeA = np.asarray(self._entity_pos("cubeA"), dtype=np.float32).reshape(-1)
    xy_err = float(np.linalg.norm(cubeA[:2] - cubeB[:2]))
    z_err = float(abs(cubeA[2] - (cubeB[2] + cube_height)))
    aligned = bool(xy_err <= 0.01 and z_err <= 0.01)
    return bool(aligned and self._is_grasping_entity("cubeA"))
