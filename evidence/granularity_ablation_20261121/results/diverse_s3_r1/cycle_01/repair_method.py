def _stage_align(self) -> bool:
    cubeB = self._entity_pos("cubeB")
    cubeA = self._entity_pos("cubeA")
    half = 0.02
    # Desired held-cube center: directly above cubeB by one cube height.
    target = np.array([cubeB[0], cubeB[1], cubeB[2] + 2.0 * half], dtype=np.float32)

    # Re-grip if the lift stage lost the grasp before aligning.
    if not self._is_grasping_entity("cubeA"):
        self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.settle_steps)

    for _ in range(20):
        if self._early_stop():
            return False
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        offset = cubeA - tcp
        err = target - cubeA
        if float(np.linalg.norm(err[:2])) <= 0.006 and float(abs(err[2])) <= 0.006:
            break
        # Move TCP so the held cube lands on target, compensating TCP-to-cube offset.
        tcp_target = target - offset
        self._move_towards(tcp_target, gripper=self.gripper_close, steps=self.move_steps)
        if not self._is_grasping_entity("cubeA"):
            self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.settle_steps)

    # Settle in place to let the held cube stabilize above cubeB.
    self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.settle_steps)

    cubeA = self._entity_pos("cubeA")
    cubeB = self._entity_pos("cubeB")
    offset = cubeA - cubeB
    xy_ok = float(np.linalg.norm(offset[:2])) <= 0.025
    z_ok = float(abs(offset[2] - 2.0 * half)) <= 0.008
    return bool(self._is_grasping_entity("cubeA") and xy_ok and z_ok)
