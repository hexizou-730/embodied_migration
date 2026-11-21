def _stage_align(self) -> bool:
        cubeB = self._entity_pos("cubeB")
        cubeA = self._entity_pos("cubeA")
        tcp = self._tcp_pos()
        # Preserve the source object-relative stack geometry: cubeA center must
        # sit 2*cube_half_size above cubeB center, xy aligned with cubeB.
        half = 0.02
        desired_cubeA = np.array(
            [cubeB[0], cubeB[1], cubeB[2] + 2.0 * half], dtype=np.float32
        )
        # Account for target TCP-to-held-object transform by re-measuring the
        # held object each iteration and commanding the TCP to compensate.
        for _ in range(18):
            if self._early_stop():
                return False
            if not self._is_grasping_entity("cubeA"):
                return False
            cubeA_now = self._entity_pos("cubeA")
            tcp_now = self._tcp_pos()
            obj_err = desired_cubeA - cubeA_now
            if float(np.linalg.norm(obj_err)) <= 0.004:
                break
            # Move TCP by the object error, clamped to bounded per-step delta.
            delta = obj_err.copy()
            max_delta = 0.04
            norm = float(np.linalg.norm(delta))
            if norm > max_delta:
                delta = delta * (max_delta / norm)
            target_tcp = tcp_now + delta
            self._move_towards(target_tcp, gripper=self.gripper_close, steps=self.move_steps)
        # Fine correction pass using measured object error directly.
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
            delta = obj_err.copy()
            norm = float(np.linalg.norm(delta))
            if norm > 0.02:
                delta = delta * (0.02 / norm)
            self._move_towards(tcp_now + delta, gripper=self.gripper_close, steps=self.move_steps)
        cubeA_now = self._entity_pos("cubeA")
        cubeB_now = self._entity_pos("cubeB")
        xy_err = float(np.linalg.norm(cubeA_now[:2] - cubeB_now[:2]))
        z_err = float(abs((cubeA_now[2] - cubeB_now[2]) - 2.0 * half))
        return bool(
            self._is_grasping_entity("cubeA")
            and xy_err <= 0.010
            and z_err <= 0.010
        )
