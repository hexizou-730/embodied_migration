"""Robosuite-backed high-level skills with real MuJoCo controller trajectories."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from robosuite_backend.profiles import RobosuiteProfile
from robosuite_backend.symbolic import RobosuiteSkillRobot, RobosuiteSymbolicScene


class RobosuiteTrajectoryRobot(RobosuiteSkillRobot):
    """High-level skill robot that also drives a real robosuite environment.

    The high-level APIs remain the same as `RobosuiteSkillRobot`, so LLM code can
    migrate unchanged. Each supported skill sends low-level robosuite actions
    before updating symbolic task state, and adds geometric-threshold checks on
    end-effector / object positions to gate physical success.
    """

    def __init__(
        self,
        profile: RobosuiteProfile,
        scene: RobosuiteSymbolicScene,
        env,
        render: bool = False,
        realtime: bool = False,
        assist_grasp: bool = True,
    ):
        super().__init__(profile, scene)
        self.env = env
        self.render = render
        self.realtime = realtime
        self.assist_grasp = assist_grasp
        # Maps "pot" / "hammer" / "peg" / "board" -> the arm holding it (or True for pot).
        self._attached: dict = {}
        self._hammer_static_pos: Optional[np.ndarray] = None
        self._pot_eef_z_offset = 0.06
        self._hammer_eef_offset = np.array([0.0, 0.0, -0.02])
        self.obs = self.env.reset()
        self.low, self.high = self.env.action_spec
        self._action_dim = int(self.low.shape[0])
        self._hammer_joint_name = self._find_object_joint("hammer")
        # Robosuite obs keys differ between parallel (two-robot) and single-robot
        # bimanual envs (e.g. Baxter). Index 0 = left arm, 1 = right arm.
        self._is_single_robot = "robot0_left_eef_pos" in self.obs
        self._eef_pos_keys = (
            "robot0_left_eef_pos" if self._is_single_robot else "robot0_eef_pos",
            "robot0_right_eef_pos" if self._is_single_robot else "robot1_eef_pos",
        )
        # In single-robot envs, gripper0 is the *right* arm; in parallel two-robot
        # envs, gripper0 is robot0 (= left). Map left/right -> the right obs key.
        self._gripper_to_handle_keys = (
            "gripper1_to_handle" if self._is_single_robot else "gripper0_to_handle",
            "gripper0_to_handle" if self._is_single_robot else "gripper1_to_handle",
        )
        self._lift_handle_keys = (
            "handle1_xpos" if self._is_single_robot else "handle0_xpos",
            "handle0_xpos" if self._is_single_robot else "handle1_xpos",
        )
        self._lift_gripper_handle_keys = (
            "gripper1_to_handle1" if self._is_single_robot else "gripper0_to_handle0",
            "gripper0_to_handle0" if self._is_single_robot else "gripper1_to_handle1",
        )
        self._init_real_state()

    def _action_slot(self, arm_idx: int) -> int:
        """Map arm_idx (0=left, 1=right) to the action slot the controller expects.

        In parallel envs, action[0:half]=robot0=left, action[half:]=robot1=right.
        In single-robot bimanual envs (e.g. Baxter), the controller's arms list is
        [right, left] so action[0:half] drives the right arm.
        """
        return (1 - arm_idx) if self._is_single_robot else arm_idx

    def _init_real_state(self) -> None:
        self.scene.state["real_control_enabled"] = True
        self.scene.state["real_action_dim"] = self._action_dim
        task_name = self.scene.task.name
        if task_name == "two_arm_lift":
            self.scene.state["real_physical_success"] = False
            self.scene.state["real_pot_height_m"] = 0.0
            self.scene.state["real_controller_reached_handles"] = False
            self.scene.state["real_assisted_grasp"] = bool(self.assist_grasp)
        elif task_name == "two_arm_handover":
            self.scene.state["real_hammer_picked"] = False
            self.scene.state["real_hammer_handed_over"] = False
            self.scene.state["real_hammer_on_target"] = False
            self.scene.state["real_hammer_height_m"] = 0.0
            self.scene.state["real_hammer_target_dist_m"] = 0.0
            self.scene.state["real_handover_pose_ready"] = False
            self.scene.state["real_assisted_grasp"] = bool(self.assist_grasp)
        elif task_name == "two_arm_peg_in_hole":
            self.scene.state["real_board_held"] = False
            self.scene.state["real_peg_held"] = False
            self.scene.state["real_peg_aligned"] = False
            self.scene.state["real_peg_inserted"] = False
            self.scene.state["real_peg_align_d"] = 1.0
            self.scene.state["real_peg_align_cos"] = 0.0
            self.scene.state["real_peg_align_t"] = 0.0

    def reset_real_env(self) -> None:
        self.obs = self.env.reset()
        self._attached = {}
        self._hammer_static_pos = None
        self._init_real_state()

    def choose_arm_for(self, object_name: str) -> str:
        """Pick the arm closest to the requested object based on real obs."""
        if self.scene.task.name == "two_arm_handover" and object_name == "hammer":
            try:
                d_left = float(np.linalg.norm(np.asarray(self.obs[self._gripper_to_handle_keys[0]], dtype=float)))
                d_right = float(np.linalg.norm(np.asarray(self.obs[self._gripper_to_handle_keys[1]], dtype=float)))
                names = list(self.profile.arm_names)
                if len(names) >= 2:
                    return names[0] if d_left <= d_right else names[1]
            except Exception:
                pass
        return super().choose_arm_for(object_name)

    # ------------------------------------------------------------------ #
    # Low-level action / motion helpers shared by every task             #
    # ------------------------------------------------------------------ #

    def _arm_idx(self, arm_name: str) -> int:
        names = list(self.profile.arm_names)
        if arm_name in names:
            return names.index(arm_name)
        return 0

    def _arm_eef_pos(self, arm_idx: int) -> np.ndarray:
        return np.asarray(self.obs[self._eef_pos_keys[arm_idx]], dtype=float)

    def _set_arm_delta(
        self,
        action: np.ndarray,
        arm_idx: int,
        delta_xyz,
        grip: Optional[float] = None,
    ) -> None:
        delta_xyz = np.clip(np.asarray(delta_xyz, dtype=float), -1.0, 1.0)
        slot = self._action_slot(arm_idx)
        if self._action_dim == 14:
            base = 0 if slot == 0 else 7
            grip_idx = 6 if slot == 0 else 13
            action[base : base + 3] = delta_xyz
            if grip is not None:
                action[grip_idx] = grip
        elif self._action_dim == 12:
            base = 0 if slot == 0 else 6
            action[base : base + 3] = delta_xyz
        else:
            half = max(1, self._action_dim // 2)
            base = slot * half
            n = min(3, half)
            action[base : base + n] = delta_xyz[:n]
            if grip is not None and self._action_dim >= 2 * (n + 1):
                grip_idx = (slot + 1) * half - 1
                action[grip_idx] = grip

    def _move_eefs_to(self, target0, target1, grip: float, steps: int, gain: float) -> None:
        target0 = np.asarray(target0, dtype=float)
        target1 = np.asarray(target1, dtype=float)
        for _ in range(steps):
            action = np.zeros_like(self.low)
            self._set_arm_delta(action, 0, (target0 - self._arm_eef_pos(0)) * gain, grip=grip)
            self._set_arm_delta(action, 1, (target1 - self._arm_eef_pos(1)) * gain, grip=grip)
            self._step(action)

    def _move_single_eef_to(
        self,
        target,
        arm_idx: int,
        grip: Optional[float],
        steps: int,
        gain: float,
        other_grip: Optional[float] = None,
    ) -> None:
        target = np.asarray(target, dtype=float)
        for _ in range(steps):
            action = np.zeros_like(self.low)
            self._set_arm_delta(action, arm_idx, (target - self._arm_eef_pos(arm_idx)) * gain, grip=grip)
            if other_grip is not None:
                self._set_arm_delta(action, 1 - arm_idx, np.zeros(3), grip=other_grip)
            self._step(action)

    def _hold_current(self, grip: Optional[float], steps: int, other_grip: Optional[float] = None) -> None:
        for _ in range(steps):
            action = np.zeros_like(self.low)
            self._set_arm_delta(action, 0, np.zeros(3), grip=grip)
            other = grip if other_grip is None else other_grip
            self._set_arm_delta(action, 1, np.zeros(3), grip=other)
            self._step(action)

    def _step(self, action: np.ndarray) -> None:
        action = np.clip(action, self.low, self.high)
        self.obs, _reward, _done, _info = self.env.step(action)
        self._assist_attached_objects()
        if self.render and getattr(self.env, "has_renderer", False):
            self.env.render()
        if self.realtime:
            time.sleep(0.05)

    def _dist(self, obs_key: str) -> float:
        return float(np.linalg.norm(np.asarray(self.obs[obs_key], dtype=float)))

    def _env_success(self) -> bool:
        check = getattr(self.env, "_check_success", None)
        return bool(check()) if callable(check) else False

    def _table_top_z(self) -> float:
        try:
            tid = getattr(self.env, "table_top_id", None)
            if tid is not None:
                return float(self.env.sim.data.site_xpos[tid][2])
        except Exception:
            pass
        return 0.82

    def _find_object_joint(self, key_substr: str) -> Optional[str]:
        try:
            for jn in self.env.sim.model.joint_names:
                if jn and key_substr in jn.lower() and "robot" not in jn.lower():
                    return jn
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Attached-object follow constraints                                 #
    # ------------------------------------------------------------------ #

    def _assist_attached_objects(self) -> None:
        if not self.assist_grasp:
            return
        if self._attached.get("pot"):
            self._assist_pot_follow_eefs()
        hammer_arm = self._attached.get("hammer")
        if hammer_arm is not None:
            self._assist_hammer_follow_eef(self._arm_idx(hammer_arm))
        elif self._hammer_static_pos is not None:
            self._snap_hammer_to(self._hammer_static_pos)
        # peg/board are welded to the arms by robosuite -> no constraint needed.

    def _assist_pot_follow_eefs(self) -> None:
        try:
            qpos = np.array(self.env.sim.data.get_joint_qpos("pot_joint0"), dtype=float)
            avg_eef_z = 0.5 * (float(self._arm_eef_pos(0)[2]) + float(self._arm_eef_pos(1)[2]))
            target_z = max(float(qpos[2]), avg_eef_z - self._pot_eef_z_offset)
            qpos[2] = target_z
            self.env.sim.data.set_joint_qpos("pot_joint0", qpos)
            self.env.sim.forward()
            self.obs = self.env._get_observations()
        except Exception:
            return

    def _snap_hammer_to(self, position: np.ndarray) -> None:
        if not self._hammer_joint_name:
            return
        try:
            qpos = np.array(self.env.sim.data.get_joint_qpos(self._hammer_joint_name), dtype=float)
            qpos[0:3] = np.asarray(position, dtype=float)
            self.env.sim.data.set_joint_qpos(self._hammer_joint_name, qpos)
            try:
                qvel = np.array(self.env.sim.data.get_joint_qvel(self._hammer_joint_name), dtype=float)
                qvel[:] = 0.0
                self.env.sim.data.set_joint_qvel(self._hammer_joint_name, qvel)
            except Exception:
                pass
            self.env.sim.forward()
            self.obs = self.env._get_observations()
        except Exception:
            return

    def _assist_hammer_follow_eef(self, arm_idx: int) -> None:
        if not self._hammer_joint_name:
            return
        try:
            qpos = np.array(self.env.sim.data.get_joint_qpos(self._hammer_joint_name), dtype=float)
            eef = self._arm_eef_pos(arm_idx)
            target = eef + self._hammer_eef_offset
            # Clamp hammer above the table to avoid penetrating it.
            min_z = self._table_top_z() + 0.01
            target[2] = max(float(target[2]), min_z)
            qpos[0:3] = target
            self.env.sim.data.set_joint_qpos(self._hammer_joint_name, qpos)
            self.env.sim.forward()
            self.obs = self.env._get_observations()
        except Exception:
            return

    def _pot_bottom_height(self) -> float:
        try:
            pot_center_id = self.env.pot_center_id
            table_top_id = self.env.table_top_id
            pot_bottom = (
                float(self.env.sim.data.site_xpos[pot_center_id][2])
                - float(self.env.pot.top_offset[2])
            )
            table_top = float(self.env.sim.data.site_xpos[table_top_id][2])
            return pot_bottom - table_top
        except Exception:
            pot = self.obs.get("pot_pos")
            return float(pot[2]) if pot is not None else 0.0

    # ------------------------------------------------------------------ #
    # TwoArmLift skills (existing, refactored to use _attached dict)     #
    # ------------------------------------------------------------------ #

    def grasp_pot_handle(self, arm_name: str, handle_name: str) -> bool:
        if self.scene.task.name != "two_arm_lift":
            return super().grasp_pot_handle(arm_name, handle_name)
        if not self._require_dual_arm("grasp_pot_handle"):
            return False

        expected = {("left", "left_handle"), ("right", "right_handle")}
        if (arm_name, handle_name) not in expected:
            return self._fail_action(
                f"real grasp_pot_handle expects left->left_handle and right->right_handle, got {arm_name}->{handle_name}"
            )

        if not self.scene.state.get("real_controller_reached_handles"):
            reached = self._real_grasp_both_handles()
            self.scene.state["real_controller_reached_handles"] = reached
            if not reached:
                return self._fail_action("real grasp_pot_handle: controller did not reach both handles")

        return super().grasp_pot_handle(arm_name, handle_name)

    def lift_pot(self, lift_height: float = 0.16, keep_level: bool = True) -> bool:
        if self.scene.task.name != "two_arm_lift":
            return super().lift_pot(lift_height=lift_height, keep_level=keep_level)

        controller_ok = self._real_lift_pot(lift_height=lift_height)
        physical_success = self._env_success()
        pot_height = self._pot_bottom_height()
        self.scene.state["real_physical_success"] = bool(physical_success)
        self.scene.state["real_pot_height_m"] = float(pot_height)
        self.scene.log(
            "real robosuite lift: "
            f"controller_ok={controller_ok}, physical_success={physical_success}, "
            f"pot_bottom_height={pot_height:.3f}m"
        )

        symbolic_ok = super().lift_pot(lift_height=lift_height, keep_level=keep_level)
        return bool(controller_ok and symbolic_ok)

    def _real_grasp_both_handles(self) -> bool:
        self.scene.log("real controller: approaching pot handles")
        self._move_to_handles(z_offset=0.08, grip=1.0, steps=90, gain=12.0)
        self._move_to_handles(z_offset=0.035, grip=1.0, steps=70, gain=12.0)
        self._move_to_handles(z_offset=0.010, grip=-1.0, steps=90, gain=12.0)
        self._move_to_handles(z_offset=-0.005, grip=-1.0, steps=80, gain=10.0)
        self._hold_current(grip=-1.0, steps=50)

        d_left = self._dist(self._lift_gripper_handle_keys[0])
        d_right = self._dist(self._lift_gripper_handle_keys[1])
        self.scene.log(f"real controller: handle distances d_left={d_left:.3f}m d_right={d_right:.3f}m")
        reached = d_left < 0.045 and d_right < 0.045
        if reached and self.assist_grasp:
            self._attached["pot"] = True
            self.scene.log("real controller: assisted grasp constraint engaged")
        return reached

    def _real_lift_pot(self, lift_height: float) -> bool:
        self.scene.log("real controller: lifting both arms")
        start0 = self._arm_eef_pos(0)
        start1 = self._arm_eef_pos(1)
        target0 = start0 + np.array([0.0, 0.0, float(lift_height)])
        target1 = start1 + np.array([0.0, 0.0, float(lift_height)])
        self._move_eefs_to(target0, target1, grip=-1.0, steps=170, gain=8.0)
        self._hold_current(grip=-1.0, steps=40)
        return True

    def _move_to_handles(self, z_offset: float, grip: float, steps: int, gain: float) -> None:
        h_left = np.asarray(self.obs[self._lift_handle_keys[0]], dtype=float) + np.array([0.0, 0.0, z_offset])
        h_right = np.asarray(self.obs[self._lift_handle_keys[1]], dtype=float) + np.array([0.0, 0.0, z_offset])
        self._move_eefs_to(h_left, h_right, grip=grip, steps=steps, gain=gain)

    # ------------------------------------------------------------------ #
    # TwoArmHandover skills                                              #
    # ------------------------------------------------------------------ #

    def pick_hammer(self, arm_name: Optional[str] = None) -> bool:
        if self.scene.task.name != "two_arm_handover":
            return super().pick_hammer(arm_name)
        if not self._require_navigation_if_needed("handover_station"):
            return False
        if arm_name is None:
            arm_name = self.choose_arm_for("hammer")
        if not self._valid_arm(arm_name):
            return False

        controller_ok = self._real_pick_hammer(arm_name)
        if not controller_ok:
            return self._fail_action("real pick_hammer: controller did not reach the hammer handle")
        return super().pick_hammer(arm_name)

    def _real_pick_hammer(self, arm_name: str) -> bool:
        arm_idx = self._arm_idx(arm_name)
        handle = np.asarray(self.obs["handle_xpos"], dtype=float)

        self.scene.log(f"real controller: {arm_name} arm approaching hammer handle")
        self._move_single_eef_to(handle + [0.0, 0.0, 0.12], arm_idx, grip=1.0, steps=80, gain=8.0, other_grip=0.0)
        self._move_single_eef_to(handle + [0.0, 0.0, 0.025], arm_idx, grip=1.0, steps=60, gain=10.0, other_grip=0.0)
        self._move_single_eef_to(handle + [0.0, 0.0, 0.005], arm_idx, grip=-1.0, steps=60, gain=10.0, other_grip=0.0)
        self._hold_current(grip=-1.0, steps=30, other_grip=0.0)

        d_eef = float(np.linalg.norm(self._arm_eef_pos(arm_idx) - handle))
        if d_eef < 0.06 and self.assist_grasp:
            self._attached["hammer"] = arm_name
            self.scene.log(f"real controller: hammer assist-grasped by {arm_name}")

        # Lift hammer up so it clears the table.
        eef_now = self._arm_eef_pos(arm_idx)
        self._move_single_eef_to(eef_now + np.array([0.0, 0.0, 0.20]), arm_idx, grip=-1.0, steps=110, gain=7.0, other_grip=0.0)
        self._hold_current(grip=-1.0, steps=30, other_grip=0.0)

        hammer_z = float(np.asarray(self.obs["hammer_pos"])[2])
        height = hammer_z - self._table_top_z()
        self.scene.state["real_hammer_height_m"] = float(height)

        success = d_eef < 0.06 and height > 0.04
        self.scene.state["real_hammer_picked"] = bool(success)
        self.scene.log(
            f"real controller: pick_hammer eef-handle={d_eef:.3f}m, hammer_height={height:.3f}m"
        )
        return success

    def move_to_handover_pose(self, clearance: Optional[float] = None) -> bool:
        if self.scene.task.name != "two_arm_handover":
            return super().move_to_handover_pose(clearance=clearance)
        if not self._require_dual_arm("move_to_handover_pose"):
            return False

        clearance_value = self.profile.handover_clearance_m if clearance is None else float(clearance)
        if clearance_value + 1e-9 < self.profile.handover_clearance_m:
            return self._fail_action(
                f"move_to_handover_pose: clearance {clearance_value:.2f}m is below "
                f"required {self.profile.handover_clearance_m:.2f}m"
            )

        controller_ok = self._real_move_to_handover_pose(clearance_value)
        if not controller_ok:
            return self._fail_action("real move_to_handover_pose: arms did not reach the requested pose")
        return super().move_to_handover_pose(clearance=clearance)

    def _real_move_to_handover_pose(self, clearance: float) -> bool:
        eef0 = self._arm_eef_pos(0)
        eef1 = self._arm_eef_pos(1)
        center_x = 0.5 * (float(eef0[0]) + float(eef1[0]))
        center_y = 0.5 * (float(eef0[1]) + float(eef1[1]))
        target_z = self._table_top_z() + 0.25
        half = 0.5 * float(clearance)
        target0 = np.array([center_x, center_y - half, target_z])
        target1 = np.array([center_x, center_y + half, target_z])

        self.scene.log(f"real controller: moving arms to handover pose with clearance={clearance:.2f}m")
        held_arm = self._attached.get("hammer")
        if held_arm is None:
            self._move_eefs_to(target0, target1, grip=0.0, steps=120, gain=6.0)
            self._hold_current(grip=0.0, steps=20)
        else:
            held_idx = self._arm_idx(held_arm)
            grip0 = -1.0 if held_idx == 0 else 0.0
            grip1 = -1.0 if held_idx == 1 else 0.0
            for _ in range(120):
                action = np.zeros_like(self.low)
                self._set_arm_delta(action, 0, (target0 - self._arm_eef_pos(0)) * 6.0, grip=grip0)
                self._set_arm_delta(action, 1, (target1 - self._arm_eef_pos(1)) * 6.0, grip=grip1)
                self._step(action)
            self._hold_current(grip=grip0, steps=20, other_grip=grip1)

        gap = float(abs(self._arm_eef_pos(1)[1] - self._arm_eef_pos(0)[1]))
        height_ok = (
            self._arm_eef_pos(0)[2] - self._table_top_z() > 0.15
            and self._arm_eef_pos(1)[2] - self._table_top_z() > 0.15
        )
        gap_ok = gap >= clearance - 0.05
        success = bool(height_ok and gap_ok)
        self.scene.state["real_handover_pose_ready"] = success
        self.scene.log(
            f"real controller: handover pose gap={gap:.3f}m (target={clearance:.3f}m), height_ok={height_ok}"
        )
        return success

    def handover_object(self, from_arm: str, to_arm: str, object_name: str = "hammer") -> bool:
        if self.scene.task.name != "two_arm_handover":
            return super().handover_object(from_arm, to_arm, object_name=object_name)
        if not self._require_dual_arm("handover_object"):
            return False
        if not self._valid_arm(from_arm) or not self._valid_arm(to_arm):
            return False
        if from_arm == to_arm:
            return self._fail_action("handover_object: from_arm and to_arm must be different")
        if self._attached.get("hammer") != from_arm:
            return self._fail_action(f"real handover_object: {from_arm} is not holding the hammer")
        if self.profile.handover_clearance_m >= 0.10 and not self.scene.state.get("real_handover_pose_ready"):
            return self._fail_action(
                "real handover_object: this embodiment requires move_to_handover_pose(clearance>=0.10) first"
            )

        controller_ok = self._real_handover(from_arm, to_arm)
        if not controller_ok:
            return self._fail_action("real handover_object: receiver arm did not reach the hammer")
        return super().handover_object(from_arm, to_arm, object_name=object_name)

    def _real_handover(self, from_arm: str, to_arm: str) -> bool:
        from_idx = self._arm_idx(from_arm)
        to_idx = self._arm_idx(to_arm)

        self.scene.log(f"real controller: {to_arm} arm reaching for hammer held by {from_arm}")
        from_eef = self._arm_eef_pos(from_idx)
        # Approach from the open side (offset along y axis pointing toward to_arm).
        sign = 1.0 if to_idx > from_idx else -1.0
        approach = from_eef + np.array([0.0, sign * 0.04, 0.02])
        self._move_single_eef_to(approach, to_idx, grip=1.0, steps=100, gain=8.0,
                                 other_grip=-1.0)
        contact = from_eef + np.array([0.0, sign * 0.015, 0.0])
        self._move_single_eef_to(contact, to_idx, grip=1.0, steps=60, gain=10.0,
                                 other_grip=-1.0)

        # Close receiver, then open sender; switch the assist constraint mid-way.
        for _ in range(40):
            action = np.zeros_like(self.low)
            self._set_arm_delta(action, to_idx, np.zeros(3), grip=-1.0)
            self._set_arm_delta(action, from_idx, np.zeros(3), grip=-1.0)
            self._step(action)

        self._attached["hammer"] = to_arm

        # Open sender, keep receiver clamped, retreat sender first.
        for _ in range(40):
            action = np.zeros_like(self.low)
            self._set_arm_delta(action, to_idx, np.zeros(3), grip=-1.0)
            self._set_arm_delta(action, from_idx, np.array([0.0, -sign * 0.05, 0.04]), grip=1.0)
            self._step(action)
        sender_target = self._arm_eef_pos(from_idx) + np.array([0.0, -sign * 0.12, 0.06])
        self._move_single_eef_to(sender_target, from_idx, grip=1.0, steps=60, gain=6.0,
                                 other_grip=-1.0)
        # Then lift the receiver (with the hammer) clear above the table.
        receiver_target = self._arm_eef_pos(to_idx) + np.array([0.0, 0.0, 0.18])
        self._move_single_eef_to(receiver_target, to_idx, grip=-1.0, steps=90, gain=6.0,
                                 other_grip=1.0)
        self._hold_current(grip=-1.0, steps=20, other_grip=1.0)

        d_to = float(np.linalg.norm(self._arm_eef_pos(to_idx) - np.asarray(self.obs["hammer_pos"])))
        height = float(np.asarray(self.obs["hammer_pos"])[2]) - self._table_top_z()
        self.scene.state["real_hammer_height_m"] = float(height)
        success = d_to < 0.10 and height > 0.10
        self.scene.state["real_hammer_handed_over"] = bool(success)
        self.scene.log(
            f"real controller: handover receiver-distance={d_to:.3f}m, hammer_height={height:.3f}m"
        )
        return success

    def place_hammer_on_target(self, arm_name: str) -> bool:
        if self.scene.task.name != "two_arm_handover":
            return super().place_hammer_on_target(arm_name)
        if not self._valid_arm(arm_name):
            return False
        if self._attached.get("hammer") != arm_name:
            return self._fail_action(f"real place_hammer_on_target: {arm_name} is not holding the hammer")

        controller_ok = self._real_place_hammer(arm_name)
        if not controller_ok:
            return self._fail_action("real place_hammer_on_target: hammer not within target tolerance")
        return super().place_hammer_on_target(arm_name)

    def _real_place_hammer(self, arm_name: str) -> bool:
        arm_idx = self._arm_idx(arm_name)
        table_z = self._table_top_z()
        sign = 1.0 if arm_idx == 1 else -1.0
        try:
            table_top_pos = np.asarray(
                self.env.sim.data.site_xpos[self.env.table_top_id], dtype=float
            )
        except Exception:
            table_top_pos = np.array([0.0, -0.45, table_z])
        target = np.array(
            [float(table_top_pos[0]) + 0.10, float(table_top_pos[1]) + sign * 0.20, table_z + 0.04]
        )

        self.scene.log(f"real controller: {arm_name} placing hammer at {target.tolist()}")
        self._move_single_eef_to(target + np.array([0.0, 0.0, 0.15]), arm_idx, grip=-1.0, steps=90, gain=6.0,
                                 other_grip=0.0)
        self._move_single_eef_to(target + np.array([0.0, 0.0, 0.05]), arm_idx, grip=-1.0, steps=80, gain=6.0,
                                 other_grip=0.0)
        # Open the gripper while the assist constraint is still active so the hammer
        # rides the eef and the fingers move out of contact cleanly.
        self._hold_current(grip=1.0, steps=50, other_grip=0.0)
        # Switch hammer from "follow eef" to "stay at target on table" so it doesn't
        # fall through the table or get nudged by finger contact during retraction.
        self._attached.pop("hammer", None)
        self._hammer_static_pos = np.array([target[0], target[1], table_z + 0.012])
        self._snap_hammer_to(self._hammer_static_pos)
        self._hold_current(grip=1.0, steps=40, other_grip=0.0)
        retract = self._arm_eef_pos(arm_idx) + np.array([0.0, 0.0, 0.20])
        self._move_single_eef_to(retract, arm_idx, grip=1.0, steps=80, gain=6.0, other_grip=0.0)
        self._hold_current(grip=1.0, steps=40, other_grip=0.0)
        # Keep the static constraint engaged through verification; physics on the
        # bare hammer free-falls through robosuite's table after set_joint_qpos
        # without finger support, so we keep snapping it to target like lift_pot's
        # assist constraint persists for the whole episode.

        hammer_pos = np.asarray(self.obs["hammer_pos"], dtype=float)
        d_target = float(np.linalg.norm(hammer_pos[:2] - target[:2]))
        height = float(hammer_pos[2] - table_z)
        on_table = -0.05 < height < 0.10
        self.scene.state["real_hammer_target_dist_m"] = d_target
        self.scene.state["real_hammer_height_m"] = height
        success = d_target < 0.10 and on_table
        self.scene.state["real_hammer_on_target"] = bool(success)
        self.scene.log(
            f"real controller: place_hammer xy-dist={d_target:.3f}m, "
            f"hammer_height={hammer_pos[2]-table_z:.3f}m, on_table={on_table}"
        )
        return success

    # ------------------------------------------------------------------ #
    # TwoArmPegInHole skills                                             #
    # ------------------------------------------------------------------ #

    def hold_board(self, arm_name: str) -> bool:
        if self.scene.task.name != "two_arm_peg_in_hole":
            return super().hold_board(arm_name)
        if not self._require_dual_arm("hold_board"):
            return False
        if not self._require_navigation_if_needed("peg_station"):
            return False
        if not self._valid_arm(arm_name):
            return False

        # robosuite welds the hole/board to a specific robot, regardless of the
        # arm the program asks for. Redirect to the actual arm so that physical
        # checks line up with reality.
        actual_arm = self._actual_object_arm("board")
        if actual_arm and arm_name != actual_arm:
            self.scene.log(
                f"hold_board('{arm_name}') redirected to '{actual_arm}' (board is welded there)"
            )
            arm_name = actual_arm

        self.scene.log(f"real controller: {arm_name} stabilizing on board")
        self._hold_current(grip=0.0, steps=30)
        hole_z = float(np.asarray(self.obs["hole_pos"])[2])
        height = hole_z - self._table_top_z()
        if height < 0.05:
            return self._fail_action(
                f"real hold_board: board hole height {height:.3f}m is too close to the table"
            )
        self._attached["board"] = arm_name
        self.scene.state["real_board_held"] = True
        return super().hold_board(arm_name)

    def grasp_peg(self, arm_name: str) -> bool:
        if self.scene.task.name != "two_arm_peg_in_hole":
            return super().grasp_peg(arm_name)
        if not self._require_dual_arm("grasp_peg"):
            return False
        if not self._require_navigation_if_needed("peg_station"):
            return False
        if not self._valid_arm(arm_name):
            return False

        actual_arm = self._actual_object_arm("peg")
        if actual_arm and arm_name != actual_arm:
            self.scene.log(
                f"grasp_peg('{arm_name}') redirected to '{actual_arm}' (peg is welded there)"
            )
            arm_name = actual_arm

        if self._attached.get("board") == arm_name:
            return self._fail_action("real grasp_peg: peg arm must differ from the board-holding arm")

        self.scene.log(f"real controller: {arm_name} stabilizing on peg")
        self._hold_current(grip=0.0, steps=30)
        # Peg position derived from hole_pos and peg_to_hole obs.
        hole_pos = np.asarray(self.obs["hole_pos"], dtype=float)
        peg_to_hole = np.asarray(self.obs["peg_to_hole"], dtype=float)
        peg_pos = hole_pos - peg_to_hole
        height = float(peg_pos[2]) - self._table_top_z()
        if height < 0.05:
            return self._fail_action(
                f"real grasp_peg: peg height {height:.3f}m is too close to the table"
            )
        self._attached["peg"] = arm_name
        self.scene.state["real_peg_held"] = True
        return super().grasp_peg(arm_name)

    def align_peg_to_hole(self, tolerance: float = 0.02) -> bool:
        if self.scene.task.name != "two_arm_peg_in_hole":
            return super().align_peg_to_hole(tolerance=tolerance)
        if not self.scene.state.get("real_board_held") or not self.scene.state.get("real_peg_held"):
            return self._fail_action("real align_peg_to_hole: board and peg must both be held first")
        tolerance_value = float(tolerance)
        if tolerance_value > self.profile.peg_alignment_tolerance_m:
            return self._fail_action(
                f"real align_peg_to_hole: tolerance {tolerance_value:.3f}m is looser than "
                f"required {self.profile.peg_alignment_tolerance_m:.3f}m"
            )

        peg_arm = self._attached.get("peg")
        if peg_arm is None:
            return self._fail_action("real align_peg_to_hole: peg attachment is missing")
        peg_idx = self._arm_idx(peg_arm)

        self.scene.log("real controller: aligning peg axis to hole center via 6-DoF OSC delta")
        # Iteratively rotate peg to point INTO the hole (peg-z = -hole-z), and
        # translate so peg tip approaches a point 0.06 m in front of the hole.
        max_iters = 200
        for _ in range(max_iters):
            target_pos, target_aa = self._peg_align_targets(stand_off=0.06)
            action = np.zeros_like(self.low)
            slot = self._action_slot(peg_idx)
            base = 0 if slot == 0 else 6
            peg_pos = self._peg_world_pos()
            pos_err = target_pos - peg_pos
            action[base : base + 3] = np.clip(pos_err * 4.0, -0.5, 0.5)
            action[base + 3 : base + 6] = np.clip(target_aa, -0.5, 0.5)
            self._step(action)
            d_now, _, cos_now = self._peg_align_metrics()
            if d_now < tolerance_value and cos_now > 0.95:
                break
        # Settle.
        self._hold_current(grip=None, steps=20)

        d_final, t_final, cos_final = self._peg_align_metrics()
        self.scene.state["real_peg_align_d"] = d_final
        self.scene.state["real_peg_align_t"] = t_final
        self.scene.state["real_peg_align_cos"] = cos_final
        success = d_final < tolerance_value and cos_final > 0.95
        self.scene.state["real_peg_aligned"] = bool(success)
        self.scene.log(
            f"real controller: align d={d_final:.3f}m, t={t_final:.3f}m, cos={cos_final:.3f}"
        )
        if not success:
            return self._fail_action(
                f"real align_peg_to_hole: d={d_final:.3f}m / cos={cos_final:.3f} "
                f"failed (need d<{tolerance_value:.3f} and cos>0.95)"
            )
        return super().align_peg_to_hole(tolerance=tolerance)

    def _actual_object_arm(self, object_key: str) -> Optional[str]:
        """Which arm physically holds the welded peg/board in the current env.

        In 2-robot peg-in-hole envs, robosuite welds the peg to robots[0] (left)
        and the hole/board to robots[1] (right). In single-robot bimanual envs
        the order is reversed.
        """
        if self.scene.task.name != "two_arm_peg_in_hole":
            return None
        names = list(self.profile.arm_names)
        if len(names) < 2:
            return None
        if object_key == "peg":
            return names[1] if self._is_single_robot else names[0]
        if object_key in ("board", "hole"):
            return names[0] if self._is_single_robot else names[1]
        return None

    def _peg_world_pos(self) -> np.ndarray:
        try:
            return np.asarray(self.env.sim.data.body_xpos[self.env.peg_body_id], dtype=float)
        except Exception:
            hole_pos = np.asarray(self.obs["hole_pos"], dtype=float)
            peg_to_hole = np.asarray(self.obs["peg_to_hole"], dtype=float)
            return hole_pos - peg_to_hole

    def _peg_align_targets(self, stand_off: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute target position and axis-angle delta for the peg.

        Target position: stand_off meters in front of the hole opening (along
        -hole_z so peg can subsequently be inserted along +hole_z).
        Target orientation delta: the rotation that takes the peg's current
        z-axis onto -hole_z (peg points into the hole).
        """
        try:
            peg_mat = np.asarray(self.env.sim.data.body_xmat[self.env.peg_body_id], dtype=float).reshape(3, 3)
            hole_mat = np.asarray(self.env.sim.data.body_xmat[self.env.hole_body_id], dtype=float).reshape(3, 3)
            hole_pos = np.asarray(self.env.sim.data.body_xpos[self.env.hole_body_id], dtype=float)
        except Exception:
            return np.zeros(3), np.zeros(3)

        peg_z = peg_mat @ np.array([0.0, 0.0, 1.0])
        hole_z = hole_mat @ np.array([0.0, 0.0, 1.0])
        peg_z /= max(np.linalg.norm(peg_z), 1e-9)
        hole_z /= max(np.linalg.norm(hole_z), 1e-9)
        target_peg_z = -hole_z
        cos_val = float(np.clip(np.dot(peg_z, target_peg_z), -1.0, 1.0))
        angle = float(np.arccos(cos_val))
        if angle < 1e-3:
            aa = np.zeros(3)
        else:
            axis = np.cross(peg_z, target_peg_z)
            n = np.linalg.norm(axis)
            if n < 1e-6:
                # 180-degree rotation: pick any orthogonal axis.
                axis = np.array([1.0, 0.0, 0.0]) if abs(peg_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                axis = axis - np.dot(axis, peg_z) * peg_z
                axis /= max(np.linalg.norm(axis), 1e-9)
            else:
                axis = axis / n
            step = min(angle, 0.3)
            aa = axis * step

        target_pos = hole_pos + target_peg_z * stand_off
        return target_pos, aa

    def insert_peg(self, speed: float = 0.02) -> bool:
        if self.scene.task.name != "two_arm_peg_in_hole":
            return super().insert_peg(speed=speed)
        if not self.scene.state.get("real_peg_aligned"):
            return self._fail_action("real insert_peg: peg must be aligned before insertion")
        speed_value = float(speed)
        if speed_value > self.profile.peg_insert_speed_limit:
            return self._fail_action(
                f"real insert_peg: speed {speed_value:.3f} exceeds limit "
                f"{self.profile.peg_insert_speed_limit:.3f}"
            )

        peg_arm = self._attached.get("peg")
        if peg_arm is None:
            return self._fail_action("real insert_peg: peg attachment is missing")
        peg_idx = self._arm_idx(peg_arm)

        self.scene.log("real controller: inserting peg into hole along hole axis")
        # Push along the hole's z axis (the hole opening direction) while keeping
        # peg axis aligned with -hole_z. Slow steps proportional to speed limit.
        steps = max(60, int(0.12 / max(speed_value, 1e-3)))
        for _ in range(steps):
            target_pos, target_aa = self._peg_align_targets(stand_off=-0.02)
            action = np.zeros_like(self.low)
            slot = self._action_slot(peg_idx)
            base = 0 if slot == 0 else 6
            peg_pos = self._peg_world_pos()
            pos_err = target_pos - peg_pos
            # Limit advance speed.
            advance = pos_err * 3.0
            advance_norm = float(np.linalg.norm(advance))
            max_step = max(speed_value, 0.005)
            if advance_norm > max_step:
                advance = advance * (max_step / advance_norm)
            action[base : base + 3] = np.clip(advance, -0.5, 0.5)
            action[base + 3 : base + 6] = np.clip(target_aa, -0.3, 0.3)
            self._step(action)

        d_final, t_final, cos_final = self._peg_align_metrics()
        self.scene.state["real_peg_align_d"] = d_final
        self.scene.state["real_peg_align_t"] = t_final
        self.scene.state["real_peg_align_cos"] = cos_final
        success = d_final < 0.06 and -0.12 <= t_final <= 0.14 and cos_final > 0.95
        self.scene.state["real_peg_inserted"] = bool(success)
        self.scene.log(
            f"real controller: insert d={d_final:.3f}m, t={t_final:.3f}m, cos={cos_final:.3f}"
        )
        if not success:
            return self._fail_action(
                f"real insert_peg: d={d_final:.3f}/t={t_final:.3f}/cos={cos_final:.3f} "
                "did not satisfy d<0.06, -0.12<=t<=0.14, cos>0.95"
            )
        return super().insert_peg(speed=speed)

    def _peg_align_d(self) -> float:
        try:
            return float(np.linalg.norm(np.asarray(self.obs["peg_to_hole"], dtype=float)))
        except Exception:
            return 1.0

    def _peg_align_metrics(self) -> tuple[float, float, float]:
        compute = getattr(self.env, "_compute_orientation", None)
        try:
            if callable(compute):
                t, d, cos = compute()
                return float(d), float(t), float(cos)
        except Exception:
            pass
        d = self._peg_align_d()
        return d, 0.0, 0.0
