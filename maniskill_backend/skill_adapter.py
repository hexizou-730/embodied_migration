"""High-level skills backed by ManiSkill actions.

This module is the bridge from LMP-style task code to real ManiSkill execution.
It supports PickCube-v1 and PegInsertionSide-v1; both adapters drive the env
through the same skill API (grasp / align_to_target / insert / place) so the
same LMP code can run across embodiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SkillTarget:
    name: str
    kind: str


class ManiSkillSceneAdapter:
    def get_object(self, name: str) -> SkillTarget:
        return SkillTarget(name=name, kind="object")

    def get_region(self, name: str) -> SkillTarget:
        return SkillTarget(name=name, kind="region")


class ManiSkillPickCubeRobot:
    """PickCube-v1 skill wrapper using ManiSkill's real env.step(action) API.

    Assumptions for the first version:
    - the environment is PickCube-v1 or one of its robot-specific variants;
    - the control mode accepts an end-effector delta-position style Box action;
    - the first three action dimensions control xyz delta and the last dimension
      controls the gripper. Different robot controllers use different gripper
      signs, so open/close commands are configurable.
    """

    def __init__(
        self,
        env: Any,
        *,
        move_steps: int = 12,
        grip_steps: int = 8,
        settle_steps: int = 8,
        max_delta_m: float = 0.08,
        pregrasp_clearance_m: float = 0.02,
        release_clearance_m: float = 0.035,
        above_clearance_m: float = 0.10,
        gripper_open: float = 1.0,
        gripper_close: float = -1.0,
        control_mode: Optional[str] = None,
    ) -> None:
        self.env = env
        self.move_steps = move_steps
        self.grip_steps = grip_steps
        self.settle_steps = settle_steps
        self.max_delta_m = max_delta_m
        self.pregrasp_clearance_m = pregrasp_clearance_m
        self.release_clearance_m = release_clearance_m
        self.above_clearance_m = above_clearance_m
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        self.control_mode = control_mode
        self.last_info: Dict[str, Any] = {}
        self.terminated: bool = False
        self.truncated: bool = False
        self.events: List[Dict[str, Any]] = []
        self.tcp_to_obj_at_grasp: Optional[np.ndarray] = None
        self._validate_action_space()

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and not self.control_mode.startswith("pd_ee_delta_"):
            raise ValueError(
                "PickCube adapter requires a pd_ee_delta_* control mode, "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("PickCube adapter currently requires a Box-like action_space.")
        if shape[-1] not in (4, 7):
            raise RuntimeError(
                "PickCube adapter expects action_space last dim in {4, 7} "
                f"(pd_ee_delta_pos or pd_ee_delta_pose), got shape {tuple(shape)!r}."
            )

    def grasp(self, obj: SkillTarget) -> bool:
        if obj.name != "cube":
            return self._fail("grasp", {"obj": obj.name}, "PickCube adapter only supports cube grasp.")

        cube_pos = self._actor_pos("cube")
        above = cube_pos + np.array([0.0, 0.0, self.above_clearance_m])
        pregrasp = cube_pos + np.array([0.0, 0.0, self.pregrasp_clearance_m])
        self._move_towards(above, gripper=self.gripper_open, steps=self.move_steps)
        self._move_towards(pregrasp, gripper=self.gripper_open, steps=self.move_steps)
        self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.grip_steps)

        grasped_after_close = self._info_bool("is_grasped") or self._agent_is_grasping("cube")
        self._log(
            "grasp_check_after_close",
            {"obj": obj.name},
            grasped_after_close,
            grasped_after_close,
            "" if grasped_after_close else "gripper closed but cube not held",
        )

        self._move_towards(above, gripper=self.gripper_close, steps=self.move_steps)
        grasped_after_lift = self._info_bool("is_grasped") or self._agent_is_grasping("cube")

        if grasped_after_lift:
            message = ""
        elif grasped_after_close:
            message = "cube slipped during lift"
        else:
            message = "cube was not grasped"
        if grasped_after_lift:
            self.tcp_to_obj_at_grasp = self._tcp_pos() - self._actor_pos("cube")
        return self._log("grasp", {"obj": obj.name}, grasped_after_lift, grasped_after_lift, message)

    def place(self, obj: SkillTarget, target: SkillTarget) -> bool:
        if obj.name != "cube":
            return self._fail("place", {"obj": obj.name, "target": target.name}, "PickCube adapter only supports cube.")
        if target.name not in {"goal", "goal_site"}:
            return self._fail("place", {"obj": obj.name, "target": target.name}, "PickCube target must be goal.")

        goal_pos = self._region_pos(target.name)
        tcp_goal = goal_pos + self._held_tcp_offset()
        above = tcp_goal + np.array([0.0, 0.0, self.above_clearance_m])
        self._move_towards(above, gripper=self.gripper_close, steps=self.move_steps)
        self._move_towards(tcp_goal, gripper=self.gripper_close, steps=self.move_steps)

        # PickCube's official success condition is "object at goal"; it does
        # not require dropping the cube. Some grippers, especially Robotiq on
        # xarm6, can disturb the cube during opening, so check while still held.
        self._repeat_action(np.zeros(3), gripper=self.gripper_close, steps=self.settle_steps)
        if self._pick_cube_success():
            return self._log(
                "place",
                {"obj": obj.name, "target": target.name},
                True,
                True,
                "cube moved to goal while held",
            )

        self._repeat_action(np.zeros(3), gripper=self.gripper_open, steps=self.grip_steps)
        self._move_towards(above, gripper=self.gripper_open, steps=self.move_steps)
        self._repeat_action(np.zeros(3), gripper=self.gripper_open, steps=self.settle_steps)

        ok = self._pick_cube_success()
        diagnostics = self._placement_diagnostics(goal_pos)
        message = "" if ok else f"cube was not placed at goal; {diagnostics}"
        return self._log("place", {"obj": obj.name, "target": target.name}, ok, ok, message)

    def align_to_target(self, obj: SkillTarget, target: SkillTarget, tolerance: float) -> bool:
        target_pos = self._region_pos(target.name) if target.kind == "region" else self._actor_pos(target.name)
        self._move_towards(
            target_pos + np.array([0.0, 0.0, 0.08]),
            gripper=self.gripper_close,
            steps=self.move_steps,
        )
        obj_pos = self._actor_pos(obj.name)
        ok = float(np.linalg.norm(obj_pos - target_pos)) <= float(tolerance)
        return self._log(
            "align_to_target",
            {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
            ok,
            ok,
            "" if ok else "object is not within requested tolerance",
        )

    def insert(self, obj: SkillTarget, target: SkillTarget, speed: float) -> bool:
        return self._fail(
            "insert",
            {"obj": obj.name, "target": target.name, "speed": float(speed)},
            "insert is not implemented for PickCube-v1",
        )

    def hook_object(self, tool: SkillTarget, obj: SkillTarget) -> bool:
        return self._fail("hook_object", {"tool": tool.name, "obj": obj.name}, "tool use is not implemented for PickCube-v1")

    def pull_with_tool(self, tool: SkillTarget, obj: SkillTarget, target: SkillTarget) -> bool:
        return self._fail(
            "pull_with_tool",
            {"tool": tool.name, "obj": obj.name, "target": target.name},
            "tool use is not implemented for PickCube-v1",
        )

    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def _move_towards(self, target_pos: np.ndarray, *, gripper: float, steps: int) -> None:
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            tcp = self._tcp_pos()
            delta = np.asarray(target_pos, dtype=np.float32) - tcp
            if np.linalg.norm(delta) < 0.01:
                break
            clipped = np.clip(delta / self.max_delta_m, -1.0, 1.0)
            self._step(self._make_action(clipped, gripper=gripper))

    def _repeat_action(self, delta_xyz: np.ndarray, *, gripper: float, steps: int) -> None:
        action = self._make_action(delta_xyz, gripper=gripper)
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            self._step(action)

    def _early_stop(self) -> bool:
        return bool(self.terminated or self.truncated)

    def _make_action(self, delta_xyz: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        shape = getattr(space, "shape", None)
        if not shape:
            raise RuntimeError("PickCube adapter currently requires a Box-like action_space.")
        action = np.zeros(shape, dtype=getattr(space, "dtype", np.float32))
        flat = action.reshape(-1)
        flat[: min(3, flat.size)] = np.asarray(delta_xyz, dtype=np.float32)[: min(3, flat.size)]
        if flat.size >= 4:
            flat[-1] = float(gripper)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def _step(self, action: Any) -> None:
        _, _, terminated, truncated, info = self.env.step(action)
        self.last_info = dict(info or {})
        self.terminated = self.terminated or _scalar_bool(terminated)
        self.truncated = self.truncated or _scalar_bool(truncated)

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _actor_pos(self, name: str) -> np.ndarray:
        actor = getattr(self._base_env(), name)
        return _to_numpy(actor.pose.p)

    def _region_pos(self, name: str) -> np.ndarray:
        if name == "goal":
            name = "goal_site"
        return self._actor_pos(name)

    def _tcp_pos(self) -> np.ndarray:
        return _to_numpy(self._base_env().agent.tcp_pose.p)

    def _agent_is_grasping(self, name: str) -> bool:
        try:
            actor = getattr(self._base_env(), name)
            value = self._base_env().agent.is_grasping(actor)
            return bool(_to_numpy(value).reshape(-1)[0])
        except Exception:
            return False

    def _info_bool(self, key: str) -> bool:
        if key not in self.last_info:
            return False
        value = self.last_info[key]
        try:
            return bool(_to_numpy(value).reshape(-1)[0])
        except Exception:
            return bool(value)

    def _pick_cube_success(self) -> bool:
        if self._info_bool("success") or self._info_bool("is_obj_placed"):
            return True
        try:
            result = self._base_env().evaluate()
            if isinstance(result, dict):
                return _dict_bool(result, "success") or _dict_bool(result, "is_obj_placed")
        except Exception:
            pass
        return False

    def _held_tcp_offset(self) -> np.ndarray:
        if self.tcp_to_obj_at_grasp is not None:
            return np.asarray(self.tcp_to_obj_at_grasp, dtype=np.float32)
        return np.array([0.0, 0.0, self.release_clearance_m], dtype=np.float32)

    def _placement_diagnostics(self, goal_pos: np.ndarray) -> str:
        try:
            cube_pos = self._actor_pos("cube")
            tcp_pos = self._tcp_pos()
            obj_goal_dist = float(np.linalg.norm(goal_pos - cube_pos))
            tcp_goal_dist = float(np.linalg.norm(goal_pos - tcp_pos))
            offset = self._held_tcp_offset()
            return (
                f"obj_goal_dist={obj_goal_dist:.4f}, "
                f"tcp_goal_dist={tcp_goal_dist:.4f}, "
                f"tcp_to_obj_offset={np.round(offset, 4).tolist()}"
            )
        except Exception:
            return "placement diagnostics unavailable"

    def _log(self, api: str, args: Dict[str, Any], result: Any, ok: bool, message: str = "") -> bool:
        self.events.append(
            {
                "step": len(self.events) + 1,
                "api": api,
                "args": dict(args),
                "result": bool(result),
                "ok": bool(ok),
                "message": message,
                "failure_type": "" if ok else "execution failure",
            }
        )
        return bool(ok)

    def _fail(self, api: str, args: Dict[str, Any], message: str) -> bool:
        return self._log(api, args, False, False, message)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape((-1, array.shape[-1]))[0]
    return array.reshape(-1)


def _scalar_bool(value: Any) -> bool:
    """Coerce a Python/numpy/torch scalar (possibly batched) to a single bool."""
    if isinstance(value, bool):
        return value
    try:
        array = _to_numpy(value)
        return bool(array[0]) if array.size else False
    except Exception:
        return bool(value)


def _dict_bool(value: Dict[str, Any], key: str) -> bool:
    if key not in value:
        return False
    try:
        array = _to_numpy(value[key])
        return bool(array[0]) if array.size else False
    except Exception:
        return bool(value[key])


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_diff_wxyz(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Return q_diff = q_target * q_current.conjugate() (wxyz, root-frame composition)."""
    qt = np.asarray(q_target, dtype=np.float64).reshape(-1)[:4]
    qc = np.asarray(q_current, dtype=np.float64).reshape(-1)[:4]
    qw1, qx1, qy1, qz1 = qt
    qw2, qx2, qy2, qz2 = qc
    qcw, qcx, qcy, qcz = qw2, -qx2, -qy2, -qz2
    dw = qw1 * qcw - qx1 * qcx - qy1 * qcy - qz1 * qcz
    dx = qw1 * qcx + qx1 * qcw + qy1 * qcz - qz1 * qcy
    dy = qw1 * qcy - qx1 * qcz + qy1 * qcw + qz1 * qcx
    dz = qw1 * qcz + qx1 * qcy - qy1 * qcx + qz1 * qcw
    if dw < 0:
        dw, dx, dy, dz = -dw, -dx, -dy, -dz
    return np.array([dw, dx, dy, dz], dtype=np.float64)


def _quat_to_xyz_euler(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to intrinsic XYZ Euler angles, matching
    ManiSkill's `matrix_to_euler_angles(..., 'XYZ')` convention (pytorch3d).

    pytorch3d's matrix_to_euler_angles 'XYZ' decomposes R = Rx(a) @ Ry(b) @ Rz(c)
    (the matrix is built left-to-right in the 'XYZ' name). This formula matches
    that decomposition.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    # R = Rx(a) @ Ry(b) @ Rz(c). Extract using standard formula.
    # b = asin(R[0,2]), a = atan2(-R[1,2], R[2,2]), c = atan2(-R[0,1], R[0,0])
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    b = float(np.arcsin(sy))
    if abs(sy) < 1.0 - 1e-7:
        a = float(np.arctan2(-R[1, 2], R[2, 2]))
        c = float(np.arctan2(-R[0, 1], R[0, 0]))
    else:
        # gimbal lock fallback
        a = float(np.arctan2(R[2, 1], R[1, 1]))
        c = 0.0
    return np.array([a, b, c], dtype=np.float64)


def _euler_xyz_rotation_delta(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """XYZ Euler angles of the delta rotation in root frame (pd_ee_delta_pose convention)."""
    q_diff = _quat_diff_wxyz(q_target, q_current)
    return _quat_to_xyz_euler(q_diff)


def _capped_unit_vector(v: np.ndarray) -> np.ndarray:
    """Cap the magnitude of v to 1.0, preserving direction (vs per-component clip)."""
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm > 1.0:
        return v / norm
    return v


def _sapien_pose_from_any(pose: Any) -> Any:
    """Convert a ManiSkill batched Pose to a plain sapien.Pose (single env)."""
    import sapien

    if isinstance(pose, sapien.Pose):
        return pose
    if hasattr(pose, "sp"):
        try:
            return pose.sp
        except Exception:
            pass
    p = _to_numpy(pose.p)[:3]
    q = _to_numpy(pose.q)[:4]
    return sapien.Pose(p.astype(np.float64), q.astype(np.float64))


class ManiSkillPegInsertionRobot:
    """PegInsertionSide-v1 skill wrapper using pd_ee_delta_pose control.

    The same LMP code (grasp / align_to_target / insert) drives Panda or any
    other robot whose action_space is the standard pd_ee_delta_pose 7-vector.

    Strategy:
    - grasp: compute a top-down grasp pose perpendicular to the peg axis using
      ManiSkill's agent.build_grasp_pose helper; descend, close gripper, lift.
      Snapshot TCP-at-grasp and peg-at-grasp so later skills can compute
      target TCP poses via rigid-body transforms.
    - align_to_target: drive the TCP so the held peg's head sits in front of
      the hole entrance, then report whether the peg-head lateral error is
      within the requested tolerance.
    - insert: push the TCP forward along the hole axis until env.evaluate
      reports success (or the move budget is exhausted).
    """

    def __init__(
        self,
        env: Any,
        *,
        move_steps: int = 50,
        grip_steps: int = 6,
        approach_clearance_m: float = 0.06,
        pre_insert_margin_m: float = 0.02,
        pos_tolerance_m: float = 0.005,
        finger_length_m: float = 0.025,
        control_mode: Optional[str] = None,
    ) -> None:
        self.env = env
        self.move_steps = move_steps
        self.grip_steps = grip_steps
        self.approach_clearance_m = approach_clearance_m
        self.pre_insert_margin_m = pre_insert_margin_m
        self.pos_tolerance_m = pos_tolerance_m
        self.finger_length_m = finger_length_m
        self.control_mode = control_mode
        self.last_info: Dict[str, Any] = {}
        self.terminated: bool = False
        self.truncated: bool = False
        self.events: List[Dict[str, Any]] = []
        self.tcp_at_grasp: Any = None
        self.peg_at_grasp: Any = None
        self._validate_action_space()

    def _validate_action_space(self) -> None:
        if self.control_mode is not None and self.control_mode != "pd_ee_pose":
            raise ValueError(
                "PegInsertion adapter requires control_mode='pd_ee_pose' "
                "(absolute target pose with internal IK), "
                f"got {self.control_mode!r}."
            )
        space = getattr(self.env, "action_space", None)
        shape = getattr(space, "shape", None)
        if not shape or shape[-1] != 7:
            raise RuntimeError(
                "PegInsertion adapter expects action_space last dim == 7 "
                f"(pd_ee_pose), got shape {shape!r}."
            )

    def grasp(self, obj: SkillTarget) -> bool:
        if obj.name != "peg":
            return self._fail(
                "grasp",
                {"obj": obj.name},
                "PegInsertion adapter only supports grasping the peg.",
            )

        import sapien

        base = self._base_env()
        peg_pose = _sapien_pose_from_any(base.peg.pose)
        peg_half_length = float(_to_numpy(base.peg_half_sizes)[0])

        peg_axis = _quat_to_rotmat(np.asarray(peg_pose.q)) @ np.array([1.0, 0.0, 0.0])
        peg_axis /= np.linalg.norm(peg_axis) + 1e-9
        horizontal = np.cross(np.array([0.0, 0.0, 1.0]), peg_axis)
        h_norm = float(np.linalg.norm(horizontal))
        if h_norm < 1e-6:
            return self._fail(
                "grasp", {"obj": obj.name}, "peg axis is vertical; cannot grasp from above"
            )
        closing = horizontal / h_norm
        approaching = np.array([0.0, 0.0, -1.0])
        # TCP is at fingertip closing midpoint (Panda's panda_hand_tcp link),
        # so placing it at peg's height puts the fingers around the peg.
        grasp_center = np.asarray(peg_pose.p) - 0.5 * peg_half_length * peg_axis

        grasp_pose = _sapien_pose_from_any(
            base.agent.build_grasp_pose(approaching, closing, grasp_center)
        )
        approach_pose = sapien.Pose(
            np.asarray(grasp_pose.p) + np.array([0.0, 0.0, self.approach_clearance_m]),
            grasp_pose.q,
        )

        self._move_tcp_to_pose(approach_pose, gripper=1.0)
        self._move_tcp_to_pose(grasp_pose, gripper=1.0)
        self._repeat_action_zero(gripper=-1.0, steps=self.grip_steps)

        grasped_after_close = self._agent_is_grasping_peg()
        self._log(
            "grasp_check_after_close",
            {"obj": obj.name},
            grasped_after_close,
            grasped_after_close,
            "" if grasped_after_close else "gripper closed but peg not held",
        )

        self.tcp_at_grasp = _sapien_pose_from_any(base.agent.tcp.pose)
        self.peg_at_grasp = _sapien_pose_from_any(base.peg.pose)

        self._move_tcp_to_pose(approach_pose, gripper=-1.0)
        grasped_after_lift = self._agent_is_grasping_peg()
        if grasped_after_lift:
            message = ""
        elif grasped_after_close:
            message = "peg slipped during lift"
        else:
            message = "peg was not grasped"
        return self._log("grasp", {"obj": obj.name}, grasped_after_lift, grasped_after_lift, message)

    def _build_top_down_grasp_at_peg_origin(
        self, peg_origin_world: Any, peg_axis_world: np.ndarray
    ) -> Any:
        """Pick the IK-friendlier of two equivalent top-down grasps.

        The peg is rotation-symmetric around its long axis, so two top-down grasp
        poses (closing = +/- horizontal⟂peg_axis) produce identical physical
        results. Panda's last wrist joint has limit ±2.8973 rad, and one
        orientation will often saturate that joint while the other won't.
        We score each candidate by the wrist joint qpos needed via IK from
        the current configuration and pick the one further from the joint limit.
        """
        base = self._base_env()
        peg_half_length = float(_to_numpy(base.peg_half_sizes)[0])
        approaching = np.array([0.0, 0.0, -1.0])
        horizontal = np.cross(np.array([0.0, 0.0, 1.0]), peg_axis_world)
        h_norm = float(np.linalg.norm(horizontal))
        if h_norm < 1e-6:
            return None
        grasp_center = (
            np.asarray(peg_origin_world) - 0.5 * peg_half_length * peg_axis_world
        )
        # Two candidates: the symmetric pair around peg axis.
        candidates: List[Tuple[Any, np.ndarray]] = []
        for sign in (+1.0, -1.0):
            closing = sign * horizontal / h_norm
            pose = _sapien_pose_from_any(
                base.agent.build_grasp_pose(approaching, closing, grasp_center)
            )
            candidates.append((pose, closing))

        # Score each by predicted wrist-7 effort. We approximate by computing the
        # target XYZ-Euler-yaw delta from current yaw. Whichever needs the
        # smaller wrist rotation is preferred.
        current_q = _to_numpy(base.agent.robot.get_qpos())
        cur_q7 = float(current_q[6]) if len(current_q) > 6 else 0.0
        best_pose, best_score = None, np.inf
        for pose, _closing in candidates:
            target_in_base = _sapien_pose_from_any(base.agent.robot.pose).inv() * pose
            euler = _quat_to_xyz_euler(np.asarray(target_in_base.q))
            # Heuristic: distance between target wrist yaw (euler[2]) and current q7,
            # accounting for ±2.8973 limits.
            yaw_target = float(euler[2])
            score = abs(yaw_target - cur_q7)
            if score < best_score:
                best_score = score
                best_pose = pose
        return best_pose

    def align_to_target(self, obj: SkillTarget, target: SkillTarget, tolerance: float) -> bool:
        import sapien

        base = self._base_env()
        if self.tcp_at_grasp is None or self.peg_at_grasp is None:
            return self._fail(
                "align_to_target",
                {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
                "align_to_target called before a successful grasp",
            )

        # Hole's x-axis in world is the direction the peg head should point into.
        hole_pose = _sapien_pose_from_any(base.box_hole_pose)
        hole_x_axis = _quat_to_rotmat(np.asarray(hole_pose.q)) @ np.array([1.0, 0.0, 0.0])
        hole_x_axis = hole_x_axis / (np.linalg.norm(hole_x_axis) + 1e-9)

        goal_pose = _sapien_pose_from_any(base.goal_pose)
        pre_insert_peg_origin = goal_pose * sapien.Pose([-self.pre_insert_margin_m, 0.0, 0.0])
        # Build a fresh top-down grasp pose at the pre-insert peg location.
        # This lets the wrist re-orient freely instead of locking it via the
        # original grasp's rigid transform (which on PegInsertion hits the
        # last Panda joint limit).
        tcp_target = self._build_top_down_grasp_at_peg_origin(
            np.asarray(pre_insert_peg_origin.p), hole_x_axis
        )
        if tcp_target is None:
            return self._fail(
                "align_to_target",
                {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
                "hole axis is vertical; top-down regrasp pose is undefined",
            )

        self._move_tcp_to_pose(tcp_target, gripper=-1.0)

        peg_head_pose = _sapien_pose_from_any(base.peg_head_pose)
        head_in_hole_frame = (hole_pose.inv() * peg_head_pose).p
        yz_err = float(np.linalg.norm(np.asarray(head_in_hole_frame)[1:3]))

        ok = yz_err <= float(tolerance)
        msg = (
            ""
            if ok
            else (
                f"alignment failure: peg-head yz error {yz_err:.4f} exceeds "
                f"tolerance {float(tolerance):.4f}"
            )
        )
        return self._log(
            "align_to_target",
            {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
            ok,
            ok,
            msg,
        )

    def insert(self, obj: SkillTarget, target: SkillTarget, speed: float) -> bool:
        base = self._base_env()
        if self.tcp_at_grasp is None or self.peg_at_grasp is None:
            return self._fail(
                "insert",
                {"obj": obj.name, "target": target.name, "speed": float(speed)},
                "insert called before a successful grasp",
            )

        hole_pose = _sapien_pose_from_any(base.box_hole_pose)
        hole_x_axis = _quat_to_rotmat(np.asarray(hole_pose.q)) @ np.array([1.0, 0.0, 0.0])
        hole_x_axis = hole_x_axis / (np.linalg.norm(hole_x_axis) + 1e-9)
        goal_pose = _sapien_pose_from_any(base.goal_pose)
        # peg.origin should end up at goal_pose (so peg.head reaches hole centre).
        tcp_target = self._build_top_down_grasp_at_peg_origin(
            np.asarray(goal_pose.p), hole_x_axis
        )
        if tcp_target is None:
            return self._fail(
                "insert",
                {"obj": obj.name, "target": target.name, "speed": float(speed)},
                "hole axis is vertical; insert pose is undefined",
            )

        # Speed acts as a soft pacing hint: smaller speed → more iterations so the
        # IK-driven controller drives joints in a smoother trajectory. We bound
        # iterations against move_steps to avoid blowing the episode budget.
        speed = max(float(speed), 1e-3)
        scaled_steps = int(min(max(self.move_steps, int(0.2 / speed)), 80))
        old_move_steps = self.move_steps
        self.move_steps = scaled_steps
        self._move_tcp_to_pose(tcp_target, gripper=-1.0)
        self.move_steps = old_move_steps

        ok = self._evaluate_success()
        msg = "" if ok else "insertion failure: peg head not inside hole tolerance"
        return self._log(
            "insert",
            {"obj": obj.name, "target": target.name, "speed": float(speed)},
            ok,
            ok,
            msg,
        )

    def place(self, obj: SkillTarget, target: SkillTarget) -> bool:
        return self._fail(
            "place",
            {"obj": obj.name, "target": target.name},
            "place is not implemented for PegInsertionSide-v1",
        )

    def hook_object(self, tool: SkillTarget, obj: SkillTarget) -> bool:
        return self._fail(
            "hook_object",
            {"tool": tool.name, "obj": obj.name},
            "tool use is not implemented for PegInsertionSide-v1",
        )

    def pull_with_tool(self, tool: SkillTarget, obj: SkillTarget, target: SkillTarget) -> bool:
        return self._fail(
            "pull_with_tool",
            {"tool": tool.name, "obj": obj.name, "target": target.name},
            "tool use is not implemented for PegInsertionSide-v1",
        )

    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def _move_tcp_to_pose(self, target_pose: Any, *, gripper: float) -> None:
        """Drive TCP to absolute target pose via pd_ee_pose (internal IK)."""
        base = self._base_env()
        robot_pose_inv = _sapien_pose_from_any(base.agent.robot.pose).inv()
        target_in_base = robot_pose_inv * target_pose
        target_pos = np.asarray(target_in_base.p)
        target_euler = _quat_to_xyz_euler(np.asarray(target_in_base.q))
        action = self._make_action(target_pos, target_euler, gripper=gripper)
        for _ in range(self.move_steps):
            if self._early_stop():
                return
            current = _sapien_pose_from_any(base.agent.tcp.pose)
            dp = np.asarray(target_pose.p) - np.asarray(current.p)
            if float(np.linalg.norm(dp)) < self.pos_tolerance_m:
                return
            self._step(action)

    def _repeat_action_hold(self, *, gripper: float, steps: int) -> None:
        """Hold the current TCP pose while changing gripper command."""
        base = self._base_env()
        robot_pose_inv = _sapien_pose_from_any(base.agent.robot.pose).inv()
        current_in_base = robot_pose_inv * _sapien_pose_from_any(base.agent.tcp.pose)
        target_pos = np.asarray(current_in_base.p)
        target_euler = _quat_to_xyz_euler(np.asarray(current_in_base.q))
        action = self._make_action(target_pos, target_euler, gripper=gripper)
        for _ in range(max(1, steps)):
            if self._early_stop():
                return
            self._step(action)

    # backward-compat alias used by older sites
    def _repeat_action_zero(self, *, gripper: float, steps: int) -> None:
        self._repeat_action_hold(gripper=gripper, steps=steps)

    def _make_action(self, pos: np.ndarray, euler: np.ndarray, *, gripper: float) -> Any:
        space = self.env.action_space
        action = np.zeros(7, dtype=getattr(space, "dtype", np.float32))
        action[0:3] = np.asarray(pos, dtype=np.float32)
        action[3:6] = np.asarray(euler, dtype=np.float32)
        action[6] = float(gripper)
        low = getattr(space, "low", None)
        high = getattr(space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high)
        return action

    def _step(self, action: Any) -> None:
        _, _, terminated, truncated, info = self.env.step(action)
        self.last_info = dict(info or {})
        self.terminated = self.terminated or _scalar_bool(terminated)
        self.truncated = self.truncated or _scalar_bool(truncated)

    def _early_stop(self) -> bool:
        return bool(self.terminated or self.truncated)

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _agent_is_grasping_peg(self) -> bool:
        try:
            value = self._base_env().agent.is_grasping(self._base_env().peg)
            return bool(_to_numpy(value)[0])
        except Exception:
            return False

    def _evaluate_success(self) -> bool:
        if "success" in self.last_info:
            try:
                return bool(_to_numpy(self.last_info["success"])[0])
            except Exception:
                pass
        try:
            result = self._base_env().evaluate()
            return bool(_to_numpy(result.get("success", False))[0])
        except Exception:
            return False

    def _log(
        self, api: str, args: Dict[str, Any], result: Any, ok: bool, message: str = ""
    ) -> bool:
        self.events.append(
            {
                "step": len(self.events) + 1,
                "api": api,
                "args": dict(args),
                "result": bool(result),
                "ok": bool(ok),
                "message": message,
                "failure_type": "" if ok else "execution failure",
            }
        )
        return bool(ok)

    def _fail(self, api: str, args: Dict[str, Any], message: str) -> bool:
        return self._log(api, args, False, False, message)
