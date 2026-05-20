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


class ManiSkillXArmPickCubePlannerRobot:
    """PickCube-v1 xarm6 wrapper backed by ManiSkill's official planner.

    The delta-EE wrapper is useful for testing raw control portability, but
    xarm6's official PickCube solution uses pd_joint_pos with a motion planner.
    This adapter keeps the same LMP-facing skill API while delegating the
    low-level path generation to ManiSkill's xarm6 planner.
    """

    def __init__(
        self,
        env: Any,
        *,
        control_mode: Optional[str] = None,
        debug: bool = False,
        vis: bool = False,
    ) -> None:
        if control_mode not in {None, "pd_joint_pos", "pd_joint_pos_vel"}:
            raise ValueError(
                "xarm6 planner PickCube adapter requires control_mode "
                f"'pd_joint_pos' or 'pd_joint_pos_vel', got {control_mode!r}."
            )
        self.env = env
        self.control_mode = control_mode
        self.debug = debug
        self.vis = vis
        self.planner: Any = None
        self.grasp_pose: Any = None
        self.last_info: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def grasp(self, obj: SkillTarget) -> bool:
        if obj.name != "cube":
            return self._fail("grasp", {"obj": obj.name}, "PickCube planner only supports cube grasp.")

        import sapien
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )

        base = self._base_env()
        planner = self._ensure_planner()
        obb = get_actor_obb(base.cube)
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = _pose_matrix(base.agent.tcp.pose)[:3, 1]
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        self.grasp_pose = base.agent.build_grasp_pose(
            approaching,
            grasp_info["closing"],
            _sapien_pose_from_any(base.cube.pose).p,
        )

        reach_pose = self.grasp_pose * sapien.Pose([0.0, 0.0, -0.05])
        if not self._capture(planner.move_to_pose_with_RRTStar(reach_pose), "reach"):
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed while reaching cube")
        if not self._capture(planner.move_to_pose_with_screw(self.grasp_pose), "pregrasp"):
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed at grasp pose")
        if not self._capture(planner.close_gripper(), "close_gripper"):
            return self._fail("grasp", {"obj": obj.name}, "close gripper command failed")

        ok = self._agent_is_grasping_cube()
        self._log(
            "grasp_check_after_close",
            {"obj": obj.name},
            ok,
            ok,
            "" if ok else "gripper closed but cube not held",
        )
        return self._log(
            "grasp",
            {"obj": obj.name},
            ok,
            ok,
            "" if ok else "cube was not grasped",
        )

    def place(self, obj: SkillTarget, target: SkillTarget) -> bool:
        if obj.name != "cube":
            return self._fail("place", {"obj": obj.name, "target": target.name}, "PickCube planner only supports cube.")
        if target.name not in {"goal", "goal_site"}:
            return self._fail("place", {"obj": obj.name, "target": target.name}, "PickCube target must be goal.")
        if self.grasp_pose is None:
            return self._fail("place", {"obj": obj.name, "target": target.name}, "place called before grasp.")

        import sapien

        base = self._base_env()
        planner = self._ensure_planner()
        goal_pose = sapien.Pose(_sapien_pose_from_any(base.goal_site.pose).p, self.grasp_pose.q)
        if not self._capture(planner.move_to_pose_with_screw(goal_pose), "place"):
            return self._fail("place", {"obj": obj.name, "target": target.name}, "motion planning failed while moving to goal")
        ok = self._pick_cube_success()
        if not ok:
            self._capture(planner.move_to_pose_with_screw(goal_pose), "place_retry")
            ok = self._pick_cube_success()
        diagnostics = self._placement_diagnostics()
        return self._log(
            "place",
            {"obj": obj.name, "target": target.name},
            ok,
            ok,
            "" if ok else f"cube was not placed at goal; {diagnostics}",
        )

    def align_to_target(self, obj: SkillTarget, target: SkillTarget, tolerance: float) -> bool:
        return self._fail(
            "align_to_target",
            {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
            "align_to_target is not implemented for PickCube-v1 planner path",
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

    def close(self) -> None:
        if self.planner is not None and hasattr(self.planner, "close"):
            self.planner.close()

    def _ensure_planner(self) -> Any:
        if self.planner is None:
            from mani_skill.examples.motionplanning.xarm6.motionplanner import (
                XArm6RobotiqMotionPlanningSolver,
            )

            base = self._base_env()
            self.planner = XArm6RobotiqMotionPlanningSolver(
                self.env,
                debug=self.debug,
                vis=self.vis,
                base_pose=base.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_acc_limits=0.5,
                joint_vel_limits=0.5,
            )
        return self.planner

    def _capture(self, result: Any, label: str) -> bool:
        if result == -1:
            self.last_info = {"planner_status": "failed", "planner_stage": label}
            return False
        if isinstance(result, tuple) and len(result) >= 5:
            self.last_info = dict(result[4] or {})
            return True
        return True

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _agent_is_grasping_cube(self) -> bool:
        try:
            value = self._base_env().agent.is_grasping(self._base_env().cube)
            return bool(_to_numpy(value)[0])
        except Exception:
            return False

    def _pick_cube_success(self) -> bool:
        try:
            result = self._base_env().evaluate()
            self.last_info = dict(result or {})
            return _dict_bool(result, "success") or _dict_bool(result, "is_obj_placed")
        except Exception:
            return False

    def _placement_diagnostics(self) -> str:
        try:
            base = self._base_env()
            goal = _to_numpy(base.goal_site.pose.p)
            cube = _to_numpy(base.cube.pose.p)
            tcp = _to_numpy(base.agent.tcp.pose.p)
            return (
                f"obj_goal_dist={float(np.linalg.norm(goal - cube)):.4f}, "
                f"tcp_goal_dist={float(np.linalg.norm(goal - tcp)):.4f}"
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


class ManiSkillStackCubePlannerRobot:
    """StackCube-v1 wrapper backed by ManiSkill's official Panda/xarm6 planners."""

    def __init__(
        self,
        env: Any,
        *,
        robot_uid: str,
        control_mode: Optional[str] = None,
        debug: bool = False,
        vis: bool = False,
    ) -> None:
        if robot_uid not in {"panda", "xarm6_robotiq"}:
            raise ValueError(f"StackCube planner does not support robot_uid={robot_uid!r}.")
        if control_mode not in {None, "pd_joint_pos", "pd_joint_pos_vel"}:
            raise ValueError(
                "StackCube planner adapter requires control_mode "
                f"'pd_joint_pos' or 'pd_joint_pos_vel', got {control_mode!r}."
            )
        self.env = env
        self.robot_uid = robot_uid
        self.control_mode = control_mode
        self.debug = debug
        self.vis = vis
        self.planner: Any = None
        self.grasp_pose: Any = None
        self.lift_pose: Any = None
        self.last_info: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def grasp(self, obj: SkillTarget) -> bool:
        if obj.name != "cubeA":
            return self._fail("grasp", {"obj": obj.name}, "StackCube planner only supports cubeA grasp.")

        import sapien
        from transforms3d.euler import euler2quat
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )

        base = self._base_env()
        planner = self._ensure_planner()
        obb = get_actor_obb(base.cubeA)
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = _pose_matrix(base.agent.tcp.pose)[:3, 1]
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        self.grasp_pose = _sapien_pose_from_any(
            base.agent.build_grasp_pose(
                approaching,
                grasp_info["closing"],
                grasp_info["center"],
            )
        )

        angles = np.arange(0.0, np.pi * 2.0 / 3.0, np.pi / 2.0)
        angles = np.repeat(angles, 2)
        angles[1::2] *= -1
        for angle in angles:
            candidate = self.grasp_pose * sapien.Pose(q=euler2quat(0, 0, float(angle)))
            if self._dry_run_pose(candidate):
                self.grasp_pose = candidate
                break
        else:
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed for all cubeA grasp poses")

        reach_pose = self.grasp_pose * sapien.Pose([0.0, 0.0, -0.05])
        if not self._capture(self._move_to_pose(reach_pose, prefer_rrt=True), "reach"):
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed while reaching cubeA")
        if not self._capture(planner.move_to_pose_with_screw(self.grasp_pose), "pregrasp"):
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed at cubeA grasp pose")
        if not self._capture(planner.close_gripper(), "close_gripper"):
            return self._fail("grasp", {"obj": obj.name}, "close gripper command failed")

        grasped_after_close = self._agent_is_grasping_cube_a()
        self._log(
            "grasp_check_after_close",
            {"obj": obj.name},
            grasped_after_close,
            grasped_after_close,
            "" if grasped_after_close else "gripper closed but cubeA not held",
        )

        self.lift_pose = sapien.Pose([0.0, 0.0, 0.1]) * self.grasp_pose
        if not self._capture(planner.move_to_pose_with_screw(self.lift_pose), "lift"):
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed while lifting cubeA")

        grasped_after_lift = self._agent_is_grasping_cube_a()
        if grasped_after_lift:
            message = ""
        elif grasped_after_close:
            message = "cubeA slipped during lift"
        else:
            message = "cubeA was not grasped"
        return self._log("grasp", {"obj": obj.name}, grasped_after_lift, grasped_after_lift, message)

    def place(self, obj: SkillTarget, target: SkillTarget) -> bool:
        if obj.name != "cubeA" or target.name != "cubeB":
            return self._fail(
                "place",
                {"obj": obj.name, "target": target.name},
                "StackCube planner only supports placing cubeA on cubeB.",
            )
        if self.lift_pose is None:
            return self._fail(
                "place",
                {"obj": obj.name, "target": target.name},
                "place called before cubeA was grasped and lifted.",
            )

        import sapien

        base = self._base_env()
        cube_half_size = _to_numpy(base.cube_half_size)
        goal_pose = _sapien_pose_from_any(base.cubeB.pose) * sapien.Pose(
            [0.0, 0.0, float(cube_half_size[2]) * 2.0]
        )
        cube_a_pose = _sapien_pose_from_any(base.cubeA.pose)
        offset = np.asarray(goal_pose.p, dtype=np.float64) - np.asarray(cube_a_pose.p, dtype=np.float64)
        align_pose = sapien.Pose(np.asarray(self.lift_pose.p, dtype=np.float64) + offset, self.lift_pose.q)

        if not self._capture(self._move_to_pose(align_pose, prefer_rrt=True), "stack_align"):
            return self._fail(
                "place",
                {"obj": obj.name, "target": target.name},
                "motion planning failed while moving cubeA above cubeB",
            )
        if not self._capture(self._ensure_planner().open_gripper(), "open_gripper"):
            return self._fail(
                "place",
                {"obj": obj.name, "target": target.name},
                "open gripper command failed",
            )

        ok = self._evaluate_success()
        diagnostics = self._stack_diagnostics()
        return self._log(
            "place",
            {"obj": obj.name, "target": target.name},
            ok,
            ok,
            "" if ok else f"cubeA was not stably stacked on cubeB; {diagnostics}",
        )

    def align_to_target(self, obj: SkillTarget, target: SkillTarget, tolerance: float) -> bool:
        return self._fail(
            "align_to_target",
            {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
            "align_to_target is not implemented for StackCube-v1",
        )

    def insert(self, obj: SkillTarget, target: SkillTarget, speed: float) -> bool:
        return self._fail(
            "insert",
            {"obj": obj.name, "target": target.name, "speed": float(speed)},
            "insert is not implemented for StackCube-v1",
        )

    def hook_object(self, tool: SkillTarget, obj: SkillTarget) -> bool:
        return self._fail("hook_object", {"tool": tool.name, "obj": obj.name}, "tool use is not implemented for StackCube-v1")

    def pull_with_tool(self, tool: SkillTarget, obj: SkillTarget, target: SkillTarget) -> bool:
        return self._fail(
            "pull_with_tool",
            {"tool": tool.name, "obj": obj.name, "target": target.name},
            "tool use is not implemented for StackCube-v1",
        )

    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def close(self) -> None:
        if self.planner is not None and hasattr(self.planner, "close"):
            self.planner.close()

    def _ensure_planner(self) -> Any:
        if self.planner is None:
            if self.robot_uid == "xarm6_robotiq":
                from mani_skill.examples.motionplanning.xarm6.motionplanner import (
                    XArm6RobotiqMotionPlanningSolver,
                )

                planner_cls = XArm6RobotiqMotionPlanningSolver
            else:
                from mani_skill.examples.motionplanning.panda.motionplanner import (
                    PandaArmMotionPlanningSolver,
                )

                planner_cls = PandaArmMotionPlanningSolver
            base = self._base_env()
            self.planner = planner_cls(
                self.env,
                debug=self.debug,
                vis=self.vis,
                base_pose=base.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
            )
        return self.planner

    def _move_to_pose(self, pose: Any, *, prefer_rrt: bool = False) -> Any:
        planner = self._ensure_planner()
        if self.robot_uid == "xarm6_robotiq" and prefer_rrt:
            return planner.move_to_pose_with_RRTStar(pose)
        return planner.move_to_pose_with_screw(pose)

    def _dry_run_pose(self, pose: Any) -> bool:
        planner = self._ensure_planner()
        if self.robot_uid == "xarm6_robotiq":
            return planner.move_to_pose_with_RRTStar(pose, dry_run=True) != -1
        return planner.move_to_pose_with_screw(pose, dry_run=True) != -1

    def _capture(self, result: Any, label: str) -> bool:
        if result == -1:
            self.last_info = {"planner_status": "failed", "planner_stage": label}
            return False
        if isinstance(result, tuple) and len(result) >= 5:
            self.last_info = dict(result[4] or {})
        return True

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _agent_is_grasping_cube_a(self) -> bool:
        try:
            value = self._base_env().agent.is_grasping(self._base_env().cubeA)
            return bool(_to_numpy(value)[0])
        except Exception:
            return False

    def _evaluate_success(self) -> bool:
        try:
            result = self._base_env().evaluate()
            self.last_info = dict(result or {})
            return _dict_bool(result, "success")
        except Exception:
            return False

    def _stack_diagnostics(self) -> str:
        try:
            result = self._base_env().evaluate()
            parts = []
            for key in ("is_cubeA_grasped", "is_cubeA_on_cubeB", "is_cubeA_static", "success"):
                if key in result:
                    parts.append(f"{key}={_to_numpy(result[key]).tolist()}")
            return ", ".join(parts) if parts else "stack diagnostics unavailable"
        except Exception:
            return "stack diagnostics unavailable"

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


class ManiSkillPullCubeToolPlannerRobot:
    """PullCubeTool-v1 wrapper backed by ManiSkill motion planners.

    The high-level LMP split is:
    - hook_object(tool, cube): grasp the L-shaped tool and position its hook
      behind the cube;
    - pull_with_tool(tool, cube, workspace): pull the cube back toward the
      robot workspace and check ManiSkill's success condition.
    """

    def __init__(
        self,
        env: Any,
        *,
        robot_uid: str,
        control_mode: Optional[str] = None,
        debug: bool = False,
        vis: bool = False,
    ) -> None:
        if robot_uid not in {"panda", "xarm6_robotiq"}:
            raise ValueError(f"PullCubeTool planner does not support robot_uid={robot_uid!r}.")
        if control_mode not in {None, "pd_joint_pos", "pd_joint_pos_vel"}:
            raise ValueError(
                "PullCubeTool planner adapter requires control_mode "
                f"'pd_joint_pos' or 'pd_joint_pos_vel', got {control_mode!r}."
            )
        self.env = env
        self.robot_uid = robot_uid
        self.control_mode = control_mode
        self.debug = debug
        self.vis = vis
        self.planner: Any = None
        self.grasp_pose: Any = None
        self.hook_pose: Any = None
        self.hook_tool_pose: Any = None
        self.tool_to_tcp: Any = None
        self.tool_tcp_scale: float = 1.0
        self.pull_start_cube_pos: Optional[np.ndarray] = None
        self.last_info: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def hook_object(
        self,
        tool: SkillTarget,
        obj: SkillTarget,
        *,
        hook_y_offset: float = -0.067,
        behind_margin: float = 0.0,
        approach_extra: float = 0.08,
        lift_height: float = 0.35,
    ) -> bool:
        if tool.name != "l_shape_tool" or obj.name != "cube":
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "PullCubeTool planner only supports l_shape_tool hooking cube.",
            )

        import sapien
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )

        base = self._base_env()
        planner = self._ensure_planner()
        hook_y_offset = float(np.clip(hook_y_offset, -0.14, 0.02))
        behind_margin = float(np.clip(behind_margin, -0.02, 0.10))
        approach_extra = float(np.clip(approach_extra, 0.02, 0.20))
        lift_height = float(np.clip(lift_height, 0.20, 0.50))
        tool_obb = get_actor_obb(base.l_shape_tool)
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = _pose_matrix(base.agent.tcp.pose)[:3, 1]
        grasp_info = compute_grasp_info_by_obb(
            tool_obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.03,
        )
        self.grasp_pose = _sapien_pose_from_any(
            base.agent.build_grasp_pose(
                approaching,
                grasp_info["closing"],
                _sapien_pose_from_any(base.l_shape_tool.pose).p,
            )
        ) * sapien.Pose([0.02, 0.0, 0.0])

        reach_pose = self.grasp_pose * sapien.Pose([0.0, 0.0, -0.05])
        if not self._capture(self._move_to_pose(reach_pose, prefer_rrt=True), "reach_tool"):
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "motion planning failed while reaching the L-shaped tool",
            )
        if not self._capture(planner.move_to_pose_with_screw(self.grasp_pose), "grasp_tool"):
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "motion planning failed at the tool grasp pose",
            )
        if not self._capture(planner.close_gripper(), "close_gripper"):
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "close gripper command failed while grasping tool",
            )

        grasped = self._agent_is_grasping_tool()
        self._log(
            "tool_grasp_check_after_close",
            {"tool": tool.name},
            grasped,
            grasped,
            "" if grasped else "gripper closed but L-shaped tool not held",
        )
        if not grasped:
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "L-shaped tool was not grasped",
            )

        lift_pose = sapien.Pose(
            np.asarray(self.grasp_pose.p, dtype=np.float64) + np.array([0.0, 0.0, lift_height]),
            self.grasp_pose.q,
        )
        if not self._capture(planner.move_to_pose_with_screw(lift_pose), "lift_tool"):
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "motion planning failed while lifting the tool",
            )
        self.tool_to_tcp = self._tool_to_tcp_transform()
        self.tool_tcp_scale = 1.0

        cube_pos = np.asarray(_sapien_pose_from_any(base.cube.pose).p, dtype=np.float64)
        hook_length = float(_to_numpy(base.hook_length)[0])
        cube_half_size = float(_to_numpy(base.cube_half_size)[0])
        behind_distance = hook_length + cube_half_size + behind_margin

        approach_tool_pose = sapien.Pose(
            cube_pos + np.array([-(behind_distance + approach_extra), 0.0, lift_height - 0.05]),
            _sapien_pose_from_any(base.l_shape_tool.pose).q,
        )
        if not self._move_to_tool_pose(
            approach_tool_pose,
            "approach_cube_with_tool",
            prefer_rrt=True,
        ):
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "motion planning failed while approaching cube with tool",
            )

        self.hook_tool_pose = sapien.Pose(
            cube_pos + np.array([-behind_distance, hook_y_offset, 0.0]),
            _sapien_pose_from_any(base.l_shape_tool.pose).q,
        )
        if not self._move_to_tool_pose(
            self.hook_tool_pose,
            "hook_cube",
            prefer_rrt=False,
            correct_tool=True,
            correction_tolerance=0.06,
        ):
            return self._fail(
                "hook_object",
                {"tool": tool.name, "obj": obj.name},
                "motion planning failed while positioning the actual tool behind cube",
            )

        return self._log(
            "hook_object",
            {
                "tool": tool.name,
                "obj": obj.name,
                "hook_y_offset": hook_y_offset,
                "behind_margin": behind_margin,
                "approach_extra": approach_extra,
                "lift_height": lift_height,
                "tool_tcp_scale": round(float(self.tool_tcp_scale), 3),
            },
            True,
            True,
            "",
        )

    def pull_with_tool(
        self,
        tool: SkillTarget,
        obj: SkillTarget,
        target: SkillTarget,
        *,
        distance: float = 0.35,
        stages: int = 1,
        pull_frame: Optional[str] = None,
    ) -> bool:
        if tool.name != "l_shape_tool" or obj.name != "cube":
            return self._fail(
                "pull_with_tool",
                {"tool": tool.name, "obj": obj.name, "target": target.name},
                "PullCubeTool planner only supports pulling cube with l_shape_tool.",
            )
        if self.hook_pose is None or self.hook_tool_pose is None:
            return self._fail(
                "pull_with_tool",
                {"tool": tool.name, "obj": obj.name, "target": target.name},
                "pull_with_tool called before hook_object succeeded.",
            )

        distance = float(np.clip(distance, 0.10, 0.70))
        stages = int(np.clip(stages, 1, 5))
        pull_frame = self._default_pull_frame(pull_frame)
        if pull_frame not in {"tool", "world", "toward_base"}:
            return self._fail(
                "pull_with_tool",
                {
                    "tool": tool.name,
                    "obj": obj.name,
                    "target": target.name,
                    "distance": distance,
                    "stages": stages,
                    "pull_frame": pull_frame,
                },
                "pull_frame must be one of: tool, world, toward_base.",
            )

        self.pull_start_cube_pos = self._cube_position()
        ok = False
        failed_stage = ""
        for stage in range(1, stages + 1):
            depth = distance * stage / stages
            target_pose = self._pull_target_pose(depth, pull_frame)
            if not self._capture(self._move_to_pose(target_pose, prefer_rrt=False), f"pull_cube_with_tool_{stage}"):
                failed_stage = f"motion planning failed while pulling cube with tool at stage {stage}"
                break
            ok = self._evaluate_success()
            if ok:
                break
        if failed_stage and not ok:
            return self._fail(
                "pull_with_tool",
                {
                    "tool": tool.name,
                    "obj": obj.name,
                    "target": target.name,
                    "distance": distance,
                    "stages": stages,
                    "pull_frame": pull_frame,
                },
                failed_stage,
            )

        diagnostics = self._pull_diagnostics()
        return self._log(
            "pull_with_tool",
            {
                "tool": tool.name,
                "obj": obj.name,
                "target": target.name,
                "distance": distance,
                "stages": stages,
                "pull_frame": pull_frame,
            },
            ok,
            ok,
            "" if ok else f"tool pull failed; cube was not pulled into workspace; {diagnostics}",
        )

    def grasp(self, obj: SkillTarget) -> bool:
        return self._fail(
            "grasp",
            {"obj": obj.name},
            "Use hook_object(tool, cube) for PullCubeTool-v1 instead of direct grasp.",
        )

    def place(self, obj: SkillTarget, target: SkillTarget) -> bool:
        return self._fail(
            "place",
            {"obj": obj.name, "target": target.name},
            "place is not implemented for PullCubeTool-v1",
        )

    def align_to_target(self, obj: SkillTarget, target: SkillTarget, tolerance: float) -> bool:
        return self._fail(
            "align_to_target",
            {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
            "align_to_target is not implemented for PullCubeTool-v1",
        )

    def insert(self, obj: SkillTarget, target: SkillTarget, speed: float) -> bool:
        return self._fail(
            "insert",
            {"obj": obj.name, "target": target.name, "speed": float(speed)},
            "insert is not implemented for PullCubeTool-v1",
        )

    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self.events)

    def close(self) -> None:
        if self.planner is not None and hasattr(self.planner, "close"):
            self.planner.close()

    def _ensure_planner(self) -> Any:
        if self.planner is None:
            if self.robot_uid == "xarm6_robotiq":
                from mani_skill.examples.motionplanning.xarm6.motionplanner import (
                    XArm6RobotiqMotionPlanningSolver,
                )

                planner_cls = XArm6RobotiqMotionPlanningSolver
                limits = {"joint_vel_limits": 0.5, "joint_acc_limits": 0.5}
            else:
                from mani_skill.examples.motionplanning.panda.motionplanner import (
                    PandaArmMotionPlanningSolver,
                )

                planner_cls = PandaArmMotionPlanningSolver
                limits = {"joint_vel_limits": 0.75, "joint_acc_limits": 0.75}
            base = self._base_env()
            self.planner = planner_cls(
                self.env,
                debug=self.debug,
                vis=self.vis,
                base_pose=base.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                **limits,
            )
        return self.planner

    def _move_to_pose(self, pose: Any, *, prefer_rrt: bool = False) -> Any:
        planner = self._ensure_planner()
        if self.robot_uid == "xarm6_robotiq" and prefer_rrt:
            return planner.move_to_pose_with_RRTStar(pose)
        return planner.move_to_pose_with_screw(pose)

    def _capture(self, result: Any, label: str) -> bool:
        if result == -1:
            self.last_info = {"planner_status": "failed", "planner_stage": label}
            return False
        if isinstance(result, tuple) and len(result) >= 5:
            self.last_info = dict(result[4] or {})
        return True

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _default_pull_frame(self, pull_frame: Optional[str]) -> str:
        if pull_frame is None:
            return "toward_base" if self.robot_uid == "xarm6_robotiq" else "tool"
        normalized = str(pull_frame).strip().lower().replace("-", "_")
        aliases = {
            "local": "tool",
            "local_x": "tool",
            "tool_local": "tool",
            "tool_x": "tool",
            "world_x": "world",
            "base": "toward_base",
            "towards_base": "toward_base",
        }
        return aliases.get(normalized, normalized)

    def _pull_target_pose(self, depth: float, pull_frame: str) -> Any:
        import sapien

        assert self.hook_pose is not None
        assert self.hook_tool_pose is not None
        if pull_frame == "tool":
            return self._tcp_pose_for_tool_pose(self.hook_tool_pose * sapien.Pose([-depth, 0.0, 0.0]))

        if pull_frame == "toward_base":
            direction = self._cube_to_base_direction()
        else:
            direction = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        tool_target = sapien.Pose(
            np.asarray(self.hook_tool_pose.p, dtype=np.float64) + direction * depth,
            self.hook_tool_pose.q,
        )
        return self._tcp_pose_for_tool_pose(tool_target)

    def _move_to_tool_pose(
        self,
        tool_pose: Any,
        label: str,
        *,
        prefer_rrt: bool,
        correct_tool: bool = False,
        correction_tolerance: float = 0.06,
    ) -> bool:
        scales = [self.tool_tcp_scale]
        if self.robot_uid == "xarm6_robotiq":
            scales.extend(scale for scale in (0.75, 0.5, 0.25, 0.0) if scale not in scales)
        for scale in scales:
            pose = self._tcp_pose_for_tool_pose(tool_pose, scale=scale)
            if self._capture(self._move_to_pose(pose, prefer_rrt=prefer_rrt), label):
                self.tool_tcp_scale = float(scale)
                self.hook_pose = pose
                if scale != scales[0]:
                    self._log(
                        "tool_tcp_compensation_fallback",
                        {"planner_stage": label, "tool_tcp_scale": round(float(scale), 3)},
                        True,
                        True,
                        "",
                    )
                if correct_tool and self.robot_uid == "xarm6_robotiq":
                    return self._correct_actual_tool_position(
                        tool_pose,
                        label,
                        tolerance=correction_tolerance,
                        prefer_rrt=prefer_rrt,
                    )
                return True
        return False

    def _correct_actual_tool_position(
        self,
        desired_tool_pose: Any,
        label: str,
        *,
        tolerance: float,
        prefer_rrt: bool,
    ) -> bool:
        import sapien

        desired_tool_pos = np.asarray(desired_tool_pose.p, dtype=np.float64)
        for attempt in range(1, 9):
            actual_tool_pos = self._tool_position()
            if actual_tool_pos is None:
                return True
            error = desired_tool_pos - actual_tool_pos
            error_norm = float(np.linalg.norm(error))
            self._log(
                "tool_position_error",
                {
                    "planner_stage": label,
                    "attempt": attempt,
                    "error_norm": round(error_norm, 4),
                    "error": np.round(error, 4).tolist(),
                    "tolerance": round(float(tolerance), 4),
                },
                True,
                True,
                "",
            )
            if error_norm <= tolerance:
                return True

            current_tcp = _sapien_pose_from_any(self._base_env().agent.tcp.pose)
            max_step = 0.035
            correction = error
            if error_norm > max_step:
                correction = error / error_norm * max_step
            correction[2] = float(np.clip(correction[2], -0.018, 0.018))

            moved = False
            for scale in (1.0, 0.5, 0.25):
                corrected_pose = sapien.Pose(
                    np.asarray(current_tcp.p, dtype=np.float64) + correction * scale,
                    current_tcp.q,
                )
                if self._capture(
                    self._move_to_pose(corrected_pose, prefer_rrt=False),
                    f"{label}_tool_position_correction_{attempt}",
                ):
                    self.hook_pose = corrected_pose
                    moved = True
                    break
            if not moved:
                return False
        actual_tool_pos = self._tool_position()
        if actual_tool_pos is None:
            return True
        return float(np.linalg.norm(desired_tool_pos - actual_tool_pos)) <= tolerance

    def _tool_to_tcp_transform(self) -> Any:
        base = self._base_env()
        tool_pose = _sapien_pose_from_any(base.l_shape_tool.pose)
        tcp_pose = _sapien_pose_from_any(base.agent.tcp.pose)
        return tool_pose.inv() * tcp_pose

    def _tcp_pose_for_tool_pose(self, tool_pose: Any, *, scale: Optional[float] = None) -> Any:
        if self.tool_to_tcp is None:
            return tool_pose
        if scale is None:
            scale = self.tool_tcp_scale
        import sapien

        scaled_tool_to_tcp = sapien.Pose(
            np.asarray(self.tool_to_tcp.p, dtype=np.float64) * float(scale),
            self.tool_to_tcp.q,
        )
        return tool_pose * scaled_tool_to_tcp

    def _cube_position(self) -> Optional[np.ndarray]:
        try:
            return np.asarray(_sapien_pose_from_any(self._base_env().cube.pose).p, dtype=np.float64)
        except Exception:
            return None

    def _tool_position(self) -> Optional[np.ndarray]:
        try:
            return np.asarray(_sapien_pose_from_any(self._base_env().l_shape_tool.pose).p, dtype=np.float64)
        except Exception:
            return None

    def _cube_to_base_direction(self) -> np.ndarray:
        try:
            base = self._base_env()
            cube_pos = np.asarray(_sapien_pose_from_any(base.cube.pose).p, dtype=np.float64)
            base_pos = np.asarray(_sapien_pose_from_any(base.agent.robot.get_links()[0].pose).p, dtype=np.float64)
            delta_xy = base_pos[:2] - cube_pos[:2]
            norm = float(np.linalg.norm(delta_xy))
            if norm > 1e-6:
                return np.array([delta_xy[0] / norm, delta_xy[1] / norm, 0.0], dtype=np.float64)
        except Exception:
            pass
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    def _agent_is_grasping_tool(self) -> bool:
        try:
            value = self._base_env().agent.is_grasping(self._base_env().l_shape_tool, max_angle=20)
            return bool(_to_numpy(value)[0])
        except Exception:
            return False

    def _evaluate_success(self) -> bool:
        try:
            result = self._base_env().evaluate()
            self.last_info = dict(result or {})
            return _dict_bool(result, "success")
        except Exception:
            return False

    def _pull_diagnostics(self) -> str:
        try:
            base = self._base_env()
            result = base.evaluate()
            parts = []
            for key in ("success", "success_once", "success_at_end", "cube_progress", "cube_distance"):
                if key in result:
                    parts.append(f"{key}={_to_numpy(result[key]).tolist()}")
            cube_pos = self._cube_position()
            if cube_pos is not None:
                parts.append(f"cube_pos={np.round(cube_pos, 4).tolist()}")
                if self.pull_start_cube_pos is not None:
                    cube_delta = cube_pos - self.pull_start_cube_pos
                    parts.append(f"cube_delta={np.round(cube_delta, 4).tolist()}")
                try:
                    tool_pos = np.asarray(_sapien_pose_from_any(base.l_shape_tool.pose).p, dtype=np.float64)
                    parts.append(f"tool_pos={np.round(tool_pos, 4).tolist()}")
                    parts.append(f"tool_cube_xy={float(np.linalg.norm(tool_pos[:2] - cube_pos[:2])):.4f}")
                    parts.append(f"tool_tcp_scale={self.tool_tcp_scale:.3f}")
                except Exception:
                    pass
                try:
                    base_pos = np.asarray(
                        _sapien_pose_from_any(base.agent.robot.get_links()[0].pose).p,
                        dtype=np.float64,
                    )
                    cube_to_base_xy = float(np.linalg.norm(cube_pos[:2] - base_pos[:2]))
                    parts.append(f"cube_to_base_xy={cube_to_base_xy:.4f}")
                except Exception:
                    pass
            return ", ".join(parts) if parts else "pull diagnostics unavailable"
        except Exception:
            return "pull diagnostics unavailable"

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


class ManiSkillPandaPegInsertionPlannerRobot:
    """PegInsertionSide-v1 wrapper backed by ManiSkill's official Panda planner."""

    def __init__(
        self,
        env: Any,
        *,
        control_mode: Optional[str] = None,
        debug: bool = False,
        vis: bool = False,
    ) -> None:
        if control_mode not in {None, "pd_joint_pos", "pd_joint_pos_vel"}:
            raise ValueError(
                "Panda planner PegInsertion adapter requires control_mode "
                f"'pd_joint_pos' or 'pd_joint_pos_vel', got {control_mode!r}."
            )
        self.env = env
        self.control_mode = control_mode
        self.debug = debug
        self.vis = vis
        self.planner: Any = None
        self.grasp_pose: Any = None
        self.insert_pose: Any = None
        self.pre_insert_pose: Any = None
        self.last_info: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []

    def grasp(self, obj: SkillTarget) -> bool:
        if obj.name != "peg":
            return self._fail("grasp", {"obj": obj.name}, "PegInsertion planner only supports peg grasp.")

        import sapien
        from mani_skill.examples.motionplanning.base_motionplanner.utils import (
            compute_grasp_info_by_obb,
            get_actor_obb,
        )

        base = self._base_env()
        planner = self._ensure_planner()
        obb = get_actor_obb(base.peg)
        approaching = np.array([0.0, 0.0, -1.0])
        target_closing = _pose_matrix(base.agent.tcp.pose)[:3, 1]
        peg_init_pose = _sapien_pose_from_any(base.peg.pose)

        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        self.grasp_pose = _sapien_pose_from_any(
            base.agent.build_grasp_pose(
                approaching,
                grasp_info["closing"],
                grasp_info["center"],
            )
        )
        peg_half_length = float(_to_numpy(base.peg_half_sizes)[0])
        self.grasp_pose = self.grasp_pose * sapien.Pose(
            [-max(0.05, peg_half_length / 2 + 0.01), 0.0, 0.0]
        )
        self.insert_pose = _sapien_pose_from_any(base.goal_pose) * peg_init_pose.inv() * self.grasp_pose

        reach_pose = self.grasp_pose * sapien.Pose([0.0, 0.0, -0.05])
        if not self._capture(planner.move_to_pose_with_screw(reach_pose), "reach"):
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed while reaching peg")
        if not self._capture(planner.move_to_pose_with_screw(self.grasp_pose), "pregrasp"):
            return self._fail("grasp", {"obj": obj.name}, "motion planning failed at grasp pose")
        if not self._capture(planner.close_gripper(), "close_gripper"):
            return self._fail("grasp", {"obj": obj.name}, "close gripper command failed")

        ok = self._agent_is_grasping_peg()
        self._log(
            "grasp_check_after_close",
            {"obj": obj.name},
            ok,
            ok,
            "" if ok else "gripper closed but peg not held",
        )
        return self._log("grasp", {"obj": obj.name}, ok, ok, "" if ok else "peg was not grasped")

    def align_to_target(self, obj: SkillTarget, target: SkillTarget, tolerance: float) -> bool:
        if obj.name != "peg" or target.name != "hole":
            return self._fail(
                "align_to_target",
                {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
                "PegInsertion planner only supports aligning peg to hole.",
            )
        if self.insert_pose is None:
            return self._fail(
                "align_to_target",
                {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
                "align_to_target called before grasp.",
            )

        import sapien

        base = self._base_env()
        planner = self._ensure_planner()
        peg_half_length = float(_to_numpy(base.peg_half_sizes)[0])
        offset = sapien.Pose([-0.01 - peg_half_length, 0.0, 0.0])
        self.pre_insert_pose = self.insert_pose * offset
        if not self._capture(planner.move_to_pose_with_screw(self.pre_insert_pose), "pre_insert"):
            return self._fail(
                "align_to_target",
                {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
                "motion planning failed while moving to pre-insert pose",
            )

        for i in range(3):
            delta_pose = _sapien_pose_from_any(base.goal_pose) * offset * _sapien_pose_from_any(base.peg.pose).inv()
            self.pre_insert_pose = delta_pose * self.pre_insert_pose
            if not self._capture(planner.move_to_pose_with_screw(self.pre_insert_pose), f"pre_insert_refine_{i + 1}"):
                return self._fail(
                    "align_to_target",
                    {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
                    "motion planning failed while refining pre-insert pose",
                )

        err = self._pre_insert_alignment_error()
        ok = err <= float(tolerance)
        message = "" if ok else f"alignment failure: pre-insert yz error {err:.4f} exceeds tolerance {float(tolerance):.4f}"
        return self._log(
            "align_to_target",
            {"obj": obj.name, "target": target.name, "tolerance": float(tolerance)},
            ok,
            ok,
            message,
        )

    def insert(self, obj: SkillTarget, target: SkillTarget, speed: float) -> bool:
        if obj.name != "peg" or target.name != "hole":
            return self._fail(
                "insert",
                {"obj": obj.name, "target": target.name, "speed": float(speed)},
                "PegInsertion planner only supports inserting peg into hole.",
            )
        if self.insert_pose is None:
            return self._fail(
                "insert",
                {"obj": obj.name, "target": target.name, "speed": float(speed)},
                "insert called before grasp.",
            )

        import sapien

        ok = False
        failed_stage = ""
        for depth in (0.05, 0.10, 0.15, 0.20):
            target_pose = self.insert_pose * sapien.Pose([depth, 0.0, 0.0])
            if not self._capture(
                self._ensure_planner().move_to_pose_with_screw(target_pose),
                f"insert_{depth:.2f}",
            ):
                failed_stage = f"motion planning failed during insertion depth {depth:.2f}m"
                break
            ok = self._evaluate_success()
            if ok:
                break
        if failed_stage and not ok:
            return self._fail(
                "insert",
                {"obj": obj.name, "target": target.name, "speed": float(speed)},
                failed_stage,
            )
        diagnostics = self._peg_diagnostics()
        return self._log(
            "insert",
            {"obj": obj.name, "target": target.name, "speed": float(speed)},
            ok,
            ok,
            "" if ok else f"peg was not inserted; {diagnostics}",
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

    def close(self) -> None:
        if self.planner is not None and hasattr(self.planner, "close"):
            self.planner.close()

    def _ensure_planner(self) -> Any:
        if self.planner is None:
            from mani_skill.examples.motionplanning.panda.motionplanner import (
                PandaArmMotionPlanningSolver,
            )

            base = self._base_env()
            self.planner = PandaArmMotionPlanningSolver(
                self.env,
                debug=self.debug,
                vis=self.vis,
                base_pose=base.agent.robot.pose,
                visualize_target_grasp_pose=False,
                print_env_info=False,
                joint_vel_limits=0.75,
                joint_acc_limits=0.75,
            )
        return self.planner

    def _capture(self, result: Any, label: str) -> bool:
        if result == -1:
            self.last_info = {"planner_status": "failed", "planner_stage": label}
            return False
        if isinstance(result, tuple) and len(result) >= 5:
            self.last_info = dict(result[4] or {})
        return True

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _agent_is_grasping_peg(self) -> bool:
        try:
            value = self._base_env().agent.is_grasping(self._base_env().peg, max_angle=20)
            return bool(_to_numpy(value)[0])
        except Exception:
            return False

    def _evaluate_success(self) -> bool:
        try:
            result = self._base_env().evaluate()
            self.last_info = dict(result or {})
            return _dict_bool(result, "success")
        except Exception:
            return False

    def _pre_insert_alignment_error(self) -> float:
        try:
            base = self._base_env()
            goal_pose = _sapien_pose_from_any(base.goal_pose)
            peg_head = goal_pose.inv() * _sapien_pose_from_any(base.peg_head_pose)
            peg_origin = goal_pose.inv() * _sapien_pose_from_any(base.peg.pose)
            return max(
                float(np.linalg.norm(np.asarray(peg_head.p)[1:3])),
                float(np.linalg.norm(np.asarray(peg_origin.p)[1:3])),
            )
        except Exception:
            return float("inf")

    def _peg_diagnostics(self) -> str:
        try:
            result = self._base_env().evaluate()
            value = _to_numpy(result.get("peg_head_pos_at_hole", []))
            return f"peg_head_pos_at_hole={np.round(value, 4).tolist()}"
        except Exception:
            return "peg diagnostics unavailable"

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


def _pose_matrix(pose: Any) -> np.ndarray:
    matrix = pose.to_transformation_matrix()
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 3:
        return matrix[0]
    return matrix


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
