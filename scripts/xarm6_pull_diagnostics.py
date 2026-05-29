"""Diagnose xarm6_robotiq contact pulling behavior in PullCube-v1.

The script is intentionally independent of the LLM loop. It runs a small set of
raw-action and closed-loop contact probes so we can see whether xarm6 can:

1. approach the cube from the positive-x contact side;
2. descend to cube height;
3. drag the cube toward the negative-x goal.

Run from the repository root:

python scripts/xarm6_pull_diagnostics.py --sim-backend auto --render-backend gpu
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from maniskill_backend.env_adapter import ManiSkillEnvAdapter
from maniskill_backend.skill_adapter import _scalar_bool, _to_numpy


ArmCommand = Tuple[float, float, float]
Phase = Tuple[str, ArmCommand, int]


def make_action(env: Any, arm: ArmCommand = (0.0, 0.0, 0.0), *, gripper: float = -1.0) -> Any:
    space = env.action_space
    action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
    flat = action.reshape(-1)
    flat[:3] = np.asarray(arm, dtype=np.float32)
    if flat.size > 3:
        flat[3:] = float(gripper)
    return np.clip(action, space.low, space.high)


def step(env: Any, arm: ArmCommand = (0.0, 0.0, 0.0), *, gripper: float = -1.0) -> Dict[str, Any]:
    _, _, terminated, truncated, info = env.step(make_action(env, arm, gripper=gripper))
    return {
        "terminated": _scalar_bool(terminated),
        "truncated": _scalar_bool(truncated),
        "info": dict(info or {}),
    }


def run_steps(env: Any, arm: ArmCommand, count: int, *, gripper: float = -1.0) -> Dict[str, Any]:
    last: Dict[str, Any] = {"terminated": False, "truncated": False, "info": {}}
    for _ in range(max(0, count)):
        last = step(env, arm, gripper=gripper)
        if last["terminated"] or last["truncated"]:
            break
    return last


def actor(base: Any, *names: str) -> Any:
    for name in names:
        value = getattr(base, name, None)
        if value is not None:
            return value
    raise AttributeError(f"Missing actor aliases: {names}")


def poses(env: Any) -> Dict[str, np.ndarray]:
    base = getattr(env, "unwrapped", env)
    tcp_pose = getattr(base.agent, "tcp_pose", None)
    if tcp_pose is None:
        tcp_pose = base.agent.tcp.pose
    return {
        "tcp": _to_numpy(tcp_pose.p),
        "cube": _to_numpy(actor(base, "cube", "obj").pose.p),
        "goal": _to_numpy(actor(base, "goal_region", "goal_site", "goal").pose.p),
    }


def metrics(start: Dict[str, np.ndarray], current: Dict[str, np.ndarray], info: Dict[str, Any]) -> Dict[str, Any]:
    cube = current["cube"]
    goal = current["goal"]
    tcp = current["tcp"]
    cube_start = start["cube"]
    return {
        "tcp": round_list(tcp),
        "cube": round_list(cube),
        "goal": round_list(goal),
        "cube_delta": round_list(cube - cube_start),
        "cube_dx": round(float(cube[0] - cube_start[0]), 4),
        "cube_goal_xy": round(float(np.linalg.norm((goal - cube)[:2])), 4),
        "tcp_cube_xy": round(float(np.linalg.norm((tcp - cube)[:2])), 4),
        "tcp_cube_z": round(float(abs(tcp[2] - cube[2])), 4),
        "far_side_for_negative_x_push": bool(tcp[0] > cube[0] + 0.025 and abs(tcp[1] - cube[1]) < 0.06),
        "success": info_bool(info, "success"),
    }


def info_bool(info: Dict[str, Any], key: str) -> bool:
    if key not in info:
        return False
    try:
        return bool(_to_numpy(info[key]).reshape(-1)[0])
    except Exception:
        return bool(info[key])


def round_list(array: np.ndarray) -> List[float]:
    return np.round(np.asarray(array, dtype=np.float32), 4).tolist()


def controller_summary(env: Any) -> Dict[str, Any]:
    base = getattr(env, "unwrapped", env)
    controller = getattr(base.agent, "controller", None)
    result: Dict[str, Any] = {
        "action_space": repr(getattr(env, "action_space", None)),
        "controller": repr(controller),
    }
    controllers = getattr(controller, "controllers", None)
    if isinstance(controllers, dict):
        result["subcontrollers"] = {
            name: {
                "type": type(sub).__name__,
                "action_space": repr(getattr(sub, "action_space", None)),
            }
            for name, sub in controllers.items()
        }
    return result


def make_env(args: argparse.Namespace) -> ManiSkillEnvAdapter:
    return ManiSkillEnvAdapter(
        "PullCube-v1",
        robot_uid="xarm6_robotiq",
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
    )


def run_raw_case(args: argparse.Namespace, name: str, phases: Sequence[Phase]) -> Dict[str, Any]:
    adapter = make_env(args)
    env = adapter.make()
    env.reset(seed=args.seed)
    start = poses(env)
    last: Dict[str, Any] = {"info": {}}
    steps_used = 0
    try:
        for _, arm, count in phases:
            last = run_steps(env, arm, count)
            steps_used += count
            if last["terminated"] or last["truncated"]:
                break
        current = poses(env)
        return {
            "name": name,
            "kind": "raw_action_probe",
            "phases": [{"name": label, "arm": list(arm), "steps": count} for label, arm, count in phases],
            "steps_used": steps_used,
            **metrics(start, current, last.get("info", {})),
        }
    finally:
        adapter.close()


def move_towards(
    env: Any,
    target: np.ndarray,
    *,
    steps: int,
    max_delta_m: float = 0.045,
    gripper: float = -1.0,
    down_bias: float | None = None,
) -> Dict[str, Any]:
    last: Dict[str, Any] = {"terminated": False, "truncated": False, "info": {}}
    for _ in range(max(1, steps)):
        tcp = poses(env)["tcp"]
        delta = np.asarray(target, dtype=np.float32) - tcp
        if np.linalg.norm(delta) < 0.008:
            break
        if down_bias is not None:
            delta[2] = min(float(delta[2]), float(down_bias))
        command = np.clip(delta / max_delta_m, -0.85, 0.85)
        last = step(env, tuple(float(x) for x in command), gripper=gripper)
        if last["terminated"] or last["truncated"]:
            break
    return last


def run_contact_candidate(args: argparse.Namespace, x_offset: float, z_offset: float) -> Dict[str, Any]:
    adapter = make_env(args)
    env = adapter.make()
    env.reset(seed=args.seed)
    start = poses(env)
    last: Dict[str, Any] = {"info": {}}
    try:
        initial = poses(env)
        cube = initial["cube"]
        goal = initial["goal"]
        contact = cube + np.array([x_offset, 0.0, z_offset], dtype=np.float32)
        pre_contact = contact + np.array([0.0, 0.0, 0.075], dtype=np.float32)
        drag_end = np.array([goal[0] - 0.03, cube[1], max(0.006, contact[2] - 0.004)], dtype=np.float32)

        last = move_towards(env, pre_contact, steps=42)
        last = move_towards(env, contact, steps=42)
        last = run_steps(env, (-0.08, 0.0, -0.05), 20)
        for stage in range(1, 7):
            waypoint = contact * (1.0 - stage / 6.0) + drag_end * (stage / 6.0)
            last = move_towards(env, waypoint, steps=18, down_bias=-0.008)
            last = run_steps(env, (-0.14, 0.0, -0.03), 4)
            if last["terminated"] or last["truncated"]:
                break
        last = run_steps(env, (0.0, 0.0, 0.0), 12)
        current = poses(env)
        return {
            "name": f"closed_loop_contact_x{x_offset:.3f}_z{z_offset:.3f}",
            "kind": "closed_loop_contact_candidate",
            "contact_x_offset": x_offset,
            "contact_z_offset": z_offset,
            **metrics(start, current, last.get("info", {})),
        }
    finally:
        adapter.close()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    adapter = make_env(args)
    env = adapter.make()
    env.reset(seed=args.seed)
    summary = {
        "env_id": "PullCube-v1",
        "robot_uid": "xarm6_robotiq",
        "seed": args.seed,
        "control_mode": args.control_mode,
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "max_episode_steps": args.max_episode_steps,
        "controller_summary": controller_summary(env),
        "initial": {key: round_list(value) for key, value in poses(env).items()},
    }
    adapter.close()

    raw_cases = [
        ("down_only", [("down", (0.0, 0.0, -0.8), 100)]),
        ("x_plus_then_down", [("x_plus", (0.8, 0.0, 0.0), 100), ("down", (0.0, 0.0, -0.8), 80)]),
        (
            "x_plus_down_drag_x_minus",
            [
                ("x_plus", (0.8, 0.0, 0.0), 100),
                ("down", (0.0, 0.0, -0.8), 80),
                ("drag_x_minus", (-0.8, 0.0, -0.05), 160),
            ],
        ),
        (
            "x_plus_y_plus_down_drag_x_minus",
            [
                ("x_plus_y_plus", (0.8, 0.4, 0.0), 100),
                ("down", (0.0, 0.0, -0.8), 80),
                ("drag_x_minus", (-0.8, 0.0, -0.05), 160),
            ],
        ),
    ]
    results: List[Dict[str, Any]] = []
    for name, phases in raw_cases:
        results.append(run_raw_case(args, name, phases))
    for x_offset, z_offset in ((0.04, 0.008), (0.055, 0.012), (0.07, 0.016)):
        results.append(run_contact_candidate(args, x_offset, z_offset))

    summary["results"] = results
    moved_negative = [item for item in results if item.get("cube_dx", 0.0) < -0.02]
    successes = [item for item in results if item.get("success")]
    summary["diagnosis"] = {
        "any_success": bool(successes),
        "best_negative_cube_dx": min((float(item.get("cube_dx", 0.0)) for item in results), default=0.0),
        "best_cube_goal_xy": min((float(item.get("cube_goal_xy", 999.0)) for item in results), default=999.0),
        "negative_motion_cases": [item["name"] for item in moved_negative],
        "success_cases": [item["name"] for item in successes],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose xarm6 PullCube contact behavior.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=repr))


if __name__ == "__main__":
    main()
