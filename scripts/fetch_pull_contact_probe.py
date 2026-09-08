"""Probe Fetch base reachability and planar cube-contact geometry.

The probe is deliberately independent of the hand-written Fetch oracle. Each
trial resets the same ManiSkill seed, applies one bounded base pulse, stops the
base, and measures a small arm contact/drag experiment. The output tells the
repair loop which physical stage failed without prescribing adapter code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.env_adapter import ManiSkillEnvAdapter
from maniskill_backend.skill_adapter import _scalar_bool, _to_numpy


ArmCommand = Tuple[float, float, float]
BaseCommand = Tuple[float, float]


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_probe_plan(text_or_path: str) -> List[Dict[str, Any]]:
    if not text_or_path:
        return []
    stripped = text_or_path.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(stripped)
    else:
        path = Path(stripped)
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(stripped)
    if isinstance(data, dict):
        data = data.get("next_probe_suggestions") or data.get("planned_probe_cases") or []
    if not isinstance(data, list):
        raise ValueError("probe plan must be a JSON list of parameter dictionaries.")
    return [dict(item) for item in data]


def make_action(
    env: Any,
    arm: ArmCommand = (0.0, 0.0, 0.0),
    *,
    gripper: float = -1.0,
    base: BaseCommand = (0.0, 0.0),
) -> Any:
    space = env.action_space
    action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
    flat = action.reshape(-1)
    if flat.size != 9:
        raise RuntimeError(f"Fetch probe requires observed 9D action space, got {space.shape!r}.")
    flat[0:3] = np.asarray(arm, dtype=np.float32)
    flat[3] = float(gripper)
    flat[7:9] = np.asarray(base, dtype=np.float32)
    return np.clip(action, space.low, space.high)


def step(
    env: Any,
    arm: ArmCommand = (0.0, 0.0, 0.0),
    *,
    gripper: float = -1.0,
    base: BaseCommand = (0.0, 0.0),
) -> Dict[str, Any]:
    _, _, terminated, truncated, info = env.step(
        make_action(env, arm, gripper=gripper, base=base)
    )
    return {
        "terminated": _scalar_bool(terminated),
        "truncated": _scalar_bool(truncated),
        "info": dict(info or {}),
    }


def run_steps(
    env: Any,
    count: int,
    *,
    arm: ArmCommand = (0.0, 0.0, 0.0),
    gripper: float = -1.0,
    base: BaseCommand = (0.0, 0.0),
) -> Dict[str, Any]:
    last: Dict[str, Any] = {"terminated": False, "truncated": False, "info": {}}
    for _ in range(max(0, count)):
        last = step(env, arm, gripper=gripper, base=base)
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


def info_bool(info: Dict[str, Any], key: str) -> bool:
    if key not in info:
        return False
    try:
        return bool(_to_numpy(info[key]).reshape(-1)[0])
    except Exception:
        return bool(info[key])


def round_list(array: np.ndarray) -> List[float]:
    return np.round(np.asarray(array, dtype=np.float32), 5).tolist()


def make_env(args: argparse.Namespace) -> ManiSkillEnvAdapter:
    return ManiSkillEnvAdapter(
        getattr(args, "env_id", "PullCube-v1"),
        robot_uid="fetch",
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
    )


def controller_summary(env: Any) -> Dict[str, Any]:
    base = getattr(env, "unwrapped", env)
    return {
        "action_space": repr(getattr(env, "action_space", None)),
        "controller": repr(getattr(base.agent, "controller", None)),
    }


def move_towards(
    env: Any,
    target: np.ndarray,
    *,
    steps: int,
    max_delta_m: float,
    gripper: float,
    tolerance: float = 0.006,
    xy_clip: float = 0.8,
    z_clip: float = 0.8,
    down_bias: float | None = None,
) -> Dict[str, Any]:
    last: Dict[str, Any] = {"terminated": False, "truncated": False, "info": {}}
    for _ in range(max(1, steps)):
        tcp = poses(env)["tcp"]
        delta = np.asarray(target, dtype=np.float32) - tcp
        if np.linalg.norm(delta) < tolerance:
            break
        if down_bias is not None:
            delta[2] = min(float(delta[2]), float(down_bias))
        command = delta / max(max_delta_m, 1e-6)
        command[:2] = np.clip(command[:2], -xy_clip, xy_clip)
        command[2] = float(np.clip(command[2], -z_clip, z_clip))
        last = step(env, tuple(float(value) for value in command), gripper=gripper)
        if last["terminated"] or last["truncated"]:
            break
    return last


def score_result(result: Dict[str, Any]) -> float:
    score = 500.0 if result.get("task_success") else 0.0
    score += float(result.get("base_reach_improvement") or 0.0) * 300.0
    score += float(result.get("cube_goal_improvement") or 0.0) * 1000.0
    score -= float(result.get("cube_goal_xy") or 0.0) * 250.0
    score -= float(result.get("tcp_cube_xy_after_base") or 0.0) * 50.0
    score -= float(result.get("tcp_contact_xy") or 0.0) * 120.0
    score -= float(result.get("tcp_contact_z") or 0.0) * 120.0
    if float(result.get("cube_delta_along_goal") or 0.0) < -0.005:
        score -= 80.0
    if result.get("terminated") or result.get("truncated"):
        score -= 25.0
    return round(score, 4)


def run_probe_case(
    args: argparse.Namespace,
    *,
    base_speed: float,
    base_steps: int,
    contact_x_offset: float,
    contact_z_offset: float,
    drag_strength: float,
    down_bias: float,
    stages: int,
) -> Dict[str, Any]:
    adapter = make_env(args)
    env = adapter.make()
    try:
        env.reset(seed=args.seed)
        initial = poses(env)
        cube_start = initial["cube"]
        goal = initial["goal"]
        initial_tcp_cube_xy = float(np.linalg.norm((initial["tcp"] - cube_start)[:2]))
        initial_cube_goal_xy = float(np.linalg.norm((goal - cube_start)[:2]))
        goal_delta = goal[:2] - cube_start[:2]
        goal_distance = float(np.linalg.norm(goal_delta))
        goal_direction = goal_delta / max(goal_distance, 1e-8)

        last = run_steps(
            env,
            int(base_steps),
            gripper=args.gripper_close,
            base=(float(base_speed), 0.0),
        )
        if not (last["terminated"] or last["truncated"]):
            last = run_steps(env, args.stop_base_steps, gripper=args.gripper_close)
        after_base = poses(env)
        tcp_cube_xy_after_base = float(np.linalg.norm((after_base["tcp"] - after_base["cube"])[:2]))
        base_reach_improvement = initial_tcp_cube_xy - tcp_cube_xy_after_base

        cube_contact_ref = after_base["cube"]
        far_side_xy = -goal_direction * float(contact_x_offset)
        contact = cube_contact_ref + np.array(
            [far_side_xy[0], far_side_xy[1], contact_z_offset], dtype=np.float32
        )
        pre_contact = contact + np.array([0.0, 0.0, args.approach_height], dtype=np.float32)

        if not (last["terminated"] or last["truncated"]):
            last = move_towards(
                env,
                pre_contact,
                steps=args.approach_steps,
                max_delta_m=args.max_delta_m,
                gripper=args.gripper_close,
            )
        if not (last["terminated"] or last["truncated"]):
            last = move_towards(
                env,
                contact,
                steps=args.descent_steps,
                max_delta_m=args.max_delta_m,
                gripper=args.gripper_close,
                z_clip=args.descent_z_clip,
            )

        contact_pose = poses(env)
        contact_error = contact_pose["tcp"] - contact
        tcp_contact_xy = float(np.linalg.norm(contact_error[:2]))
        tcp_contact_z = float(abs(contact_error[2]))
        far_side_projection = float(
            np.dot((contact_pose["tcp"] - cube_contact_ref)[:2], -goal_direction)
        )

        if not (last["terminated"] or last["truncated"]):
            inward = goal_direction * min(float(drag_strength) * 0.25, 0.2)
            last = run_steps(
                env,
                args.contact_steps,
                arm=(float(inward[0]), float(inward[1]), float(down_bias)),
                gripper=args.gripper_close,
            )

        stages = max(1, int(stages))
        drag_end = contact + np.array(
            [
                goal_direction[0] * (goal_distance + args.drag_extra),
                goal_direction[1] * (goal_distance + args.drag_extra),
                -0.004,
            ],
            dtype=np.float32,
        )
        for stage_index in range(1, stages + 1):
            if last["terminated"] or last["truncated"]:
                break
            alpha = stage_index / stages
            waypoint = contact * (1.0 - alpha) + drag_end * alpha
            last = move_towards(
                env,
                waypoint,
                steps=max(1, args.drag_steps // stages),
                max_delta_m=args.max_delta_m,
                gripper=args.gripper_close,
                down_bias=down_bias,
            )
            if last["terminated"] or last["truncated"]:
                break
            pulse_xy = goal_direction * float(drag_strength)
            last = run_steps(
                env,
                args.drag_pulse_steps,
                arm=(float(pulse_xy[0]), float(pulse_xy[1]), float(down_bias)),
                gripper=args.gripper_close,
            )

        if not (last["terminated"] or last["truncated"]):
            last = run_steps(env, args.settle_steps, gripper=args.gripper_close)

        final = poses(env)
        cube = final["cube"]
        tcp = final["tcp"]
        cube_goal_xy = float(np.linalg.norm((goal - cube)[:2]))
        cube_delta_xy = (cube - cube_start)[:2]
        result: Dict[str, Any] = {
            "base_speed": round(float(base_speed), 5),
            "base_steps": int(base_steps),
            "contact_x_offset": round(float(contact_x_offset), 5),
            "contact_z_offset": round(float(contact_z_offset), 5),
            "drag_strength": round(float(drag_strength), 5),
            "down_bias": round(float(down_bias), 5),
            "stages": stages,
            "task_success": info_bool(last.get("info", {}), "success"),
            "tcp_cube_xy_initial": round(initial_tcp_cube_xy, 5),
            "tcp_cube_xy_after_base": round(tcp_cube_xy_after_base, 5),
            "base_reach_improvement": round(base_reach_improvement, 5),
            "base_motion_helpful": bool(base_reach_improvement > 0.005),
            "tcp_contact_xy": round(tcp_contact_xy, 5),
            "tcp_contact_z": round(tcp_contact_z, 5),
            "far_side_at_contact": bool(far_side_projection > 0.01),
            "cube_goal_xy_initial": round(initial_cube_goal_xy, 5),
            "cube_goal_xy": round(cube_goal_xy, 5),
            "cube_goal_improvement": round(initial_cube_goal_xy - cube_goal_xy, 5),
            "cube_delta_x": round(float(cube[0] - cube_start[0]), 5),
            "cube_delta_along_goal": round(float(np.dot(cube_delta_xy, goal_direction)), 5),
            "tcp_cube_xy": round(float(np.linalg.norm((tcp - cube)[:2])), 5),
            "terminated": bool(last.get("terminated")),
            "truncated": bool(last.get("truncated")),
            "cube_start": round_list(cube_start),
            "goal": round_list(goal),
            "tcp_initial": round_list(initial["tcp"]),
            "tcp_after_base": round_list(after_base["tcp"]),
            "contact_target": round_list(contact),
            "tcp_at_contact": round_list(contact_pose["tcp"]),
            "cube_final": round_list(cube),
            "tcp_final": round_list(tcp),
        }
        result["score"] = score_result(result)
        return result
    finally:
        adapter.close()


def build_prompt_feedback(results: Sequence[Dict[str, Any]], *, top_k: int = 8) -> str:
    ranked = sorted(results, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    successes = [item for item in ranked if item.get("task_success")]
    lines = [
        "Structured Fetch base/contact probe results.",
        "These are measured simulator observations, not adapter source code.",
        "",
        f"total_probe_cases={len(results)}",
        f"successful_probe_cases={len(successes)}",
        "",
        "top_probe_cases:",
    ]
    for item in ranked[:top_k]:
        lines.append(
            "- "
            f"base={item['base_speed']}x{item['base_steps']}, "
            f"x={item['contact_x_offset']}, z={item['contact_z_offset']}, "
            f"drag={item['drag_strength']}, success={item['task_success']}, "
            f"base_improvement={item['base_reach_improvement']}, "
            f"contact_xy={item['tcp_contact_xy']}, contact_z={item['tcp_contact_z']}, "
            f"goal_progress={item['cube_goal_improvement']}, score={item['score']}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    explicit_plan = list(getattr(args, "probe_plan", None) or [])
    if not explicit_plan:
        explicit_plan = load_probe_plan(getattr(args, "probe_plan_json", ""))

    adapter = make_env(args)
    env = adapter.make()
    env.reset(seed=args.seed)
    summary: Dict[str, Any] = {
        "env_id": getattr(args, "env_id", "PullCube-v1"),
        "robot_uid": "fetch",
        "seed": args.seed,
        "control_mode": args.control_mode,
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "max_episode_steps": args.max_episode_steps,
        "controller_summary": controller_summary(env),
        "initial": {key: round_list(value) for key, value in poses(env).items()},
        "probe_plan_mode": "explicit" if explicit_plan else "grid",
    }
    adapter.close()

    if explicit_plan:
        grid: Iterable[Tuple[float, int, float, float, float, float, int]] = (
            (
                float(item["base_speed"]),
                int(item["base_steps"]),
                float(item["contact_x_offset"]),
                float(item["contact_z_offset"]),
                float(item["drag_strength"]),
                float(item["down_bias"]),
                int(item["stages"]),
            )
            for item in explicit_plan
        )
        summary["probe_plan"] = explicit_plan
    else:
        grid = (
            (base_speed, base_steps, x, z, drag, down, stages)
            for base_speed in parse_float_list(args.base_speeds)
            for base_steps in parse_int_list(args.base_steps)
            for x in parse_float_list(args.contact_x_offsets)
            for z in parse_float_list(args.contact_z_offsets)
            for drag in parse_float_list(args.drag_strengths)
            for down in parse_float_list(args.down_biases)
            for stages in parse_int_list(args.stages)
        )

    results: List[Dict[str, Any]] = []
    for index, values in enumerate(grid, start=1):
        if args.max_cases and index > args.max_cases:
            break
        result = run_probe_case(
            args,
            base_speed=values[0],
            base_steps=values[1],
            contact_x_offset=values[2],
            contact_z_offset=values[3],
            drag_strength=values[4],
            down_bias=values[5],
            stages=values[6],
        )
        result["case_index"] = index
        results.append(result)
        if args.stop_on_success and result.get("task_success"):
            break

    prompt_feedback = build_prompt_feedback(results, top_k=args.top_k)
    payload = {
        **summary,
        "results": sorted(results, key=lambda item: float(item.get("score") or 0.0), reverse=True),
        "prompt_feedback": prompt_feedback,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_slug = getattr(args, "output_slug", "fetch_pull_contact_probe")
    json_path = output_dir / f"{output_slug}.json"
    prompt_path = output_dir / f"{output_slug}_prompt.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    prompt_path.write_text(prompt_feedback + "\n", encoding="utf-8")
    payload["wrote"] = {"json": str(json_path), "prompt_feedback": str(prompt_path)}
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--env-id", default="PullCube-v1")
    parser.add_argument("--output-slug", default="fetch_pull_contact_probe")
    parser.add_argument("--base-speeds", default="-0.30,0.15,0.30")
    parser.add_argument("--base-steps", default="8,20")
    parser.add_argument("--contact-x-offsets", default="0.015,0.035")
    parser.add_argument("--contact-z-offsets", default="0.006,0.010")
    parser.add_argument("--drag-strengths", default="0.45,0.65")
    parser.add_argument("--down-biases", default="-0.04")
    parser.add_argument("--stages", default="4")
    parser.add_argument("--approach-height", type=float, default=0.06)
    parser.add_argument("--stop-base-steps", type=int, default=4)
    parser.add_argument("--approach-steps", type=int, default=36)
    parser.add_argument("--descent-steps", type=int, default=36)
    parser.add_argument("--contact-steps", type=int, default=8)
    parser.add_argument("--drag-steps", type=int, default=80)
    parser.add_argument("--drag-pulse-steps", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--drag-extra", type=float, default=0.03)
    parser.add_argument("--max-delta-m", type=float, default=0.05)
    parser.add_argument("--descent-z-clip", type=float, default=0.65)
    parser.add_argument("--gripper-close", type=float, default=-1.0)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--probe-plan-json", default="")
    return parser


def main() -> None:
    print(json.dumps(run(build_arg_parser().parse_args()), indent=2, ensure_ascii=False, default=repr))


if __name__ == "__main__":
    main()
