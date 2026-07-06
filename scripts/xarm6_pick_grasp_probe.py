"""Probe xarm6_robotiq PickCube close-envelope parameters.

This script is intentionally independent of the LLM generation loop. It runs a
small fixed-XY sweep over grasp height and gripper close parameters, then writes
structured evidence that can be fed back into the LLM prompt.

Run from the repository root:

python scripts/xarm6_pick_grasp_probe.py --sim-backend auto --render-backend gpu
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


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_probe_plan(text_or_path: str) -> List[Dict[str, Any]]:
    """Load an explicit probe plan from a JSON string or JSON file path."""

    if not text_or_path:
        return []
    stripped = text_or_path.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        data = json.loads(stripped)
    else:
        path = Path(stripped)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = json.loads(stripped)
    if isinstance(data, dict):
        data = data.get("next_probe_suggestions") or data.get("planned_probe_cases") or data.get("probe_plan") or []
    if not isinstance(data, list):
        raise ValueError("probe plan must be a JSON list of parameter dictionaries.")
    return [dict(item) for item in data]


def make_action(env: Any, arm: ArmCommand = (0.0, 0.0, 0.0), *, gripper: float = -1.0) -> Any:
    space = env.action_space
    action = np.zeros(space.shape, dtype=getattr(space, "dtype", np.float32))
    flat = action.reshape(-1)
    flat[:3] = np.asarray(arm, dtype=np.float32)
    if flat.size > 3:
        flat[3] = float(gripper)
    return np.clip(action, space.low, space.high)


def step(env: Any, arm: ArmCommand = (0.0, 0.0, 0.0), *, gripper: float = -1.0) -> Dict[str, Any]:
    _, _, terminated, truncated, info = env.step(make_action(env, arm, gripper=gripper))
    return {
        "terminated": _scalar_bool(terminated),
        "truncated": _scalar_bool(truncated),
        "info": dict(info or {}),
    }


def run_steps(env: Any, arm: ArmCommand, count: int, *, gripper: float) -> Dict[str, Any]:
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
        "goal": _to_numpy(actor(base, "goal_site", "goal_region", "goal").pose.p),
    }


def is_grasping(env: Any) -> bool:
    try:
        base = getattr(env, "unwrapped", env)
        return _scalar_bool(base.agent.is_grasping(actor(base, "cube", "obj")))
    except Exception:
        return False


def info_bool(info: Dict[str, Any], key: str) -> bool:
    if key not in info:
        return False
    try:
        return bool(_to_numpy(info[key]).reshape(-1)[0])
    except Exception:
        return bool(info[key])


def round_list(array: np.ndarray) -> List[float]:
    return np.round(np.asarray(array, dtype=np.float32), 4).tolist()


def move_towards(
    env: Any,
    target: np.ndarray,
    *,
    steps: int,
    max_delta_m: float,
    gripper: float,
    tolerance: float = 0.004,
    xy_clip: float = 0.8,
    z_clip: float = 0.8,
) -> Dict[str, Any]:
    last: Dict[str, Any] = {"terminated": False, "truncated": False, "info": {}}
    for _ in range(max(1, steps)):
        tcp = poses(env)["tcp"]
        delta = np.asarray(target, dtype=np.float32) - tcp
        if np.linalg.norm(delta) < tolerance:
            break
        command = delta / max(max_delta_m, 1e-6)
        command[:2] = np.clip(command[:2], -xy_clip, xy_clip)
        command[2] = float(np.clip(command[2], -z_clip, z_clip))
        last = step(env, tuple(float(x) for x in command), gripper=gripper)
        if last["terminated"] or last["truncated"]:
            break
    return last


def make_env(args: argparse.Namespace) -> ManiSkillEnvAdapter:
    return ManiSkillEnvAdapter(
        "PickCube-v1",
        robot_uid="xarm6_robotiq",
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
    )


def controller_summary(env: Any) -> Dict[str, Any]:
    base = getattr(env, "unwrapped", env)
    controller = getattr(base.agent, "controller", None)
    return {
        "action_space": repr(getattr(env, "action_space", None)),
        "controller": repr(controller),
    }


def score_result(result: Dict[str, Any]) -> float:
    score = 0.0
    if result.get("is_grasping_after_close"):
        score += 100.0
    if result.get("is_grasping_after_lift"):
        score += 200.0
    if result.get("task_success"):
        score += 300.0
    score += float(result.get("cube_lift_delta_z") or 0.0) * 500.0
    score -= float(result.get("cube_disp_xy") or 0.0) * 1000.0
    score -= float(result.get("tcp_grasp_xy") or 0.0) * 200.0
    score -= float(result.get("tcp_grasp_z") or 0.0) * 200.0
    if result.get("terminated") or result.get("truncated"):
        score -= 25.0
    return round(score, 4)


def run_probe_case(
    args: argparse.Namespace,
    *,
    grasp_z_offset: float,
    close_steps: int,
    close_command: float,
    settle_steps: int,
) -> Dict[str, Any]:
    adapter = make_env(args)
    env = adapter.make()
    try:
        env.reset(seed=args.seed)
        initial = poses(env)
        cube_start = initial["cube"]
        grasp_pos = cube_start + np.array([0.0, 0.0, grasp_z_offset], dtype=np.float32)
        high = grasp_pos + np.array([0.0, 0.0, args.approach_height], dtype=np.float32)

        last: Dict[str, Any] = {"terminated": False, "truncated": False, "info": {}}
        last = run_steps(env, (0.0, 0.0, 0.0), args.open_steps, gripper=args.gripper_open)
        if not (last["terminated"] or last["truncated"]):
            last = move_towards(
                env,
                high,
                steps=args.move_steps,
                max_delta_m=args.max_delta_m,
                gripper=args.gripper_open,
                xy_clip=args.move_xy_clip,
                z_clip=args.move_z_clip,
            )
        if not (last["terminated"] or last["truncated"]):
            last = move_towards(
                env,
                grasp_pos,
                steps=args.descend_steps,
                max_delta_m=args.descend_max_delta_m,
                gripper=args.gripper_open,
                xy_clip=args.descend_xy_clip,
                z_clip=args.descend_z_clip,
                tolerance=args.preclose_tolerance,
            )

        before_close = poses(env)
        tcp_before = before_close["tcp"]
        cube_before = before_close["cube"]
        tcp_grasp_xy = float(np.linalg.norm((tcp_before - grasp_pos)[:2]))
        tcp_grasp_z = float(abs(tcp_before[2] - grasp_pos[2]))

        if not (last["terminated"] or last["truncated"]):
            if args.staged_close:
                half_command = close_command * 0.5
                last = run_steps(env, (0.0, 0.0, 0.0), max(1, close_steps // 2), gripper=half_command)
                if not (last["terminated"] or last["truncated"]):
                    last = run_steps(
                        env,
                        (0.0, 0.0, 0.0),
                        max(1, close_steps - max(1, close_steps // 2)),
                        gripper=close_command,
                    )
            else:
                last = run_steps(env, (0.0, 0.0, 0.0), close_steps, gripper=close_command)

        if not (last["terminated"] or last["truncated"]):
            last = run_steps(env, (0.0, 0.0, 0.0), settle_steps, gripper=close_command)

        after_close = poses(env)
        cube_after_close = after_close["cube"]
        cube_disp_xy = float(np.linalg.norm((cube_after_close - cube_before)[:2]))
        is_grasping_after_close = is_grasping(env)

        if not (last["terminated"] or last["truncated"]):
            lift_target = after_close["tcp"] + np.array([0.0, 0.0, args.lift_height], dtype=np.float32)
            last = move_towards(
                env,
                lift_target,
                steps=args.lift_steps,
                max_delta_m=args.max_delta_m,
                gripper=close_command,
                xy_clip=args.move_xy_clip,
                z_clip=args.move_z_clip,
            )

        after_lift = poses(env)
        is_grasping_after_lift = is_grasping(env)
        cube_lift_delta_z = float(after_lift["cube"][2] - cube_after_close[2])
        task_success = info_bool(last.get("info", {}), "success")

        result: Dict[str, Any] = {
            "grasp_z_offset": round(float(grasp_z_offset), 5),
            "close_steps": int(close_steps),
            "close_command": round(float(close_command), 4),
            "settle_steps": int(settle_steps),
            "staged_close": bool(args.staged_close),
            "tcp_grasp_xy": round(tcp_grasp_xy, 5),
            "tcp_grasp_z": round(tcp_grasp_z, 5),
            "cube_disp_xy": round(cube_disp_xy, 5),
            "is_grasping_after_close": bool(is_grasping_after_close),
            "is_grasping_after_lift": bool(is_grasping_after_lift),
            "cube_lift_delta_z": round(cube_lift_delta_z, 5),
            "task_success": bool(task_success),
            "terminated": bool(last.get("terminated")),
            "truncated": bool(last.get("truncated")),
            "tcp_before_close": round_list(tcp_before),
            "cube_before_close": round_list(cube_before),
            "cube_after_close": round_list(cube_after_close),
            "cube_after_lift": round_list(after_lift["cube"]),
        }
        result["score"] = score_result(result)
        return result
    finally:
        adapter.close()


def build_prompt_feedback(results: Sequence[Dict[str, Any]], *, top_k: int = 8) -> str:
    ranked = sorted(results, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    successes = [item for item in ranked if item.get("is_grasping_after_lift") or item.get("is_grasping_after_close")]
    lines = [
        "Structured xarm6 PickCube grasp probe results.",
        "Use these real simulation measurements instead of guessing close-envelope parameters.",
        "",
        f"total_probe_cases={len(results)}",
        f"grasping_cases={len(successes)}",
    ]
    if ranked:
        best = ranked[0]
        lines.extend(
            [
                "",
                "best_probe_case:",
                (
                    f"  grasp_z_offset={best['grasp_z_offset']}, close_steps={best['close_steps']}, "
                    f"close_command={best['close_command']}, settle_steps={best['settle_steps']}, "
                    f"tcp_grasp_xy={best['tcp_grasp_xy']}, tcp_grasp_z={best['tcp_grasp_z']}, "
                    f"cube_disp_xy={best['cube_disp_xy']}, "
                    f"is_grasping_after_close={best['is_grasping_after_close']}, "
                    f"is_grasping_after_lift={best['is_grasping_after_lift']}, "
                    f"cube_lift_delta_z={best['cube_lift_delta_z']}, score={best['score']}"
                ),
            ]
        )
    lines.extend(["", "top_probe_cases:"])
    for item in ranked[:top_k]:
        lines.append(
            (
                f"- z={item['grasp_z_offset']}, close_steps={item['close_steps']}, "
                f"close={item['close_command']}, settle={item['settle_steps']}, "
                f"grasp_close={item['is_grasping_after_close']}, "
                f"grasp_lift={item['is_grasping_after_lift']}, "
                f"disp={item['cube_disp_xy']}, tcp_xy={item['tcp_grasp_xy']}, "
                f"tcp_z={item['tcp_grasp_z']}, lift_dz={item['cube_lift_delta_z']}, score={item['score']}"
            )
        )
    lines.extend(
        [
            "",
            "LLM adapter guidance:",
            "- Prefer parameter choices near the best_probe_case.",
            "- Do not use candidates with large cube_disp_xy as first attempts.",
            "- If no probe case achieves is_grasping=True, report close-envelope/force failure and use the least destructive high-score case for the next bounded attempt.",
            "- Keep the high-level LMP program unchanged; only adapt the target-side grasp/place adapter.",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    results = payload["results"]
    ranked = sorted(results, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    lines = [
        "# xArm6 PickCube Grasp Probe",
        "",
        "This file records fixed-XY close-envelope probe results.",
        "",
        "## Best Cases",
        "",
        "| rank | score | z_offset | close_steps | close_command | settle | grasp_close | grasp_lift | cube_disp_xy | tcp_grasp_xy | tcp_grasp_z | lift_dz |",
        "|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for idx, item in enumerate(ranked[:20], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(item["score"]),
                    str(item["grasp_z_offset"]),
                    str(item["close_steps"]),
                    str(item["close_command"]),
                    str(item["settle_steps"]),
                    str(item["is_grasping_after_close"]),
                    str(item["is_grasping_after_lift"]),
                    str(item["cube_disp_xy"]),
                    str(item["tcp_grasp_xy"]),
                    str(item["tcp_grasp_z"]),
                    str(item["cube_lift_delta_z"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Prompt Feedback", "", "```text", payload["prompt_feedback"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    z_offsets = parse_float_list(args.grasp_z_offsets)
    close_steps = parse_int_list(args.close_steps)
    close_commands = parse_float_list(args.close_commands)
    settle_steps = parse_int_list(args.settle_steps)
    explicit_plan = list(getattr(args, "probe_plan", None) or [])
    if not explicit_plan:
        explicit_plan = load_probe_plan(getattr(args, "probe_plan_json", ""))

    adapter = make_env(args)
    env = adapter.make()
    env.reset(seed=args.seed)
    summary = {
        "env_id": "PickCube-v1",
        "robot_uid": "xarm6_robotiq",
        "seed": args.seed,
        "control_mode": args.control_mode,
        "obs_mode": args.obs_mode,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "max_episode_steps": args.max_episode_steps,
        "controller_summary": controller_summary(env),
        "initial": {key: round_list(value) for key, value in poses(env).items()},
        "grid": {
            "grasp_z_offsets": z_offsets,
            "close_steps": close_steps,
            "close_commands": close_commands,
            "settle_steps": settle_steps,
        },
        "probe_plan_mode": "explicit" if explicit_plan else "grid",
    }
    if explicit_plan:
        summary["probe_plan"] = explicit_plan
    adapter.close()

    results: List[Dict[str, Any]] = []
    if explicit_plan:
        grid: Iterable[Tuple[float, int, float, int]] = (
            (
                float(item["grasp_z_offset"]),
                int(item["close_steps"]),
                float(item["close_command"]),
                int(item["settle_steps"]),
            )
            for item in explicit_plan
        )
    else:
        grid = (
            (z, steps, command, settle)
            for z in z_offsets
            for steps in close_steps
            for command in close_commands
            for settle in settle_steps
        )
    for idx, (z, steps, command, settle) in enumerate(grid, start=1):
        if args.max_cases and idx > args.max_cases:
            break
        result = run_probe_case(
            args,
            grasp_z_offset=z,
            close_steps=steps,
            close_command=command,
            settle_steps=settle,
        )
        result["case_index"] = idx
        results.append(result)
        if args.stop_on_grasp and result.get("is_grasping_after_lift"):
            break

    prompt_feedback = build_prompt_feedback(results, top_k=args.top_k)
    payload = {
        **summary,
        "results": sorted(results, key=lambda item: float(item.get("score") or 0.0), reverse=True),
        "prompt_feedback": prompt_feedback,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "xarm6_pick_grasp_probe.json"
    md_path = output_dir / "xarm6_pick_grasp_probe.md"
    prompt_path = output_dir / "xarm6_pick_grasp_probe_prompt.txt"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    prompt_path.write_text(prompt_feedback + "\n", encoding="utf-8")
    payload["wrote"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "prompt_feedback": str(prompt_path),
    }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=220)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--grasp-z-offsets", default="0.004,0.008,0.012,0.016")
    parser.add_argument("--close-steps", default="12,24")
    parser.add_argument("--close-commands", default="-0.6,-1.0")
    parser.add_argument("--settle-steps", default="8,16")
    parser.add_argument("--open-steps", type=int, default=8)
    parser.add_argument("--move-steps", type=int, default=34)
    parser.add_argument("--descend-steps", type=int, default=42)
    parser.add_argument("--lift-steps", type=int, default=28)
    parser.add_argument("--approach-height", type=float, default=0.12)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--max-delta-m", type=float, default=0.045)
    parser.add_argument("--descend-max-delta-m", type=float, default=0.03)
    parser.add_argument("--preclose-tolerance", type=float, default=0.003)
    parser.add_argument("--move-xy-clip", type=float, default=0.75)
    parser.add_argument("--move-z-clip", type=float, default=0.75)
    parser.add_argument("--descend-xy-clip", type=float, default=0.08)
    parser.add_argument("--descend-z-clip", type=float, default=0.55)
    parser.add_argument("--gripper-open", type=float, default=1.0)
    parser.add_argument("--staged-close", action="store_true", default=True)
    parser.add_argument("--no-staged-close", action="store_false", dest="staged_close")
    parser.add_argument("--stop-on-grasp", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--probe-plan-json",
        default="",
        help="Optional JSON list/path containing explicit probe cases to run instead of the Cartesian grid.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
