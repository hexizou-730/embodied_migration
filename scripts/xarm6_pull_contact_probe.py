"""Probe xarm6_robotiq PullCube contact and drag parameters.

This script is independent of the LLM generation loop. It runs a bounded
contact-geometry sweep in the real ManiSkill PullCube-v1 environment and writes
structured evidence for the generic probe runner.

Run from the repository root:

python scripts/xarm6_pull_contact_probe.py --sim-backend auto --render-backend gpu
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
        "PullCube-v1",
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


def move_towards(
    env: Any,
    target: np.ndarray,
    *,
    steps: int,
    max_delta_m: float,
    gripper: float,
    tolerance: float = 0.006,
    xy_clip: float = 0.85,
    z_clip: float = 0.85,
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
    score = 0.0
    if result.get("task_success"):
        score += 500.0
    score += float(result.get("cube_goal_improvement") or 0.0) * 1000.0
    score -= float(result.get("cube_goal_xy") or 0.0) * 250.0
    score -= float(result.get("tcp_contact_xy") or 0.0) * 120.0
    score -= float(result.get("tcp_contact_z") or 0.0) * 120.0
    score -= float(result.get("tcp_cube_xy") or 0.0) * 80.0
    if float(result.get("cube_delta_x") or 0.0) > 0.005:
        score -= 80.0
    if result.get("terminated") or result.get("truncated"):
        score -= 25.0
    return round(score, 4)


def run_probe_case(
    args: argparse.Namespace,
    *,
    contact_x_offset: float,
    contact_z_offset: float,
    approach_height: float,
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
        initial_cube_goal_xy = float(np.linalg.norm((goal - cube_start)[:2]))

        contact = cube_start + np.array([contact_x_offset, 0.0, contact_z_offset], dtype=np.float32)
        pre_contact = contact + np.array([0.0, 0.0, approach_height], dtype=np.float32)
        drag_end = np.array([goal[0] - args.drag_extra, cube_start[1], contact[2]], dtype=np.float32)

        last: Dict[str, Any] = {"terminated": False, "truncated": False, "info": {}}
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
        tcp_contact_error = contact_pose["tcp"] - contact
        tcp_contact_xy = float(np.linalg.norm(tcp_contact_error[:2]))
        tcp_contact_z = float(abs(tcp_contact_error[2]))
        far_side_at_contact = bool(
            contact_pose["tcp"][0] > cube_start[0] + 0.025
            and abs(contact_pose["tcp"][1] - cube_start[1]) < 0.07
        )

        if not (last["terminated"] or last["truncated"]):
            last = run_steps(env, (0.0, 0.0, down_bias), args.contact_steps, gripper=args.gripper_close)

        stages = max(1, int(stages))
        for stage in range(1, stages + 1):
            if last["terminated"] or last["truncated"]:
                break
            alpha = stage / stages
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
            last = run_steps(
                env,
                (drag_strength, 0.0, down_bias),
                args.drag_pulse_steps,
                gripper=args.gripper_close,
            )

        if not (last["terminated"] or last["truncated"]):
            last = run_steps(env, (0.0, 0.0, 0.0), args.settle_steps, gripper=args.gripper_close)

        final = poses(env)
        cube = final["cube"]
        tcp = final["tcp"]
        cube_goal_xy = float(np.linalg.norm((goal - cube)[:2]))
        result: Dict[str, Any] = {
            "contact_x_offset": round(float(contact_x_offset), 5),
            "contact_z_offset": round(float(contact_z_offset), 5),
            "approach_height": round(float(approach_height), 5),
            "drag_strength": round(float(drag_strength), 5),
            "down_bias": round(float(down_bias), 5),
            "stages": int(stages),
            "task_success": info_bool(last.get("info", {}), "success"),
            "cube_goal_xy_initial": round(initial_cube_goal_xy, 5),
            "cube_goal_xy": round(cube_goal_xy, 5),
            "cube_goal_improvement": round(initial_cube_goal_xy - cube_goal_xy, 5),
            "cube_delta_x": round(float(cube[0] - cube_start[0]), 5),
            "tcp_contact_xy": round(tcp_contact_xy, 5),
            "tcp_contact_z": round(tcp_contact_z, 5),
            "tcp_cube_xy": round(float(np.linalg.norm((tcp - cube)[:2])), 5),
            "tcp_cube_z": round(float(abs(tcp[2] - cube[2])), 5),
            "far_side_at_contact": far_side_at_contact,
            "terminated": bool(last.get("terminated")),
            "truncated": bool(last.get("truncated")),
            "cube_start": round_list(cube_start),
            "goal": round_list(goal),
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
        "Structured xarm6 PullCube contact probe results.",
        "Use these real simulation measurements instead of guessing contact geometry.",
        "",
        f"total_probe_cases={len(results)}",
        f"successful_probe_cases={len(successes)}",
    ]
    if ranked:
        best = ranked[0]
        lines.extend(
            [
                "",
                "best_probe_case:",
                (
                    f"  contact_x_offset={best['contact_x_offset']}, "
                    f"contact_z_offset={best['contact_z_offset']}, "
                    f"approach_height={best['approach_height']}, "
                    f"drag_strength={best['drag_strength']}, down_bias={best['down_bias']}, "
                    f"stages={best['stages']}, task_success={best['task_success']}, "
                    f"cube_goal_xy={best['cube_goal_xy']}, "
                    f"cube_goal_improvement={best['cube_goal_improvement']}, "
                    f"tcp_contact_xy={best['tcp_contact_xy']}, "
                    f"tcp_contact_z={best['tcp_contact_z']}, score={best['score']}"
                ),
            ]
        )
    lines.extend(["", "top_probe_cases:"])
    for item in ranked[:top_k]:
        lines.append(
            (
                f"- x={item['contact_x_offset']}, z={item['contact_z_offset']}, "
                f"approach={item['approach_height']}, drag={item['drag_strength']}, "
                f"down={item['down_bias']}, stages={item['stages']}, "
                f"success={item['task_success']}, goal_xy={item['cube_goal_xy']}, "
                f"progress={item['cube_goal_improvement']}, tcp_contact_xy={item['tcp_contact_xy']}, "
                f"tcp_contact_z={item['tcp_contact_z']}, score={item['score']}"
            )
        )
    lines.extend(
        [
            "",
            "LLM adapter guidance:",
            "- Prefer successful probe cases; otherwise prefer low cube_goal_xy with positive cube_goal_improvement.",
            "- If tcp_contact_xy/z stay large, repair approach/descent reachability before changing drag.",
            "- If contact is good but progress is small, repair drag pulse strength/down-bias/stages.",
            "- Keep the high-level LMP program unchanged; only adapt the target-side pull adapter.",
        ]
    )
    return "\n".join(lines)


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    results = payload["results"]
    ranked = sorted(results, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    lines = [
        "# xArm6 PullCube Contact Probe",
        "",
        "This file records contact-geometry and drag-parameter probe results.",
        "",
        "## Best Cases",
        "",
        "| rank | score | x_offset | z_offset | approach | drag | down | stages | success | cube_goal_xy | progress | tcp_contact_xy | tcp_contact_z |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for idx, item in enumerate(ranked[:20], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(item["score"]),
                    str(item["contact_x_offset"]),
                    str(item["contact_z_offset"]),
                    str(item["approach_height"]),
                    str(item["drag_strength"]),
                    str(item["down_bias"]),
                    str(item["stages"]),
                    str(item["task_success"]),
                    str(item["cube_goal_xy"]),
                    str(item["cube_goal_improvement"]),
                    str(item["tcp_contact_xy"]),
                    str(item["tcp_contact_z"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Prompt Feedback", "", "```text", payload["prompt_feedback"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    x_offsets = parse_float_list(args.contact_x_offsets)
    z_offsets = parse_float_list(args.contact_z_offsets)
    approach_heights = parse_float_list(args.approach_heights)
    drag_strengths = parse_float_list(args.drag_strengths)
    down_biases = parse_float_list(args.down_biases)
    stages_values = parse_int_list(args.stages)
    explicit_plan = list(getattr(args, "probe_plan", None) or [])
    if not explicit_plan:
        explicit_plan = load_probe_plan(getattr(args, "probe_plan_json", ""))

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
        "grid": {
            "contact_x_offsets": x_offsets,
            "contact_z_offsets": z_offsets,
            "approach_heights": approach_heights,
            "drag_strengths": drag_strengths,
            "down_biases": down_biases,
            "stages": stages_values,
        },
        "probe_plan_mode": "explicit" if explicit_plan else "grid",
    }
    if explicit_plan:
        summary["probe_plan"] = explicit_plan
    adapter.close()

    if explicit_plan:
        grid: Iterable[Tuple[float, float, float, float, float, int]] = (
            (
                float(item["contact_x_offset"]),
                float(item["contact_z_offset"]),
                float(item["approach_height"]),
                float(item["drag_strength"]),
                float(item["down_bias"]),
                int(item["stages"]),
            )
            for item in explicit_plan
        )
    else:
        grid = (
            (x, z, approach, drag, down, stages)
            for x in x_offsets
            for z in z_offsets
            for approach in approach_heights
            for drag in drag_strengths
            for down in down_biases
            for stages in stages_values
        )

    results: List[Dict[str, Any]] = []
    for idx, (x, z, approach, drag, down, stages) in enumerate(grid, start=1):
        if args.max_cases and idx > args.max_cases:
            break
        result = run_probe_case(
            args,
            contact_x_offset=x,
            contact_z_offset=z,
            approach_height=approach,
            drag_strength=drag,
            down_bias=down,
            stages=stages,
        )
        result["case_index"] = idx
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
    json_path = output_dir / "xarm6_pull_contact_probe.json"
    md_path = output_dir / "xarm6_pull_contact_probe.md"
    prompt_path = output_dir / "xarm6_pull_contact_probe_prompt.txt"
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
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--contact-x-offsets", default="0.08,0.10,0.12,0.14")
    parser.add_argument("--contact-z-offsets", default="0.010,0.014")
    parser.add_argument("--approach-heights", default="0.06,0.09")
    parser.add_argument("--drag-strengths", default="-0.6,-0.8")
    parser.add_argument("--down-biases", default="-0.02")
    parser.add_argument("--stages", default="5")
    parser.add_argument("--approach-steps", type=int, default=36)
    parser.add_argument("--descent-steps", type=int, default=36)
    parser.add_argument("--contact-steps", type=int, default=12)
    parser.add_argument("--drag-steps", type=int, default=100)
    parser.add_argument("--drag-pulse-steps", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=12)
    parser.add_argument("--drag-extra", type=float, default=0.03)
    parser.add_argument("--max-delta-m", type=float, default=0.045)
    parser.add_argument("--descent-z-clip", type=float, default=0.65)
    parser.add_argument("--gripper-close", type=float, default=-1.0)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--probe-plan-json",
        default="",
        help="Optional JSON list/path containing explicit probe cases to run instead of the Cartesian grid.",
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_arg_parser().parse_args()), ensure_ascii=False, indent=2, default=repr))


if __name__ == "__main__":
    main()
