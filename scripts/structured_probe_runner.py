"""Run structured simulation probes for failed adapter migrations.

This is the generic entry point for bounded probing. It selects a probe
specification from the migration case, runs the task-specific probe backend
when available, and writes a standardized JSON/Markdown/prompt artifact.

Example:

python scripts/structured_probe_runner.py \
  --case case03_pick_cube_panda_to_xarm6 \
  --sim-backend auto \
  --render-backend gpu

Use --dry-run on machines without ManiSkill/GPU access to inspect the selected
probe grid without running simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.cases import get_full_migration_case
from maniskill_backend.structured_probe import (
    build_probe_feedback,
    get_probe_spec,
    probe_grid,
    suggest_next_probe_cases,
    summarize_probe_results,
    write_probe_outputs,
)


def _join_grid_values(values: Any) -> str:
    return ",".join(str(item) for item in values)


def _load_json(text_or_path: str) -> Dict[str, Any]:
    if not text_or_path:
        return {}
    stripped = text_or_path.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)
    path = Path(text_or_path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(stripped)


def _pick_cube_xarm6_args(args: argparse.Namespace, spec: Any, *, probe_plan: Optional[Any] = None) -> Namespace:
    grid = spec.parameter_grid
    return Namespace(
        seed=args.seed,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        output_dir=args.legacy_output_dir,
        grasp_z_offsets=args.grasp_z_offsets or _join_grid_values(grid["grasp_z_offset"]),
        close_steps=args.close_steps or _join_grid_values(grid["close_steps"]),
        close_commands=args.close_commands or _join_grid_values(grid["close_command"]),
        settle_steps=args.settle_steps or _join_grid_values(grid["settle_steps"]),
        open_steps=args.open_steps,
        move_steps=args.move_steps,
        descend_steps=args.descend_steps,
        lift_steps=args.lift_steps,
        approach_height=args.approach_height,
        lift_height=args.lift_height,
        max_delta_m=args.max_delta_m,
        descend_max_delta_m=args.descend_max_delta_m,
        preclose_tolerance=args.preclose_tolerance,
        move_xy_clip=args.move_xy_clip,
        move_z_clip=args.move_z_clip,
        descend_xy_clip=args.descend_xy_clip,
        descend_z_clip=args.descend_z_clip,
        gripper_open=args.gripper_open,
        staged_close=args.staged_close,
        stop_on_grasp=args.stop_on_grasp,
        max_cases=args.max_cases,
        top_k=args.top_k,
        probe_plan=probe_plan or [],
        probe_plan_json="",
    )


def _pull_cube_xarm6_args(args: argparse.Namespace, spec: Any, *, probe_plan: Optional[Any] = None) -> Namespace:
    grid = spec.parameter_grid
    return Namespace(
        seed=args.seed,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        output_dir=args.legacy_output_dir,
        contact_x_offsets=args.contact_x_offsets or _join_grid_values(grid["contact_x_offset"]),
        contact_z_offsets=args.contact_z_offsets or _join_grid_values(grid["contact_z_offset"]),
        approach_heights=args.approach_heights or _join_grid_values(grid["approach_height"]),
        drag_strengths=args.drag_strengths or _join_grid_values(grid["drag_strength"]),
        down_biases=args.down_biases or _join_grid_values(grid["down_bias"]),
        stages=args.stages or _join_grid_values(grid["stages"]),
        approach_steps=args.approach_steps,
        descent_steps=args.descend_steps,
        contact_steps=args.contact_steps,
        drag_steps=args.drag_steps,
        drag_pulse_steps=args.drag_pulse_steps,
        settle_steps=args.pull_settle_steps,
        drag_extra=args.drag_extra,
        max_delta_m=args.max_delta_m,
        descent_z_clip=args.descent_z_clip,
        gripper_close=args.gripper_close,
        stop_on_success=args.stop_on_success,
        max_cases=args.max_cases,
        top_k=args.top_k,
        probe_plan=probe_plan or [],
        probe_plan_json="",
    )


def run_structured_probe(args: argparse.Namespace) -> Dict[str, Any]:
    case = get_full_migration_case(args.case)
    diagnosis = _load_json(args.failure_diagnosis_json) if args.failure_diagnosis_json else {}
    spec = get_probe_spec(case, diagnosis=diagnosis)
    adaptive_source: Dict[str, Any] = _load_json(args.adaptive_from) if args.adaptive_from else {}
    adaptive_plan = []
    if adaptive_source:
        adaptive_plan = suggest_next_probe_cases(
            spec,
            adaptive_source.get("all_probe_cases") or adaptive_source.get("results") or [],
            budget=args.suggestion_budget,
        )

    if args.dry_run or args.suggest_only:
        payload = summarize_probe_results(spec, [], diagnosis=diagnosis, top_k=args.top_k, dry_run=True)
        planned = adaptive_plan or probe_grid(spec, max_cases=args.max_cases)
        payload["num_planned_cases"] = len(planned)
        payload["planned_probe_cases"] = planned
        payload["next_probe_suggestions"] = planned
        if adaptive_source:
            payload["learning_guided_source"] = args.adaptive_from
            payload["learning_guided_mode"] = "suggest_only" if args.suggest_only else "dry_run_adaptive_plan"
        payload["prompt_feedback"] = build_probe_feedback(payload, top_k=args.top_k)
    elif spec.probe_id == "pick_cube_xarm6_close_envelope":
        from scripts import xarm6_pick_grasp_probe

        task_payload = xarm6_pick_grasp_probe.run(_pick_cube_xarm6_args(args, spec, probe_plan=adaptive_plan))
        payload = summarize_probe_results(
            spec,
            task_payload.get("results") or [],
            diagnosis=diagnosis,
            top_k=args.top_k,
            dry_run=False,
        )
        if adaptive_plan:
            payload["learning_guided_source"] = args.adaptive_from
            payload["learning_guided_probe_plan"] = adaptive_plan
        payload["legacy_probe_wrote"] = task_payload.get("wrote", {})
        payload["controller_summary"] = task_payload.get("controller_summary", {})
        payload["initial"] = task_payload.get("initial", {})
    elif spec.probe_id == "pull_cube_xarm6_contact_geometry":
        from scripts import xarm6_pull_contact_probe

        task_payload = xarm6_pull_contact_probe.run(_pull_cube_xarm6_args(args, spec, probe_plan=adaptive_plan))
        payload = summarize_probe_results(
            spec,
            task_payload.get("results") or [],
            diagnosis=diagnosis,
            top_k=args.top_k,
            dry_run=False,
        )
        if adaptive_plan:
            payload["learning_guided_source"] = args.adaptive_from
            payload["learning_guided_probe_plan"] = adaptive_plan
        payload["legacy_probe_wrote"] = task_payload.get("wrote", {})
        payload["controller_summary"] = task_payload.get("controller_summary", {})
        payload["initial"] = task_payload.get("initial", {})
    else:
        raise RuntimeError(f"No executable probe backend for {spec.probe_id!r}.")

    output_dir = Path(args.output_dir) / case.case_id
    payload["wrote"] = write_probe_outputs(output_dir, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Migration case id, e.g. case03_pick_cube_panda_to_xarm6.")
    parser.add_argument(
        "--failure-diagnosis-json",
        default="",
        help="Optional failure diagnosis JSON string or path to a JSON file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Select and write the probe grid without simulation.")
    parser.add_argument(
        "--adaptive-from",
        default="",
        help="Optional previous structured probe JSON path/string used to generate a score-guided next probe plan.",
    )
    parser.add_argument(
        "--suggest-only",
        action="store_true",
        help="Only write the learning-guided next probe plan; do not run simulation.",
    )
    parser.add_argument("--suggestion-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=220)
    parser.add_argument("--output-dir", default="results/structured_probes")
    parser.add_argument("--legacy-output-dir", default="results")

    # Optional overrides for the currently implemented close-envelope backend.
    parser.add_argument("--grasp-z-offsets", default="")
    parser.add_argument("--close-steps", default="")
    parser.add_argument("--close-commands", default="")
    parser.add_argument("--settle-steps", default="")
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

    # Optional overrides for the xarm6 PullCube contact-geometry backend.
    parser.add_argument("--contact-x-offsets", default="")
    parser.add_argument("--contact-z-offsets", default="")
    parser.add_argument("--approach-heights", default="")
    parser.add_argument("--drag-strengths", default="")
    parser.add_argument("--down-biases", default="")
    parser.add_argument("--stages", default="")
    parser.add_argument("--approach-steps", type=int, default=36)
    parser.add_argument("--contact-steps", type=int, default=12)
    parser.add_argument("--drag-steps", type=int, default=100)
    parser.add_argument("--drag-pulse-steps", type=int, default=4)
    parser.add_argument("--pull-settle-steps", type=int, default=12)
    parser.add_argument("--drag-extra", type=float, default=0.03)
    parser.add_argument("--gripper-close", type=float, default=-1.0)
    parser.add_argument("--stop-on-success", action="store_true")

    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_structured_probe(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
