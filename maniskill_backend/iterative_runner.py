"""Iterative LLM program-migration runner for the full ManiSkill stack.

This runner covers the LMP-program layer of cross-embodiment migration:
1. verify the source robot succeeds with the source program;
2. ask an LLM to write target-robot code;
3. execute it in real ManiSkill simulation;
4. feed the failure log back to the LLM;
5. repeat up to N attempts.

If a failure is caused by target grasp geometry, contact primitives, planner
routes, or controller assumptions, migrate ``skill_adapter.py`` as well; LMP
rewriting alone is not the full migration experiment.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cases import PRIMARY_FULL_MIGRATION_CASE
from .llm import gen_code
from .profiles import get_robot_profile
from .real_runner import _default_control_mode, run_real_code_trial, run_real_trial
from .tasks import TaskSpec, get_task_spec


def build_iterative_prompt(
    *,
    task: TaskSpec,
    source_robot: str,
    target_robot: str,
    previous_attempts: List[Dict[str, Any]],
) -> str:
    """Build the per-attempt prompt for iterative target-code generation."""

    source_profile = get_robot_profile(source_robot)
    target_profile = get_robot_profile(target_robot)
    lines = [
        "You are doing iterative robot code migration in real ManiSkill simulation.",
        "A source robot already completed the task with the source program.",
        "Write executable Python LMP code for the target robot to complete the SAME task.",
        "",
        "# Allowed API",
        "- scene.get_object(name)",
        "- scene.get_region(name)",
        "- robot.pull(obj, target)",
        "",
        "# Safety constraints",
        "- Output only executable Python code. Do not include Markdown.",
        "- Do not fake success, bypass task outcomes, or directly modify simulator state.",
        "- If the target cannot realize the task with the exposed API, set ret_val to a string beginning `infeasible:` and briefly state the reason.",
        "",
        "# Code-generation constraints",
        "- Objects returned by scene are opaque handles. Do not call methods on them.",
        "- Do not use obj.get_position(), obj.pose, obj.position, or distance math.",
        "- Do not import packages.",
        "- Set ret_val to the final success/failure value.",
        "- You may change the target code across attempts based on simulator feedback.",
        "",
        task.to_prompt_section(),
        "",
        source_profile.to_prompt_section(),
        "",
        target_profile.to_prompt_section(),
    ]
    if task.task_id == "pull_cube":
        lines.extend(
            [
                "",
                "# Extra tunable API for this task",
                "- robot.pull(cube, goal, contact_x_offset=0.07, contact_z_offset=0.02, drag_extra=0.02, stages=4)",
                "- You may tune contact_x_offset, contact_z_offset, drag_extra, and stages.",
                "- Do not invent grasp/place/tool APIs for PullCube-v1; this is a contact-pulling task.",
                "- If simulator feedback shows the failure is in target contact/controller reachability, you may return `infeasible: target adapter/controller migration required`.",
            ]
        )

    if previous_attempts:
        lines.extend(["", "# Previous target attempts and simulator feedback"])
        for attempt in previous_attempts:
            result = attempt["result"]
            lines.extend(
                [
                    "",
                    f"## Attempt {attempt['attempt']}",
                    f"success: {result.get('success')}",
                    f"failure_type: {result.get('failure_type')}",
                    f"failure_layer: {result.get('failure_layer')}",
                    f"message: {result.get('message')}",
                    "code:",
                    "```python",
                    str(attempt.get("code", "")).strip(),
                    "```",
                ]
            )
            execution_log = result.get("execution_log") or []
            if execution_log:
                lines.extend(["execution_log:"])
                lines.extend(f"- {line}" for line in _format_execution_log(execution_log))
            final_info = result.get("final_info")
            if final_info:
                lines.append(f"final_info: {json.dumps(final_info, ensure_ascii=False, default=repr)}")
        lines.extend(
            [
                "",
                "Revise the code for the next attempt. Do not simply repeat code that failed unless the feedback proves the remaining failure is in the target skill adapter or controller primitive rather than the LMP program.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "# First target attempt",
                "Start from the source program if appropriate, but adapt it for the target robot when useful.",
            ]
        )

    lines.extend(["", "# Required output", "Return only the Python code for the next target attempt."])
    return "\n".join(lines)


def run_iterative_migration(
    *,
    task_id: str,
    source_robot: Optional[str] = None,
    target_robot: str,
    max_attempts: int = 3,
    seed: int = 0,
    source_control_mode: Optional[str] = None,
    target_control_mode: Optional[str] = None,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    max_episode_steps: int = 300,
    dry_run: bool = False,
    source_check: bool = True,
) -> Dict[str, Any]:
    task = get_task_spec(task_id)
    source_robot = source_robot or task.source_robot
    if source_control_mode is None:
        source_control_mode = _default_control_mode(task.task_id, source_robot)
    if target_control_mode is None:
        target_control_mode = _default_control_mode(task.task_id, target_robot)

    source_result: Optional[Dict[str, Any]] = None
    if source_check:
        source_result = run_real_trial(
            task_id=task.task_id,
            robot_uid=source_robot,
            method="source-copy",
            seed=seed,
            control_mode=source_control_mode,
            obs_mode=obs_mode,
            sim_backend=sim_backend,
            render_backend=render_backend,
            max_episode_steps=max_episode_steps,
        )
        if not bool(source_result.get("success", False)):
            return {
                "task_id": task.task_id,
                "source_robot": source_robot,
                "target_robot": target_robot,
                "method": "iterative_llm",
                "success": False,
                "aborted": True,
                "message": "source robot did not succeed; target migration was not attempted",
                "source_result": source_result,
                "attempts": [],
            }

    attempts: List[Dict[str, Any]] = []
    last_code: Optional[str] = None
    success = False
    successful_attempt: Optional[int] = None

    for attempt_idx in range(1, max_attempts + 1):
        prompt = build_iterative_prompt(
            task=task,
            source_robot=source_robot,
            target_robot=target_robot,
            previous_attempts=attempts,
        )
        generated = gen_code(
            prompt=prompt,
            fallback_code=last_code or task.source_program.strip(),
            dry_run=dry_run,
        )
        code = generated.code
        diff = _code_diff(last_code or "", code)
        result = run_real_code_trial(
            task_id=task.task_id,
            robot_uid=target_robot,
            method="iterative_llm",
            code=code,
            prompt=prompt,
            seed=seed,
            control_mode=target_control_mode,
            obs_mode=obs_mode,
            sim_backend=sim_backend,
            render_backend=render_backend,
            max_episode_steps=max_episode_steps,
            extra_result={
                "attempt": attempt_idx,
                "used_llm": generated.used_llm,
                "llm_model": generated.model,
                "llm_reason": generated.reason,
                "llm_raw_text": generated.raw_text,
            },
        )
        attempt_record = {
            "attempt": attempt_idx,
            "code": code,
            "code_changed_from_previous": bool(last_code is not None and code.strip() != last_code.strip()),
            "diff_from_previous": diff,
            "result": result,
            "used_llm": generated.used_llm,
            "llm_model": generated.model,
            "llm_reason": generated.reason,
            "llm_raw_text": generated.raw_text,
        }
        attempts.append(attempt_record)
        last_code = code
        if bool(result.get("success", False)):
            success = True
            successful_attempt = attempt_idx
            break

    final_result = attempts[-1]["result"] if attempts else {}
    return {
        "task_id": task.task_id,
        "task_name": task.display_name,
        "task_name_cn": task.name_cn,
        "env_id": task.maniskill_env_id,
        "source_robot": source_robot,
        "target_robot": target_robot,
        "method": "iterative_llm",
        "seed": seed,
        "source_control_mode": source_control_mode,
        "target_control_mode": target_control_mode,
        "max_attempts": max_attempts,
        "attempts_run": len(attempts),
        "success": success,
        "successful_attempt": successful_attempt,
        "final_failure_type": final_result.get("failure_type", ""),
        "final_failure_layer": final_result.get("failure_layer", ""),
        "final_message": final_result.get("message", ""),
        "source_result": source_result,
        "attempts": attempts,
    }


def write_iterative_outputs(
    result: Dict[str, Any],
    *,
    jsonl_path: Path,
    md_path: Path,
    summary_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=repr) + "\n")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(iterative_result_to_md(result) + "\n", encoding="utf-8")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task_id",
            "source_robot",
            "target_robot",
            "success",
            "attempts_run",
            "successful_attempt",
            "final_failure_type",
            "final_failure_layer",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: result.get(key, "") for key in fieldnames})


def iterative_result_to_md(result: Dict[str, Any]) -> str:
    lines = [
        "# Iterative Migration Trial",
        "",
        f"- **task**: `{result.get('task_id')}`",
        f"- **source_robot**: `{result.get('source_robot')}`",
        f"- **target_robot**: `{result.get('target_robot')}`",
        f"- **success**: `{result.get('success')}`",
        f"- **attempts_run**: `{result.get('attempts_run')}`",
        f"- **successful_attempt**: `{result.get('successful_attempt')}`",
        f"- **final_failure_type**: `{result.get('final_failure_type')}`",
        f"- **final_failure_layer**: `{result.get('final_failure_layer')}`",
        f"- **final_message**: `{result.get('final_message')}`",
        "",
    ]
    source = result.get("source_result") or {}
    if source:
        lines.extend(
            [
                "## Source Check",
                "",
                f"- **success**: `{source.get('success')}`",
                f"- **message**: `{source.get('message')}`",
                "",
            ]
        )
    for attempt in result.get("attempts", []):
        trial = attempt.get("result", {})
        lines.extend(
            [
                f"## Attempt {attempt.get('attempt')}",
                "",
                f"- **success**: `{trial.get('success')}`",
                f"- **failure_type**: `{trial.get('failure_type')}`",
                f"- **failure_layer**: `{trial.get('failure_layer')}`",
                f"- **message**: `{trial.get('message')}`",
                f"- **code_changed_from_previous**: `{attempt.get('code_changed_from_previous')}`",
                "",
            ]
        )
        if attempt.get("diff_from_previous"):
            lines.extend(["### Diff", "", "```diff", attempt["diff_from_previous"].strip(), "```", ""])
        execution_log = trial.get("execution_log")
        if execution_log:
            lines.extend(["### Execution Log", "", "```text"])
            lines.extend(_format_execution_log(execution_log))
            lines.extend(["```", ""])
        final_info = trial.get("final_info")
        if final_info:
            lines.extend(["### Final Info", "", "```json", json.dumps(final_info, ensure_ascii=False, indent=2, default=repr), "```", ""])
        lines.extend(["### Code", "", "```python", str(attempt.get("code", "")).strip(), "```", ""])
        raw = attempt.get("llm_raw_text")
        if raw:
            lines.extend(["### Raw LLM Text", "", "```text", str(raw).strip(), "```", ""])
    return "\n".join(lines)


def _format_execution_log(execution_log: Any) -> List[str]:
    if not isinstance(execution_log, list):
        return []
    lines = []
    for event in execution_log:
        if not isinstance(event, dict):
            continue
        args = event.get("args") or {}
        arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(args.items()))
        line = (
            f"{event.get('step', '?')}. {event.get('api', 'unknown')}({arg_text}) "
            f"-> {event.get('result')} ok={event.get('ok')}"
        )
        if event.get("message"):
            line += f" {event.get('message')}"
        lines.append(line)
    return lines


def _code_diff(previous: str, current: str) -> str:
    if not previous:
        return ""
    return "\n".join(
        difflib.unified_diff(
            previous.strip().splitlines(),
            current.strip().splitlines(),
            fromfile="previous_attempt.py",
            tofile="current_attempt.py",
            lineterm="",
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative LLM robot-code migration.")
    parser.add_argument("--task", default=PRIMARY_FULL_MIGRATION_CASE.task_id)
    parser.add_argument("--source-robot", default=PRIMARY_FULL_MIGRATION_CASE.source_robot)
    parser.add_argument("--target-robot", default=PRIMARY_FULL_MIGRATION_CASE.target_robot)
    parser.add_argument("--max-attempts", type=int, default=PRIMARY_FULL_MIGRATION_CASE.max_attempts)
    parser.add_argument("--seed", type=int, default=PRIMARY_FULL_MIGRATION_CASE.seed)
    parser.add_argument("--source-control-mode", default=PRIMARY_FULL_MIGRATION_CASE.source_control_mode)
    parser.add_argument("--target-control-mode", default=PRIMARY_FULL_MIGRATION_CASE.target_control_mode)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=PRIMARY_FULL_MIGRATION_CASE.max_episode_steps)
    parser.add_argument("--jsonl", default="results/iterative_trials.jsonl")
    parser.add_argument("--md", default="results/iterative_trials.md")
    parser.add_argument("--summary", default="results/iterative_summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_iterative_migration(
        task_id=args.task,
        source_robot=args.source_robot,
        target_robot=args.target_robot,
        max_attempts=args.max_attempts,
        seed=args.seed,
        source_control_mode=args.source_control_mode,
        target_control_mode=args.target_control_mode,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
        dry_run=args.dry_run,
        source_check=not args.no_source_check,
    )
    write_iterative_outputs(
        result,
        jsonl_path=Path(args.jsonl),
        md_path=Path(args.md),
        summary_path=Path(args.summary),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=repr))
    print(f"Wrote: {args.jsonl}")
    print(f"Wrote: {args.md}")
    print(f"Wrote: {args.summary}")


if __name__ == "__main__":
    main()
