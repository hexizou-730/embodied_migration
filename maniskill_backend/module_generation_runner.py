"""LLM target-module generation workflow for full ManiSkill migration.

This runner is the direct-generation alternative to patch loops. The LLM does
not return a unified diff. Instead, it returns the complete target adapter
module for one case. The runner writes that module, runs tests, executes the
real target simulation, and records a source-vs-target migration analysis.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .cases import PRIMARY_FULL_MIGRATION_CASE_ID, FullMigrationCase, get_full_migration_case
from .llm import gen_text
from .profiles import get_robot_profile
from .tasks import get_task_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_CONTEXT_PATH = "maniskill_backend/skill_adapter.py"
ADAPTER_CONTEXT_WINDOWS = ((1, 380),)
MODULE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ALLOWED_IMPORT_PREFIXES = (
    "__future__",
    "math",
    "typing",
    "numpy",
    "sapien",
    "mani_skill",
    "maniskill_backend.skill_adapter",
)
FORBIDDEN_CALLS = {"eval", "exec", "compile", "input", "open", "__import__"}
FORBIDDEN_TEXT = (
    "subprocess",
    "os.",
    "sys.",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "pathlib",
)


def extract_python_module(text: str) -> str:
    """Extract a complete Python module from raw LLM text."""

    candidates = [match.group(1).strip() for match in MODULE_FENCE.finditer(text)]
    if candidates:
        for candidate in candidates:
            if "def build_robot" in candidate:
                return candidate
        return candidates[0]
    return text.strip()


def validate_generated_adapter_module(code: str) -> None:
    """Reject unsafe or structurally invalid generated adapter modules."""

    if not code.strip():
        raise ValueError("Generated adapter module is empty.")
    for snippet in FORBIDDEN_TEXT:
        if snippet in code:
            raise ValueError(f"Generated adapter module contains forbidden text: {snippet}")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Generated adapter module is not valid Python: {exc}") from exc

    has_factory = any(isinstance(node, ast.FunctionDef) and node.name == "build_robot" for node in tree.body)
    if not has_factory:
        raise ValueError("Generated adapter module must define build_robot(env, *, control_mode, robot_uid).")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Generated adapter module must use absolute imports only.")
            _validate_import(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                raise ValueError(f"Generated adapter module calls forbidden function: {func.id}")


def _target_specific_generation_lines(case: FullMigrationCase) -> List[str]:
    """Return prompt constraints that are specific to the target embodiment."""

    common = [
        "# Mandatory target adapter constraints",
        "- Do not return an empty pass-through subclass. A module that only inherits ManiSkillPullCubeRobot without overriding behavior is a failed migration.",
        "- Preserve action clipping against env.action_space.low/high.",
        "- Do not import os, pathlib, subprocess, requests, urllib, socket, or shutil. Do not read environment variables.",
        "- Diagnostic pattern: cube_goal_xy remains about 0.20m, tcp_cube_xy remains large, or cube position is nearly unchanged. This means the TCP did not make effective contact.",
        "- You may increase move/contact/drag/settle steps, but step-count changes alone are not sufficient if tcp_cube_xy stays large.",
        "- Keep using the real ManiSkill success evaluation; do not infer success from distances you compute yourself.",
        "",
    ]
    if case.target_robot == "fetch":
        return [
            *common,
            "# Mandatory Fetch action-space migration for this case",
            "- The latest failure shows Fetch uses action_space.shape == (9,), while the Panda source adapter assumes 4D or 7D actions.",
            "- The generated Fetch adapter must override _validate_action_space and _make_action.",
            "- Accept Fetch 9D actions in _validate_action_space in addition to any base-compatible dimensions you intentionally keep.",
            "- Fetch pd_ee_delta_pos action layout is exactly: arm[0:3], gripper[3], body[4:7], base[7:9].",
            "- In _make_action for 9D Fetch: write clipped xyz delta to action[0:3], write gripper command to action[3], keep action[4:7] at zero unless you give a concrete reason, and allow base commands in action[7:9] when implementing mobile-base approach.",
            "",
            "# Mandatory Fetch contact-geometry migration for this case",
            "- If a previous attempt already fixed the 9D action mapping but failed with contact execution failure, continue by changing Fetch contact geometry rather than only restating the action mapping.",
            "- For Fetch, Panda's default contact_x_offset=0.07 and contact_z_offset=0.02 may miss the cube. You must try target-specific contact offsets inside the adapter.",
            "- Prefer closer/lower contact candidates such as contact_x_offset in [0.03, 0.06] and contact_z_offset in [0.005, 0.02], while staying physically safe.",
            "- You may override pull(...) to use a small set of staged contact attempts, for example approaching from multiple x offsets or adding a short lateral/forward sweep before the drag.",
            "",
            "# Mandatory Fetch mobile-base migration for this case",
            "- Fetch is a mobile manipulator. If tcp_cube_xy remains large (for example >0.15m or around 0.35m), arm-only/contact-only migration is insufficient.",
            "- The generated adapter may and should use base[7:9] in the 9D action to move the base closer before fine arm contact.",
            "- In Fetch's 9D layout, base[7:9] is a PDBaseForwardVelController action. Use bounded, staged base commands, then stop the base with zeros before arm contact/drag.",
            "- Empirical seed-0 base diagnostic: base=[+0.3, 0.0] for 30 steps reduced tcp_cube_xy from 0.1847m to 0.1476m.",
            "- Empirical seed-0 base diagnostic: base=[-0.3, 0.0] increased tcp_cube_xy to 0.5897m; base=[0.0, +0.3] and base=[0.0, -0.3] increased it above 1.17m.",
            "- Therefore use short positive base[7] pulses for coarse approach, keep base[8]=0 unless you implement an explicit measured correction, and never use negative base[7] for initial approach.",
            "- Avoid over-driving the base: previous generated adapters moved the TCP farther away (tcp_cube_xy grew to about 0.42m and later 0.77m). Stop base motion once tcp_cube_xy is around 0.12-0.15m, then rely on arm/contact refinement.",
            "- You may add helper methods such as _make_fetch_action(delta_xyz, gripper, base=None) or _drive_base_towards_cube(...), as long as all motion still uses env.step(action).",
            "- A good strategy is coarse mobile-base approach until tcp_cube_xy is below a contact threshold, then arm/contact geometry refinement, then drag.",
            "- Do not keep base[7:9] permanently zero if the latest failure shows the TCP never gets near the cube.",
            "- Required closed-loop base guard: use base[7] pulses no longer than 5-10 env.step calls before recomputing tcp_cube_xy.",
            "- Required closed-loop base guard: cap total base approach to 40 env.step calls, use base[7] in [0.1, 0.3], and keep base[8]=0.",
            "- Required closed-loop base guard: continue base approach only while tcp_cube_xy decreases; if it increases or improves by less than 0.005m for two checks, stop base motion.",
            "- Required closed-loop base guard: after base approach stops, send zero base action for at least 5 steps before arm-only contact/drag.",
            "- Do not combine long base driving, five contact retries, and overshoot in one attempt; preserve episode budget for contact and drag.",
        ]
    if case.target_robot == "xarm6_robotiq":
        return [
            *common,
            "# Mandatory xarm6_robotiq migration constraints for this case",
            "- xarm6_robotiq is a fixed-base single-arm target. Do not invent mobile-base actions, navigation APIs, or Fetch-style base/body control.",
            "- Real xarm6_robotiq diagnostic for this case: action_space.shape == (4,), with arm[0:3] and gripper_active[3].",
            "- Compared with Panda, xarm6 has less kinematic redundancy. Prefer conservative staged moves, smaller max_delta_m, and a few contact-offset candidates rather than aggressive one-shot motion.",
            "- The high-level program remains robot.pull(cube, goal). Focus migration on the target adapter: action validation, contact side, contact height, staged drag, and settle behavior.",
            "- Start from compact pd_ee_delta_pos assumptions: arm delta xyz plus gripper. Map xyz to action[0:3] and gripper to action[3] for the observed 4D controller.",
            "- Measured successful seed-0 raw action sequence: x_plus=(0.8,0,0) for 100 steps, then down=(0,0,-0.8) for 80 steps, then drag_x_minus=(-0.8,0,-0.05) for 160 steps.",
            "- A closed-loop target-position controller failed where this raw contact sequence succeeded; prefer reproducing the measured contact sequence before adding complex waypoint controllers.",
            "- For PullCube seed 0, the cube must move toward negative x. Preserve the correct contact side: approach from the positive-x side of the cube and drag toward the goal.",
            "- Tune contact_x_offset around 0.04-0.08m and contact_z_offset around 0.006-0.02m; try only a small candidate set before declaring failure.",
            "- If target execution fails, report whether the failure is reachability, action-space mapping, contact establishment, or task outcome.",
        ]
    return common


def build_module_generation_prompt(
    *,
    case: FullMigrationCase,
    target_result: Dict[str, Any],
    attempts: Sequence[Dict[str, Any]],
) -> str:
    """Prompt the LLM for a complete generated target adapter module."""

    task = get_task_spec(case.task_id)
    source_profile = get_robot_profile(case.source_robot)
    target_profile = get_robot_profile(case.target_robot)
    current_module = _read_file(case.target_adapter_path)
    target_program = _read_file(case.target_program_path)
    source_adapter_context = _read_context(ADAPTER_CONTEXT_PATH, ADAPTER_CONTEXT_WINDOWS)

    lines = [
        "You are generating target-specific robot execution code for a real ManiSkill migration case.",
        "This is direct module generation, not a patch loop.",
        "",
        "# Required output",
        "Return one complete Python module only. Do not return a diff, JSON, Markdown, or explanation.",
        f"The module will overwrite `{case.target_adapter_path}`.",
        "The module must define:",
        "def build_robot(env, *, control_mode: str, robot_uid: str):",
        "    ...",
        "",
        "# Non-negotiable safety constraints",
        "- Do not fake success, bypass evaluate(), disable failure checks, force ret_val, or edit simulator state directly to make success true.",
        "- Do not read files, call subprocesses, use network libraries, or write outside this module.",
        "- All claimed success must come from real planner/env.step execution and the task's real success evaluation.",
        "",
        "# Migration design space",
        f"- Implement target-side execution behavior for {case.target_robot} in this generated module.",
        "- Keep the public high-level LMP API compatible with the fixed target program, but you may reinterpret skill defaults and optional parameters inside the adapter.",
        "- You may subclass ManiSkillPullCubeRobot and override methods such as pull, _move_towards, _pull_cube_success, or _pull_diagnostics.",
        "- You may change contact side, contact height, intermediate waypoints, drag distance, staged motion, gripper state during contact, and controller-level fallback logic.",
        "- You may import numpy, sapien, mani_skill motion-planning helpers, and maniskill_backend.skill_adapter symbols.",
        "",
        *_target_specific_generation_lines(case),
        "",
        "# Infeasibility policy",
        "- If the target embodiment cannot physically realize the task under the current scene geometry, return a real failure from the relevant skill with a message beginning `infeasible:` and include the reachability/planner evidence.",
        "- Prefer explicit infeasibility evidence over repeated tiny contact corrections that do not change the measured cube/goal error.",
        "- Do not mark infeasible cases as success.",
        "",
        "# Case",
        f"case_id: {case.case_id}",
        f"task_id: {case.task_id}",
        f"source_robot: {case.source_robot}",
        f"target_robot: {case.target_robot}",
        f"source_control_mode: {case.source_control_mode}",
        f"target_control_mode: {case.target_control_mode}",
        f"seed: {case.seed}",
        f"episode_budget: {case.max_episode_steps}",
        "",
        "# Task Spec",
        task.to_prompt_section(),
        "",
        "# Source Robot Profile",
        source_profile.to_prompt_section(),
        "",
        "# Target Robot Profile",
        target_profile.to_prompt_section(),
        "",
        "# Unchanged high-level target LMP program",
        "```python",
        target_program,
        "```",
        "",
        "# Latest target failure",
        "```json",
        _json_dump(_result_digest(target_result)),
        "```",
    ]
    if attempts:
        lines.extend(["", "# Previous generated-module attempts"])
        for attempt in attempts[-3:]:
            lines.extend(
                [
                    f"## Attempt {attempt.get('round')}",
                    f"module_valid: {attempt.get('module_valid')}",
                    f"module_kept: {attempt.get('module_kept')}",
                    f"verification_ok: {attempt.get('verification_ok')}",
                    f"module_error: {attempt.get('module_error', '')}",
                ]
            )
            if attempt.get("target_result"):
                lines.extend(["target_result:", "```json", _json_dump(_result_digest(attempt["target_result"])), "```"])
            if attempt.get("verification") and not attempt["verification"].get("ok"):
                lines.extend(
                    [
                        "test_failure:",
                        "```text",
                        _trim_text(str(attempt["verification"].get("output", "")), 6000),
                        "```",
                    ]
                )
    lines.extend(
        [
            "",
            "# Current generated target adapter module",
            "```python",
            current_module,
            "```",
            "",
            "# Reference source/target adapter context",
            "Only use this as implementation context; return the generated module, not this whole file.",
            "```python",
            source_adapter_context,
            "```",
        ]
    )
    return "\n".join(lines)


def build_migration_analysis_prompt(
    *,
    case: FullMigrationCase,
    result: Dict[str, Any],
) -> str:
    """Ask the LLM to explain the concrete code migration after generation."""

    source_adapter_context = _read_context(ADAPTER_CONTEXT_PATH, ADAPTER_CONTEXT_WINDOWS)
    generated_module = _read_file(case.target_adapter_path)
    return "\n".join(
        [
            "Analyze this robot code migration for a paper experiment.",
            "Write concise Markdown in Chinese.",
            "",
            "# What to compare",
            "- source Panda execution assumptions",
            "- generated target adapter changes",
            "- failure evidence that motivated the target-side changes",
            "- which layer changed: program, skill adapter, controller primitive, contact geometry, or infeasibility policy",
            "",
            "# Case result",
            "```json",
            _json_dump(_result_digest(result.get("final_target_result") or {})),
            "```",
            "",
            "# Attempts",
            "```json",
            _json_dump([_attempt_digest(attempt) for attempt in result.get("attempts") or []]),
            "```",
            "",
            "# Reference source adapter context",
            "```python",
            source_adapter_context,
            "```",
            "",
            "# Generated target adapter module",
            "```python",
            generated_module,
            "```",
        ]
    )


def run_module_generation_migration(
    *,
    case_id: str = PRIMARY_FULL_MIGRATION_CASE_ID,
    max_attempts: int | None = None,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    trial_timeout_s: int = 900,
    test_timeout_s: int = 240,
    dry_run: bool = False,
    source_check: bool = True,
) -> Dict[str, Any]:
    """Generate, test, and evaluate complete target adapter modules."""

    case = get_full_migration_case(case_id)
    rounds = max_attempts if max_attempts is not None else case.max_attempts
    source_result: Dict[str, Any] | None = None
    if source_check:
        source_result = _run_source_trial(
            case=case,
            obs_mode=obs_mode,
            sim_backend=sim_backend,
            render_backend=render_backend,
            timeout_s=trial_timeout_s,
        )
        if not bool(source_result.get("success", False)):
            return _base_result(
                case=case,
                source_result=source_result,
                initial_target_result={},
                final_target_result={},
                attempts=[],
                success=False,
                message="source robot did not succeed; target module generation was not attempted",
            )

    target_result = _run_target_program_trial(
        case=case,
        obs_mode=obs_mode,
        sim_backend=sim_backend,
        render_backend=render_backend,
        timeout_s=trial_timeout_s,
    )
    initial_target_result = target_result
    attempts: List[Dict[str, Any]] = []
    if bool(target_result.get("success", False)):
        result = _base_result(
            case=case,
            source_result=source_result,
            initial_target_result=initial_target_result,
            final_target_result=target_result,
            attempts=attempts,
            success=True,
            message="target generated adapter already succeeded before regeneration",
        )
        return _attach_analysis(case=case, result=result, dry_run=dry_run)

    module_path = REPO_ROOT / case.target_adapter_path
    for round_idx in range(1, max(0, rounds) + 1):
        prompt = build_module_generation_prompt(case=case, target_result=target_result, attempts=attempts)
        current_module = module_path.read_text(encoding="utf-8")
        generated = gen_text(
            prompt=prompt,
            system=(
                "You generate complete Python modules for target robot execution adapters. "
                "Return only Python module text."
            ),
            fallback_text=current_module,
            dry_run=dry_run,
        )
        module_code = extract_python_module(generated.text)
        attempt: Dict[str, Any] = {
            "round": round_idx,
            "used_llm": generated.used_llm,
            "llm_model": generated.model,
            "llm_reason": generated.reason,
            "llm_raw_text": generated.raw_text,
            "llm_response_preview": _trim_text(generated.text, 5000),
            "prompt": prompt,
            "generated_module_preview": _trim_text(module_code, 5000),
            "module_path": case.target_adapter_path,
            "module_applied": False,
            "module_valid": False,
            "module_kept": False,
        }
        snapshot = current_module
        try:
            validate_generated_adapter_module(module_code)
            attempt["module_valid"] = True
            module_path.write_text(module_code.rstrip() + "\n", encoding="utf-8")
            attempt["module_applied"] = True
            verification = _run_command(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                timeout_s=test_timeout_s,
            )
            attempt["verification"] = verification
            attempt["verification_ok"] = verification["ok"]
            if not verification["ok"]:
                module_path.write_text(snapshot, encoding="utf-8")
                attempts.append(attempt)
                continue

            attempt["module_kept"] = True
            target_result = _run_target_program_trial(
                case=case,
                obs_mode=obs_mode,
                sim_backend=sim_backend,
                render_backend=render_backend,
                timeout_s=trial_timeout_s,
            )
            attempt["target_result"] = target_result
            attempts.append(attempt)
            if bool(target_result.get("success", False)):
                break
        except Exception as exc:
            if module_path.read_text(encoding="utf-8") != snapshot:
                module_path.write_text(snapshot, encoding="utf-8")
            attempt["module_error"] = repr(exc)
            attempts.append(attempt)

    result = _base_result(
        case=case,
        source_result=source_result,
        initial_target_result=initial_target_result,
        final_target_result=target_result,
        attempts=attempts,
        success=bool(target_result.get("success", False)),
        message="target success reached" if target_result.get("success") else "module generation budget exhausted",
    )
    return _attach_analysis(case=case, result=result, dry_run=dry_run)


def write_module_generation_outputs(
    result: Dict[str, Any],
    *,
    jsonl_path: Path,
    md_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=repr) + "\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(module_generation_result_to_md(result) + "\n", encoding="utf-8")


def module_generation_result_to_md(result: Dict[str, Any]) -> str:
    lines = [
        "# Target Adapter Module Generation",
        "",
        f"- **case_id**: `{result.get('case_id')}`",
        f"- **task**: `{result.get('task_id')}`",
        f"- **source_robot**: `{result.get('source_robot')}`",
        f"- **target_robot**: `{result.get('target_robot')}`",
        f"- **target_adapter_module**: `{result.get('target_adapter_module')}`",
        f"- **success**: `{result.get('success')}`",
        f"- **message**: `{result.get('message')}`",
        f"- **attempts**: `{len(result.get('attempts') or [])}`",
        "",
        "## Initial Target Result",
        "",
        "```json",
        _json_dump(_result_digest(result.get("initial_target_result") or {})),
        "```",
        "",
    ]
    for attempt in result.get("attempts") or []:
        lines.extend(
            [
                f"## Generated Module Attempt {attempt.get('round')}",
                "",
                f"- **module_valid**: `{attempt.get('module_valid')}`",
                f"- **module_applied**: `{attempt.get('module_applied')}`",
                f"- **module_kept**: `{attempt.get('module_kept')}`",
                f"- **verification_ok**: `{attempt.get('verification_ok')}`",
                f"- **module_error**: `{attempt.get('module_error', '')}`",
                "",
                "### Generated Module Preview",
                "",
                "```python",
                str(attempt.get("generated_module_preview") or "").strip(),
                "```",
                "",
            ]
        )
        if attempt.get("target_result"):
            lines.extend(
                [
                    "### Target Result",
                    "",
                    "```json",
                    _json_dump(_result_digest(attempt["target_result"])),
                    "```",
                    "",
                ]
            )
    if result.get("migration_analysis"):
        lines.extend(["## Migration Analysis", "", str(result["migration_analysis"]).strip(), ""])
    return "\n".join(lines)


def _attach_analysis(*, case: FullMigrationCase, result: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    fallback = _fallback_analysis(result)
    generated = gen_text(
        prompt=build_migration_analysis_prompt(case=case, result=result),
        system="You write concise robotics migration analysis for research notes.",
        fallback_text=fallback,
        dry_run=dry_run,
    )
    result.update(
        migration_analysis=generated.text,
        analysis_used_llm=generated.used_llm,
        analysis_llm_model=generated.model,
        analysis_llm_reason=generated.reason,
    )
    return result


def _base_result(
    *,
    case: FullMigrationCase,
    source_result: Dict[str, Any] | None,
    initial_target_result: Dict[str, Any],
    final_target_result: Dict[str, Any],
    attempts: List[Dict[str, Any]],
    success: bool,
    message: str,
) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "task_id": case.task_id,
        "source_robot": case.source_robot,
        "target_robot": case.target_robot,
        "target_program_path": case.target_program_path,
        "target_adapter_module": case.target_adapter_module,
        "target_adapter_path": case.target_adapter_path,
        "success": success,
        "message": message,
        "source_result": source_result,
        "initial_target_result": initial_target_result,
        "final_target_result": final_target_result,
        "attempts": attempts,
        "tracked_diff_after_run": _git_diff([case.target_adapter_path]),
    }


def _run_source_trial(
    *,
    case: FullMigrationCase,
    obs_mode: str,
    sim_backend: str,
    render_backend: str,
    timeout_s: int,
) -> Dict[str, Any]:
    return _run_real_runner_json(
        [
            "--task",
            case.task_id,
            "--robot",
            case.source_robot,
            "--method",
            "source-copy",
            "--seed",
            str(case.seed),
            "--control-mode",
            case.source_control_mode,
            "--obs-mode",
            obs_mode,
            "--sim-backend",
            sim_backend,
            "--render-backend",
            render_backend,
            "--max-episode-steps",
            str(case.max_episode_steps),
        ],
        timeout_s=timeout_s,
    )


def _run_target_program_trial(
    *,
    case: FullMigrationCase,
    obs_mode: str,
    sim_backend: str,
    render_backend: str,
    timeout_s: int,
) -> Dict[str, Any]:
    return _run_real_runner_json(
        [
            "--task",
            case.task_id,
            "--robot",
            case.target_robot,
            "--method",
            "target-module-generation",
            "--seed",
            str(case.seed),
            "--control-mode",
            case.target_control_mode,
            "--obs-mode",
            obs_mode,
            "--sim-backend",
            sim_backend,
            "--render-backend",
            render_backend,
            "--max-episode-steps",
            str(case.max_episode_steps),
            "--code-file",
            case.target_program_path,
            "--adapter-module",
            case.target_adapter_module,
        ],
        timeout_s=timeout_s,
    )


def _run_real_runner_json(args: Sequence[str], *, timeout_s: int) -> Dict[str, Any]:
    completed = _run_command(
        [sys.executable, "-m", "maniskill_backend.real_runner", *args],
        timeout_s=timeout_s,
    )
    result = _extract_json_object(completed["output"])
    if result is not None:
        result["command_ok"] = completed["ok"]
        return result
    return {
        "success": False,
        "failure_type": "execution failure",
        "failure_layer": "runtime_setup",
        "message": "real_runner did not return a JSON trial result",
        "command_ok": completed["ok"],
        "command_output": _trim_text(completed["output"], 12000),
    }


def _run_command(command: Sequence[str], *, timeout_s: int) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "command": list(command),
            "output": _trim_text(output, 20000),
        }
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            part
            for part in (
                _decode_timeout_output(exc.stdout),
                _decode_timeout_output(exc.stderr),
            )
            if part
        )
        return {
            "ok": False,
            "returncode": None,
            "command": list(command),
            "output": _trim_text(f"timeout after {timeout_s}s\n{output}", 20000),
        }


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index in (idx for idx, char in enumerate(text) if char == "{"):
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and ("task_id" in value or "success" in value):
            return value
    return None


def _validate_import(module_name: str) -> None:
    if not module_name:
        raise ValueError("Generated adapter module contains an empty import.")
    if not any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in ALLOWED_IMPORT_PREFIXES):
        raise ValueError(f"Generated adapter module imports disallowed module: {module_name}")


def _read_file(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _read_context(path: str, windows: Sequence[tuple[int, int]]) -> str:
    lines = _read_file(path).splitlines()
    chunks = []
    for start, end in windows:
        chosen = lines[max(0, start - 1) : min(len(lines), end)]
        numbered = [f"{line_no:04d}: {line}" for line_no, line in enumerate(chosen, start=start)]
        chunks.append("\n".join(numbered))
    return "\n\n".join(chunks)


def _result_digest(result: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "task_id",
        "robot_uid",
        "method",
        "adapter_module",
        "control_mode",
        "success",
        "failure_type",
        "failure_layer",
        "message",
        "execution_log",
        "final_info",
        "command_output",
    )
    return {key: result[key] for key in keys if key in result}


def _attempt_digest(attempt: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "round",
        "used_llm",
        "llm_model",
        "llm_reason",
        "module_valid",
        "module_applied",
        "module_kept",
        "verification_ok",
        "module_error",
        "target_result",
    )
    return {key: attempt[key] for key in keys if key in attempt}


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=repr)


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...<trimmed>...\n" + text[-limit // 2 :]


def _decode_timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _git_diff(paths: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "diff", "--", *paths],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout


def _fallback_analysis(result: Dict[str, Any]) -> str:
    final_target = result.get("final_target_result") or {}
    return "\n".join(
        [
            "### 迁移分析",
            "",
            f"- 最终成功: `{result.get('success')}`",
            f"- 最终失败层: `{final_target.get('failure_layer', '')}`",
            f"- 最终信息: `{final_target.get('message', '')}`",
            "- 本路线直接生成目标机器人 adapter 模块，而不是修改 patch diff。",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct LLM target-adapter module generation.")
    parser.add_argument("--case", default=PRIMARY_FULL_MIGRATION_CASE_ID)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--trial-timeout-s", type=int, default=900)
    parser.add_argument("--test-timeout-s", type=int, default=240)
    parser.add_argument("--jsonl", default="results/module_generation_trials.jsonl")
    parser.add_argument("--md", default="results/module_generation_trials.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-source-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_module_generation_migration(
        case_id=args.case,
        max_attempts=args.max_attempts,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        trial_timeout_s=args.trial_timeout_s,
        test_timeout_s=args.test_timeout_s,
        dry_run=args.dry_run,
        source_check=not args.no_source_check,
    )
    write_module_generation_outputs(
        result,
        jsonl_path=Path(args.jsonl),
        md_path=Path(args.md),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=repr))
    print(f"Wrote: {args.jsonl}")
    print(f"Wrote: {args.md}")


if __name__ == "__main__":
    main()
