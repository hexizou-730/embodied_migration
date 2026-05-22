"""Bounded LLM-driven full-stack repair loop for ManiSkill migration cases."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .cases import PRIMARY_FULL_MIGRATION_CASE_ID, FullMigrationCase, get_full_migration_case
from .llm import gen_text


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_ALLOWED_PATHS = (
    "maniskill_backend/case_programs/case01_pull_cube_tool.py",
    "maniskill_backend/profiles.py",
    "maniskill_backend/real_runner.py",
    "maniskill_backend/skill_adapter.py",
)
PATCH_CONTEXT_WINDOWS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "maniskill_backend/case_programs/case01_pull_cube_tool.py": ((1, 120),),
    "maniskill_backend/profiles.py": ((1, 260),),
    "maniskill_backend/real_runner.py": ((1, 440),),
    # The PullCubeTool planner wrapper currently lives in this window.
    "maniskill_backend/skill_adapter.py": ((810, 1415),),
}

_PATCH_FENCE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def extract_unified_diff(text: str) -> str:
    """Extract one git-style unified diff from raw LLM text."""

    fenced = _PATCH_FENCE.search(text)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    start = candidate.find("diff --git ")
    return candidate[start:].strip() if start >= 0 else ""


def patch_paths(patch: str) -> Tuple[str, ...]:
    """Return changed repo paths from a git-style patch."""

    paths: List[str] = []
    for old_path, new_path in _DIFF_HEADER.findall(patch):
        for path in (old_path, new_path):
            if path != "/dev/null" and path not in paths:
                paths.append(path)
    return tuple(paths)


def validate_patch(patch: str, allowed_paths: Iterable[str] = PATCH_ALLOWED_PATHS) -> Tuple[str, ...]:
    """Reject empty or out-of-scope patches before calling ``git apply``."""

    paths = patch_paths(patch)
    if not paths:
        raise ValueError("LLM response did not contain a git-style unified diff.")

    allowed = set(allowed_paths)
    blocked = [path for path in paths if path not in allowed]
    if blocked:
        raise ValueError(f"Patch touched non-Case-01 paths: {', '.join(blocked)}")
    return paths


def build_full_stack_patch_prompt(
    *,
    case: FullMigrationCase,
    target_result: Dict[str, Any],
    attempts: Sequence[Dict[str, Any]],
) -> str:
    """Ask the LLM to migrate the failing layer with one bounded patch."""

    lines = [
        "You are an autonomous robot migration engineer.",
        "Migrate every layer needed for the target embodiment to complete the same task.",
        "Do not stop at the high-level LMP program if the failure is in a skill adapter, planner, controller primitive, tool pose, grasp geometry, or contact path.",
        "",
        "# Required output",
        "Return exactly one git-style unified diff. Do not include Markdown or explanation.",
        "",
        "# Case",
        f"case_id: {case.case_id}",
        f"task_id: {case.task_id}",
        f"source_robot: {case.source_robot}",
        f"target_robot: {case.target_robot}",
        f"target_program_path: {case.target_program_path}",
        f"source_control_mode: {case.source_control_mode}",
        f"target_control_mode: {case.target_control_mode}",
        f"seed: {case.seed}",
        "",
        "# Patch scope",
        "You may change only these files:",
    ]
    lines.extend(f"- {path}" for path in PATCH_ALLOWED_PATHS)
    lines.extend(
        [
            "",
            "# Migration rules",
            "- Preserve the source-side Panda baseline; only migrate target-side code and target execution assumptions.",
            "- Keep changes focused on the observed failure evidence.",
            "- Prefer real robot execution semantics over text-only workarounds.",
            "- Do not disable success checks, fake simulator outcomes, or mark failure as success.",
            "- Do not edit tests, docs, result files, environment files, or dependency files.",
            "- The patch will be tested and then rerun in real ManiSkill simulation.",
            "",
            "# Latest target failure",
            _json_dump(_result_digest(target_result)),
        ]
    )
    if attempts:
        lines.extend(["", "# Previous full-stack patch attempts"])
        for attempt in attempts[-3:]:
            lines.extend(
                [
                    f"## Round {attempt.get('round')}",
                    f"patch_applied: {attempt.get('patch_applied')}",
                    f"patch_kept: {attempt.get('patch_kept')}",
                    f"verification_ok: {attempt.get('verification_ok')}",
                    f"patch_paths: {attempt.get('patch_paths')}",
                ]
            )
            if attempt.get("patch_error"):
                lines.append(f"patch_error: {attempt['patch_error']}")
            verification = attempt.get("verification") or {}
            if verification and not verification.get("ok"):
                lines.extend(
                    [
                        "verification_failure:",
                        _trim_text(str(verification.get("output", "")), 5000),
                    ]
                )
            if attempt.get("target_result"):
                lines.extend(
                    [
                        "target_result_after_patch:",
                        _json_dump(_result_digest(attempt["target_result"])),
                    ]
                )
    lines.extend(["", "# In-scope code context"])
    for path in PATCH_ALLOWED_PATHS:
        lines.extend(["", f"## {path}", "```python"])
        lines.append(_read_context(path, PATCH_CONTEXT_WINDOWS.get(path, ())))
        lines.append("```")
    return "\n".join(lines)


def run_full_stack_migration(
    *,
    case_id: str = PRIMARY_FULL_MIGRATION_CASE_ID,
    max_repair_rounds: int | None = None,
    obs_mode: str = "state",
    sim_backend: str = "auto",
    render_backend: str = "gpu",
    trial_timeout_s: int = 900,
    test_timeout_s: int = 240,
    dry_run: bool = False,
    source_check: bool = True,
    allow_dirty: bool = False,
) -> Dict[str, Any]:
    """Run source check, target failure, LLM repo patches, and target retries."""

    case = get_full_migration_case(case_id)
    rounds = max_repair_rounds if max_repair_rounds is not None else case.max_attempts
    if not allow_dirty:
        dirty_paths = _tracked_dirty_paths()
        if dirty_paths:
            raise RuntimeError(
                "full_stack_runner needs a clean tracked worktree before it edits code: "
                + ", ".join(dirty_paths)
            )

    source_result: Dict[str, Any] | None = None
    if source_check:
        source_result = _run_real_trial(
            case=case,
            robot=case.source_robot,
            method="source-copy",
            control_mode=case.source_control_mode,
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
                message="source robot did not succeed; full-stack repair was not attempted",
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
        return _base_result(
            case=case,
            source_result=source_result,
            initial_target_result=initial_target_result,
            final_target_result=target_result,
            attempts=attempts,
            success=True,
            message="target program already succeeded before repair",
        )

    for round_idx in range(1, max(0, rounds) + 1):
        prompt = build_full_stack_patch_prompt(
            case=case,
            target_result=target_result,
            attempts=attempts,
        )
        generated = gen_text(
            prompt=prompt,
            system=(
                "You migrate robot programs and execution layers. "
                "Return only one git-style unified diff within the allowed files."
            ),
            fallback_text="",
            dry_run=dry_run,
        )
        attempt: Dict[str, Any] = {
            "round": round_idx,
            "used_llm": generated.used_llm,
            "llm_model": generated.model,
            "llm_reason": generated.reason,
            "llm_raw_text": generated.raw_text,
            "prompt": prompt,
            "patch": extract_unified_diff(generated.text),
            "patch_applied": False,
            "patch_kept": False,
        }
        try:
            attempt["patch_paths"] = list(validate_patch(attempt["patch"]))
            _git_apply(attempt["patch"])
            attempt["patch_applied"] = True
            verification = _run_command(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                timeout_s=test_timeout_s,
            )
            attempt["verification"] = verification
            attempt["verification_ok"] = verification["ok"]
            if not verification["ok"]:
                _git_apply(attempt["patch"], reverse=True)
                attempt["patch_kept"] = False
                attempts.append(attempt)
                continue

            attempt["patch_kept"] = True
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
            attempt["patch_error"] = repr(exc)
            attempts.append(attempt)

    return _base_result(
        case=case,
        source_result=source_result,
        initial_target_result=initial_target_result,
        final_target_result=target_result,
        attempts=attempts,
        success=bool(target_result.get("success", False)),
        message="target success reached" if target_result.get("success") else "repair budget exhausted",
    )


def write_full_stack_outputs(
    result: Dict[str, Any],
    *,
    jsonl_path: Path,
    md_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=repr) + "\n")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(full_stack_result_to_md(result) + "\n", encoding="utf-8")


def full_stack_result_to_md(result: Dict[str, Any]) -> str:
    lines = [
        "# Full-Stack LLM Migration",
        "",
        f"- **case_id**: `{result.get('case_id')}`",
        f"- **task**: `{result.get('task_id')}`",
        f"- **source_robot**: `{result.get('source_robot')}`",
        f"- **target_robot**: `{result.get('target_robot')}`",
        f"- **success**: `{result.get('success')}`",
        f"- **message**: `{result.get('message')}`",
        f"- **repair_rounds**: `{len(result.get('attempts') or [])}`",
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
                f"## Repair Round {attempt.get('round')}",
                "",
                f"- **patch_paths**: `{attempt.get('patch_paths')}`",
                f"- **patch_applied**: `{attempt.get('patch_applied')}`",
                f"- **patch_kept**: `{attempt.get('patch_kept')}`",
                f"- **verification_ok**: `{attempt.get('verification_ok')}`",
                f"- **patch_error**: `{attempt.get('patch_error', '')}`",
                "",
                "### Patch",
                "",
                "```diff",
                str(attempt.get("patch") or "").strip(),
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
    return "\n".join(lines)


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
        "success": success,
        "message": message,
        "source_result": source_result,
        "initial_target_result": initial_target_result,
        "final_target_result": final_target_result,
        "attempts": attempts,
        "tracked_diff_after_run": _git_diff(),
    }


def _run_real_trial(
    *,
    case: FullMigrationCase,
    robot: str,
    method: str,
    control_mode: str,
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
            robot,
            "--method",
            method,
            "--seed",
            str(case.seed),
            "--control-mode",
            control_mode,
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
            "full-stack-llm",
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


def _git_apply(patch: str, *, reverse: bool = False) -> None:
    command = ["git", "apply"]
    if reverse:
        command.append("--reverse")
    check = subprocess.run(
        [*command, "--check", "-"],
        cwd=REPO_ROOT,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(_trim_text(check.stderr or check.stdout, 4000))
    apply_result = subprocess.run(
        [*command, "-"],
        cwd=REPO_ROOT,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )
    if apply_result.returncode != 0:
        raise RuntimeError(_trim_text(apply_result.stderr or apply_result.stdout, 4000))


def _tracked_dirty_paths() -> Tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _git_diff() -> str:
    completed = subprocess.run(
        ["git", "diff", "--", *PATCH_ALLOWED_PATHS],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout


def _read_context(path: str, windows: Sequence[Tuple[int, int]]) -> str:
    lines = (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded LLM full-stack migration.")
    parser.add_argument("--case", default=PRIMARY_FULL_MIGRATION_CASE_ID)
    parser.add_argument("--max-repair-rounds", type=int, default=None)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--trial-timeout-s", type=int, default=900)
    parser.add_argument("--test-timeout-s", type=int, default=240)
    parser.add_argument("--jsonl", default="results/full_stack_trials.jsonl")
    parser.add_argument("--md", default="results/full_stack_trials.md")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-source-check", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_full_stack_migration(
        case_id=args.case,
        max_repair_rounds=args.max_repair_rounds,
        obs_mode=args.obs_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        trial_timeout_s=args.trial_timeout_s,
        test_timeout_s=args.test_timeout_s,
        dry_run=args.dry_run,
        source_check=not args.no_source_check,
        allow_dirty=args.allow_dirty,
    )
    write_full_stack_outputs(
        result,
        jsonl_path=Path(args.jsonl),
        md_path=Path(args.md),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=repr))
    print(f"Wrote: {args.jsonl}")
    print(f"Wrote: {args.md}")


if __name__ == "__main__":
    main()
