"""Static migration runner for early WSL2 development.

This runner deliberately avoids real ManiSkill simulation. It executes LMP code
against a tiny fake scene/robot API so we can validate prompt construction,
source-copy/oracle behavior, failure classification, and JSONL logging before
the GPU simulator is ready.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from lmp.executor import execute_lmp
from lmp.failure_report import FailureReport

from .evaluation import TrialRecord, classify_failure
from .llm import gen_code
from .migration import MigrationRequest, build_migration_prompt, get_source_copy_code, norm_method
from .profiles import RobotProfile
from .tasks import TaskSpec


@dataclass(frozen=True)
class FakeEntity:
    name: str
    kind: str = "object"


@dataclass(frozen=True)
class ExecutionEvent:
    step: int
    api: str
    args: Dict[str, Any]
    result: Any
    ok: bool
    message: str = ""
    failure_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "api": self.api,
            "args": dict(self.args),
            "result": self.result,
            "ok": self.ok,
            "message": self.message,
            "failure_type": self.failure_type,
        }


class FakeScene:
    def get_object(self, name: str) -> FakeEntity:
        return FakeEntity(name=name, kind="object")

    def get_region(self, name: str) -> FakeEntity:
        return FakeEntity(name=name, kind="region")


class StaticRobot:
    def __init__(self, profile: RobotProfile, task_id: str = "") -> None:
        self.profile = profile
        self.card = profile.card
        self.task_id = task_id
        self.last_failure: Optional[str] = None
        self.last_aligned = False
        self.last_hooked = False
        self.align_count = 0
        self.events: List[ExecutionEvent] = []

    @property
    def needs_empirical_margin(self) -> bool:
        return "PegMulti" in self.task_id or "PlugMulti" in self.task_id

    @property
    def requires_two_alignments(self) -> bool:
        return "PegMulti" in self.task_id or "PlugMulti" in self.task_id

    @property
    def recommended_alignment_tolerance(self) -> float:
        card_value = float(
            getattr(
                self.card,
                "recommended_alignment_tolerance_m",
                self.card.extra.get(
                    "recommended_alignment_tolerance_m",
                    self.card.extra.get("alignment_tolerance_m", 0.01),
                ),
            )
        )
        base = max(card_value, float(self.card.ik_accuracy_m))
        if self.needs_empirical_margin:
            return round(base + 0.01, 3)
        return base

    @property
    def safe_insertion_speed(self) -> float:
        base = float(
            getattr(
                self.card,
                "insertion_speed_limit_mps",
                self.card.extra.get("insertion_speed_limit_mps", 0.015),
            )
        )
        if self.needs_empirical_margin:
            return round(base * 0.75, 4)
        return base

    @property
    def tool_workspace_required(self) -> float:
        return float(self.card.extra.get("tool_workspace_required_m", 0.42))

    def grasp(self, obj: Any) -> bool:
        self.last_failure = None
        self.last_aligned = False
        self.last_hooked = False
        self.align_count = 0
        self._log("grasp", {"obj": _entity_label(obj)}, True, True)
        return True

    def align_to_target(self, obj: Any, target: Any, tolerance: float) -> bool:
        required = self.recommended_alignment_tolerance
        ok = float(tolerance) >= required
        self.last_aligned = ok
        message = ""
        if ok:
            self.align_count += 1
        if not ok:
            message = (
                f"alignment failure: tolerance {float(tolerance):.3f} is tighter "
                f"than {self.profile.name} can reliably achieve ({required:.3f})"
            )
            self.last_failure = message
        self._log(
            "align_to_target",
            {
                "obj": _entity_label(obj),
                "target": _entity_label(target),
                "tolerance": round(float(tolerance), 6),
            },
            ok,
            ok,
            message=message,
            failure_type="alignment failure" if not ok else "",
        )
        return ok

    def insert(self, obj: Any, target: Any, speed: float) -> bool:
        limit = self.safe_insertion_speed
        if not self.last_aligned:
            self.last_failure = "alignment failure: insert called before successful alignment"
            self._log(
                "insert",
                {"obj": _entity_label(obj), "target": _entity_label(target), "speed": round(float(speed), 6)},
                False,
                False,
                message=self.last_failure,
                failure_type="alignment failure",
            )
            return False
        if self.requires_two_alignments and self.align_count < 2:
            self.last_failure = (
                f"alignment failure: {self.task_id} requires two successful "
                f"alignment stages before insertion, got {self.align_count}"
            )
            self._log(
                "insert",
                {"obj": _entity_label(obj), "target": _entity_label(target), "speed": round(float(speed), 6)},
                False,
                False,
                message=self.last_failure,
                failure_type="alignment failure",
            )
            return False
        ok = float(speed) <= limit
        message = ""
        if not ok:
            message = (
                f"insertion speed failure: speed {float(speed):.4f} exceeds "
                f"{self.profile.name} limit {limit:.4f}"
            )
            self.last_failure = message
        self._log(
            "insert",
            {"obj": _entity_label(obj), "target": _entity_label(target), "speed": round(float(speed), 6)},
            ok,
            ok,
            message=message,
            failure_type="insertion speed failure" if not ok else "",
        )
        return ok

    def place(self, obj: Any, target: Any) -> bool:
        if "PickCube" not in self.task_id:
            self.last_failure = "api mismatch: place is only implemented for PickCube-v1 in the static runner"
            self._log(
                "place",
                {"obj": _entity_label(obj), "target": _entity_label(target)},
                False,
                False,
                message=self.last_failure,
                failure_type="api mismatch",
            )
            return False
        self.last_failure = None
        self._log("place", {"obj": _entity_label(obj), "target": _entity_label(target)}, True, True)
        return True

    def hook_object(self, tool: Any, obj: Any) -> bool:
        if not self.last_aligned:
            self.last_failure = (
                "tool-use ordering failure: call align_to_target(tool, object, "
                "tolerance=robot.recommended_alignment_tolerance) before hook_object"
            )
            self.last_hooked = False
            self._log(
                "hook_object",
                {"tool": _entity_label(tool), "obj": _entity_label(obj)},
                False,
                False,
                message=self.last_failure,
                failure_type="tool-use ordering failure",
            )
            return False
        if self.card.workspace_radius_m < self.tool_workspace_required:
            self.last_failure = "reachability failure: tool hook pose is outside workspace"
            self.last_hooked = False
            self._log(
                "hook_object",
                {"tool": _entity_label(tool), "obj": _entity_label(obj)},
                False,
                False,
                message=self.last_failure,
                failure_type="reachability failure",
            )
            return False
        self.last_failure = None
        self.last_hooked = True
        self._log("hook_object", {"tool": _entity_label(tool), "obj": _entity_label(obj)}, True, True)
        return True

    def pull_with_tool(self, tool: Any, obj: Any, target: Any) -> bool:
        if not self.last_hooked:
            self.last_failure = "tool-use ordering failure: pull_with_tool called before hook_object"
            self._log(
                "pull_with_tool",
                {"tool": _entity_label(tool), "obj": _entity_label(obj), "target": _entity_label(target)},
                False,
                False,
                message=self.last_failure,
                failure_type="tool-use ordering failure",
            )
            return False
        if self.card.workspace_radius_m < self.tool_workspace_required:
            self.last_failure = "reachability failure: target pull path is outside workspace"
            self._log(
                "pull_with_tool",
                {"tool": _entity_label(tool), "obj": _entity_label(obj), "target": _entity_label(target)},
                False,
                False,
                message=self.last_failure,
                failure_type="reachability failure",
            )
            return False
        self.last_failure = None
        self._log(
            "pull_with_tool",
            {"tool": _entity_label(tool), "obj": _entity_label(obj), "target": _entity_label(target)},
            True,
            True,
        )
        return True

    def execution_log(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def _log(
        self,
        api: str,
        args: Dict[str, Any],
        result: Any,
        ok: bool,
        *,
        message: str = "",
        failure_type: str = "",
    ) -> None:
        self.events.append(
            ExecutionEvent(
                step=len(self.events) + 1,
                api=api,
                args=args,
                result=result,
                ok=ok,
                message=message,
                failure_type=failure_type,
            )
        )


def _entity_label(value: Any) -> str:
    if isinstance(value, FakeEntity):
        return value.name
    return str(value)


def build_oracle_code(task: TaskSpec) -> str:
    if "PickCube" in task.task_id:
        return """
cube = scene.get_object("cube")
goal = scene.get_region("goal")

ok = robot.grasp(cube)
if ok:
    ret_val = robot.place(cube, goal)
else:
    ret_val = "failure: grasp"
""".strip()

    if "PullCubeTool" in task.task_id:
        return """
tool = scene.get_object("tool")
cube = scene.get_object("cube")
target = scene.get_region("goal")

ok = robot.grasp(tool)
if ok:
    aligned = robot.align_to_target(
        tool,
        cube,
        tolerance=robot.recommended_alignment_tolerance,
    )
    if aligned:
        hooked = robot.hook_object(tool, cube)
        if hooked:
            ret_val = robot.pull_with_tool(tool, cube, target)
        else:
            ret_val = "failure: tool hook"
    else:
        ret_val = "failure: tool alignment"
else:
    ret_val = "failure: grasp tool"
""".strip()

    if "PegMulti" in task.task_id:
        return """
peg = scene.get_object("peg")
hole = scene.get_object("hole")

ok = robot.grasp(peg)
if ok:
    coarse = robot.align_to_target(
        peg,
        hole,
        tolerance=robot.recommended_alignment_tolerance,
    )
    if coarse:
        fine = robot.align_to_target(
            peg,
            hole,
            tolerance=robot.recommended_alignment_tolerance,
        )
        if fine:
            ret_val = robot.insert(peg, hole, speed=robot.safe_insertion_speed)
        else:
            ret_val = "failure: fine alignment"
    else:
        ret_val = "failure: coarse alignment"
else:
    ret_val = "failure: grasp"
""".strip()

    if "PlugMulti" in task.task_id:
        return """
charger = scene.get_object("charger")
socket = scene.get_object("socket")

ok = robot.grasp(charger)
if ok:
    prealign = robot.align_to_target(
        charger,
        socket,
        tolerance=robot.recommended_alignment_tolerance,
    )
    if prealign:
        seated = robot.align_to_target(
            charger,
            socket,
            tolerance=robot.recommended_alignment_tolerance,
        )
        if seated:
            ret_val = robot.insert(charger, socket, speed=robot.safe_insertion_speed)
        else:
            ret_val = "failure: seating alignment"
    else:
        ret_val = "failure: prealign"
else:
    ret_val = "failure: grasp"
""".strip()

    obj_name = "charger" if "PlugCharger" in task.task_id else "peg"
    target_name = "socket" if "PlugCharger" in task.task_id else "hole"
    return f"""
obj = scene.get_object("{obj_name}")
target = scene.get_object("{target_name}")

ok = robot.grasp(obj)
if ok:
    aligned = robot.align_to_target(
        obj,
        target,
        tolerance=robot.recommended_alignment_tolerance,
    )
    if aligned:
        ret_val = robot.insert(obj, target, speed=robot.safe_insertion_speed)
    else:
        ret_val = "failure: alignment"
else:
    ret_val = "failure: grasp"
""".strip()


def build_static_report(
    *,
    task: TaskSpec,
    target_profile: RobotProfile,
    failed_record: TrialRecord,
) -> FailureReport:
    execution_log = failed_record.info.get("execution_log", [])
    failed_event = _first_failed_event(execution_log)
    message = _sanitize_failure_message(
        str(
            (failed_event or {}).get("message")
            or failed_record.message
            or failed_record.failure_type
        )
    )
    failed_step = _format_failed_step(failed_event)
    if "PickCube" in task.task_id:
        failed_api = str((failed_event or {}).get("api", "unknown skill"))
        return FailureReport(
            task_name=task.task_id,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected={
                "execution_result": "success",
                "grasp_cube": "robot.grasp(cube) returns True",
                "place_cube": "robot.place(cube, goal) returns True after grasp succeeds",
            },
            actual={
                "execution_result": "failure",
                "failure_type": failed_record.failure_type,
                "message": message,
                "failed_skill_call": failed_step,
            },
            diagnosis=[
                f"Execution log failed at {failed_step}.",
                message,
                (
                    "The failure happened inside a real high-level skill wrapper, "
                    f"not in object lookup or distance math. The failing API was {failed_api}."
                ),
            ],
            suggestions=[
                "Keep the grasp guard: only call place after robot.grasp(cube) returns True.",
                "If grasp fails, set ret_val to a clear failure string instead of pretending the task succeeded.",
                "Do not invent object pose APIs or direct distance calculations; use only the allowed high-level skill API.",
                "For persistent grasp failures, tune the target robot skill wrapper or control mode rather than only rewriting LMP code.",
            ],
        )

    if "PullCubeTool" in task.task_id:
        return FailureReport(
            task_name=task.task_id,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected={
                "execution_result": "success",
                "tool_aligned_before_hook": True,
                "tool_workspace": "within target robot Capability Card limits",
            },
            actual={
                "execution_result": "failure",
                "failure_type": failed_record.failure_type,
                "message": message,
                "failed_skill_call": failed_step,
            },
            diagnosis=[
                f"Execution log failed at {failed_step}.",
                message,
            ],
            suggestions=[
                "Call align_to_target(tool, cube, tolerance=...) before hook_object.",
                "Choose tolerance using max(target ik_accuracy_m, target recommended_alignment_tolerance_m) from the target Capability Card.",
                "Only call pull_with_tool after hook_object returns True.",
                "Use only the allowed high-level skill API.",
            ],
        )

    if "PegMulti" in task.task_id or "PlugMulti" in task.task_id:
        source_tolerances = "0.010 and 0.006"
        source_speed = "0.020"
        if "PlugMulti" in task.task_id:
            source_tolerances = "0.012 and 0.010"
            source_speed = "0.014"
        return FailureReport(
            task_name=task.task_id,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected={
                "execution_result": "success",
                "all_alignment_tolerances_m": "max(target ik_accuracy_m, target recommended_alignment_tolerance_m) plus empirical contact margin",
                "all_insertion_speeds_mps": "target speed limit with empirical slowdown",
            },
            actual={
                "execution_result": "failure",
                "failure_type": failed_record.failure_type,
                "message": message,
                "failed_skill_call": failed_step,
                "source_tolerances_m": source_tolerances,
                "source_speed_mps": source_speed,
            },
            diagnosis=[
                f"Execution log failed at {failed_step}.",
                message,
                "This contact-rich task needs a safety margin beyond the nominal Capability Card values.",
                "The source program has two alignment stages; both must use the same empirical tolerance margin.",
                "After alignment is fixed, insertion must also use an empirical speed slowdown.",
            ],
            suggestions=[
                "Update every align_to_target call using max(target ik_accuracy_m, target recommended_alignment_tolerance_m) plus a 0.010 m empirical contact margin.",
                "Update every insert call using 75% of the target robot insertion speed limit, or slower.",
                "Do not fix only the first failing line; this task has multiple coupled failure causes.",
            ],
        )

    return FailureReport(
            task_name=task.task_id,
            instruction=task.instruction,
            robot_name=target_profile.name,
            expected={
                "execution_result": "success",
                "recommended_alignment_tolerance_m": "no tighter than target Capability Card permits",
                "insertion_speed_mps": "no faster than target Capability Card permits",
            },
            actual={
                "execution_result": "failure",
                "failure_type": failed_record.failure_type,
                "message": message,
                "failed_skill_call": failed_step,
            },
        diagnosis=[
            f"Execution log failed at {failed_step}.",
            message,
        ],
        suggestions=[
            "Use align_to_target(..., tolerance=...) with max(target ik_accuracy_m, target recommended_alignment_tolerance_m) from the target Capability Card.",
            "Use insert(..., speed=...) with a value derived from the target Capability Card.",
            "Use only the allowed high-level skill API.",
        ],
    )


def _sanitize_failure_message(message: str) -> str:
    """Remove target-specific answer values from reports while keeping the cause."""

    text = re.sub(
        r"than [\w_]+ can reliably achieve \([0-9.]+\)",
        "than the target robot can reliably achieve",
        message,
    )
    text = re.sub(
        r"exceeds [\w_]+ limit [0-9.]+",
        "exceeds the target robot insertion speed limit",
        text,
    )
    return text


def _first_failed_event(execution_log: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(execution_log, list):
        return None
    for event in execution_log:
        if isinstance(event, dict) and event.get("ok") is False:
            return event
    return None


def _format_failed_step(event: Optional[Dict[str, Any]]) -> str:
    if not event:
        return "unknown skill call"
    step = event.get("step", "?")
    api = event.get("api", "unknown_api")
    args = event.get("args", {})
    if isinstance(args, dict) and args:
        arg_text = ", ".join(f"{key}={value!r}" for key, value in sorted(args.items()))
        return f"step {step}: {api}({arg_text})"
    return f"step {step}: {api}(...)"


def _success_from_ret_val(ret_val: Any) -> bool:
    if ret_val is True:
        return True
    if isinstance(ret_val, str):
        return not ret_val.lower().startswith("failure")
    return bool(ret_val)


def run_static_trial(
    *,
    task_id: str,
    target_robot: str,
    method: str,
    seed: int = 0,
    dry_run: bool = False,
) -> TrialRecord:
    method = norm_method(method)
    if method not in {
        "source-copy",
        "oracle",
        "llm_no_card",
        "llm_card_only",
        "llm_report_only",
        "llm_card_report",
    }:
        raise ValueError(
            "static runner supports source-copy, oracle, llm_no_card, "
            "llm_card_only, llm_report_only, llm_card_report"
        )

    request = MigrationRequest.from_ids(
        task_id=task_id,
        target_robot=target_robot,
        method=method,
    )
    report = None
    report_source_record = None
    if method in {"llm_report_only", "llm_card_report"}:
        failed_record = run_static_trial(
            task_id=task_id,
            target_robot=target_robot,
            method="source-copy",
            seed=seed,
            dry_run=True,
        )
        report_source_record = failed_record
        report = build_static_report(
            task=request.task,
            target_profile=request.target_profile,
            failed_record=failed_record,
        )
        request = MigrationRequest(
            task=request.task,
            source_profile=request.source_profile,
            target_profile=request.target_profile,
            method=method,
            failure_report=report,
        )
    prompt = build_migration_prompt(request)
    llm_info: Dict[str, Any] = {}
    if method == "source-copy":
        code = get_source_copy_code(task_id)
    elif method == "oracle":
        code = build_oracle_code(request.task)
    else:
        result = gen_code(
            prompt=prompt,
            fallback_code=build_oracle_code(request.task),
            dry_run=dry_run,
        )
        code = result.code
        llm_info = {
            "used_llm": result.used_llm,
            "llm_model": result.model,
            "llm_reason": result.reason,
            "llm_raw_text": result.raw_text,
        }

    robot = StaticRobot(request.target_profile, task_id=request.task.task_id)
    globals_dict = {
        "scene": FakeScene(),
        "robot": robot,
    }
    code_ok, message, locals_dict = execute_lmp(code, globals_dict, verbose=False)
    ret_val = locals_dict.get("ret_val")
    success = bool(code_ok and _success_from_ret_val(ret_val))
    failure_message = robot.last_failure or message
    failure_type = classify_failure(
        success=success,
        code_ok=code_ok,
        message=failure_message,
        info={"ret_val": repr(ret_val), "method": method},
    )

    info = {
        "ret_val": repr(ret_val),
        "code_ok": code_ok,
        "robot_last_failure": robot.last_failure,
        "execution_log": robot.execution_log(),
        "static_runner": True,
        "capability_card": request.target_profile.to_prompt_section(),
        **llm_info,
    }
    if report_source_record is not None:
        info["report_source_method"] = report_source_record.method
        info["report_source_failure_type"] = report_source_record.failure_type
        info["report_source_message"] = report_source_record.message
        info["report_source_log"] = report_source_record.info.get("execution_log", [])

    record = TrialRecord(
        task_id=request.task.task_id,
        source_robot=request.source_profile.name,
        target_robot=request.target_profile.name,
        method=method,
        seed=seed,
        generated_code=code,
        success=success,
        failure_type=failure_type,
        message=failure_message,
        prompt=prompt,
        info=info,
    )
    if report is not None:
        record.failure_report = report.to_prompt_section()
    return record


def append_jsonl(path: Path, records: Iterable[TrialRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static migration trials.")
    parser.add_argument("--task", default="PegInsertionSide-v1")
    parser.add_argument("--target", default="so100")
    parser.add_argument(
        "--method",
        choices=(
            "source-copy",
            "oracle",
            "llm_no_card",
            "llm_card_only",
            "llm_report_only",
            "llm_card_report",
        ),
        default="source-copy",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/trials.jsonl")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    record = run_static_trial(
        task_id=args.task,
        target_robot=args.target,
        method=args.method,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    if args.print_prompt:
        print(record.prompt)
        print("\n" + "=" * 80 + "\n")

    print(
        json.dumps(
            {
                "task_id": record.task_id,
                "target_robot": record.target_robot,
                "method": record.method,
                "success": record.success,
                "failure_type": record.failure_type,
                "message": record.message,
                "used_llm": record.info.get("used_llm"),
                "llm_reason": record.info.get("llm_reason"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not args.no_log:
        append_jsonl(Path(args.out), [record])
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
