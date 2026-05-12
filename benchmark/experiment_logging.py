"""
Experiment logging for benchmark runs.

Each run gets a timestamped directory with:
- metadata.json
- trials/*.json
- prompts/*.txt
- generated_code/*.py
- raw_responses/*.txt
- summary.csv
"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


SUMMARY_FIELDS = [
    "run_id",
    "trial_id",
    "mode",
    "canonical_mode",
    "robot",
    "task",
    "task_family",
    "scene_variant",
    "scene_seed",
    "trial_index",
    "success",
    "info",
    "attempts",
    "llm_model",
    "llm_temperature",
    "llm_cache_enabled",
    "llm_cache_hits",
    "use_capability_card",
    "include_few_shot",
    "use_failure_report",
    "final_reason",
    "failure_type",
    "failure_subtype",
    "exec_error",
    "check_failure",
    "action_failure",
    "ret_val",
    "error_excerpt",
    "used_mobile_navigate_to",
    "used_mobile_is_reachable",
    "used_dual_arm_api",
    "used_dual_left_arm",
    "used_dual_right_arm",
    "used_dual_choose_arm",
    "used_dual_hold",
    "used_dual_coordinated_lift",
    "used_dual_coordinated_place",
    "used_pick_and_place",
    "used_pick",
    "used_place",
    "used_move_ee_to",
    "used_low_release_height",
    "checked_return_value",
    "used_numpy",
    "used_scene_get_names",
    "used_scene_get_position",
    "used_loop",
    "used_conditional",
    "used_refusal_ret_val",
    "lines_of_code",
]


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_slug(value: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return slug[:max_len] or "item"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def snapshot_scene(scene) -> Dict[str, Any]:
    objects = {}
    for name in scene.get_object_names():
        objects[name] = {
            "position": scene.get_object_position(name).round(6).tolist(),
        }
    return {
        "table_position": np.asarray(scene.table_position).round(6).tolist(),
        "table_top_z": float(getattr(scene, "table_top_z", 0.0)),
        "scene_variant": getattr(scene, "scene_variant", ""),
        "scene_seed": getattr(scene, "scene_seed", None),
        "objects": objects,
    }


def analyze_code(code: str) -> Dict[str, Any]:
    lowered = code.lower()
    lines = [
        line for line in code.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return {
        "used_mobile_navigate_to": "mobile.navigate_to" in code or "robot.navigate_to" in code,
        "used_mobile_is_reachable": "mobile.is_reachable" in code or "robot.is_reachable" in code,
        "used_dual_arm_api": (
            "pick_with_arm" in code
            or "place_with_arm" in code
            or "pick_and_place_with_arm" in code
            or "choose_arm_for" in code
            or "is_reachable_by" in code
            or "lift_two_objects" in code
            or "pick_two_objects" in code
            or "place_two_objects" in code
            or "pick_and_place_two_objects" in code
            or "robot.left" in code
            or "robot.right" in code
        ),
        "used_dual_left_arm": "robot.left" in code or "left." in code or "'left'" in code or '"left"' in code,
        "used_dual_right_arm": "robot.right" in code or "right." in code or "'right'" in code or '"right"' in code,
        "used_dual_choose_arm": "choose_arm_for" in code or "is_reachable_by" in code,
        "used_dual_hold": "hold_with_arm" in code or "release_arm" in code,
        "used_dual_coordinated_lift": "lift_two_objects" in code or "pick_two_objects" in code,
        "used_dual_coordinated_place": "place_two_objects" in code or "pick_and_place_two_objects" in code,
        "used_pick_and_place": ".pick_and_place" in code,
        "used_pick": ".pick(" in code,
        "used_place": ".place(" in code,
        "used_move_ee_to": ".move_ee_to" in code,
        "used_low_release_height": (
            "place_release_height" in code
            or "pre_release_height" in code
            or "0.005" in code
        ),
        "checked_return_value": (
            "if " in code
            or "success" in lowered
            or "ret_val" in code
            or "_ok" in code
        ),
        "used_numpy": "np." in code,
        "used_scene_get_names": "scene.get_object_names" in code,
        "used_scene_get_position": "scene.get_object_position" in code,
        "used_loop": "for " in code or "while " in code,
        "used_conditional": "if " in code,
        "used_refusal_ret_val": "refuse" in lowered,
        "lines_of_code": len(lines),
    }


class BenchmarkLogger:
    def __init__(
        self,
        root: str = "results/runs",
        run_id: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.run_id = run_id or make_run_id()
        self.run_dir = Path(root) / self.run_id
        self.summary_rows = []

        if self.enabled:
            for dirname in ("trials", "prompts", "generated_code", "raw_responses"):
                (self.run_dir / dirname).mkdir(parents=True, exist_ok=True)

    def path_for_display(self) -> str:
        return str(self.run_dir)

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(metadata)
        payload["run_id"] = self.run_id
        payload["created_at"] = datetime.now().isoformat(timespec="seconds")
        self._write_json(self.run_dir / "metadata.json", payload)

    def save_attempt_artifacts(
        self,
        trial_id: str,
        attempt: int,
        prompt: str,
        raw_response: str,
        code: str,
    ) -> Dict[str, str]:
        if not self.enabled:
            return {}

        stem = f"{safe_slug(trial_id)}_attempt_{attempt:02d}"
        prompt_path = self.run_dir / "prompts" / f"{stem}.txt"
        raw_path = self.run_dir / "raw_responses" / f"{stem}.txt"
        code_path = self.run_dir / "generated_code" / f"{stem}.py"

        prompt_path.write_text(prompt, encoding="utf-8")
        raw_path.write_text(raw_response, encoding="utf-8")
        code_path.write_text(code + "\n", encoding="utf-8")

        return {
            "prompt_path": str(prompt_path),
            "raw_response_path": str(raw_path),
            "code_path": str(code_path),
        }

    def write_trial(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        trial_id = safe_slug(record.get("trial_id", "trial"))
        self._write_json(self.run_dir / "trials" / f"{trial_id}.json", record)

    def trial_path(self, trial_id: str) -> Path:
        return self.run_dir / "trials" / f"{safe_slug(trial_id)}.json"

    def read_trial(self, trial_id: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self.trial_path(trial_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def add_summary_row(self, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        normalized = {field: jsonable(row.get(field, "")) for field in SUMMARY_FIELDS}
        self.summary_rows.append(normalized)

    def write_summary(self) -> None:
        if not self.enabled:
            return
        summary_path = self.run_dir / "summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(self.summary_rows)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(
            json.dumps(jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def choose_failure_type(record: Dict[str, Any]) -> str:
    if record.get("success"):
        return ""
    if record.get("llm_error"):
        return "llm_error"
    if record.get("exec_error"):
        return "exec_error"
    if record.get("action_failure"):
        return "action_failure"
    if record.get("check_failure"):
        return "check_failure"
    if record.get("ret_val_failed"):
        return "ret_val_failure"
    return "unknown_failure"


def choose_failure_subtype(record: Dict[str, Any]) -> str:
    """Return a paper-friendly, fine-grained failure label.

    The top-level failure_type says where the failure surfaced. This subtype says
    what the failure looked like in the log, which is what we need for failure
    case analysis tables.
    """
    if record.get("success"):
        return ""

    attempts = list(record.get("attempts") or [])
    last_attempt = attempts[-1] if attempts else {}
    exec_message = str(last_attempt.get("exec_message") or record.get("llm_error") or "")
    action_failures = " | ".join(
        str(x) for attempt in attempts for x in attempt.get("action_failures", [])
    )
    action_lower = action_failures.lower()
    code_features = flatten_code_features(attempts)
    task_name = str(record.get("task") or "")

    if record.get("llm_error"):
        return "llm_call_error"

    if record.get("exec_error"):
        msg = exec_message.lower()
        if "safetyerror" in msg or "forbidden pattern" in msg:
            return "safety_violation"
        if "syntaxerror" in msg:
            return "syntax_error"
        if "nameerror" in msg or "attributeerror" in msg:
            return "api_misuse"
        if "keyerror" in msg:
            return "missing_object_or_key"
        if "typeerror" in msg or "valueerror" in msg:
            return "bad_argument_or_value"
        return "runtime_exception"

    if record.get("action_failure"):
        if "navigate_to" in action_lower:
            if "holding" in action_lower:
                return "navigation_while_holding"
            return "navigation_failure"
        if (
            record.get("robot") == "mobile"
            and not code_features.get("used_mobile_navigate_to")
            and ("move_ee_to" in action_lower or "unreachable" in action_lower)
        ):
            return "missing_mobile_navigation"
        if (
            record.get("robot") in {"dual_arm", "mobile_dual_arm", "dual_franka"}
            and not code_features.get("used_dual_arm_api")
            and ("outside arm workspace" in action_lower or "unreachable" in action_lower)
        ):
            return "missing_dual_arm_selection"
        if "failed to grasp" in action_lower or "no object within" in action_lower:
            return "grasp_failure"
        if "pick:" in action_lower:
            return "pick_failure"
        if "place:" in action_lower:
            return "place_failure"
        if "move_ee_to" in action_lower or "unreachable" in action_lower:
            return "ik_or_workspace_failure"
        return "robot_api_returned_false"

    if record.get("ret_val_failed"):
        actual = record.get("actual") or {}
        if actual.get("ret_val") in (None, "", "None"):
            return "missing_ret_val"
        if "refuse" in task_name:
            return "incorrect_refusal_decision"
        return "wrong_ret_val"

    if record.get("check_failure"):
        if task_name.startswith("arrange_") or task_name in {"mirror_layout", "sort_left_to_right"}:
            return "geometric_layout_mismatch"
        if task_name in {"wide_blue_to_tray", "collect_red_and_blue_to_tray"}:
            return "mobility_goal_mismatch"
        if task_name in {
            "hold_red_while_place_green",
            "lift_red_and_green_together",
            "lift_red_green_together_to_tray",
        }:
            return "bimanual_goal_mismatch"
        if task_name == "stack_two":
            return "stack_instability_or_offset"
        if task_name in {"pick_red_to_tray", "move_green_right"}:
            return "goal_state_mismatch"
        return "task_checker_mismatch"

    return "unknown_failure"


def extract_error_excerpt(record: Dict[str, Any], max_chars: int = 220) -> str:
    if record.get("llm_error"):
        text = str(record["llm_error"])
    else:
        attempts = list(record.get("attempts") or [])
        last_attempt = attempts[-1] if attempts else {}
        parts = []
        if last_attempt.get("exec_message"):
            parts.append(str(last_attempt.get("exec_message")))
        for failure in last_attempt.get("action_failures", []) or []:
            parts.append(str(failure))
        text = " | ".join(parts)
    text = " ".join(text.split())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def flatten_code_features(attempts: Iterable[Dict[str, Any]]) -> Dict[str, bool]:
    merged = {
        "used_mobile_navigate_to": False,
        "used_mobile_is_reachable": False,
        "used_dual_arm_api": False,
        "used_dual_left_arm": False,
        "used_dual_right_arm": False,
        "used_dual_choose_arm": False,
        "used_dual_hold": False,
        "used_dual_coordinated_lift": False,
        "used_dual_coordinated_place": False,
        "used_pick_and_place": False,
        "used_pick": False,
        "used_place": False,
        "used_move_ee_to": False,
        "used_low_release_height": False,
        "checked_return_value": False,
        "used_numpy": False,
        "used_scene_get_names": False,
        "used_scene_get_position": False,
        "used_loop": False,
        "used_conditional": False,
        "used_refusal_ret_val": False,
        "lines_of_code": 0,
    }
    for attempt in attempts:
        for k, v in attempt.get("code_features", {}).items():
            if k in merged:
                if k == "lines_of_code":
                    merged[k] = max(int(merged[k]), int(v or 0))
                else:
                    merged[k] = merged[k] or bool(v)
    return merged


def summary_row_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    attempts = list(record.get("attempts") or [])
    features = flatten_code_features(attempts)
    last_attempt = attempts[-1] if attempts else {}
    cache_hits = sum(1 for attempt in attempts if attempt.get("llm_cache_hit"))
    row = {
        "run_id": record.get("run_id", ""),
        "trial_id": record.get("trial_id", ""),
        "mode": record.get("mode", ""),
        "canonical_mode": record.get("canonical_mode", ""),
        "robot": record.get("robot", ""),
        "task": record.get("task", ""),
        "task_family": record.get("task_family", ""),
        "scene_variant": record.get("scene_variant", ""),
        "scene_seed": record.get("scene_seed", ""),
        "trial_index": record.get("trial_index", ""),
        "success": record.get("success", False),
        "info": record.get("info", ""),
        "attempts": len(attempts),
        "llm_model": record.get("llm_model", ""),
        "llm_temperature": record.get("llm_temperature", ""),
        "llm_cache_enabled": record.get("llm_cache_enabled", ""),
        "llm_cache_hits": cache_hits,
        "use_capability_card": record.get("use_capability_card", ""),
        "include_few_shot": record.get("include_few_shot", ""),
        "use_failure_report": record.get("use_failure_report", ""),
        "final_reason": record.get("final_reason", ""),
        "failure_type": record.get("failure_type", ""),
        "failure_subtype": record.get("failure_subtype", ""),
        "exec_error": record.get("exec_error", ""),
        "check_failure": record.get("check_failure", ""),
        "action_failure": record.get("action_failure", ""),
        "ret_val": repr(last_attempt.get("ret_val", "")),
        "error_excerpt": record.get("error_excerpt", ""),
        **features,
    }
    return {field: jsonable(row.get(field, "")) for field in SUMMARY_FIELDS}
