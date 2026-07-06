"""LLM planner for the bottom-interface autonomous harness.

The planner receives only ``agent_observation``: task facts, constraints,
available tools, and raw simulator results. It returns a small JSON action list.
It should not receive the human report.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence

from maniskill_backend.llm import gen_text


ALLOWED_AGENT_TOOLS = {
    "run_single_seed",
    "run_multi_seed",
    "run_structured_probe",
    "run_llm_repair",
    "inspect_results",
    "stop",
}


def build_agent_planner_prompt(observation: Mapping[str, Any], *, max_actions: int = 2) -> str:
    """Build a machine-facing prompt with no human report or repair hint."""

    payload = dict(observation)
    return (
        "You are an autonomous robotics migration agent.\n"
        "You receive a JSON observation exposing only simulator facts, constraints, and allowed tools.\n"
        "Decide the next bounded tool action(s). Do not explain in prose.\n\n"
        "Hard constraints:\n"
        "- Do not modify simulator, controller, success signal, or high-level program.\n"
        "- Do not fake success or edit object state directly.\n"
        "- You may decide which generated adapter behavior to keep, remove, or rewrite.\n"
        "- Use only tools listed in observation.allowed_tools.\n"
        f"- Return at most {max_actions} actions.\n\n"
        "Return only JSON in this exact shape:\n"
        "{\n"
        '  "actions": [\n'
        '    {"tool": "run_multi_seed", "args": {}},\n'
        '    {"tool": "run_structured_probe", "args": {"seed": 0}}\n'
        "  ],\n"
        '  "stop_reason": ""\n'
        "}\n\n"
        "Observation JSON:\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```"
    )


def fallback_agent_actions(observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback used for dry-runs or missing API keys."""

    latest = observation.get("latest_results") or {}
    multiseed = latest.get("multiseed") or {}
    probe = latest.get("structured_probe") or {}
    if not multiseed:
        return {"actions": [{"tool": "run_multi_seed", "args": {}}], "stop_reason": ""}
    if multiseed.get("success_rate") is not None and multiseed.get("success_rate", 0.0) >= 0.8:
        return {"actions": [{"tool": "stop", "args": {"reason": "success_threshold_met"}}], "stop_reason": "success_threshold_met"}
    if not probe:
        failure_rows = list(multiseed.get("failure_rows") or [])
        seed = failure_rows[0].get("seed") if failure_rows else 0
        return {"actions": [{"tool": "run_structured_probe", "args": {"seed": seed}}], "stop_reason": ""}
    return {"actions": [{"tool": "run_llm_repair", "args": {}}], "stop_reason": ""}


def plan_agent_actions(
    observation: Mapping[str, Any],
    *,
    max_actions: int = 2,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Ask the LLM for the next actions and validate the returned JSON."""

    fallback = fallback_agent_actions(observation)
    generated = gen_text(
        prompt=build_agent_planner_prompt(observation, max_actions=max_actions),
        system=(
            "You are a tool-planning agent. Return only valid JSON. "
            "Do not include Markdown or prose."
        ),
        fallback_text=json.dumps(fallback, ensure_ascii=False),
        dry_run=dry_run,
    )
    raw = generated.text or generated.raw_text or ""
    planner_error = ""
    try:
        parsed = _parse_plan_json(raw)
    except Exception as exc:
        parsed = fallback
        planner_error = repr(exc)
    validated = validate_agent_plan(parsed, observation, max_actions=max_actions)
    validated["used_llm"] = bool(generated.used_llm)
    validated["llm_model"] = generated.model
    validated["llm_reason"] = generated.reason
    if planner_error:
        validated["planner_error"] = planner_error
    return validated


def validate_agent_plan(
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    max_actions: int = 2,
) -> Dict[str, Any]:
    """Validate a planner result against the observation tool surface."""

    allowed = {tool.get("name") for tool in observation.get("allowed_tools") or []}
    allowed.add("stop")
    actions = list(plan.get("actions") or [])
    if len(actions) > max_actions:
        actions = actions[:max_actions]
    normalized: List[Dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("tool") or "").strip()
        if name not in ALLOWED_AGENT_TOOLS or name not in allowed:
            continue
        args = item.get("args") or {}
        if not isinstance(args, Mapping):
            args = {}
        normalized.append({"tool": name, "args": dict(args)})
    if not normalized:
        normalized = fallback_agent_actions(observation)["actions"]
    return {
        "schema": "agent_action_plan.v1",
        "actions": normalized,
        "stop_reason": str(plan.get("stop_reason") or ""),
    }


def _parse_plan_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
        if match:
            stripped = match.group(1).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Agent planner must return a JSON object.")
    return parsed
