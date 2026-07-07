"""Tiny Guava-style harness demo for this project.

Default:
    python demo.py

Run one selected simulator tool:
    python demo.py --run
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from maniskill_backend.agent_planner import fallback_agent_actions, plan_agent_actions
from maniskill_backend.autonomous_harness import build_harness_plan


REPO_ROOT = Path(__file__).resolve().parent


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _tool_by_name(observation: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for tool in observation.get("allowed_tools") or []:
        if tool.get("name") == name:
            return tool
    return {}


def _run_command(command: str, *, run_dir: Path, execute: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "command": command,
        "executed": bool(execute),
        "returncode": None,
        "stdout_file": None,
    }
    if not execute:
        result["note"] = "dry run: command was selected but not executed"
        return result

    stdout_path = run_dir / "tool_stdout.txt"
    process = subprocess.run(
        shlex.split(command),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    result["returncode"] = int(process.returncode)
    result["stdout_file"] = str(stdout_path)
    result["stdout_tail"] = "\n".join((process.stdout or "").splitlines()[-40:])
    return result


def _write_readme(run_dir: Path, *, plan: Mapping[str, Any], tool_result: Mapping[str, Any]) -> None:
    action = (plan.get("actions") or [{}])[0]
    lines = [
        "# Simple Harness Demo",
        "",
        "This is a minimal Guava-style loop for this project:",
        "",
        "```text",
        "agent_observation.json -> agent_plan.json -> selected simulator tool -> tool_result.json",
        "```",
        "",
        "## Selected Tool",
        "",
        f"- tool: `{action.get('tool')}`",
        f"- executed: `{tool_result.get('executed')}`",
        f"- returncode: `{tool_result.get('returncode')}`",
        "",
        "## Files",
        "",
        "- `agent_observation.json`: what the LLM agent sees",
        "- `agent_plan.json`: the next tool action selected by the agent",
        "- `selected_tool_command.txt`: the command exposed by the harness",
        "- `tool_result.json`: dry-run or real simulator execution result",
        "",
        "## Next Commands",
        "",
        "Dry-run demo:",
        "",
        "```bash",
        "python demo.py",
        "```",
        "",
        "Run one real simulator tool on the remote GPU machine:",
        "",
        "```bash",
        "python demo.py --run",
        "```",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal harness demo.")
    parser.add_argument("--case", default="case02_pull_cube_panda_to_xarm6")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="results/simple_demo")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--llm", action="store_true", help="Use the configured LLM planner instead of fallback planning.")
    parser.add_argument("--run", action="store_true", help="Execute the selected tool command.")
    args = parser.parse_args()

    run_name = args.run_name or f"demo_{_timestamp()}"
    run_dir = REPO_ROOT / args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = REPO_ROOT / args.output_root / "latest.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(str(run_dir), encoding="utf-8")

    bundle = build_harness_plan(
        case_id=args.case,
        include_existing_probe=False,
        seeds=args.seeds,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
    )
    observation = bundle["agent_observation"]
    plan = (
        plan_agent_actions(observation, max_actions=1, dry_run=False)
        if args.llm
        else {
            **fallback_agent_actions(observation),
            "schema": "agent_action_plan.v1",
            "used_llm": False,
            "llm_model": "",
        }
    )
    action = (plan.get("actions") or [{"tool": "stop", "args": {}}])[0]
    selected_tool = _tool_by_name(observation, str(action.get("tool") or ""))
    command = str(selected_tool.get("command_template") or "")
    tool_result = _run_command(command, run_dir=run_dir, execute=args.run and bool(command))

    _write_json(run_dir / "agent_observation.json", observation)
    _write_json(run_dir / "agent_plan.json", plan)
    _write_json(run_dir / "tool_result.json", tool_result)
    (run_dir / "selected_tool_command.txt").write_text(command + "\n", encoding="utf-8")
    _write_readme(run_dir, plan=plan, tool_result=tool_result)

    print(json.dumps({
        "demo_dir": str(run_dir),
        "selected_tool": action.get("tool"),
        "executed": bool(tool_result.get("executed")),
        "returncode": tool_result.get("returncode"),
        "open": str(run_dir / "README.md"),
    }, indent=2, ensure_ascii=False))
    if args.run and tool_result.get("returncode") not in (0, None):
        sys.exit(int(tool_result["returncode"]))


if __name__ == "__main__":
    main()
