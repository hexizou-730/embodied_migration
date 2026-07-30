"""Reproducibility and adapter-provenance evidence ledger."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from maniskill_backend.cases import FullMigrationCase, iter_full_migration_cases


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_docstring(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    return ast.get_docstring(tree) or ""


def detect_adapter_provenance(path: Path, *, seed_path: Path | None = None) -> str:
    docstring = _module_docstring(path).lower()
    if "hand-written oracle" in docstring or "hand written oracle" in docstring:
        return "hand_written_oracle"
    if seed_path and path.exists() and seed_path.exists() and _sha256(path) == _sha256(seed_path):
        return "neutral_seed"
    if "neutral" in docstring and "seed adapter" in docstring:
        return "neutral_seed"
    if "generated" in docstring or "llm" in docstring:
        return "generated_claim_unverified"
    return "unknown"


def _tracked_files(repo_root: Path) -> List[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _tracked_result_artifacts(case: FullMigrationCase, tracked: Iterable[str]) -> List[str]:
    case_id = case.case_id
    return sorted(
        path
        for path in tracked
        if path.startswith("results/")
        and case_id in path
        and path.endswith((".json", ".jsonl", ".md", ".py"))
    )


def _tracked_evidence_bundles(
    case: FullMigrationCase,
    tracked: Iterable[str],
    repo_root: Path,
) -> List[Dict[str, Any]]:
    tracked_set = set(tracked)
    bundles = []
    for relative_path in tracked_set:
        if not relative_path.startswith("evidence/runs/") or not relative_path.endswith("/manifest.json"):
            continue
        path = repo_root / relative_path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("case_id") != case.case_id:
            continue
        artifacts = payload.get("artifacts") or {}
        required = (
            "adapter",
            "module_generation_jsonl",
            "development_jsonl",
            "held_out_jsonl",
        )
        complete = all(
            artifacts.get(key) in tracked_set
            for key in required
        )
        adapter_snapshot = repo_root / str(artifacts.get("adapter") or "")
        hash_matches = (
            adapter_snapshot.is_file()
            and _sha256(adapter_snapshot) == payload.get("adapter_sha256")
        )
        bundles.append(
            {
                "manifest": relative_path,
                "status": payload.get("status"),
                "success": bool(payload.get("success")) and complete and hash_matches,
                "bundle_complete": complete,
                "adapter_hash_matches": hash_matches,
                "adapter_sha256": payload.get("adapter_sha256"),
                "adapter_provenance": payload.get("adapter_provenance"),
                "development_success_rate": payload.get("development_success_rate"),
                "held_out_success_rate": payload.get("held_out_success_rate"),
            }
        )
    return bundles


def build_evidence_ledger(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    tracked = _tracked_files(repo_root)
    cases = []
    for case in iter_full_migration_cases():
        adapter_path = repo_root / case.target_adapter_path
        seed_path = repo_root / case.seed_adapter_path if case.seed_adapter_path else None
        artifacts = _tracked_result_artifacts(case, tracked)
        bundles = _tracked_evidence_bundles(case, tracked, repo_root)
        provenance = detect_adapter_provenance(adapter_path, seed_path=seed_path)
        reproducible_success = any(bool(item.get("success")) for item in bundles)
        cases.append(
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "source_robot": case.source_robot,
                "target_robot": case.target_robot,
                "adapter_path": case.target_adapter_path,
                "adapter_sha256": _sha256(adapter_path),
                "seed_adapter_path": case.seed_adapter_path,
                "seed_adapter_sha256": _sha256(seed_path) if seed_path else "",
                "adapter_provenance": provenance,
                "tracked_result_artifacts": artifacts,
                "tracked_evidence_bundles": bundles,
                "reproducible_success_in_tracked_artifacts": reproducible_success,
                "evidence_status": (
                    "tracked_accepted_evidence"
                    if reproducible_success
                    else
                    "oracle_upper_bound_without_tracked_run"
                    if provenance == "hand_written_oracle" and not artifacts
                    else "no_tracked_success_evidence"
                    if not artifacts
                    else "tracked_artifacts_require_manual_result_audit"
                ),
            }
        )
    return {
        "schema": "migration_evidence_ledger.v1",
        "policy": {
            "success_claim_requires": [
                "non-dry-run simulator log",
                "adapter SHA matching the evaluated artifact",
                "seed and simulator configuration",
                "real task success signal",
            ],
            "provenance_labels": [
                "neutral_seed",
                "llm_generated_cegis",
                "hand_written_oracle",
                "human_modified",
                "preexisting_adapter_unverified",
                "unknown",
            ],
            "oracle_is_not_llm_success": True,
        },
        "cases": cases,
    }


def _git_rev(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def archive_cegis_evidence(
    *,
    evidence_root: Path,
    run_name: str,
    case: FullMigrationCase,
    adapter_path: Path,
    cycle_dir: Path,
    command_log: Path,
    summary: Mapping[str, Any],
    adapter_provenance: str,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Copy one accepted run into a compact, Git-trackable evidence bundle."""

    bundle_dir = evidence_root / run_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    sources = {
        "adapter": adapter_path,
        "module_generation_jsonl": cycle_dir / "module_generation.jsonl",
        "module_generation_md": cycle_dir / "module_generation.md",
        "development_jsonl": cycle_dir / "development.jsonl",
        "development_md": cycle_dir / "development.md",
        "held_out_jsonl": cycle_dir / "held_out.jsonl",
        "held_out_md": cycle_dir / "held_out.md",
        "commands_log": command_log,
    }
    for key, source in sources.items():
        if not source.exists():
            continue
        suffix = source.suffix or ".txt"
        target_name = "final_adapter.py" if key == "adapter" else f"{key}{suffix}"
        target = bundle_dir / target_name
        shutil.copy2(source, target)
        copied[key] = str(target.relative_to(repo_root)) if target.is_relative_to(repo_root) else str(target)

    cycles = list(summary.get("cycles") or [])
    accepted_cycle = cycles[-1] if cycles else {}
    development = accepted_cycle.get("development_summary") or {}
    held_out = (summary.get("held_out") or {}).get("summary") or {}
    manifest = {
        "schema": "accepted_migration_evidence.v1",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "source_robot": case.source_robot,
        "target_robot": case.target_robot,
        "status": summary.get("status"),
        "success": bool(summary.get("success")),
        "adapter_sha256": _sha256(adapter_path),
        "adapter_provenance": adapter_provenance,
        "git_rev_at_run": _git_rev(repo_root),
        "development_seeds": (summary.get("config") or {}).get("development_seeds"),
        "held_out_seeds": (summary.get("config") or {}).get("held_out_seeds"),
        "development_success_rate": development.get("success_rate"),
        "held_out_success_rate": held_out.get("success_rate"),
        "held_out_used_for_repair": False,
        "artifacts": copied,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=repr),
        encoding="utf-8",
    )
    return {
        "bundle_dir": str(bundle_dir),
        "manifest": str(manifest_path),
        "artifacts": copied,
    }


def evidence_ledger_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 实验证据账本",
        "",
        "本文件只记录当前 Git 仓库中可追溯的证据，不把口头运行结果、未提交的远程日志或 oracle 当成 LLM 迁移成功。",
        "",
        "## 证据规则",
        "",
        "一次可复现成功必须同时包含：非 dry-run 仿真日志、与日志一致的 adapter SHA、seed/环境配置，以及 ManiSkill 真实 success signal。",
        "",
        "## 当前状态",
        "",
        "| Case | 迁移 | 当前 adapter 来源 | 已跟踪结果 | 可复现成功证据 | 状态 |",
        "|---|---|---|---:|---|---|",
    ]
    for item in payload.get("cases") or []:
        artifacts = item.get("tracked_result_artifacts") or []
        lines.append(
            f"| `{item.get('case_id')}` | {item.get('source_robot')} -> {item.get('target_robot')} "
            f"({item.get('task_id')}) | `{item.get('adapter_provenance')}` | {len(artifacts)} | "
            f"`{item.get('reproducible_success_in_tracked_artifacts')}` | "
            f"`{item.get('evidence_status')}` |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- Fetch 当前 adapter 明确标注为 hand-written oracle，只能作为性能上界。",
            "- xArm6 当前提交文件若与 seed adapter 相同，只代表中性起点，不代表 LLM 成功代码。",
            "- 远程产生的新成功结果必须把运行 summary、adapter snapshot 和 SHA 一起带回仓库，才能升级本账本状态。",
            "",
        ]
    )
    return "\n".join(lines)


def write_evidence_ledger(
    json_path: Path,
    markdown_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    payload = build_evidence_ledger(repo_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown_path.write_text(evidence_ledger_markdown(payload), encoding="utf-8")
    return payload
