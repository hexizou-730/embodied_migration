"""Recompute this pilot's summaries and verify its retained raw evidence offline."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))

from maniskill_backend.selective_comparison import build_comparison_summary, summarize_repetitions
from maniskill_backend.selective_migration import method_hashes


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    group = ROOT / "results/selective_comparison/source_facts_comparison"
    checks = []
    comparisons = []
    prompt_count = 0
    for rep in ("r01", "r02", "r03"):
        directory = group / rep
        results = {method: read(directory / method / "summary.json")
                   for method in ("target_only", "whole_rewrite", "selective")}
        comparisons.append(build_comparison_summary(
            method_results=results, output_path=directory / "verified_summary.json",
            protocol=read(directory / "summary.json")["protocol"],
        ))
        for method, result in results.items():
            checks.append({"name": f"{rep}/{method}/input_filter",
                           "pass": result.get("source_metadata_filter") == "extracted_source_facts_only"})
            verify_run(directory / method, result, checks)
            for prompt in (directory / method).glob("cycle_*/user_prompt.txt"):
                prompt_count += 1
                text = prompt.read_text(encoding="utf-8")
                checks.append({"name": f"{rep}/{method}/{prompt.parent.name}/no_human_labels",
                               "pass": not any(tag in text for tag in
                                   ("manual_evaluation_only", "expected_stage_order", "expected_entities"))})
    summarize_repetitions(comparisons, output_path=group / "verified_summary.json")
    for run in sorted((ROOT / "results/selective_migration").glob("method_patch_r*")):
        verify_run(run, read(run / "summary.json"), checks)
    smoke = ROOT / "results/fixed_adapter_checks"
    smoke_summary = read(smoke / "summary.json")
    candidate = ROOT / smoke_summary["candidate"]
    checks.append({"name": "fixed/candidate_sha256", "pass":
        hashlib.sha256(candidate.read_bytes()).hexdigest() == smoke_summary["adapter_sha256"]})
    checks.append({"name": "fixed/no_repair_feedback", "pass":
        smoke_summary["llm_calls"] == 0 and smoke_summary["used_for_repair_feedback"] is False})
    for path in sorted(smoke.glob("seed_*.json")):
        trial = read(path)
        checks.append({"name": f"fixed/{path.stem}", "pass": bool(
            trial.get("success") and trial.get("official_success") and trial.get("action_steps", 0) > 0)})
    snapshot = ROOT / "executed_runtime_source_facts"
    hashes = {str(path.relative_to(snapshot)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in snapshot.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    payload = {
        "schema": "stackcube_automatic_evidence_verification.v2",
        "primary_comparison": str(group.relative_to(ROOT)),
        "expected_primary_runs": 9, "primary_prompt_count": prompt_count,
        "checks": checks, "all_checks_pass": all(row["pass"] for row in checks),
        "executed_runtime_snapshot_sha256": hashes,
        "ast_hash_note": "Both compared files are parsed by this same interpreter; remote AST hashes can differ across Python versions.",
        "raw_execution_summaries_retained": True,
        "exploratory_comparison": "corrected_comparison retained but excluded from primary results after input audit",
    }
    (ROOT / "verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": len(checks), "all_checks_pass": payload["all_checks_pass"],
                      "primary_prompt_count": prompt_count}))
    if not payload["all_checks_pass"]:
        raise SystemExit(1)


def verify_run(directory, result, checks):
    previous = None
    if result.get("resume_replay"):
        previous = method_hashes((REPO / result["resume_replay"]["adapter"]).read_text(encoding="utf-8"))
    for cycle in result["cycles"]:
        cycle_dir = directory / f"cycle_{cycle['cycle']:02d}"
        if cycle.get("success"):
            trial = read(cycle_dir / "trial_result.json")
            checks.append({"name": f"{directory.name}/{cycle_dir.name}/official_success",
                           "pass": bool(cycle.get("used_llm") and trial.get("success")
                                        and trial.get("official_success") and trial.get("action_steps", 0) > 0)})
        if not cycle.get("valid") or not cycle.get("method_hashes"):
            continue
        current = method_hashes((cycle_dir / "target_adapter.py").read_text(encoding="utf-8"))
        if previous is not None and cycle.get("allowed_stage"):
            changed = sorted(name for name in set(previous) | set(current)
                             if previous.get(name) != current.get(name))
            checks.append({"name": f"{directory.name}/{cycle_dir.name}/local_scope",
                           "changed_methods": changed,
                           "pass": changed == [f"_stage_{cycle['allowed_stage']}"]})
        previous = current


if __name__ == "__main__":
    main()
