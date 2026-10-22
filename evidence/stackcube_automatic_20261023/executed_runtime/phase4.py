#!/usr/bin/env python3
"""Run the three-method StackCube comparison pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from maniskill_backend.selective_comparison import build_comparison_summary, run_baseline, summarize_repetitions, validate_reference_result
from maniskill_backend.selective_migration import generation_settings, run_selective_migration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=Path("benchmark_tasks/t07_stack_cube/selective_migration_study.json"),
    )
    parser.add_argument(
        "--selective-result",
        type=Path,
        help="Reuse only compatible selective evidence; by default run all three methods afresh.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1, help="Independent generation runs; environment seed remains fixed.")
    args = parser.parse_args()
    if args.repetitions < 1 or args.max_cycles < 1:
        parser.error("repetitions and max-cycles must be positive")
    if args.repetitions > 1 and args.selective_result is not None:
        parser.error("Independent repetitions cannot reuse a selective result.")
    selective_reference = None
    if args.selective_result is not None:
        selective_reference = json.loads(args.selective_result.read_text(encoding="utf-8"))
        study = json.loads(args.study.read_text(encoding="utf-8"))
        validate_reference_result(selective_reference, study, seed=args.seed, max_cycles=args.max_cycles)
    name = args.run_name or datetime.now(timezone.utc).strftime("stackcube_%Y%m%d_%H%M%S")
    root = Path("results/selective_comparison") / name
    if args.repetitions > 1:
        if root.exists():
            parser.error("Use a new run-name to preserve the existing repetition evidence.")
        root.mkdir(parents=True)
        settings = generation_settings()
        manifest = {
            "schema": "selective_comparison_repetition_plan.v1", "environment_seed": args.seed,
            "sampling_seeds": [settings["sampling_seed"] + index for index in range(args.repetitions)],
            "max_generation_calls": args.repetitions * 3 * args.max_cycles,
            "generation_settings": settings,
        }
        (root / "repetition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        comparisons = []
        for index, sampling_seed in enumerate(manifest["sampling_seeds"], start=1):
            repetition_dir = root / f"r{index:02d}"
            repetition_dir.mkdir()
            command = [sys.executable, str(Path(__file__).resolve()), "--study", str(args.study.resolve()),
                       "--run-name", f"{name}/r{index:02d}", "--max-cycles", str(args.max_cycles),
                       "--seed", str(args.seed)]
            print(f"[repetition {index}/{args.repetitions}] sampling_seed={sampling_seed}", flush=True)
            with (repetition_dir / "run.log").open("w", encoding="utf-8") as output:
                process = subprocess.run(command, env={**os.environ, "EM_LLM_SEED": str(sampling_seed)},
                                         stdout=output, stderr=subprocess.STDOUT)
            if process.returncode != 0:
                raise RuntimeError(f"Repetition {index} exited {process.returncode}; inspect {repetition_dir / 'run.log'}")
            comparisons.append(json.loads((repetition_dir / "summary.json").read_text(encoding="utf-8")))
            summarize_repetitions(comparisons, output_path=root / "summary.json")
            print(f"[repetition {index}] completed", flush=True)
        print(json.dumps(summarize_repetitions(comparisons, output_path=root / "summary.json"), indent=2))
        return
    results = {}
    for method_id in ("target_only", "whole_rewrite"):
        print(f"[{method_id}] running", flush=True)
        results[method_id] = run_baseline(
            method_id=method_id,
            study_path=args.study,
            run_dir=root / method_id,
            max_cycles=args.max_cycles,
            seed=args.seed,
        )
        print(
            f"[{method_id}] success={results[method_id]['success']} "
            f"status={results[method_id]['status']}",
            flush=True,
        )
    if selective_reference is not None:
        results["selective"] = selective_reference
    else:
        print("[selective] running", flush=True)
        results["selective"] = run_selective_migration(
            study_path=args.study, run_dir=root / "selective",
            max_cycles=args.max_cycles, seed=args.seed,
        )
    models = sorted({
        str(cycle["model"])
        for result in results.values()
        for cycle in result.get("cycles", [])
        if cycle.get("model")
    })
    protocol = {
        "task": "StackCube-v1",
        "source_robot": "panda",
        "target_robot": "xarm6_robotiq",
        "target_control_mode": "pd_ee_delta_pose",
        "seed": args.seed,
        "max_generation_cycles": args.max_cycles,
        "llm_models_observed": models,
        "generation_settings": generation_settings(),
        "official_success": "env.unwrapped.evaluate()['success']",
        "selective_result_reused": str(args.selective_result) if args.selective_result else None,
        "experimental_unit": "one independent generation run per method",
        "method_information": {
            "target_only": "task definition plus target interface; global rewrite",
            "whole_rewrite": "complete source code plus source-derived stage representation; global rewrite",
            "selective": "source-derived stage representation; fixed stage order and failed-stage-only repair",
        },
    }
    summary = build_comparison_summary(
        method_results=results,
        output_path=root / "summary.json",
        protocol=protocol,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
