"""Select an adapter strategy from an existing multi-seed JSONL result.

This script does not run ManiSkill. It reads the JSONL produced by
`scripts/pullcube_multiseed_eval.py`, rebuilds the multi-seed summary, and
prints the automatic generalization strategy selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pullcube_multiseed_eval import build_summary, summary_to_markdown


def read_multiseed_jsonl(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata: Dict[str, Any] = {}
    trials: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_type = record.pop("type", "")
        if record_type == "metadata":
            metadata = record
        elif record_type == "trial":
            trials.append(record)
    if not metadata:
        raise ValueError(f"No metadata record found in {path}.")
    return metadata, trials


def run(args: argparse.Namespace) -> Dict[str, Any]:
    metadata, trials = read_multiseed_jsonl(Path(args.input))
    summary = build_summary(
        metadata,
        trials,
        success_threshold=args.success_threshold,
        min_trials_for_accept=args.min_trials_for_accept,
    )
    output = {
        "summary": summary,
        "generalization_strategy": summary.get("generalization_strategy") or {},
    }
    if args.markdown:
        Path(args.markdown).write_text(summary_to_markdown(summary), encoding="utf-8")
        output["markdown"] = args.markdown
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a multiseed JSONL file.")
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--min-trials-for-accept", type=int, default=5)
    parser.add_argument("--markdown", default="", help="Optional path to write a refreshed Markdown report.")
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
