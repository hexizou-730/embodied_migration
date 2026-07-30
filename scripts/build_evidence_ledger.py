"""Regenerate the tracked adapter provenance and experiment evidence ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maniskill_backend.evidence import write_evidence_ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="docs/EVIDENCE_LEDGER.json")
    parser.add_argument("--markdown", default="docs/EVIDENCE_LEDGER_CN.md")
    args = parser.parse_args()
    payload = write_evidence_ledger(
        REPO_ROOT / args.json,
        REPO_ROOT / args.markdown,
        repo_root=REPO_ROOT,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
