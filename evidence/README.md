# Reproducible Experiment Evidence

Accepted CEGIS runs are copied to `evidence/runs/<run_name>/`.

Each bundle contains:

- `manifest.json`;
- final adapter snapshot and SHA;
- development and held-out JSONL/Markdown;
- module-generation output;
- executed command log.

Only bundles whose manifest reports real held-out acceptance should be cited as
successful LLM migration evidence. Hand-written oracle results must use a
different provenance label.
