# Qualitative Analysis Template

Use this after Stage 8 generates:

```text
results/runs/<run_id>/casebook/
```

## What To Select For The Paper

Pick 3-5 examples from `qualitative_casebook.md`:

- one mobile case where the corrected code adds `navigate_to` or reachability
  handling;
- one precise placement case where the corrected code adds low release height;
- one geometric case where the corrected code adds NumPy/loop structure;
- one refusal case where the program avoids unsafe or impossible motion;
- one persistent failure case to discuss limitations.

## Suggested Narrative

The qualitative examples should support the claim that the method performs
structured code migration rather than merely producing different code strings.
For each case, describe:

1. the embodiment or task condition that made the first program fail;
2. the signal exposed by the Failure Report or Capability Card;
3. the concrete code-level adaptation in the final program;
4. the resulting physical or task-level outcome.

## Minimal Case Format

```text
Case: <case_id>
Robot / task: <robot> / <task>
Failure: <failure_subtype>
Adaptation: <added feature or code behavior>
Outcome: <success/failure and why>
```

## Files

- `qualitative_cases.csv`: compact index of selected cases.
- `qualitative_casebook.md`: readable casebook with code and diffs.
- `qualitative_casebook.tex`: LaTeX snippets for appendix material.

Avoid placing long generated programs in the main paper. Use short snippets or
diffs that reveal the adaptation.
