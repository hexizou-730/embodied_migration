"""Build Figure 2: per-pair repair differences from the archived ablation analysis.

Reads the pre-registered n=30 analysis JSON and plots module-minus-method
differences per sampling seed for tokens and executed actions. Zero cost,
fully offline; rerun after any analysis change.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = (
    ROOT / "evidence" / "granularity_ablation_20261121" / "results"
    / "granularity_ablation_n30" / "granularity_analysis.json"
)
OUT = Path(__file__).resolve().with_name("fig2_paired_differences.png")

payload = json.loads(ANALYSIS.read_text(encoding="utf-8"))
manifest = json.loads((ANALYSIS.parent / "repetition_manifest.json").read_text(encoding="utf-8"))
paired = payload["paired"]["per_repetition"]
seeds = manifest["sampling_seeds"]
token_diffs = [row["total_tokens"][1] - row["total_tokens"][0] for row in paired]
action_diffs = [row["action_steps"][1] - row["action_steps"][0] for row in paired]
failed = [
    not (row["method_patch_success"] and row["module_rewrite_success"]) for row in paired
]

fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)

token_ci = payload["paired"]["token_difference_bootstrap_95"]
token_mean = payload["paired"]["mean_token_difference_module_minus_method"]
ax = axes[0]
ax.axhline(0, color="0.4", linewidth=1.0, linestyle="--", zorder=1)
ax.axhspan(token_ci[0], token_ci[1], color="tab:blue", alpha=0.12, zorder=0,
           label=f"bootstrap 95% [{token_ci[0]:.0f}, {token_ci[1]:.0f}]")
ax.axhline(token_mean, color="tab:blue", linewidth=1.2, zorder=2,
           label=f"mean {token_mean:.0f}")
for seed, diff, bad in zip(seeds, token_diffs, failed):
    ax.scatter(seed, diff, s=52 if bad else 30,
               color="crimson" if bad else "tab:blue",
               marker="x" if bad else "o", zorder=3)
ax.set_ylabel("token difference\n(module − method)")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_title("Per-pair repair cost differences, pre-registered n=30 (seeds 10–39)",
             fontsize=11)

action_ci = payload["paired"]["action_difference_bootstrap_95"]
action_mean = payload["paired"]["mean_action_difference_module_minus_method"]
ax = axes[1]
ax.axhline(0, color="0.4", linewidth=1.0, linestyle="--", zorder=1)
ax.axhspan(action_ci[0], action_ci[1], color="tab:orange", alpha=0.12, zorder=0,
           label=f"bootstrap 95% [{action_ci[0]:.1f}, {action_ci[1]:.1f}]")
ax.axhline(action_mean, color="tab:orange", linewidth=1.2, zorder=2,
           label=f"mean {action_mean:.1f}")
for seed, diff, bad in zip(seeds, action_diffs, failed):
    ax.scatter(seed, diff, s=52 if bad else 30,
               color="crimson" if bad else "tab:orange",
               marker="x" if bad else "o", zorder=3)
ax.set_ylabel("action difference\n(module − method)")
ax.set_xlabel("LLM sampling seed")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

red = plt.Line2D([], [], color="crimson", marker="x", linestyle="",
                 markersize=8, label="pair with an arm failure")
neutral = plt.Line2D([], [], color="0.3", marker="o", linestyle="",
                    markersize=6, label="both arms succeeded")
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [red, neutral], fontsize=8, framealpha=0.9,
              loc="upper right")

fig.tight_layout()
fig.savefig(OUT, dpi=300)
print(f"saved {OUT}")
print(f"pairs={len(paired)} failed_pairs={sum(failed)}")
