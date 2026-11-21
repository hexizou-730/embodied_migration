"""Build Figure 1: the measured-evidence repair pipeline (fixed-grid layout).

Offline, rerunnable schematic of the migration loop described in the paper.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().with_name("fig1_pipeline.png")

BLUE = "#dce9f7"
BLUE_EDGE = "#4a7fb5"
ORANGE = "#fdebd0"
ORANGE_EDGE = "#c07f24"
GREEN = "#d8f0d8"
GREEN_EDGE = "#3f8f4f"
GREY = "#eeeeee"
GREY_EDGE = "#888888"

fig, ax = plt.subplots(figsize=(12.5, 8.0))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 8.6)
ax.axis("off")


def box(x, y, w, h, text, face, edge, fontsize=9, style="round,pad=0.06"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.3,
                           facecolor=face, edgecolor=edge, mutation_scale=1.0)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x, y, w, h)


def arrow(x1, y1, x2, y2, color="#333333", style="-|>", connectionstyle="arc3,rad=0", lw=1.6, ls="-"):
    patch = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                            linewidth=lw, linestyle=ls, connectionstyle=connectionstyle,
                            mutation_scale=16, zorder=3)
    ax.add_patch(patch)


# ---- top band: source analysis (no LLM) ----
box(0.3, 6.9, 3.2, 1.25, "Successful source program\n(Panda, frozen, 10/10 seeds)", BLUE, BLUE_EDGE)
box(4.4, 6.9, 3.2, 1.25, "Deterministic analysis\nAST + dataflow, no LLM", BLUE, BLUE_EDGE)
box(8.5, 6.9, 3.7, 1.25, "Stage program\nstages + per-action attribution\n+ stage-end official states", BLUE, BLUE_EDGE)
arrow(3.5, 7.52, 4.4, 7.52)
arrow(7.6, 7.52, 8.5, 7.52)

# ---- middle band: failure replay and localization ----
box(0.3, 4.6, 3.2, 1.25, "Failed target candidate\n(hash-pinned bytes)", ORANGE, ORANGE_EDGE)
box(4.4, 4.6, 3.2, 1.25, "Replay\nmeasured stage-boundary\nstates + official booleans", ORANGE, ORANGE_EDGE)
box(8.5, 4.6, 3.7, 1.25, "Localization\nfirst runtime failure\n+ source-postcondition reopening", ORANGE, ORANGE_EDGE)
arrow(3.5, 5.22, 4.4, 5.22)
arrow(7.6, 5.22, 8.5, 5.22)
arrow(10.35, 6.9, 10.35, 5.85, color=BLUE_EDGE, ls="--")
ax.text(10.55, 6.35, "stage-end states\nas reference", fontsize=7.5, ha="left", color=BLUE_EDGE)

# ---- bottom band: bounded repair loop ----
box(0.3, 2.3, 3.2, 1.25, "LLM rewrites\nonly the localized stage", ORANGE, ORANGE_EDGE)
box(4.4, 2.3, 3.2, 1.25, "Structural validation\nsplice + AST hashes;\nno-op candidates rejected", ORANGE, ORANGE_EDGE)
box(8.5, 2.3, 3.7, 1.25, "Simulator execution\nunchanged controller\n+ official evaluator", ORANGE, ORANGE_EDGE)
box(4.4, 0.35, 3.2, 1.05, "Official success", GREEN, GREEN_EDGE)
arrow(3.5, 2.92, 4.4, 2.92)
arrow(7.6, 2.92, 8.5, 2.92)
arrow(10.35, 2.3, 10.35, 1.45, color=GREEN_EDGE)
ax.text(10.55, 1.75, "success", fontsize=8, color=GREEN_EDGE, ha="left")
arrow(10.35, 0.9, 7.6, 0.9, color=GREEN_EDGE)
arrow(1.9, 4.6, 1.9, 3.55, color=ORANGE_EDGE)
ax.text(2.1, 4.0, "repair target", fontsize=7.5, color=ORANGE_EDGE, ha="left")
# feedback loop: official evaluator -> localization (failure path)
arrow(9.2, 3.55, 9.2, 4.6, color=ORANGE_EDGE, connectionstyle="arc3,rad=0.0")
ax.text(9.35, 4.1, "failure feedback\n(bounded cycles)", fontsize=7.5, color=ORANGE_EDGE, ha="left")

# ---- interface probe note ----
box(0.3, 0.35, 3.2, 1.05,
    "Interface facts measured first\n(gripper endpoints, stationary close)",
    GREY, GREY_EDGE, fontsize=8)
ax.text(6.25, 8.35, "Measured-evidence repair pipeline", fontsize=12, ha="center", weight="bold")
ax.text(6.25, -0.25,
        "The LLM proposes code only; acceptance belongs to the unchanged official evaluator. "
        "Analysis is deterministic; localization is measured, never asked of the model.",
        fontsize=8, ha="center", style="italic", color="#444444")

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"saved {OUT}")
