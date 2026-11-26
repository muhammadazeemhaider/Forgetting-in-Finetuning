"""Regenerate the figures used in the README from the CSVs in results/.

Usage (from finetuning_forgetting/):
    python analysis/make_figures.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGURES, LORA, METHODS, RESULTS, load_runs

# Categorical palette (validated for colour-vision deficiency); markers add a second cue.
COLOR = {"Full FT": "#2a78d6", "LoRA r16": "#eb6834", "LoRA r64": "#1baf7a", "LoRA r256": "#eda100"}
MARKER = {"Full FT": "o", "LoRA r16": "s", "LoRA r64": "^", "LoRA r256": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 160,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / f"{name}.png")


def reversal(runs: pd.DataFrame) -> None:
    """Same cell, full FT forgetting (x) vs LoRA forgetting (y). Above the diagonal = LoRA forgets less."""
    panels = [("rr_tuned", "AdamW, tuned learning rates\n7 models x 4 target tasks"),
              ("mm4", "Lion (Biderman et al. recipe)\n5 models, MetaMathQA, epochs 1/2/4")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, (regime, title) in zip(axes, panels):
        r = runs[runs.regime == regime]
        keys = ["model", "target", "epoch"]
        ft = r[r.method == "Full FT"].set_index(keys)["forget"]
        lo_all, ft_all = [], []
        for m in LORA:
            lo = r[r.method == m].set_index(keys)["forget"]
            both = ft.to_frame("ft").join(lo.to_frame("lo"), how="inner")
            ax.scatter(both.ft, both.lo, s=34, color=COLOR[m], marker=MARKER[m], label=m,
                       edgecolor="white", linewidth=0.8, zorder=3)
            lo_all += list(both.lo)
            ft_all += list(both.ft)
        lim = [min(lo_all + ft_all) - 1.5, max(lo_all + ft_all) + 1.5]
        ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.fill_between(lim, lim, [lim[1]] * 2, color="#2a78d6", alpha=0.05, zorder=1)
        ax.text(0.03, 0.96, "LoRA forgets less", transform=ax.transAxes, va="top", color=MUTED)
        ax.text(0.97, 0.04, "full FT forgets less", transform=ax.transAxes, ha="right", color=MUTED)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlabel("Full fine-tuning: change on non-target tasks (pp)")
        ax.set_ylabel("LoRA: change on non-target tasks (pp)")
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
    fig.suptitle("Which method forgets less flips with the optimizer", fontsize=12, color=INK, y=1.02)
    save(fig, "reversal_adamw_vs_lion")


def hellaswag_optimizer(runs: pd.DataFrame) -> None:
    """Llama-2-7B, MetaMathQA, one pass, same LR and batch: only the optimizer (and format) differ."""
    l2 = runs[runs.model == "Llama-2-7B"]
    adamw = l2[(l2.regime == "mm1_l2")].set_index("method")["d_hellaswag"].reindex(METHODS)
    lion = l2[(l2.regime == "mm4") & (l2.epoch == 1)].set_index("method")["d_hellaswag"].reindex(METHODS)
    x = np.arange(len(METHODS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - w / 2 - 0.01, adamw, w, color="#86b6ef", label="AdamW (8-bit)")
    b2 = ax.bar(x + w / 2 + 0.01, lion, w, color="#1c5cab", label="Lion")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v - 0.25, f"{v:.1f}", ha="center", va="top",
                    fontsize=9, color=INK)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(x, METHODS)
    ax.set_ylim(min(lion.min(), adamw.min()) - 1.6, 0.8)
    ax.set_ylabel("HellaSwag change (pp)")
    ax.set_title("Llama-2-7B after one pass over MetaMathQA (same LR, same batch)", fontsize=10.5)
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    save(fig, "hellaswag_adamw_vs_lion")


def displacement(runs: pd.DataFrame) -> None:
    """How far each fine-tune moved the weights vs how much HellaSwag it lost."""
    n = pd.read_csv(RESULTS / "NORMS_JOINED.csv")
    n = n[~n.coll.astype(bool)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in METHODS:
        for regime, filled in [("AdamW", False), ("Lion", True)]:
            s = n[(n.method == m) & n.regime.str.contains(regime)]
            ax.scatter(s.rel_fro, s.hs, s=36, marker=MARKER[m], zorder=3,
                       facecolor=COLOR[m] if filled else "white", edgecolor=COLOR[m], linewidth=1.3,
                       label=f"{m}, {regime}")
    ax.set_xscale("log")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("Relative weight displacement ||ΔW|| / ||W₀|| (log scale)")
    ax.set_ylabel("HellaSwag change (pp)")
    ax.set_title("Within each method, moving the weights further costs more HellaSwag\n"
                 "(hollow = AdamW, filled = Lion)", fontsize=10.5)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    save(fig, "displacement_vs_hellaswag")


def learning_vs_forgetting(runs: pd.DataFrame) -> None:
    """Every checkpoint: learning on the target vs mean change on non-targets."""
    panels = [("rr_tuned", "AdamW round robin (7 models x 4 targets)"),
              ("mm4", "Lion, MetaMathQA (5 models x 3 epochs)")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (regime, title) in zip(axes, panels):
        r = runs[runs.regime == regime]
        for m in METHODS:
            s = r[r.method == m]
            ax.scatter(s.learn, s.forget, s=30, color=COLOR[m], marker=MARKER[m], label=m,
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Learning: target-task gain (pp)")
    axes[0].set_ylabel("Forgetting: mean non-target change (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    save(fig, "learning_vs_forgetting")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    reversal(runs)
    hellaswag_optimizer(runs)
    displacement(runs)
    learning_vs_forgetting(runs)


if __name__ == "__main__":
    main()
