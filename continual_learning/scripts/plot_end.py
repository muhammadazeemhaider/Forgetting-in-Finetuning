#!/usr/bin/env python3
"""Figures for the end-of-sequence repair experiment -> results/figures/end_*.png
  end_aa_bwt.png        AA and BWT per method: mean with 95% bootstrap CI, one faint dot per (order, seed) unit
  end_paired.png        paired difference vs replay and vs delayed (mean, 95% CI, Holm p from report_end)
  end_forgetting.png    forgetting of the 1st / 2nd / 3rd task in the order (final - just learned)
  end_pre_post.png      AA before vs after the end repair, per unit
  end_single.png        end_sum / end_mult: accuracy change per task from each per-task repair alone vs all combined
Usage: python scripts/plot_end.py [--runs runs] [--out results]   (run report_end.py first for the p-values)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import boot, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["seq", "replay", "delayed", "repair", "end_joint", "end_sum", "end_mult"]
NEW = ["end_joint", "end_sum", "end_mult"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"seq": "#b5b4ae", "replay": "#7d7c77", "delayed": "#52514e",            # baselines: neutral greys
       "repair": "#2a78d6", "end_joint": "#eb6834", "end_sum": "#1baf7a", "end_mult": "#6250d6"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK})


def load(runs):
    raw, rows = {}, {}
    for d in sorted((ROOT / runs).glob("order*_s[0-9]")):
        for m in METHODS + ["mtl"]:
            f = d / f"{m}.json"
            if f.exists():
                r = json.loads(f.read_text())
                raw[(d.name, m)] = r
                rows[(d.name, m)] = metrics(r)
    present = [m for m in METHODS if any(k[1] == m for k in rows)]
    units = sorted({u for u, _ in rows})
    units = [u for u in units if all((u, m) in rows for m in present)]
    return raw, rows, present, units


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig_aa_bwt(rows, present, units, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(present) + 1.6), sharey=True)
    y = np.arange(len(present))[::-1]
    for ax, key, lab in ((axes[0], "AA", "AA: final average accuracy (%)"), (axes[1], "BWT", "BWT: forgetting (pp, < 0 = forgot)")):
        for yi, m in zip(y, present):
            v = np.array([rows[(u, m)][key] for u in units])
            ax.scatter(v, yi - 0.24 + np.linspace(-0.06, 0.06, len(v)), s=14, color=COL[m], alpha=0.45, lw=0)
            lo, hi = boot(v)
            ax.plot([lo, hi], [yi, yi], color=COL[m], lw=2, solid_capstyle="round")
            ax.scatter([v.mean()], [yi], s=64, color=COL[m], edgecolor=SURF, lw=2, zorder=3)
            ax.annotate(f"{v.mean():.2f}" if key == "AA" else f"{v.mean():+.2f}", (v.mean(), yi), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK2)
        if key == "AA" and any((u, "mtl") in rows for u in units):
            mt = np.mean([rows[(u, "mtl")]["AA"] for u in units if (u, "mtl") in rows])
            ax.axvline(mt, color=INK2, ls=(0, (4, 3)), lw=1)
            ax.annotate(f"multi-task reference {mt:.2f}", (mt, y.max() + 0.55), ha="right", fontsize=8, color=INK2)
        if key == "BWT":
            ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(lab); grid(ax)
    axes[0].set_yticks(y, present); axes[0].set_ylim(-0.6, len(present) - 0.1)
    fig.suptitle(f"Mean and 95% bootstrap CI over {len(units)} (order, seed) runs; faint dots = single runs",
                 fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / "end_aa_bwt.png", dpi=160); plt.close(fig)


def fig_paired(rows, present, units, out, stats, ours=None, fname="end_paired.png"):
    ours = [m for m in (ours or ["repair"] + NEW) if m in present]
    base = [b for b in ("replay", "delayed") if b in present]
    if not ours or not base:
        return
    fig, axes = plt.subplots(1, len(base), figsize=(5 * len(base), 0.55 * len(ours) + 1.6), sharey=True, squeeze=False)
    y = np.arange(len(ours))[::-1]
    for ax, b in zip(axes[0], base):
        for yi, m in zip(y, ours):
            d = np.array([rows[(u, m)]["AA"] - rows[(u, b)]["AA"] for u in units])
            lo, hi = boot(d)
            ax.scatter(d, yi - 0.24 + np.linspace(-0.06, 0.06, len(d)), s=14, color=COL[m], alpha=0.45, lw=0)
            ax.plot([lo, hi], [yi, yi], color=COL[m], lw=2, solid_capstyle="round")
            ax.scatter([d.mean()], [yi], s=64, color=COL[m], edgecolor=SURF, lw=2, zorder=3)
            t = stats.get(f"AA:{m}-{b}")
            p = (", Holm p<0.001" if t["p_holm"] < 1e-3 else f", Holm p={t['p_holm']:.3f}") if t and "p_holm" in t else ""
            ax.annotate(f"{d.mean():+.2f}  ({int((d > 0).sum())}/{len(d)} runs better{p})", (d.mean(), yi),
                        xytext=(0, 7), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK2,
                        bbox=dict(fc=SURF, ec="none", pad=1))
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"ΔAA vs {b} (pp; > 0 = beats {b})"); grid(ax)
        lim = max(0.5, *(abs(rows[(u, m)]["AA"] - rows[(u, b)]["AA"]) for u in units for m in ours)) * 1.6
        ax.set_xlim(-lim, lim)
    axes[0][0].set_yticks(y, ours); axes[0][0].set_ylim(-0.6, len(ours) - 0.4)
    fig.tight_layout(); fig.savefig(out / fname, dpi=160); plt.close(fig)


def fig_forgetting(raw, present, units, out):
    ms = [m for m in present if m != "seq"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    pos = np.arange(3)
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        vals = []
        for p in range(3):
            v = []
            for u in units:
                r = raw[(u, m)]; o, A = r["order"], r["acc_matrix"]
                v.append(100 * (A[-1]["acc"][o[p]] - A[p]["acc"][o[p]]))
            vals.append(np.mean(v))
        x = pos - 0.4 + w * (i + 0.5)
        ax.bar(x, vals, width=w - 0.02, color=COL[m], label=m, edgecolor=SURF, lw=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(pos, ["1st task in order", "2nd task", "3rd task"])
    ax.set_ylabel("final − just learned (pp)"); grid(ax, "y")
    ax.legend(frameon=False, ncol=len(ms), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(out / "end_forgetting.png", dpi=160); plt.close(fig)


def fig_pre_post(raw, present, units, out):
    ms = [m for m in NEW if m in present]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms), 3.4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ms):
        for u in units:
            r = raw[(u, m)]
            pre = 100 * np.mean([r["pre_repair"]["acc"][t] for t in r["order"]])
            post = 100 * np.mean([r["acc_matrix"][-1]["acc"][t] for t in r["order"]])
            ax.plot([0, 1], [pre, post], color=COL[m], alpha=0.5, lw=1.2, marker="o", ms=4)
        for b, ls in (("replay", (0, (1, 2))), ("delayed", (0, (4, 3)))):
            if b in present:
                ax.axhline(np.mean([metrics(raw[(u, b)])["AA"] for u in units]), color=INK2, ls=ls, lw=1, label=f"{b} mean")
        ax.set_xticks([0, 1], ["before\nend repair", "after"]); ax.set_xlim(-0.3, 1.3)
        ax.set_title(m, fontsize=10, color=INK); grid(ax, "y")
    axes[0][0].set_ylabel("AA (%)"); axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "end_pre_post.png", dpi=160); plt.close(fig)


def fig_single(raw, present, units, out):
    ms = [m for m in ("end_sum", "end_mult") if m in present]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(5.2 * len(ms), 3.6), squeeze=False)
    for ax, m in zip(axes[0], ms):
        M = np.zeros((5, 4))
        for u in units:
            r = raw[(u, m)]; o = r["order"]; pre = r["pre_repair"]["acc"]
            for j, t in enumerate(o):
                one = r["single_repair_acc"][f"end_{t}"]
                M[j] += [100 * (one[s] - pre[s]) for s in o]
            M[4] += [100 * (r["acc_matrix"][-1]["acc"][s] - pre[s]) for s in o]
        M /= len(units)
        lim = max(1.0, np.abs(M).max())
        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(5):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=8.5,
                        color="white" if abs(M[i, j]) > 0.6 * lim else INK)
        ax.set_xticks(range(4), ["task 1", "task 2", "task 3", "task 4"])
        ax.set_yticks(range(5), ["repair for task 1 alone", "repair for task 2 alone", "repair for task 3 alone",
                                 "repair for task 4 alone", "all four combined"])
        ax.axhline(3.5, color=SURF, lw=3)
        ax.set_title(f"{m}: Δ accuracy vs before repair (pp)", fontsize=10)
        ax.spines[:].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(out / "end_single.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = ROOT / a.out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    raw, rows, present, units = load(a.runs)
    if not units:
        sys.exit("no complete units")
    sf = ROOT / a.out / "end_stats.json"
    stats = json.loads(sf.read_text())["tests"] if sf.exists() else {}
    fig_aa_bwt(rows, present, units, out)
    fig_paired(rows, present, units, out, stats)
    fig_paired(rows, present, units, out, stats, ours=["repair", "end_joint"], fname="end_paired_zoom.png")
    fig_forgetting(raw, present, units, out)
    fig_pre_post(raw, present, units, out)
    fig_single(raw, present, units, out)
    print("wrote", sorted(p.name for p in out.glob("end_*.png")))


if __name__ == "__main__":
    main()
