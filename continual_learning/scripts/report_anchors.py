#!/usr/bin/env python3
"""How many stored examples per task are needed? Compares the same methods at 128 / 64 / 32 anchors per task.
-> results/ANCHORS.md, results/anchors_stats.json, results/figures/anchors_*.png

Every anchor count reuses the same first-task adapter and the same 9 (order, seed) units, so differences are paired.
Tests (paired over units; mean, 95% bootstrap CI, units better, exact sign-flip p, Wilcoxon p, t p, Holm on t p):
  shrink    each method at N vs the same method at 128 (does less memory hurt?)       Holm over methods x {64, 32}
  vs_base   end_joint vs replay and vs delayed at each N (does the ranking change?)  Holm over 2 x 3
Usage: python scripts/report_anchors.py [--roots 128=runs 64=runs_a64 32=runs_a32]
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
from report_end import fmt_p, holm, test  # noqa: E402
from plot_end import COL, GRID, INK2, SURF, grid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["replay", "delayed", "repair", "end_joint"]


def load(roots):
    get = {}
    for n, r in roots.items():
        for d in sorted((ROOT / r).glob("order*_s[0-9]")):
            for m in METHODS + ["seq"]:
                f = d / f"{m}.json"
                if f.exists():
                    x = json.loads(f.read_text())
                    v = metrics(x)
                    if "pre_repair" in x:
                        v["AA_pre"] = 100 * np.mean([x["pre_repair"]["acc"][t] for t in x["order"]])
                    get[(n, d.name, m)] = v
    return get


def table(L, head, rows):
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["128=runs", "64=runs_a64", "32=runs_a32"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--md", action="store_true", help="also write ANCHORS.md (default: print only)")
    a = ap.parse_args()
    roots = {int(k): v for k, v in (s.split("=") for s in a.roots)}
    Ns = sorted(roots, reverse=True)
    get = load(roots)
    units = sorted({u for _, u, _ in get})
    units = [u for u in units if all((n, u, m) in get for n in Ns for m in METHODS)]
    if not units:
        sys.exit("no complete units across all anchor counts")
    out = ROOT / a.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seq = np.mean([get[(Ns[0], u, "seq")]["AA"] for u in units if (Ns[0], u, "seq") in get])
    L = ["# How many stored examples are needed? 128 vs 64 vs 32 per task\n",
         f"Same first-task adapter and same {len(units)} (order, seed) units at every count; only the stored-example "
         f"memory shrinks (class-balanced subsets of the 128). seq (no memory) AA = {seq:.2f}.\n",
         "## Summary (mean ± SD over units)\n"]
    rows = []
    for m in METHODS:
        for k in ("AA", "BWT"):
            r = [m, k]
            for n in Ns:
                v = [get[(n, u, m)][k] for u in units]
                r.append(f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}")
            rows.append(r)
    table(L, ["Method", "metric"] + [f"{n} per task" for n in Ns], rows)

    L.append("## end_joint: how much of the forgetting the end repair wins back\n")
    rows = []
    for n in Ns:
        pre = np.mean([get[(n, u, "end_joint")]["AA_pre"] for u in units])
        post = np.mean([get[(n, u, "end_joint")]["AA"] for u in units])
        ref = np.mean([get[(Ns[0], u, "end_joint")]["AA"] for u in units])
        rows.append([str(n), str(4 * n), f"{pre:.2f}", f"{post:.2f}", f"{post - pre:+.2f}",
                     f"{100 * (post - pre) / (ref - pre):.0f}%"])
    table(L, ["per task", "total stored", "AA before repair", "AA after", "recovered (pp)",
              f"share of the {Ns[0]}-example recovery"], rows)

    res = {}
    fams = {"shrink": [(m, n) for m in METHODS for n in Ns[1:]], "vs_base": [(b, n) for b in ("replay", "delayed") for n in Ns]}
    for metric in ("AA", "BWT"):
        L.append(f"## Significance — {metric}\n")
        for fam, pairs in fams.items():
            tests = []
            for x, n in pairs:
                if fam == "shrink":
                    d = [get[(n, u, x)][metric] - get[(Ns[0], u, x)][metric] for u in units]
                else:
                    d = [get[(n, u, "end_joint")][metric] - get[(n, u, x)][metric] for u in units]
                tests.append(test(d))
            for t, h in zip(tests, holm([t["p_t"] for t in tests])):
                t["p_holm"] = h
            L.append("**Each method with fewer examples vs itself at %d** (Holm over %d)\n" % (Ns[0], len(pairs))
                     if fam == "shrink" else "**end_joint vs replay / delayed at the same count** (Holm over %d)\n" % len(pairs))
            rows = []
            for (x, n), t in zip(pairs, tests):
                lab = f"{x} @ {n} − {x} @ {Ns[0]}" if fam == "shrink" else f"end_joint − {x} @ {n}"
                rows.append([lab, f"{t['mean']:+.2f}", f"[{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]", f"{t['wins']}/{t['n']}",
                             fmt_p(t["p_exact"]), fmt_p(t["p_wilcoxon"]), fmt_p(t["p_t"]),
                             fmt_p(t["p_holm"]) + (" **\\***" if t["p_holm"] < 0.05 else "")])
                res[f"{metric}:{lab}"] = {**t, "family": fam}
            table(L, ["comparison", "mean Δ", "95% CI", "A better", "p exact", "p Wilcoxon", "p t", "p Holm (t)"], rows)
    L.append("\\* = significant at 0.05 after Holm. Units = (order, seed) runs; with n = 9 the exact p is ≥ 0.0039.")
    if a.md:                                              # tables go into results/RESULTS.md by hand
        (out / "ANCHORS.md").write_text("\n".join(L) + "\n")
    json.dump({"units": units, "anchors": Ns, "tests": res}, open(out / "anchors_stats.json", "w"), indent=1)

    # figure: AA and BWT vs anchors per task, mean + 95% CI; end_joint before-repair as a dashed line
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = np.arange(len(Ns))
    off = np.linspace(-0.12, 0.12, len(METHODS))
    for ax, k, lab in ((axes[0], "AA", "AA: final average accuracy (%)"), (axes[1], "BWT", "BWT: forgetting (pp)")):
        for o, m in zip(off, METHODS):
            mu, lo, hi = [], [], []
            for n in Ns:
                v = np.array([get[(n, u, m)][k] for u in units])
                c = boot(v); mu.append(v.mean()); lo.append(c[0]); hi.append(c[1])
            ax.vlines(xs + o, lo, hi, color=COL[m], lw=2)
            ax.plot(xs + o, mu, color=COL[m], lw=2, marker="o", ms=7, mec=SURF, mew=1.5, label=m)
        if k == "BWT":                                   # AA panel stays zoomed; before-repair / seq are in the tables
            ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(xs, [f"{n}\nper task" for n in Ns]); ax.set_ylabel(lab); grid(ax, "y")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "figures" / "anchors_aa_bwt.png", dpi=160); plt.close(fig)
    print("\n".join(L))


if __name__ == "__main__":
    main()
