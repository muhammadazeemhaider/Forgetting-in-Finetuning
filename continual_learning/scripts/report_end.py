#!/usr/bin/env python3
"""End-of-sequence repair experiment -> results/END_REPAIR.md (+ end_per_run.csv, end_stats.json).

Replication unit = one (order, seed) run (9 units). For every comparison A - B on AA / BWT / LA:
  mean paired difference, 95% percentile bootstrap CI (10k), units where A > B,
  exact two-sided sign-flip permutation p (all 2^n sign patterns), Wilcoxon signed-rank p, paired t-test p.
Holm correction (on the paired t p) inside each pre-declared family:
  primary   = each repair-style method vs replay and vs delayed (the claim to beat)
  secondary = each end_* vs direct repair, end_sum vs end_mult, end_* vs end_joint
  sanity    = every protection method vs seq
With n = 9 the exact tests cannot go below p = 2/512 = 0.0039, so after Holm only the t-test p can clear 0.05
in a large family; the exact p is reported as the assumption-free check.
Usage: python scripts/report_end.py [--runs runs] [--out results]
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import boot, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = ["seq", "replay", "delayed", "repair"]
NEW = ["end_joint", "end_sum", "end_mult"]
METHODS = BASE + NEW + ["mtl"]
OURS = ["repair"] + NEW


def sign_flip_p(d):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    flips = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float(np.mean(np.abs((flips * d).mean(1)) >= obs - 1e-12))


def test(d):
    d = np.asarray(d, float)
    lo, hi = boot(d)
    w = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    t = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else (0.0 if d.mean() else 1.0)
    return {"mean": float(d.mean()), "ci": [lo, hi], "wins": int((d > 0).sum()), "n": len(d),
            "p_exact": sign_flip_p(d), "p_wilcoxon": float(w), "p_t": float(t)}


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def fmt_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--md", action="store_true", help="also write END_REPAIR.md (default: print only)")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(exist_ok=True)
    rows = []
    for d in sorted((ROOT / a.runs).glob("order*_s[0-9]")):
        for m in METHODS:
            f = d / f"{m}.json"
            if f.exists():
                r = json.loads(f.read_text())
                x = {"unit": d.name, "order": d.name.split("_")[0], "seed": r["seed"], "method": m, **metrics(r)}
                if "pre_repair" in r:
                    pre = r["pre_repair"]["acc"]
                    x["AA_pre_repair"] = 100 * np.mean([pre[t] for t in r["order"]])
                rows.append(x)
    if not rows:
        sys.exit("no runs yet")
    get = {(r["unit"], r["method"]): r for r in rows}
    units = sorted({r["unit"] for r in rows})
    present = [m for m in METHODS if any((u, m) in get for u in units)]
    complete = [u for u in units if all((u, m) in get for m in present)]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("unit", "order", "seed", "method"), k))
    with open(out / "end_per_run.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    L = ["# End-of-sequence repair vs replay / delayed distillation (Llama-3.2-1B, Standard CL Benchmark)\n",
         f"Config: learner rank 8 + repair rank 8 per task, LoRA on all 7 linear layers (q,k,v,o,gate,up,down). Complete units: {len(complete)} "
         f"({', '.join(complete)}). Methods present: {', '.join(present)}.\n",
         "## Summary (mean ± SD over units)\n",
         "| Method | AA | LA | BWT | worst task drop | AA before end repair |", "|---|---|---|---|---|---|"]
    for m in present:
        v = [get[(u, m)] for u in complete]
        ms = lambda k: (f"{np.nanmean([x.get(k, np.nan) for x in v]):.2f} ± {np.nanstd([x.get(k, np.nan) for x in v], ddof=1):.2f}"
                        if not np.all(np.isnan([x.get(k, np.nan) for x in v])) else "—")
        L.append(f"| {m} | {ms('AA')} | {ms('LA')} | {ms('BWT')} | {ms('worst_drop')} | {ms('AA_pre_repair')} |")

    tasks = sorted({k[6:] for r in rows for k in r if k.startswith("final_")})
    L += ["\n## Final accuracy per task (mean over units)\n", "| Method | " + " | ".join(tasks) + " |",
          "|---|" + "---|" * len(tasks)]
    for m in present:
        L.append(f"| {m} | " + " | ".join(f"{np.mean([get[(u, m)][f'final_{t}'] for u in complete]):.2f}" for t in tasks) + " |")

    L += ["\n## AA per order (mean over seeds)\n", "| Order | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for o in sorted({u.split("_")[0] for u in complete}):
        us = [u for u in complete if u.startswith(o)]
        L.append(f"| {o} | " + " | ".join(f"{np.mean([get[(u, m)]['AA'] for u in us]):.2f}" for m in present) + " |")

    fam = {"primary": [(x, b) for x in OURS for b in ("replay", "delayed")],
           "secondary": [(x, "repair") for x in NEW] + [("end_sum", "end_mult"), ("end_sum", "end_joint"), ("end_mult", "end_joint")],
           "sanity": [(x, "seq") for x in ("replay", "delayed") + tuple(OURS)]}
    res = {}
    for fname, pairs in fam.items():
        pairs = [(x, b) for x, b in pairs if x in present and b in present]
        if not pairs:
            continue
        L.append(f"\n## Significance — {fname} family (Holm over {len(pairs)} comparisons, per metric)\n")
        for metric in ("AA", "BWT", "LA"):
            tests = []
            for x, b in pairs:
                d = [get[(u, x)][metric] - get[(u, b)][metric] for u in complete]
                tests.append(test(d) if not np.any(np.isnan(d)) else None)
            ok = [i for i, t in enumerate(tests) if t]
            adj = holm([tests[i]["p_t"] for i in ok]) if ok else []
            for i, h in zip(ok, adj):
                tests[i]["p_holm"] = h
            L += [f"**{metric}**\n", "| A − B | mean Δ | 95% CI | A better | p exact | p Wilcoxon | p t | p Holm (t) |",
                  "|---|---|---|---|---|---|---|---|"]
            for (x, b), t in zip(pairs, tests):
                if not t:
                    continue
                star = " **\\***" if t["p_holm"] < 0.05 else ""
                L.append(f"| {x} − {b} | {t['mean']:+.2f} | [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}] | {t['wins']}/{t['n']} | "
                         f"{fmt_p(t['p_exact'])} | {fmt_p(t['p_wilcoxon'])} | {fmt_p(t['p_t'])} | {fmt_p(t['p_holm'])}{star} |")
                res[f"{metric}:{x}-{b}"] = {**t, "family": fname}
            L.append("")
    L.append("\\* = significant at 0.05 after Holm. Units are (order, seed) runs; n = 9 bounds the exact p at ≥ 0.0039.")
    if a.md:                                              # tables go into results/RESULTS.md by hand
        (out / "END_REPAIR.md").write_text("\n".join(L) + "\n")
    json.dump({"complete_units": complete, "tests": res}, open(out / "end_stats.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
