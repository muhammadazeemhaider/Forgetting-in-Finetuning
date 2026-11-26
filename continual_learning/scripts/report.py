#!/usr/bin/env python3
"""Aggregate all (order, seed) runs -> results/: per-run metrics, method summary, paired comparisons, gate check.

AA  = average accuracy over all 4 tasks after the last task (the benchmark's headline metric)
LA  = average of a_{i,i}, accuracy on each task right after learning it
BWT = mean over the first 3 tasks of a_{T,i} - a_{i,i}  (negative = forgetting)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["seq", "replay", "full_replay", "joint", "delayed", "repair", "olora", "mtl"]
CONTROLS = ["seq", "replay", "joint", "delayed", "olora", "full_replay"]


def metrics(r):
    order, rows = r["order"], r["acc_matrix"]
    if r["method"] == "mtl":
        fin = rows[-1]["acc"]
        return {"AA": 100 * np.mean([fin[t] for t in order]), "LA": np.nan, "BWT": np.nan,
                **{f"final_{t}": 100 * fin[t] for t in order}}
    fin = rows[-1]["acc"]
    diag = [rows[i]["acc"][order[i]] for i in range(len(order))]
    bwt = [fin[order[i]] - diag[i] for i in range(len(order) - 1)]
    return {"AA": 100 * np.mean([fin[t] for t in order]), "LA": 100 * np.mean(diag), "BWT": 100 * np.mean(bwt),
            "worst_drop": 100 * min(bwt), **{f"final_{t}": 100 * fin[t] for t in order},
            **{f"learned_{t}": 100 * diag[i] for i, t in enumerate(order)}}


def boot(d, B=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (B, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for d in sorted((ROOT / "runs").glob("order*_s[0-9]")):
        for m in METHODS:
            f = d / f"{m}.json"
            if f.exists():
                r = json.loads(f.read_text())
                rows.append({"unit": d.name, "order": d.name.split("_")[0], "seed": r["seed"], "method": m,
                             "seconds": r["seconds"], **metrics(r)})
    if not rows:
        sys.exit("no runs yet")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("unit", "order", "seed", "method"), k))
    with open(out / "per_run.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    units = sorted({r["unit"] for r in rows})
    get = {(r["unit"], r["method"]): r for r in rows}
    complete = [u for u in units if all((u, m) in get for m in METHODS)]

    L = ["# Standard CL Benchmark: compensated-LoRA repair vs controls (Llama-3.2-1B)\n",
         f"Complete (order, seed) units: {len(complete)} of {len(units)} started: {', '.join(complete)}\n"]
    L.append("## Method summary (mean ± SD over complete units)\n")
    L.append("| Method | AA (final avg acc) | LA (acc when learned) | BWT (forgetting <0) | worst task drop | train+eval time (min) |")
    L.append("|---|---|---|---|---|---|")
    summ = {}
    for m in METHODS:
        v = [get[(u, m)] for u in complete]
        if not v:
            continue
        f = lambda k: (np.nanmean([x[k] for x in v]), np.nanstd([x[k] for x in v], ddof=1) if len(v) > 1 else np.nan)
        summ[m] = {k: f(k) for k in ("AA", "LA", "BWT")}
        wd = np.nanmean([x.get("worst_drop", np.nan) for x in v])
        L.append(f"| {m} | {summ[m]['AA'][0]:.2f} ± {summ[m]['AA'][1]:.2f} | {summ[m]['LA'][0]:.2f} ± {summ[m]['LA'][1]:.2f} | "
                 f"{summ[m]['BWT'][0]:+.2f} ± {summ[m]['BWT'][1]:.2f} | {wd:+.2f} | {np.mean([x['seconds'] for x in v]) / 60:.1f} |")

    L.append("\n## Per order (mean over seeds)\n")
    L.append("| Order | " + " | ".join(METHODS) + " |")
    L.append("|---|" + "---|" * len(METHODS))
    for o in sorted({u.split("_")[0] for u in complete}):
        us = [u for u in complete if u.startswith(o)]
        L.append(f"| {o} AA | " + " | ".join(f"{np.mean([get[(u, m)]['AA'] for u in us]):.2f}" for m in METHODS) + " |")
        L.append(f"| {o} BWT | " + " | ".join(
            "—" if m == "mtl" else f"{np.mean([get[(u, m)]['BWT'] for u in us]):+.2f}" for m in METHODS) + " |")

    seq_bwt = np.mean([get[(u, "seq")]["BWT"] for u in complete]) if complete else np.nan
    gap = (np.mean([get[(u, "replay")]["AA"] for u in complete]) - np.mean([get[(u, "mtl")]["AA"] for u in complete])) if complete else np.nan
    L.append("\n## Gate: is this a setting where a repair method can show anything?\n")
    L.append(f"- Unprotected (seq) mean BWT: **{seq_bwt:+.2f} pp** (gate: ≤ −10)")
    L.append(f"- Matched replay AA minus multi-task reference AA: **{gap:+.2f} pp** (a gap means replay leaves room)")

    L.append("\n## Paired: repair minus each control (over complete units; 95% bootstrap CI over units)\n")
    L.append("| Control | ΔAA | 95% CI | units repair better | ΔBWT | 95% CI | ΔLA |")
    L.append("|---|---|---|---|---|---|---|")
    pairs = {}
    for c in CONTROLS:
        dA = [get[(u, "repair")]["AA"] - get[(u, c)]["AA"] for u in complete]
        dB = [get[(u, "repair")]["BWT"] - get[(u, c)]["BWT"] for u in complete]
        dL = [get[(u, "repair")]["LA"] - get[(u, c)]["LA"] for u in complete]
        if not dA:
            continue
        ciA, ciB = boot(dA), boot(dB)
        pairs[c] = {"dAA": float(np.mean(dA)), "ciAA": ciA, "dBWT": float(np.mean(dB)), "ciBWT": ciB, "dLA": float(np.mean(dL)),
                    "wins": int(sum(x > 0 for x in dA)), "n": len(dA)}
        L.append(f"| {c} | {np.mean(dA):+.2f} | [{ciA[0]:+.2f}, {ciA[1]:+.2f}] | {pairs[c]['wins']}/{len(dA)} | "
                 f"{np.mean(dB):+.2f} | [{ciB[0]:+.2f}, {ciB[1]:+.2f}] | {np.mean(dL):+.2f} |")
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    json.dump({"complete_units": complete, "summary": {m: {k: list(map(float, v)) for k, v in s.items()} for m, s in summ.items()},
               "paired_repair_minus": pairs, "seq_BWT": float(seq_bwt), "replay_minus_mtl_AA": float(gap)},
              open(out / "summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
