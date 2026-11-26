#!/usr/bin/env python3
"""Pick TIES (density, lambda) per method by VALIDATION final average accuracy; write them into config.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
rows = []
for d in sorted((ROOT / "runs_tune").glob("d*_l*")):
    dens, lam = float(d.name.split("_")[0][1:]), float(d.name.split("_")[1][1:])
    for m in ["ties_only", "ties_repair", "ties_repair_lwf"]:
        f = d / "order1_s0" / f"{m}.json"
        if f.exists():
            r = json.loads(f.read_text())
            mt = metrics(r)
            conf = np.mean([s["ties"]["sign_conflict_frac"] for s in r["task_stats"] if "ties" in s])
            rows.append({"method": m, "density": dens, "lambda": lam, "dev_AA": mt["AA"], "dev_BWT": mt["BWT"],
                         "dev_LA": mt["LA"], "sign_conflict": conf})
best = {}
print("| method | density | lambda | dev final avg acc | dev forgetting | dev learning acc | sign-conflict frac |")
print("|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["method"], -r["dev_AA"])):
    print(f"| {r['method']} | {r['density']} | {r['lambda']} | {r['dev_AA']:.2f} | {r['dev_BWT']:+.2f} | {r['dev_LA']:.2f} | {r['sign_conflict']:.3f} |")
    if r["method"] not in best or r["dev_AA"] > best[r["method"]]["dev_AA"]:
        best[r["method"]] = r
print("\nselected:", {m: (b["density"], b["lambda"]) for m, b in best.items()})
if len(sys.argv) > 1 and sys.argv[1] == "--write":
    cfg = json.loads((ROOT / "config.json").read_text())
    cfg["ties"] = {m: {"density": b["density"], "lambda": b["lambda"], "selected_on": "dev split, order1 seed0"}
                   for m, b in best.items()}
    (ROOT / "config.json").write_text(json.dumps(cfg, indent=2))
    (ROOT / "results" / "TIES_TUNING.json").write_text(json.dumps({"grid": rows, "selected": best}, indent=1))
    print("written to config.json")
