#!/usr/bin/env python3
"""Validation split for hyperparameter selection (TIES density / lambda): 100 per class per task,
drawn from each dataset's TRAIN split, disjoint from the 1000/class training subset. Login node."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import LOADERS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
for task in ["dbpedia", "amazon", "yahoo", "agnews"]:
    out = ROOT / "data" / f"{task}_dev.json"
    if out.exists():
        print(task, "dev exists"); continue
    used = {r["src_idx"] for r in json.loads((ROOT / "data" / f"{task}_train.json").read_text())}
    rows = LOADERS[task]("train")
    rng = np.random.default_rng(4242)
    by = {}
    for i, r in enumerate(rows):
        if i not in used:
            by.setdefault(r["label"], []).append(i)
    pick = [i for lab in sorted(by) for i in rng.choice(by[lab], 100, replace=False)]
    rng.shuffle(pick)
    dev = [dict(rows[i], src_idx=int(i)) for i in pick]
    assert not ({r["src_idx"] for r in dev} & used)
    out.write_text(json.dumps(dev))
    print(task, len(dev), "dev examples, disjoint from train")
