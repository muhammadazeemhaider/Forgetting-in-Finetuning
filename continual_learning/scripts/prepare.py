#!/usr/bin/env python3
"""Fixed data for the Standard CL Benchmark (login node; downloads ~650 MB, streams Amazon).

Per task (data_seed 42): train 1000/class from `train`, test 500/class from `test`, and 128
class-balanced anchors drawn FROM the train subset (the stored historical memory). Refuses to overwrite.
Amazon: Zhang et al. amazon_review_full (5 classes); only the first 40k rows of each CSV are streamed
(the file is shuffled: classes are within 7,595-8,212 per 40k), not the 1.6 GB file.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / "config.json").read_text())
D = cfg["data"]
OUT = ROOT / (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "data")


def amazon(split):
    import pandas as pd
    from huggingface_hub import hf_hub_url
    u = hf_hub_url("yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes", f"{split}.csv",
                   repo_type="dataset")
    df = pd.read_csv(u, nrows=40000, storage_options={"Authorization": "Bearer " + os.environ.get("HF_TOKEN", "")})
    return [{"title": str(r.review_title), "text": str(r.review_text), "label": int(r.class_index) - 1}
            for r in df.itertuples()]


def hf(name, config, split, fn):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    return [fn(r) for r in ds]


LOADERS = {
    "agnews": lambda s: hf("fancyzhx/ag_news", None, s, lambda r: {"text": r["text"], "label": r["label"]}),
    "dbpedia": lambda s: hf("fancyzhx/dbpedia_14", None, s,
                            lambda r: {"title": r["title"], "text": r["content"], "label": r["label"]}),
    "yahoo": lambda s: hf("community-datasets/yahoo_answers_topics", None, s,
                          lambda r: {"title": r["question_title"], "question": r["question_content"],
                                     "text": r["best_answer"], "label": r["topic"]}),
    "amazon": amazon,
}


def balanced(rows, per_class, rng):
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["label"], []).append(i)
    out = []
    for lab in sorted(by):
        idx = by[lab]
        assert len(idx) >= per_class, f"label {lab}: only {len(idx)}"
        out += rng.choice(idx, per_class, replace=False).tolist()
    rng.shuffle(out)
    return [dict(rows[i], src_idx=int(i)) for i in out]


def main():
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for task in cfg["tasks"]:
        if all((OUT / f"{task}_{s}.json").exists() for s in ("train", "test", "anchors")):
            print(task, "already prepared"); continue
        rng = np.random.default_rng(D["data_seed"])
        tr = balanced(LOADERS[task]("train"), D["train_per_class"], rng)
        te = balanced(LOADERS[task]("test"), D["test_per_class"], rng)
        labs = sorted({r["label"] for r in tr})
        k = D["anchors_per_task"]
        per = k // len(labs)
        anc = []
        for lab in labs:
            anc += [r for r in tr if r["label"] == lab][:per]
        rest = [r for r in tr if r not in anc]
        anc += [rest[i] for i in rng.choice(len(rest), k - len(anc), replace=False)]
        for s, rows in (("train", tr), ("test", te), ("anchors", anc)):
            (OUT / f"{task}_{s}.json").write_text(json.dumps(rows))
        manifest[task] = {"n_labels": len(labs), "train": len(tr), "test": len(te), "anchors": len(anc)}
        print(task, manifest[task], flush=True)
    if manifest:
        (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
