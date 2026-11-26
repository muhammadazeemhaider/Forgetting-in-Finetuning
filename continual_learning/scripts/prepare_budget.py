#!/usr/bin/env python3
"""Data-budget ladder for the Standard CL Benchmark (login node: label columns + selected rows only, Amazon streamed).

Per task (data_seed 42), every class's train rows are permuted once. Then:
  anchors   the first 128 rows taken class-balanced (128 // C per class, the remainder to the first classes): the
            stored memory. Never used for fine-tuning, identical at every budget.
  train     budget B examples per task, class-balanced (B // C per class, the remainder to the first classes), taken
            from the permuted rows AFTER the anchors, so 5k is a subset of 10k is a subset of 100k.
  test      copied unchanged from data/ (500 per class, as in every earlier run).
Writes data_b5k/, data_b10k/, data_b100k/ (or --budgets) with MANIFEST.json; refuses to overwrite.
Usage: python scripts/prepare_budget.py [--budgets 5000 10000 100000] [--amazon-rows 150000]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TASKS = ["dbpedia", "amazon", "yahoo", "agnews"]
SEED, N_ANCHORS = 42, 128


def hf_rows(name, label_col, fn):
    """Label column for sampling, and a row getter that converts only the selected rows."""
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    return np.asarray(ds[label_col]), lambda idx: [fn(r) for r in ds.select(idx)]


def amazon_rows(n_rows):
    import pandas as pd
    from huggingface_hub import hf_hub_url
    u = hf_hub_url("yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes", "train.csv",
                   repo_type="dataset")
    df = pd.read_csv(u, nrows=n_rows, storage_options={"Authorization": "Bearer " + os.environ.get("HF_TOKEN", "")})
    rows = [{"title": str(r.review_title), "text": str(r.review_text), "label": int(r.class_index) - 1}
            for r in df.itertuples()]
    return np.array([r["label"] for r in rows]), lambda idx: [rows[i] for i in idx]


def quotas(n, n_cls):
    return [n // n_cls + (1 if c < n % n_cls else 0) for c in range(n_cls)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[5000, 10000, 100000])
    ap.add_argument("--amazon-rows", type=int, default=150000)
    a = ap.parse_args()
    outs = {b: ROOT / f"data_b{b // 1000}k" for b in a.budgets}
    for d in outs.values():
        assert not d.exists(), f"{d} exists; refusing to overwrite"
    src = {
        "agnews": lambda: hf_rows("fancyzhx/ag_news", "label", lambda r: {"text": r["text"], "label": r["label"]}),
        "dbpedia": lambda: hf_rows("fancyzhx/dbpedia_14", "label",
                                   lambda r: {"title": r["title"], "text": r["content"], "label": r["label"]}),
        "yahoo": lambda: hf_rows("community-datasets/yahoo_answers_topics", "topic",
                                 lambda r: {"title": r["question_title"], "question": r["question_content"],
                                            "text": r["best_answer"], "label": r["topic"]}),
        "amazon": lambda: amazon_rows(a.amazon_rows),
    }
    for d in outs.values():
        d.mkdir()
    manifest = {b: {} for b in a.budgets}
    top = max(a.budgets)
    for task in TASKS:
        labels, get = src[task]()
        cls = sorted(set(labels.tolist()))
        rng = np.random.default_rng(SEED)
        perm = {c: rng.permutation(np.where(labels == c)[0]).tolist() for c in cls}
        qa = quotas(N_ANCHORS, len(cls))
        need = [qa[i] + quotas(top, len(cls))[i] for i in range(len(cls))]
        short = {c: (len(perm[c]), need[i]) for i, c in enumerate(cls) if len(perm[c]) < need[i]}
        assert not short, f"{task}: not enough rows per class (have, need): {short}"
        anc_idx = sorted(i for k, c in enumerate(cls) for i in perm[c][:qa[k]])
        anchors = [dict(r, src_idx=int(i)) for i, r in zip(anc_idx, get(anc_idx))]
        for b in a.budgets:
            qb = quotas(b, len(cls))
            idx = [i for k, c in enumerate(cls) for i in perm[c][qa[k]:qa[k] + qb[k]]]
            assert not set(idx) & set(anc_idx)
            order = np.random.default_rng(SEED + 1).permutation(len(idx))   # shuffled once; same order rule per budget
            idx = [idx[j] for j in order]
            train = [dict(r, src_idx=int(i)) for i, r in zip(idx, get(idx))]
            (outs[b] / f"{task}_train.json").write_text(json.dumps(train))
            (outs[b] / f"{task}_anchors.json").write_text(json.dumps(anchors))
            shutil.copy(ROOT / "data" / f"{task}_test.json", outs[b] / f"{task}_test.json")
            manifest[b][task] = {"n_labels": len(cls), "train": len(train), "train_per_class": qb,
                                 "anchors": len(anchors), "test": len(json.loads((ROOT / "data" / f"{task}_test.json").read_text())),
                                 "pool_rows": int(len(labels))}
        print(task, {b: manifest[b][task]["train"] for b in a.budgets}, "anchors", len(anchors), flush=True)
    sets = {b: {t: {r["src_idx"] for r in json.loads((outs[b] / f"{t}_train.json").read_text())} for t in TASKS}
            for b in a.budgets}
    bs = sorted(a.budgets)
    for lo, hi in zip(bs, bs[1:]):
        assert all(sets[lo][t] <= sets[hi][t] for t in TASKS), f"{lo} is not nested in {hi}"
    for b in a.budgets:
        (outs[b] / "MANIFEST.json").write_text(json.dumps(
            {"seed": SEED, "budget_per_task": b, "nested_in": [x for x in bs if x > b], "tasks": manifest[b]}, indent=2))
    print("nested and disjoint from anchors: ok")


if __name__ == "__main__":
    main()
