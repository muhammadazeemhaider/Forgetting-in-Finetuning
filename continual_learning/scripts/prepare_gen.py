#!/usr/bin/env python3
"""Fixed data for the generative CL benchmark (login node; ~1 MB download for WMT14, the rest is cached).

Per task (data_seed 42): train_per_task training examples, test_per_task test examples, and anchors_per_task
stored examples drawn FROM the training set (the memory for replay / distillation / repair). Every example is
{"prompt", "target"} plus what its metric needs. Examples whose prompt+target exceed max_length tokens are dropped
before sampling. Refuses to overwrite.

  gsm8k      openai/gsm8k main: train -> train, test -> test. Target = full worked solution ending "#### N".
  triviaqa   mandarjoshi/trivia_qa rc.nocontext (closed book): train -> train, validation -> test. Target = answer.
  hellaswag  Rowan/hellaswag: train -> train, validation -> test. Target = the correct ending; scored by choosing
             among the 4 endings by mean token log-likelihood.
  wmt14      wmt/wmt14 de-en, English->German: newstest2013 (validation, 3000 pairs) -> train,
             newstest2014 (test, 3003 pairs) -> test. Scored with chrF (and BLEU) via sacrebleu.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CFG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "config_gen.json"
cfg = json.loads((ROOT / CFG).read_text())
D = cfg["data"]
OUT = ROOT / D["dir"]

INSTR = {
    "gsm8k": "Solve the math problem. End with '#### <answer>'.",
    "triviaqa": "Answer the trivia question.",
    "hellaswag": "Complete the passage with the most plausible continuation.",
    "wmt14": "Translate the English sentence into German.",
}


def gsm8k(split):
    from datasets import load_dataset
    out = []
    for r in load_dataset("openai/gsm8k", "main", split=split):
        final = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"prompt": f"{INSTR['gsm8k']}\nQuestion: {r['question']}\nAnswer:", "target": " " + r["answer"],
                    "final": final})
    return out


def triviaqa(split):
    from datasets import load_dataset
    out = []
    for r in load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split):
        a = r["answer"]
        out.append({"prompt": f"{INSTR['triviaqa']}\nQuestion: {r['question']}\nAnswer:", "target": " " + a["value"],
                    "aliases": sorted(set(a["normalized_aliases"] + [a["normalized_value"]]))})
    return out


def _hs_clean(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub(r"\[.*?\]", "", t)
    return t.replace("  ", " ")


def hellaswag(split):
    from datasets import load_dataset
    out = []
    for r in load_dataset("Rowan/hellaswag", split=split):
        if r["label"] == "":
            continue
        ctx = _hs_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        ends = [" " + _hs_clean(e) for e in r["endings"]]
        lab = int(r["label"])
        out.append({"prompt": f"{INSTR['hellaswag']}\nPassage: {ctx}\nContinuation:", "target": ends[lab],
                    "endings": ends, "label": lab})
    return out


def wmt14(split):
    from datasets import load_dataset
    ds = load_dataset("wmt/wmt14", "de-en", split=split)
    if split == "train":                                   # 4.5M pairs: draw candidates first, keep clean ones
        idx = np.random.default_rng(D["data_seed"] + 1).choice(len(ds), D.get("wmt14_candidates", 20000), replace=False)
        ds = ds.select(sorted(idx.tolist()))
    out = []
    for r in ds:
        en, de = r["translation"]["en"], r["translation"]["de"]
        if split == "train" and not (5 <= len(en.split()) <= 60 and 0.5 <= len(de.split()) / len(en.split()) <= 2.0):
            continue                                       # drop fragments and misaligned web-crawl pairs
        out.append({"prompt": f"{INSTR['wmt14']}\nEnglish: {en}\nGerman:", "target": " " + de, "reference": de})
    return out


LOADERS = {"gsm8k": (gsm8k, "train", "test"), "triviaqa": (triviaqa, "train", "validation"),
           "hellaswag": (hellaswag, "train", "validation"), "wmt14": (wmt14, "validation", "test")}


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    OUT.mkdir(exist_ok=True)
    manifest = {}
    for task in cfg["tasks"]:
        if all((OUT / f"{task}_{s}.json").exists() for s in ("train", "test", "anchors")):
            print(task, "already prepared"); continue
        fn, tr_split, te_split = LOADERS[task]
        if task == "wmt14":
            tr_split = D.get("wmt14_train_split", tr_split)
        rng = np.random.default_rng(D["data_seed"])
        sets = {}
        for name, split, n in (("train", tr_split, D["train_per_task"]), ("test", te_split, D["test_per_task"])):
            rows = fn(split)
            keep = [r for r in rows
                    if len(tok(r["prompt"] + max([r["target"]] + r.get("endings", []), key=len))["input_ids"]) < D["max_length"]]
            idx = rng.choice(len(keep), n, replace=False)
            sets[name] = [keep[i] for i in sorted(idx)]
            print(task, name, f"{len(rows)} rows, {len(keep)} fit, sampled {n}", flush=True)
        a = rng.choice(len(sets["train"]), D["anchors_per_task"], replace=False)
        sets["anchors"] = [sets["train"][i] for i in sorted(a)]
        for s, rows in sets.items():
            (OUT / f"{task}_{s}.json").write_text(json.dumps(rows))
        manifest[task] = {s: len(v) for s, v in sets.items()}
    if manifest:
        (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
