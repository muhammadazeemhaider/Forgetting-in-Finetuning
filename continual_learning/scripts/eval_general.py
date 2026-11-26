#!/usr/bin/env python3
"""General-ability retention: score the base model and every saved final model of one (order, seed) unit on held-out
benchmarks none of the methods trained on. Zero-shot, log-likelihood multiple choice (no generation):

  arc_easy, arc_challenge   "Question: q\nAnswer:" + " choice"; pick the highest mean token log-prob (length-normalised)
  winogrande                partial scoring: fill the blank with each option, score the shared continuation after it
  mmlu                      question + lettered options + "Answer:"; pick the letter (A-D) with the highest logit

Models: base (no adapters) and <method>_adapters.pt for each method (end_joint also before its end repair, i.e. with
the end_all branch switched off). Writes <runs>/<unit>/general.json; finished models are skipped on rerun.
Usage: python scripts/eval_general.py --runs runs_gen5k --order order1 --seed 0 [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clb.core import ROOT, clear_branches, inject, load_branch_state  # noqa
from clb.gen import lm_batch, target_logits  # noqa

METHODS = ["seq", "replay", "delayed", "repair", "end_joint", "df_lwf", "df_gen", "df_diff"]


def build(tok, limit):
    """Each item: list of (prompt_ids, continuation_ids) options + gold index; mmlu items use letter scoring."""
    from datasets import load_dataset
    enc = lambda s, bos=True: tok(s, add_special_tokens=bos)["input_ids"]
    B = {}
    for name, cfg in (("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")):
        items = []
        for r in load_dataset("allenai/ai2_arc", cfg, split="test"):
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            p = enc(f"Question: {r['question']}\nAnswer:")
            items.append(([(p, enc(" " + t, False)) for t in r["choices"]["text"]], labels.index(r["answerKey"])))
        B[name] = items
    items = []
    for r in load_dataset("allenai/winogrande", "winogrande_xl", split="validation"):
        pre, post = r["sentence"].split("_", 1)
        cont = enc(" " + post.strip(), False)
        items.append(([(enc(pre + r["option1"]), cont), (enc(pre + r["option2"]), cont)], int(r["answer"]) - 1))
    B["winogrande"] = items
    items = []
    for r in load_dataset("cais/mmlu", "all", split="test"):
        opts = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", r["choices"]))
        items.append((enc(f"The following is a multiple choice question about {r['subject'].replace('_', ' ')}.\n\n"
                          f"{r['question']}\n{opts}\nAnswer:"), r["answer"]))
    B["mmlu"] = items
    if limit:
        B = {k: v[:limit] for k, v in B.items()}
    return B


@torch.no_grad()
def score_mc(model, items, pad, dev, bs=64):
    """Mean token log-prob of each option's continuation; accuracy of the argmax."""
    flat = [(i, j, p, c) for i, (opts, _) in enumerate(items) for j, (p, c) in enumerate(opts)]
    sc = {}
    for b in range(0, len(flat), bs):
        ch = flat[b:b + bs]
        ids, mask, pos, lab = lm_batch([x[2] for x in ch], [x[3] for x in ch], pad, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, y, sel = target_logits(model, ids, mask, pos, lab)
        lp = F.log_softmax(lg.float(), -1).gather(1, y[:, None])[:, 0]
        per = torch.zeros(len(ch), device=dev).index_add_(0, sel.nonzero()[:, 0], lp) / sel.sum(1)
        for (i, j, _, _), v in zip(ch, per.tolist()):
            sc[(i, j)] = v
    right = sum(int(max(range(len(opts)), key=lambda j: sc[(i, j)]) == g) for i, (opts, g) in enumerate(items))
    return right / len(items)


@torch.no_grad()
def score_mmlu(model, tok, items, pad, dev, bs=32):
    letters = torch.tensor([tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in "ABCD"], device=dev)
    right = 0
    for b in range(0, len(items), bs):
        ch = items[b:b + bs]
        n = max(len(p) for p, _ in ch)
        ids = torch.full((len(ch), n), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, (p, _) in enumerate(ch):
            ids[i, n - len(p):] = torch.tensor(p); mask[i, n - len(p):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev)).last_hidden_state[:, -1]
            lg = model.lm_head(h)[:, letters]
        right += sum(int(a == g) for a, (_, g) in zip(lg.argmax(-1).tolist(), ch))
    return right / len(items)


def evaluate_all(c, B):
    out = {k: score_mc(c.model, v, c.pad, c.dev) for k, v in B.items() if k != "mmlu"}
    out["mmlu"] = score_mmlu(c.model, c.tok, B["mmlu"], c.pad, c.dev)
    out["avg"] = sum(out.values()) / len(out)
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_gen5k")
    ap.add_argument("--order", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="config_gen5k.json")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N items per benchmark")
    a = ap.parse_args()

    class C:
        pass
    c = C()
    c.cfg = json.loads((ROOT / a.config).read_text())
    c.dev = "cuda"
    unit = ROOT / a.runs / f"{a.order}_s{a.seed}"
    f = unit / ("general_smoke.json" if a.limit else "general.json")
    res = json.loads(f.read_text()) if f.exists() else {}
    c.tok = AutoTokenizer.from_pretrained(c.cfg["model"])
    c.pad = c.tok.eos_token_id
    c.model = AutoModelForCausalLM.from_pretrained(c.cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa").to(c.dev).eval()
    c.model.requires_grad_(False)
    c.mods = inject(c.model, c.cfg)
    B = build(c.tok, a.limit)
    print({k: len(v) for k, v in B.items()}, flush=True)
    jobs = [("base", None, ())] + [(m, m, ()) for m in METHODS] + [("end_joint_pre_repair", "end_joint", ("end_all",))]
    for name, src, off in jobs:
        if name in res:
            print(f"{name}: done", flush=True); continue
        if src is not None and not (unit / f"{src}_adapters.pt").exists():
            print(f"{name}: no adapters, skipped", flush=True); continue
        t = time.time()
        clear_branches(c.mods)
        if src is not None:
            load_branch_state(c.mods, torch.load(unit / f"{src}_adapters.pt", weights_only=False), c.cfg, c.dev)
        for m in c.mods:
            m.off = set(off)
        res[name] = evaluate_all(c, B)
        res[name]["seconds"] = round(time.time() - t, 1)
        print(f"{name}: " + " ".join(f"{k}={v:.4f}" for k, v in res[name].items() if k != "seconds"), flush=True)
        json.dump(res, open(f, "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
