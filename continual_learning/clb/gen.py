"""Generative CL benchmark (GSM8K, TriviaQA, HellaSwag, WMT14 en->de): data, target-only LM loss,
token-level distillation against an exact past model, and generation-based evaluation.

Teacher for task j = the model right after task j was learned. Adapters are frozen additive branches, so that
model is recovered exactly by switching off every branch added after task j (LoRALinear.off); no logits are stored.
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter

import torch
import torch.nn.functional as F

from clb.core import ROOT

GENERATIVE = ("gsm8k", "triviaqa", "wmt14")


class GenData:
    """Tokenised prompt / target pairs for one task split (target ends with EOS)."""

    def __init__(self, tok, task, split, cfg):
        self.task = task
        self.rows = json.loads((ROOT / cfg["data"]["dir"] / f"{task}_{split}.json").read_text())
        L = cfg["data"]["max_length"]
        self.prompt = [tok(r["prompt"])["input_ids"] for r in self.rows]                       # with BOS
        self.target = [tok(r["target"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for r in self.rows]
        for i in range(len(self.rows)):                                                         # keep the target whole
            keep = L - len(self.target[i])
            self.prompt[i] = self.prompt[i][-keep:]
        if task == "hellaswag":
            self.endings = [[tok(e, add_special_tokens=False)["input_ids"] for e in r["endings"]] for r in self.rows]

    def __len__(self):
        return len(self.rows)


def lm_batch(prompts, targets, pad_id, device):
    """Left-padded prompt+target sequences; labels are -100 everywhere except target tokens."""
    seqs = [p + t for p, t in zip(prompts, targets)]
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    lab = torch.full_like(ids, -100)
    for i, (p, t) in enumerate(zip(prompts, targets)):
        s = p + t
        ids[i, n - len(s):] = torch.tensor(s)
        mask[i, n - len(s):] = 1
        lab[i, n - len(t):] = torch.tensor(t)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device), lab.to(device)


def target_logits(model, ids, mask, pos, lab):
    """Full-vocabulary logits at the positions that predict target tokens: [n_target_tokens, V], and the targets."""
    h = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, :-1]
    y = lab[:, 1:]
    sel = y != -100
    return model.lm_head(h[sel]), y[sel], sel


def set_off(mods, names):
    for m in mods:
        m.off = set(names)


def distill_loss(c, anchors, rng):
    """Mean token-level KL(teacher || model) over an anchor batch pooled from `anchors` (task -> dict with
    'data' GenData and 'teacher' = branch names that existed right after that task was learned)."""
    pool = [(t, i) for t in anchors for i in range(len(anchors[t]["data"]))]
    pick = [pool[j] for j in rng.choice(len(pool), min(len(pool), c.cfg["training"]["anchor_batch_size"]), replace=False)]
    all_names = set(c.mods[0].branches) | set(c.mods[0].mult)
    total, n_tok = 0.0, 0
    for t in sorted({t for t, _ in pick}):
        idx = [i for tt, i in pick if tt == t]
        d = anchors[t]["data"]
        ids, mask, pos, lab = lm_batch([d.prompt[i] for i in idx], [d.target[i] for i in idx], c.pad, c.dev)
        set_off(c.mods, all_names - set(anchors[t]["teacher"]))
        with torch.no_grad():
            tl, _, _ = target_logits(c.model, ids, mask, pos, lab)
            p_log = F.log_softmax(tl.float(), -1)
        set_off(c.mods, ())
        sl, _, _ = target_logits(c.model, ids, mask, pos, lab)
        q_log = F.log_softmax(sl.float(), -1)
        total = total + (p_log.exp() * (p_log - q_log)).sum()
        n_tok += sl.shape[0]
    return total / max(n_tok, 1)


# ------------------------------------------------------------------------------------------- evaluation

def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _gsm_num(text):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _same_num(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


@torch.no_grad()
def generate(c, data, max_new):
    out, bs = [], c.cfg["eval"]["batch_size"]
    for b in range(0, len(data), bs):
        ps = data.prompt[b:b + bs]
        n = max(map(len, ps))
        ids = torch.full((len(ps), n), c.pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(ps):
            ids[i, n - len(p):] = torch.tensor(p)
            mask[i, n - len(p):] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = c.model.generate(input_ids=ids.to(c.dev), attention_mask=mask.to(c.dev), max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=c.pad, eos_token_id=c.tok.eos_token_id)
        out += c.tok.batch_decode(g[:, n:], skip_special_tokens=True)
    return out


@torch.no_grad()
def hellaswag_choice(c, data):
    """Pick the ending with the highest mean token log-likelihood (acc_norm-style length normalisation)."""
    preds, bs = [], max(1, c.cfg["eval"]["batch_size"] // 4)
    for b in range(0, len(data), bs):
        P, T, owner = [], [], []
        for i in range(b, min(b + bs, len(data))):
            for e in data.endings[i]:
                P.append(data.prompt[i]); T.append(e); owner.append(i)
        ids, mask, pos, lab = lm_batch(P, T, c.pad, c.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, y, sel = target_logits(c.model, ids, mask, pos, lab)
        lp = F.log_softmax(lg.float(), -1).gather(1, y[:, None])[:, 0]
        per = torch.zeros(len(P), device=c.dev).index_add_(0, sel.nonzero()[:, 0], lp) / sel.sum(1)
        preds += per.view(-1, 4).argmax(1).tolist()
    return preds


def evaluate_task(c, task, data):
    """Score in [0, 1] plus the raw outputs. gsm8k: exact final number; triviaqa: normalised exact match against
    aliases; hellaswag: 4-way choice accuracy; wmt14: corpus chrF / 100 (BLEU kept alongside)."""
    if task == "hellaswag":
        p = hellaswag_choice(c, data)
        return sum(int(a == r["label"]) for a, r in zip(p, data.rows)) / len(data), {"pred": p}
    gens = generate(c, data, c.cfg["eval"]["max_new_tokens"][task])
    if task == "gsm8k":
        nums = [_gsm_num(g.split("\n\n")[0] if "####" not in g else g) for g in gens]
        score = sum(_same_num(n, r["final"]) for n, r in zip(nums, data.rows)) / len(data)
        return score, {"gen": gens}
    first = [g.strip().split("\n")[0].strip() for g in gens]
    if task == "triviaqa":
        return sum(_norm(a) in r["aliases"] for a, r in zip(first, data.rows)) / len(data), {"gen": first}
    import sacrebleu
    refs = [r["reference"] for r in data.rows]
    chrf = sacrebleu.corpus_chrf(first, [refs]).score
    bleu = sacrebleu.corpus_bleu(first, [refs]).score
    return chrf / 100, {"gen": first, "bleu": bleu}
