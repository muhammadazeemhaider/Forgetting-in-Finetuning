"""Standard CL Benchmark x compensated-LoRA methods: prompts, stackable LoRA branches, label scoring.

Scoring/training follow the compensated-LoRA protocol: the answer is one letter; the decision, CE and
KL are restricted to the current task's valid letters, computed from the last non-padding hidden
state through only those output-head rows (no full-vocabulary softmax, no generation).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
LETTERS = "ABCDEFGHIJKLMN"

LABELS = {
    "agnews": ["World", "Sports", "Business", "Science/Technology"],
    "dbpedia": ["Company", "Educational institution", "Artist", "Athlete", "Office holder", "Means of transportation", "Building", "Natural place", "Village", "Animal", "Plant", "Album", "Film", "Written work"],
    "yahoo": ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"],
    "amazon": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
}
INSTR = {
    "agnews": "Classify the topic of the news article.",
    "dbpedia": "Classify the category of the entity described in the text.",
    "yahoo": "Classify the topic of the question and its reply.",
    "amazon": "Rate the product review from 1 to 5 stars.",
}


def load_config():
    return json.loads((ROOT / "config.json").read_text())


def load_task(task, split, data_dir="data"):
    return json.loads((ROOT / data_dir / f"{task}_{split}.json").read_text())


def _trunc(tok, s, n):
    ids = tok(s or "", add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:n]) if len(ids) > n else (s or "")


def render(tok, task, ex, n_text):
    if task == "agnews":
        body = f"Article: {_trunc(tok, ex['text'], n_text)}"
    elif task == "dbpedia":
        body = f"Title: {_trunc(tok, ex['title'], 32)}\nText: {_trunc(tok, ex['text'], n_text)}"
    elif task == "yahoo":
        q = (ex["title"] + " " + (ex["question"] or "")).strip()
        body = f"Question: {_trunc(tok, q, 64)}\nReply: {_trunc(tok, ex['text'], n_text - 32)}"
    else:
        body = f"Review title: {_trunc(tok, ex['title'], 32)}\nReview: {_trunc(tok, ex['text'], n_text)}"
    opts = "\n".join(f"{LETTERS[i]}. {name}" for i, name in enumerate(LABELS[task]))
    return f"{INSTR[task]}\nOptions:\n{opts}\n\n{body}\nAnswer:"


def letter_ids(tok):
    ids = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in LETTERS]
    assert all(len(i) == 1 for i in ids), "answer letters must be single tokens"
    return torch.tensor([i[0] for i in ids])


class TaskData:
    """Tokenised prompts + labels for one task split."""

    def __init__(self, tok, task, split, cfg):
        rows = load_task(task, split, cfg["data"].get("dir", "data"))
        self.task, self.n_labels = task, len(LABELS[task])
        self.ids = [tok(render(tok, task, r, cfg["data"]["text_max_tokens"]), add_special_tokens=True)["input_ids"]
                    [-cfg["data"]["max_length"]:] for r in rows]
        self.labels = torch.tensor([r["label"] for r in rows])

    def __len__(self):
        return len(self.ids)


def batchify(seqs, pad_id, device):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        mask[i, n - len(s):] = 1
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device)


# ------------------------------------------------------------------------------ stackable LoRA

class Branch(nn.Module):
    def __init__(self, in_f, out_f, rank, scale):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_f, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)) # PEFT default; B = 0 -> zero initial output
        self.scale = scale


class LoRALinear(nn.Module):
    """W x + sum_k scale_k B_k A_k x over a dict of named branches (frozen or trainable), then optional
    multiplicative branches applied to the output: y <- (I + scale_j B_j A_j) y for each j (product composition).
    Branch names in `off` are skipped (used to train per-task repairs independently)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.branches = nn.ModuleDict()
        self.mult = nn.ModuleDict()  # output-side multiplicative branches, [out, out] low-rank
        self.off = set()
        self.delta = None          # optional frozen dense update (e.g. a TIES-merged adapter), bf16 [out, in]

    def forward(self, x):
        y = self.base(x)
        if self.delta is not None:
            y = y + x @ self.delta.to(x.dtype).T
        for n, b in self.branches.items():
            if n not in self.off:
                y = y + ((x @ b.A.to(x.dtype).T) @ b.B.to(x.dtype).T) * b.scale
        for n, b in self.mult.items():
            if n not in self.off:
                y = y + ((y @ b.A.to(y.dtype).T) @ b.B.to(y.dtype).T) * b.scale
        return y


def inject(model, cfg):
    mods = []
    for layer in model.model.layers:
        for name in cfg["adapter"]["target_modules"]:
            parent = layer.mlp if name in ("gate_proj", "up_proj", "down_proj") else layer.self_attn
            lin = getattr(parent, name)
            wrapped = LoRALinear(lin)
            setattr(parent, name, wrapped)
            mods.append(wrapped)
    return mods


def add_branch(mods, name, rank, cfg, device):
    for m in mods:
        m.branches[name] = Branch(m.base.in_features, m.base.out_features, rank,
                                  cfg["adapter"]["alpha_over_r"]).to(device)


def add_mult_branch(mods, name, rank, cfg, device):
    for m in mods:
        m.mult[name] = Branch(m.base.out_features, m.base.out_features, rank,
                              cfg["adapter"]["alpha_over_r"]).to(device)


def branch_params(mods, names):
    return [p for m in mods for n in names for p in (m.branches[n] if n in m.branches else m.mult[n]).parameters()]


def clear_branches(mods):
    for m in mods:
        m.branches = nn.ModuleDict()
        m.mult = nn.ModuleDict()
        m.off = set()
        m.delta = None


def branch_state(mods, names):
    return {f"{i}.{n}.{k}": getattr(m.branches[n], k).detach().cpu().clone()
            for i, m in enumerate(mods) for n in names for k in ("A", "B")}


def load_branch_state(mods, sd, cfg, device):
    names = sorted({k.split(".")[1] for k in sd})
    for n in names:
        rank = sd[f"0.{n}.A"].shape[0]
        add_branch(mods, n, rank, cfg, device)
        for i, m in enumerate(mods):
            m.branches[n].A.data.copy_(sd[f"{i}.{n}.A"])
            m.branches[n].B.data.copy_(sd[f"{i}.{n}.B"])
    return names


# ------------------------------------------------------------------------------ scoring

def label_logits(model, head, ids, mask, pos, n_labels_per_ex):
    """Logits over each example's valid letters: [B, n_max] with -inf padding beyond its label count."""
    h = model.model(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1]
    with torch.autocast("cuda", enabled=False):
        logits = h.float() @ head.T
    n = torch.as_tensor(n_labels_per_ex, device=logits.device)
    valid = torch.arange(head.shape[0], device=logits.device)[None, :] < n[:, None]
    return logits.masked_fill(~valid, float("-inf"))


@torch.no_grad()
def evaluate(model, head, data: TaskData, pad_id, bs, device):
    preds, probs = [], []
    for b in range(0, len(data), bs):
        ids, mask, pos = batchify(data.ids[b:b + bs], pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = label_logits(model, head, ids, mask, pos, [data.n_labels] * ids.shape[0])
        lg = lg[:, :data.n_labels]
        preds.append(lg.argmax(-1).cpu())
        probs.append(F.softmax(lg, -1).cpu())
    preds = torch.cat(preds)
    return float((preds == data.labels).float().mean()), preds.numpy(), torch.cat(probs).numpy()
