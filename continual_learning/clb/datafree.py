"""Data-free protection examples for the generative benchmark: no example of an old task is ever stored.

Each builder returns a Pseudo set (prompt/target token lists) for one old task t, distilled against the exact model
right after task t (branch switch-off, see clb/gen.py) with delayed-distillation timing:

  lwf   inputs of the CURRENT task rewritten in task t's prompt; target = snapshot t's greedy answer (LwF-style)
  gen   snapshot t writes its own task-t examples from the instruction line (generative replay, LAMOL / SSR-style);
        sampled right after task t, kept if well-formed
  diff  like gen, but from 4x more candidates keep those whose mean token log-likelihood rose most from the model
        before task t to snapshot t:  log p_t(x) - log p_{t-1}(x)  -- samples carrying what task t changed
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from clb.gen import lm_batch, set_off, target_logits

INSTR = {
    "gsm8k": "Solve the math problem. End with '#### <answer>'.",
    "triviaqa": "Answer the trivia question.",
    "hellaswag": "Complete the passage with the most plausible continuation.",
    "wmt14": "Translate the English sentence into German.",
}
FIELD = {"gsm8k": ("Question", "Answer"), "triviaqa": ("Question", "Answer"),
         "hellaswag": ("Passage", "Continuation"), "wmt14": ("English", "German")}
MAX_ANS = {"gsm8k": 256, "triviaqa": 24, "hellaswag": 48, "wmt14": 128}


class Pseudo:
    def __init__(self, prompts, targets, texts):
        self.prompt, self.target, self.texts = prompts, targets, texts

    def __len__(self):
        return len(self.prompt)


def prompt_text(task, content):
    q, a = FIELD[task]
    return f"{INSTR[task]}\n{q}: {content}\n{a}:"


def content_of(row):
    """The task input inside a prepared prompt: the text after '<Field>: ' on the middle line(s)."""
    body = row["prompt"].split("\n", 1)[1].rsplit("\n", 1)[0]
    return body.split(": ", 1)[1] if ": " in body else body


def _with_branches(c, on):
    set_off(c.mods, (set(c.mods[0].branches) | set(c.mods[0].mult)) - set(on))


@torch.no_grad()
def _generate(c, prompts, max_new, sample, bs=48):
    out = []
    for b in range(0, len(prompts), bs):
        ps = prompts[b:b + bs]
        n = max(map(len, ps))
        ids = torch.full((len(ps), n), c.pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(ps):
            ids[i, n - len(p):] = torch.tensor(p); mask[i, n - len(p):] = 1
        kw = dict(do_sample=True, temperature=1.0, top_p=0.95) if sample else dict(do_sample=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = c.model.generate(input_ids=ids.to(c.dev), attention_mask=mask.to(c.dev), max_new_tokens=max_new,
                                 pad_token_id=c.pad, eos_token_id=c.tok.eos_token_id, **kw)
        out += c.tok.batch_decode(g[:, n:], skip_special_tokens=True)
    return out


def _finish(c, task, prompt_strs, answers):
    P, T, X = [], [], []
    for p, a in zip(prompt_strs, answers):
        a = a.strip() if task == "gsm8k" else a.strip().split("\n")[0].strip()
        if not a or (task == "gsm8k" and "####" not in a):
            continue
        if task == "gsm8k":
            a = a[:a.index("####")] + "#### " + a.split("####", 1)[1].strip().split("\n")[0]
        P.append(c.tok(p)["input_ids"]); T.append(c.tok(" " + a, add_special_tokens=False)["input_ids"] + [c.tok.eos_token_id])
        X.append(p + " " + a)
    return P, T, X


def build_lwf(c, old_task, teacher, new_rows, n, rng):
    """Current-task inputs in old_task's prompt; targets = old snapshot's greedy answers."""
    pick = rng.choice(len(new_rows), min(n, len(new_rows)), replace=False)
    prompts = [prompt_text(old_task, content_of(new_rows[i])) for i in pick]
    _with_branches(c, teacher)
    ans = _generate(c, [c.tok(p)["input_ids"] for p in prompts], MAX_ANS[old_task], sample=False)
    set_off(c.mods, ())
    P, T, X = _finish(c, old_task, prompts, ans)
    if old_task == "gsm8k" and len(P) < n // 4:          # keep it data-free even when '####' is rarely produced
        P, T, X = [], [], []
        for p, a in zip(prompts, ans):
            a = a.strip()
            if a:
                P.append(c.tok(p)["input_ids"]); T.append(c.tok(" " + a, add_special_tokens=False)["input_ids"] + [c.tok.eos_token_id]); X.append(p + " " + a)
    return Pseudo(P, T, X)


def _self_generate(c, task, teacher, n_cand):
    """Snapshot `teacher` writes n_cand task examples from the instruction + field name, then answers them."""
    q, a = FIELD[task]
    head = f"{INSTR[task]}\n{q}:"
    _with_branches(c, teacher)
    raw = _generate(c, [c.tok(head)["input_ids"]] * n_cand, 96 if task != "gsm8k" else 128, sample=True)
    contents = []
    for r in raw:
        body = r.split(f"\n{a}:")[0].split("\n")[0].strip()          # the input the model wrote, one line
        if len(body.split()) >= 3:
            contents.append(body)
    prompts = [prompt_text(task, x) for x in contents]
    ans = _generate(c, [c.tok(p)["input_ids"] for p in prompts], MAX_ANS[task], sample=False)
    set_off(c.mods, ())
    return _finish(c, task, prompts, ans)


@torch.no_grad()
def _mean_logp(c, branches, P, T, bs=32):
    _with_branches(c, branches)
    out = []
    for b in range(0, len(P), bs):
        # score the whole written example (input + answer) after the instruction line
        ids, mask, pos, lab = lm_batch(P[b:b + bs], T[b:b + bs], c.pad, c.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, y, sel = target_logits(c.model, ids, mask, pos, lab)
        lp = F.log_softmax(lg.float(), -1).gather(1, y[:, None])[:, 0]
        per = torch.zeros(ids.shape[0], device=c.dev).index_add_(0, sel.nonzero()[:, 0], lp) / sel.sum(1)
        out += per.tolist()
    set_off(c.mods, ())
    return out


def build_gen(c, task, teacher, n):
    P, T, X = _self_generate(c, task, teacher, 2 * n)
    return Pseudo(P[:n], T[:n], X[:n])


def build_diff(c, task, teacher, previous, n):
    P, T, X = _self_generate(c, task, teacher, 4 * n)
    if not P:
        return Pseudo([], [], []), []
    head = c.tok(INSTR[task])["input_ids"]                           # instruction = shared, unscored prefix
    full = [c.tok(x, add_special_tokens=False)["input_ids"][len(head) - 1:] for x in X]
    hp = [head] * len(full)
    d = [a - b for a, b in zip(_mean_logp(c, teacher, hp, full), _mean_logp(c, previous, hp, full))]
    keep = sorted(range(len(P)), key=lambda i: -d[i])[:n]
    return Pseudo([P[i] for i in keep], [T[i] for i in keep], [X[i] for i in keep]), [d[i] for i in keep]
