#!/usr/bin/env python3
"""One (order, seed) of the generative CL benchmark: train the first task once (shared adapter), then every method
over tasks 2..4 from that same start, plus the multi-task reference. Same method definitions as run_order.py:

  seq        rank-16 adapter per task, target-only LM loss
  replay     rank-16; + KL(teacher || model) on a 16-example stored batch every 4th step
  delayed    rank-16; + that KL on the last 25% of each task's steps
  repair     rank-8 learner for 75% of steps, frozen; new rank-8 repair for the last 25% (LM loss + KL)
  end_joint  rank-8 learners, plain training; after the last task ONE rank-24 repair trained with KL on the stored
             examples of all 4 tasks (incl. the last), for the same total repair steps direct repair uses
  mtl        reference: one rank-64 adapter trained on all four tasks shuffled together

KL is token-level over the full vocabulary on the stored examples' target tokens. The teacher for task j is the model
exactly as it was right after task j, rebuilt by switching off every branch added later (see clb/gen.py).
Scores per task in [0, 1]: gsm8k exact match, triviaqa exact match, hellaswag accuracy, wmt14 chrF/100.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clb.core import ROOT, add_branch, branch_params, branch_state, clear_branches, inject, load_branch_state  # noqa
from clb.gen import GenData, distill_loss, evaluate_task, lm_batch, target_logits  # noqa
from clb.datafree import build_diff, build_gen, build_lwf  # noqa

DF = ("df_lwf", "df_gen", "df_diff")    # data-free: delayed timing, no stored old-task examples (clb/datafree.py)


class Ctx:
    pass


def phase_of(method, s, phase_start):
    if method == "repair":
        return ("repair", True) if s >= phase_start else ("learn", False)
    if method == "replay":
        return "learn", (s + 1) % 4 == 0
    if method == "delayed" or method in DF:
        return "learn", s >= phase_start
    return "learn", False


def n_steps(c, task):
    T = c.cfg["training"]
    return math.ceil(len(c.train[task]) * T["epochs_per_task"] / T["batch_size"])   # epochs are concatenated


def train_task(c, method, k, task, anchors, log):
    T = c.cfg["training"]
    data, bs = c.train[task], T["batch_size"]
    steps = n_steps(c, task)
    phase_start = steps - round(T["protect_fraction"] * steps)
    rng = np.random.default_rng(1000 * c.seed + k)
    torch.manual_seed(1000 * c.seed + k)
    perm = np.concatenate([rng.permutation(len(data)) for _ in range(T["epochs_per_task"])]).tolist()
    A = c.cfg["adapter"]
    if method in ("repair", "end_joint"):
        learn = f"t{k}_learn"; add_branch(c.mods, learn, A["learn_rank"], c.cfg, c.dev)
    else:
        learn = f"t{k}_main"; add_branch(c.mods, learn, A["total_rank_per_task"], c.cfg, c.dev)
    rep = f"t{k}_repair"

    def opt_for(name):
        return torch.optim.AdamW(branch_params(c.mods, [name]), lr=T["lr"], betas=tuple(T["betas"]),
                                 weight_decay=T["weight_decay"])

    def activate(name):
        for m in c.mods:
            for n, b in m.branches.items():
                b.requires_grad_(n == name)

    opts, current = {"learn": opt_for(learn)}, None
    st = {"steps": steps, "learn_steps": 0, "repair_steps": 0, "protected_steps": 0}
    for s in range(steps):
        phase, protect = phase_of(method, s, phase_start)
        if phase == "repair" and "repair" not in opts:
            add_branch(c.mods, rep, A["repair_rank"], c.cfg, c.dev)
            opts["repair"] = opt_for(rep)
        name = rep if phase == "repair" else learn
        if name != current:
            activate(name); current = name
        st[f"{phase}_steps"] += 1
        idx = perm[s * bs:(s + 1) * bs]
        ids, mask, pos, lab = lm_batch([data.prompt[i] for i in idx], [data.target[i] for i in idx], c.pad, c.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, y, _ = target_logits(c.model, ids, mask, pos, lab)
            ce = F.cross_entropy(lg.float(), y)
            loss, kl = ce, None
            if anchors and protect:
                kl = distill_loss(c, anchors, rng)
                loss = loss + T["lambda_old"] * kl
                st["protected_steps"] += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_params(c.mods, [name]), T["grad_clip"])
        opts[phase].step(); opts[phase].zero_grad(set_to_none=True)
        if s % 25 == 0 or s == steps - 1:
            log.append({"task": task, "step": s, "phase": phase, "ce": float(ce.detach()),
                        "kl": None if kl is None else float(kl.detach())})
    for m in c.mods:
        m.requires_grad_(False)
    return st


def end_repair(c, order, anchors, log):
    """One rank-3*repair_rank branch after the last task; KL on all tasks' stored examples (teacher of the last task
    = the current model, so its examples only anchor it). Steps = what direct repair spends over tasks 2..4."""
    T, A = c.cfg["training"], c.cfg["adapter"]
    steps = sum(round(T["protect_fraction"] * n_steps(c, t)) for t in order[1:])
    rng = np.random.default_rng(1000 * c.seed + 900)
    torch.manual_seed(1000 * c.seed + 900)
    add_branch(c.mods, "end_all", 3 * A["repair_rank"], c.cfg, c.dev)
    for m in c.mods:
        for n, b in m.branches.items():
            b.requires_grad_(n == "end_all")
    opt = torch.optim.AdamW(branch_params(c.mods, ["end_all"]), lr=T["lr"], betas=tuple(T["betas"]),
                            weight_decay=T["weight_decay"])
    for s in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            kl = distill_loss(c, anchors, rng)
        kl.backward()
        torch.nn.utils.clip_grad_norm_(branch_params(c.mods, ["end_all"]), T["grad_clip"])
        opt.step(); opt.zero_grad(set_to_none=True)
        if s % 25 == 0 or s == steps - 1:
            log.append({"task": "end_all", "step": s, "phase": "end_repair", "kl": float(kl.detach())})
    for m in c.mods:
        m.requires_grad_(False)
    return {"repair_steps": steps, "rank": 3 * A["repair_rank"]}


def eval_seen(c, tasks, outs=None, tag=""):
    res = {}
    for t in tasks:
        score, raw = evaluate_task(c, t, c.test[t])
        res[t] = score
        if outs is not None:
            outs[f"{tag}{t}"] = raw
    return res


def run_method(c, method, order, t0, out_dir):
    f = out_dir / f"{method}.json"
    if f.exists():
        print(f"  {method}: done"); return
    t_start = time.time()
    clear_branches(c.mods)
    load_branch_state(c.mods, t0["state"], c.cfg, c.dev)
    for m in c.mods:
        m.requires_grad_(False)
    anchors = {order[0]: {"data": c.anchors[order[0]], "teacher": sorted(c.mods[0].branches)}}
    rows = [{"after": order[0], "acc": {order[0]: t0["score"]}}]
    log, stats, outs, extra = [], [], {}, {}
    n_df = c.cfg["data"]["anchors_per_task"]
    rng_df = np.random.default_rng(1000 * c.seed + 700)
    pseudo, dump = {}, {}

    def snapshot_examples(t, teacher, previous):        # right after task t: the snapshot writes its own examples
        if method == "df_gen":
            d, sc = build_gen(c, t, teacher, n_df), None
        else:
            d, sc = build_diff(c, t, teacher, previous, n_df)
        pseudo[t] = {"data": d, "teacher": teacher}
        dump[t] = {"n": len(d), "examples": d.texts[:20], "diff_scores": sc[:20] if sc else None}
        print(f"  {method}: {len(d)} self-generated examples for {t}", flush=True)

    if method in ("df_gen", "df_diff"):
        snapshot_examples(order[0], anchors[order[0]]["teacher"], [])
    for k, task in enumerate(order[1:], start=1):
        if method == "df_lwf":                          # rebuilt for each new task from its own inputs
            use = {}
            for t in order[:k]:
                d = build_lwf(c, t, anchors[t]["teacher"], c.train[task].rows, n_df, rng_df)
                if len(d):
                    use[t] = {"data": d, "teacher": anchors[t]["teacher"]}
                dump[f"{t}<-{task}"] = {"n": len(d), "examples": d.texts[:20]}
        elif method in ("df_gen", "df_diff"):
            use = {t: v for t, v in pseudo.items() if len(v["data"])}
        else:
            use = dict(anchors)
        before = sorted(c.mods[0].branches)
        stats.append(train_task(c, method, k, task, use, log))
        stats[-1]["protect_examples"] = {t: len(v["data"]) for t, v in use.items()}
        anchors[task] = {"data": c.anchors[task], "teacher": sorted(c.mods[0].branches)}
        if method in ("df_gen", "df_diff") and k < len(order) - 1:
            snapshot_examples(task, anchors[task]["teacher"], before)
        last = k == len(order) - 1
        ev = eval_seen(c, order[:k + 1], outs if last else None, "final_")
        rows.append({"after": task, "acc": ev})
        print(f"  {method} after {task}: " + " ".join(f"{t}={v:.3f}" for t, v in ev.items()), flush=True)
    if method == "end_joint":
        extra["pre_repair"] = rows[-1]
        extra["end_repair"] = end_repair(c, order, anchors, log)
        ev = eval_seen(c, order, outs, "final_")
        rows[-1] = {"after": order[-1], "acc": ev}
        print(f"  {method} after end repair: " + " ".join(f"{t}={v:.3f}" for t, v in ev.items()), flush=True)
    if dump:
        json.dump(dump, open(out_dir / f"{method}_pseudo.json", "w"), indent=1)
    json.dump(outs, open(out_dir / f"{method}_outputs.json", "w"))
    torch.save(branch_state(c.mods, sorted(c.mods[0].branches)), out_dir / f"{method}_adapters.pt")
    json.dump({"method": method, "order": order, "seed": c.seed, "acc_matrix": rows, "task_stats": stats, "log": log,
               "seconds": round(time.time() - t_start, 1), **extra}, open(f, "w"), indent=1)


def run_mtl(c, order, out_dir):
    f = out_dir / "mtl.json"
    if f.exists():
        print("  mtl: done"); return
    t_start, T = time.time(), c.cfg["training"]
    clear_branches(c.mods)
    torch.manual_seed(1000 * c.seed + 99)
    add_branch(c.mods, "mtl", 4 * c.cfg["adapter"]["total_rank_per_task"], c.cfg, c.dev)
    for m in c.mods:
        m.branches["mtl"].requires_grad_(True)
    opt = torch.optim.AdamW(branch_params(c.mods, ["mtl"]), lr=T["lr"], weight_decay=T["weight_decay"])
    rng = np.random.default_rng(1000 * c.seed + 99)
    items = [(t, i) for _ in range(T["epochs_per_task"]) for t in order for i in range(len(c.train[t]))]
    rng.shuffle(items)
    bs = T["batch_size"]
    for s in range(math.ceil(len(items) / bs)):
        ch = items[s * bs:(s + 1) * bs]
        ids, mask, pos, lab = lm_batch([c.train[t].prompt[i] for t, i in ch], [c.train[t].target[i] for t, i in ch],
                                       c.pad, c.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, y, _ = target_logits(c.model, ids, mask, pos, lab)
            loss = F.cross_entropy(lg.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_params(c.mods, ["mtl"]), T["grad_clip"])
        opt.step(); opt.zero_grad(set_to_none=True)
    for m in c.mods:
        m.requires_grad_(False)
    ev = eval_seen(c, order)
    print("  mtl: " + " ".join(f"{t}={v:.3f}" for t, v in ev.items()), flush=True)
    json.dump({"method": "mtl", "order": order, "seed": c.seed, "acc_matrix": [{"after": "all", "acc": ev}],
               "seconds": round(time.time() - t_start, 1)}, open(f, "w"), indent=1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--out-root", default="runs_gen")
    ap.add_argument("--config", default="config_gen.json")
    ap.add_argument("--max-train", type=int, default=None, help="smoke tests: cap train and test examples per task")
    a = ap.parse_args()
    c = Ctx()
    c.cfg = json.loads((ROOT / a.config).read_text())
    c.seed, c.dev = a.seed, "cuda"
    order = c.cfg["orders"][a.order]
    out_dir = ROOT / a.out_root / (f"{a.order}_s{a.seed}" + ("_smoke" if a.max_train else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    c.tok = AutoTokenizer.from_pretrained(c.cfg["model"])
    c.pad = c.tok.eos_token_id
    c.model = AutoModelForCausalLM.from_pretrained(c.cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa").to(c.dev).eval()
    c.model.requires_grad_(False)
    c.model.generation_config.temperature = None
    c.model.generation_config.top_p = None
    c.mods = inject(c.model, c.cfg)
    c.train = {t: GenData(c.tok, t, "train", c.cfg) for t in order}
    c.test = {t: GenData(c.tok, t, "test", c.cfg) for t in order}
    c.anchors = {t: GenData(c.tok, t, "anchors", c.cfg) for t in order}
    if a.max_train:
        for d in list(c.train.values()) + list(c.test.values()):
            n = a.max_train
            d.rows, d.prompt, d.target = d.rows[:n], d.prompt[:n], d.target[:n]
            if hasattr(d, "endings"):
                d.endings = d.endings[:n]
    json.dump({"order": order, "seed": a.seed, "config": c.cfg, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0)}, open(out_dir / "run_config.json", "w"), indent=1)

    t0f = out_dir / "task0.pt"
    if not t0f.exists():
        log = []
        st = train_task(c, "seq", 0, order[0], None, log)
        outs = {}
        score = eval_seen(c, order[:1], outs)[order[0]]
        torch.save({"state": branch_state(c.mods, sorted(c.mods[0].branches)), "score": score, "stats": st,
                    "log": log}, t0f)
    t0 = torch.load(t0f, weights_only=False)
    print(f"task0 {order[0]} score {t0['score']:.3f}", flush=True)
    for m in (a.methods.split(",") if a.methods else c.cfg["methods"]):
        run_mtl(c, order, out_dir) if m == "mtl" else run_method(c, m, order, t0, out_dir)
    print("done")


if __name__ == "__main__":
    main()
