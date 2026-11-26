#!/usr/bin/env python3
"""One (order, seed): train the first task once (shared rank-12 adapter = the 'acquired' history), then
run every method over tasks 2..4 from that same starting point, and the multi-task reference.

Methods for each later task k (total trainable rank 12 per task, alpha/r = 1, AdamW 1e-4 constant, bs 16,
one epoch; 'protected' steps use a 16-example anchor batch from ALL previous tasks):
  seq          rank-12 branch, CE only
  replay       rank-12, CE every step + KL(teacher||model) on anchors every 4th step (25% of steps)
  full_replay  rank-12, CE + KL every step (4x the old-task exposure; not exposure-matched)
  joint        rank-8 + rank-4 branches trained together, replay's schedule
  delayed      rank-12, CE for the first 75% of steps, CE + KL for the last 25%
  repair       THE METHOD (compensated LoRA, post-training repair): rank-8 learner, CE for 75% of steps;
               freeze it; new rank-4 repair branch trained with KL + CE for the last 25%
  olora        rank-12, CE + 0.5 * sum |A_prev A_new^T| over all previous branches (released O-LoRA trainer), no memory
  end_joint    learners only during the sequence (rank learn_rank, CE on all steps); AFTER the last task one repair
               branch (rank 3 x repair_rank) trained with KL on the stored anchors of ALL 4 tasks (incl. the last)
  end_sum      as end_joint, but one repair per task (4 x rank 3*repair_rank/4), each trained alone on that task's
               anchors against the final learner stack, then all four SUMMED and applied together
  end_mult     as end_sum, but each repair is multiplicative on the layer output, y <- (I + B_j A_j) y, and the four
               are composed as a PRODUCT  W_eff -> (I+R_4)(I+R_3)(I+R_2)(I+R_1) W_eff
               End-repair steps = the total repair steps direct repair uses over tasks 2..4 (same anchor batch 16),
               so stored-memory exposure is matched; only its timing and the protected set change.
  mtl          reference: one rank-48 adapter trained on all four tasks shuffled together (not sequential)
Teachers: each task's anchor label-distribution recorded with THAT method's model at the end of that task.
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
from clb.core import (ROOT, TaskData, add_branch, batchify, branch_params, branch_state, clear_branches,  # noqa
                      add_mult_branch, evaluate, inject, label_logits, letter_ids, load_branch_state, load_config)


class Ctx:
    pass


def kl_loss(c, anchors, rng):
    """Mean KL(teacher || model) over a 16-example anchor batch pooled from all previous tasks."""
    pool = [(t, i) for t in anchors for i in range(len(anchors[t]["ids"]))]
    pick = [pool[j] for j in rng.choice(len(pool), c.cfg["training"]["anchor_batch_size"], replace=False)]
    seqs = [anchors[t]["ids"][i] for t, i in pick]
    nl = [anchors[t]["n"] for t, _ in pick]
    ids, mask, pos = batchify(seqs, c.pad, c.dev)
    lg = label_logits(c.model, c.head, ids, mask, pos, nl)
    loss = 0.0
    for k, (t, i) in enumerate(pick):
        p = torch.as_tensor(anchors[t]["teacher"][i], device=c.dev)
        logq = F.log_softmax(lg[k, :anchors[t]["n"]], -1)
        loss = loss + (p * (torch.log(p.clamp_min(1e-12)) - logq)).sum()
    return loss / len(pick)


def olora_loss(c, prev_names, new_name):
    tot = 0.0
    for m in c.mods:
        An = m.branches[new_name].A
        for pn in prev_names:
            tot = tot + (m.branches[pn].A.detach() @ An.T).abs().sum()
    return c.cfg["training"]["olora_lambda"] * tot


TIES_METHODS = ("ties_only", "ties_repair", "ties_repair_lwf")


def ties_params(c, method):
    t = c.cfg.get("ties", {})
    d = t.get(method, t.get("default", {"density": 0.2, "lambda": 1.0}))
    return (c.ties_override or {}).get("density", d["density"]), (c.ties_override or {}).get("lambda", d["lambda"])


@torch.no_grad()
def ties_merge(c, method, learn_names, k):
    """TIES-merge, per adapted matrix, the OLD update (dense merged delta + every frozen branch from earlier tasks)
    with the NEW learner update (B A of learn_names). Trim each to its top `density` entries by magnitude,
    elect a sign per entry by summed mass, average the entries that agree (disjoint mean), scale by lambda.
    The result replaces both as one frozen dense delta (Yadav et al. 2023, TIES-Merging)."""
    density, lam = ties_params(c, method)
    stats = []
    for m in c.mods:
        old = m.delta.float().clone() if m.delta is not None else torch.zeros(m.base.out_features, m.base.in_features, device=c.dev)
        new = torch.zeros_like(old)
        for n, b in m.branches.items():
            upd = (b.B.float() @ b.A.float()) * b.scale
            if n in learn_names:
                new += upd
            else:
                old += upd
        trimmed = []
        for v in (old, new):
            kk = max(1, int(round(density * v.numel())))
            if kk >= v.numel():
                trimmed.append(v)
            else:
                thr = v.abs().flatten().kthvalue(v.numel() - kk + 1).values
                trimmed.append(torch.where(v.abs() >= thr, v, torch.zeros_like(v)))
        T = torch.stack(trimmed)
        sign = torch.sign(T.sum(0))
        agree = (torch.sign(T) == sign) & (T != 0)
        merged = lam * (T * agree).sum(0) / agree.sum(0).clamp(min=1)
        stats.append(float((torch.sign(old) * torch.sign(new) < 0).float().mean()))
        m.delta = merged.to(torch.bfloat16)
        m.branches = torch.nn.ModuleDict()                    # old branches + learner now live in the delta
    return {"density": density, "lambda": lam, "sign_conflict_frac": float(np.mean(stats))}


REPAIR_METHODS = ("repair", "alternating", "shared_repair")       # learner then separate repair branch
SHARED = ("shared_seq", "shared_replay", "shared_delayed", "shared_repair")
END_METHODS = ("end_joint", "end_sum", "end_mult")                 # learners only; all repair after the last task


def phase_of(method, s, n_steps, phase_start, n_cycles):
    """'learn' or 'repair' for repair-style methods (who trains), and whether the step uses old anchors."""
    if method == "alternating":
        b = [round(i * n_steps / n_cycles) for i in range(n_cycles + 1)]
        for a, e in zip(b[:-1], b[1:]):
            if a <= s < e:
                return ("repair", True) if s >= a + round(0.75 * (e - a)) else ("learn", False)
    if method in ("repair", "shared_repair"):
        return ("repair", True) if s >= phase_start else ("learn", False)
    if method in ("replay", "joint", "shared_replay"):
        return "learn", (s + 1) % 4 == 0
    if method == "full_replay":
        return "learn", True
    if method in ("delayed", "shared_delayed"):
        return "learn", s >= phase_start
    return "learn", False


def train_task(c, method, k, task, anchors, log):
    """One task for one method. Learner: a new per-task branch, or for shared_* the task-0 adapter 't0_main'
    which keeps training across every task (never frozen). Repair-style methods add a fresh rank-4 branch
    t{k}_repair, trained only in repair phases (KL on pooled old anchors + CE on the new task)."""
    T = c.cfg["training"]
    sched = ("repair" if method.startswith("repair_") or method in ("ties_repair", "ties_repair_lwf")
             else "seq" if method == "ties_only" else method)     # memory / TIES variants of direct repair
    data = c.train[task]
    bs = T["batch_size"]
    n_steps = math.ceil(len(data) / bs)
    n_prot = round(T["protect_fraction"] * n_steps)
    phase_start = n_steps - n_prot
    n_cycles = T.get("alternating_cycles", 5)
    g = torch.Generator().manual_seed(1000 * c.seed + k)
    order = torch.randperm(len(data), generator=g).tolist()
    rng = np.random.default_rng(1000 * c.seed + k)
    torch.manual_seed(1000 * c.seed + k)
    prev = sorted({n for m in c.mods[:1] for n in m.branches})
    r_all, r_l, r_r = c.cfg["adapter"]["total_rank_per_task"], c.cfg["adapter"]["learn_rank"], c.cfg["adapter"]["repair_rank"]
    rep_name = f"t{k}_repair"
    if method in SHARED:
        learn_names = ["t0_main"]                         # the running shared adapter W_{t-1}
    elif sched in ("joint", "repair", "alternating") or method in END_METHODS:
        add_branch(c.mods, f"t{k}_learn", r_l, c.cfg, c.dev)
        learn_names = [f"t{k}_learn"]
        if method == "joint":
            add_branch(c.mods, rep_name, r_r, c.cfg, c.dev)
            learn_names = [f"t{k}_learn", rep_name]
    else:
        add_branch(c.mods, f"t{k}_main", r_all, c.cfg, c.dev)
        learn_names = [f"t{k}_main"]

    def make_opt(names):
        return torch.optim.AdamW(branch_params(c.mods, names), lr=T["lr"], betas=tuple(T["betas"]),
                                 weight_decay=T["weight_decay"])

    def activate(names):
        for m in c.mods:
            for n, b in m.branches.items():
                b.requires_grad_(n in names)

    opts = {"learn": make_opt(learn_names)}
    current = None
    stats = {"steps": n_steps, "protected_steps": 0, "old_exposures": 0, "learn_steps": 0, "repair_steps": 0}
    for s in range(n_steps):
        phase, protect = phase_of(sched, s, n_steps, phase_start, n_cycles)
        if sched not in REPAIR_METHODS:
            phase = "learn"
        if phase == "repair" and "repair" not in opts:    # first repair phase: fresh rank-4 branch (B=0 -> no-op)
            if method in TIES_METHODS:                    # TIES(old adapters, learner) before repairing
                stats["ties"] = ties_merge(c, method, learn_names, k)
            add_branch(c.mods, rep_name, r_r, c.cfg, c.dev)
            opts["repair"] = make_opt([rep_name])        # optimizer states persist across alternating cycles
        names = [rep_name] if phase == "repair" else learn_names
        if names != current:
            activate(names)
            current = names
        opt = opts[phase]
        stats[f"{phase}_steps"] += 1
        idx = order[s * bs:(s + 1) * bs]
        ids, mask, pos = batchify([data.ids[i] for i in idx], c.pad, c.dev)
        y = data.labels[idx].to(c.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = label_logits(c.model, c.head, ids, mask, pos, [data.n_labels] * len(idx))
            ce = F.cross_entropy(lg[:, :data.n_labels], y)
            loss = ce
            kl = None
            if anchors and protect:
                kl = kl_loss(c, anchors, rng)
                loss = loss + T["lambda_old"] * kl
                stats["protected_steps"] += 1
                stats["old_exposures"] += T["anchor_batch_size"]
            if method == "olora" and prev:
                loss = loss + olora_loss(c, prev, learn_names[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_params(c.mods, names), T["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if s % 50 == 0 or s == n_steps - 1:
            log.append({"task": task, "step": s, "phase": phase, "ce": float(ce.detach()),
                        "kl": None if kl is None else float(kl.detach())})
    if method == "ties_only":                             # no repair: merge learner into the old update at task end
        stats["ties"] = ties_merge(c, method, learn_names, k)
    for m in c.mods:
        m.requires_grad_(False)
    return stats


def record_teacher(c, task):
    d = c.anchor_data[task]
    _, _, probs = evaluate(c.model, c.head, d, c.pad, c.cfg["eval_batch_size"], c.dev)
    return {"ids": d.ids, "n": d.n_labels, "teacher": probs}


def eval_seen(c, tasks):
    out = {}
    for t in tasks:
        acc, preds, _ = evaluate(c.model, c.head, c.test[t], c.pad, c.cfg["eval_batch_size"], c.dev)
        out[t] = {"acc": acc, "preds": preds}
    return out


MEMORY_VARIANTS = ("repair_last1", "repair_budget", "repair_prevteacher", "repair_lwf", "ties_repair_lwf")


def teacher_probs(c, ids, n):
    """Current model's label distribution (over n letters) for a list of tokenised prompts."""
    out = []
    bs = c.cfg["eval_batch_size"]
    with torch.no_grad():
        for b in range(0, len(ids), bs):
            i, m, p = batchify(ids[b:b + bs], c.pad, c.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = label_logits(c.model, c.head, i, m, p, [n] * len(ids[b:b + bs]))
            out.append(F.softmax(lg[:, :n].float(), -1).cpu().numpy())
    return np.concatenate(out)


def select_anchors(c, method, anchors, order, k):
    """Old-task memory the repair sees while learning task k (0-based; tasks order[:k] are old).
      repair             all old tasks, 128 stored examples each, teacher = model right after that task
      repair_last1       only task k-1's stored examples (sliding window of 1)
      repair_budget      fixed TOTAL of 128 stored examples, split evenly over the old tasks
      repair_prevteacher all stored examples, but teacher refreshed = model just before task k (compounding)
      repair_lwf         NO stored data: 128 of the NEW task's inputs rendered with each old task's prompt,
                         teacher = model just before task k (compounding, rehearsal-free)"""
    from clb.core import load_task, render
    old = order[:k]
    if method == "repair_last1":
        return {old[-1]: anchors[old[-1]]}
    if method == "repair_budget":
        # class-balanced subset: anchors are stored grouped by class, so a prefix would cover only a few classes
        per = c.cfg["data"]["anchors_per_task"] // len(old)
        out = {}
        for t in old:
            labs = c.anchor_data[t].labels.numpy()
            rng = np.random.default_rng(1000 * c.seed + 700 + k)
            by = {l: rng.permutation(np.where(labs == l)[0]).tolist() for l in np.unique(labs)}
            idx = []
            while len(idx) < per:                        # round-robin over classes
                for l in sorted(by):
                    if by[l] and len(idx) < per:
                        idx.append(by[l].pop())
            out[t] = {"ids": [anchors[t]["ids"][i] for i in idx], "n": anchors[t]["n"],
                      "teacher": anchors[t]["teacher"][idx]}
        return out
    if method == "repair_prevteacher":
        return {t: {"ids": anchors[t]["ids"], "n": anchors[t]["n"],
                    "teacher": teacher_probs(c, anchors[t]["ids"], anchors[t]["n"])} for t in old}
    if method in ("repair_lwf", "ties_repair_lwf"):
        rows = load_task(order[k], "train", c.cfg["data"].get("dir", "data"))
        rng = np.random.default_rng(1000 * c.seed + 500 + k)
        out = {}
        for j, t in enumerate(old):
            pick = rng.choice(len(rows), c.cfg["data"]["anchors_per_task"], replace=False)
            ids = []
            for i in pick:
                r = rows[i]
                ex = {"text": r["text"], "title": r.get("title") or r["text"][:80], "question": r.get("question") or ""}
                ids.append(c.tok(render(c.tok, t, ex, c.cfg["data"]["text_max_tokens"]),
                                 add_special_tokens=True)["input_ids"][-c.cfg["data"]["max_length"]:])
            n = anchors[t]["n"]
            out[f"pseudo_{t}"] = {"ids": ids, "n": n, "teacher": teacher_probs(c, ids, n)}
        return out
    return anchors


def end_repair(c, method, order, anchors, log):
    """Repair after the last task, protecting all tasks (incl. the last) with their stored anchors and the teacher
    recorded right after each task was learned. Returns stats and the names of the repair branches."""
    T = c.cfg["training"]
    bs, r_r = T["batch_size"], c.cfg["adapter"]["repair_rank"]
    n_total = sum(round(T["protect_fraction"] * math.ceil(len(c.train[t]) / bs)) for t in order[1:])
    rng = np.random.default_rng(1000 * c.seed + 900)
    torch.manual_seed(1000 * c.seed + 900)
    if method == "end_joint":
        jobs = [("end_all", list(order), n_total)]
    else:
        split = [n_total // len(order) + (j < n_total % len(order)) for j in range(len(order))]
        jobs = [(f"end_{t}", [t], n) for t, n in zip(order, split)]
    names = [n for n, _, _ in jobs]
    rank = 3 * r_r if method == "end_joint" else max(1, round(3 * r_r / len(order)))
    for name, tasks, n_steps in jobs:
        (add_mult_branch if method == "end_mult" else add_branch)(c.mods, name, rank, c.cfg, c.dev)
        for m in c.mods:                                   # train each repair alone on the final learner stack
            m.off = set(names) - {name}
            for n, b in list(m.branches.items()) + list(m.mult.items()):
                b.requires_grad_(n == name)
        opt = torch.optim.AdamW(branch_params(c.mods, [name]), lr=T["lr"], betas=tuple(T["betas"]),
                                weight_decay=T["weight_decay"])
        anc = {t: anchors[t] for t in tasks}
        for s in range(n_steps):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                kl = kl_loss(c, anc, rng)
            kl.backward()
            torch.nn.utils.clip_grad_norm_(branch_params(c.mods, [name]), T["grad_clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            if s % 50 == 0 or s == n_steps - 1:
                log.append({"task": name, "step": s, "phase": "end_repair", "ce": None, "kl": float(kl.detach())})
    for m in c.mods:
        m.off = set()
        m.requires_grad_(False)
    return {"repair_steps": n_total, "old_exposures": n_total * T["anchor_batch_size"], "rank_each": rank,
            "branches": names, "steps_each": [n for _, _, n in jobs]}, names


def run_method(c, method, order, t0_state, t0_teacher, out_dir):
    f = out_dir / f"{method}.json"
    if f.exists():
        print(f"  {method}: done"); return
    t0 = time.time()
    clear_branches(c.mods)
    load_branch_state(c.mods, t0_state, c.cfg, c.dev)
    for m in c.mods:
        m.requires_grad_(False)
    anchors = {order[0]: t0_teacher}
    acc_rows = [{"after": order[0], "acc": {order[0]: c.t0_acc}}]
    preds = {f"after0_{order[0]}": c.t0_preds}
    log, stats = [], []
    for k, task in enumerate(order[1:], start=1):
        use = select_anchors(c, method, anchors, order, k) if method in MEMORY_VARIANTS else anchors
        st = train_task(c, method, k, task, use, log)
        st["memory_tasks"] = sorted(use)
        st["memory_examples"] = int(sum(len(v["ids"]) for v in use.values()))
        stats.append(st)
        anchors[task] = record_teacher(c, task)
        ev = eval_seen(c, order[:k + 1])
        acc_rows.append({"after": task, "acc": {t: v["acc"] for t, v in ev.items()}})
        for t, v in ev.items():
            preds[f"after{k}_{t}"] = v["preds"]
        print(f"  {method} after {task}: " + " ".join(f"{t}={v['acc']:.3f}" for t, v in ev.items()), flush=True)
    extra = {}
    if method in END_METHODS:                              # final row = after the end repair; keep the pre-repair row
        extra["pre_repair"] = acc_rows[-1]
        extra["end_repair"], names = end_repair(c, method, order, anchors, log)
        if method != "end_joint":                          # diagnostic: each per-task repair applied alone
            extra["single_repair_acc"] = {}
            for n in names:
                for m in c.mods:
                    m.off = set(names) - {n}
                extra["single_repair_acc"][n] = {t: v["acc"] for t, v in eval_seen(c, order).items()}
            for m in c.mods:
                m.off = set()
        ev = eval_seen(c, order)
        acc_rows[-1] = {"after": order[-1], "acc": {t: v["acc"] for t, v in ev.items()}}
        for t, v in ev.items():
            preds[f"after{len(order) - 1}_{t}"] = v["preds"]
        print(f"  {method} after end repair: " + " ".join(f"{t}={v['acc']:.3f}" for t, v in ev.items()), flush=True)
    np.savez_compressed(out_dir / f"{method}_preds.npz", **preds)
    sd = branch_state(c.mods, sorted(c.mods[0].branches))
    if c.mods[0].delta is not None:
        sd = {"branches": sd, "deltas": [m.delta.cpu() for m in c.mods]}
    if len(c.mods[0].mult):
        sd = {"branches": sd, "mult": {f"{i}.{n}.{k}": getattr(m.mult[n], k).detach().cpu().clone()
                                       for i, m in enumerate(c.mods) for n in m.mult for k in ("A", "B")}}
    torch.save(sd, out_dir / f"{method}_adapters.pt")
    json.dump({"method": method, "order": order, "seed": c.seed, "acc_matrix": acc_rows, "task_stats": stats,
               "log": log, "seconds": round(time.time() - t0, 1), **extra}, open(f, "w"), indent=1)


def run_mtl(c, order, out_dir):
    f = out_dir / "mtl.json"
    if f.exists():
        print("  mtl: done"); return
    t0 = time.time()
    T = c.cfg["training"]
    clear_branches(c.mods)
    torch.manual_seed(1000 * c.seed + 99)
    add_branch(c.mods, "mtl", 4 * c.cfg["adapter"]["total_rank_per_task"], c.cfg, c.dev)
    for m in c.mods:
        m.branches["mtl"].requires_grad_(True)
    opt = torch.optim.AdamW(branch_params(c.mods, ["mtl"]), lr=T["lr"], weight_decay=T["weight_decay"])
    items = [(t, i) for t in order for i in range(len(c.train[t]))]
    rng = np.random.default_rng(1000 * c.seed + 99)
    rng.shuffle(items)
    bs = T["batch_size"]
    for s in range(math.ceil(len(items) / bs)):
        chunk = items[s * bs:(s + 1) * bs]
        ids, mask, pos = batchify([c.train[t].ids[i] for t, i in chunk], c.pad, c.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = label_logits(c.model, c.head, ids, mask, pos, [c.train[t].n_labels for t, _ in chunk])
            y = torch.stack([c.train[t].labels[i] for t, i in chunk]).to(c.dev)
            loss = F.cross_entropy(lg, y)          # invalid letters are -inf, so this is each task's restricted CE
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_params(c.mods, ["mtl"]), T["grad_clip"])
        opt.step(); opt.zero_grad(set_to_none=True)
    for m in c.mods:
        m.requires_grad_(False)
    ev = eval_seen(c, order)
    print("  mtl: " + " ".join(f"{t}={v['acc']:.3f}" for t, v in ev.items()), flush=True)
    json.dump({"method": "mtl", "order": order, "seed": c.seed, "acc_matrix": [{"after": "all", "acc": {t: v["acc"] for t, v in ev.items()}}],
               "seconds": round(time.time() - t0, 1)}, open(f, "w"), indent=1)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--methods", default=None, help="comma list; default = all in config")
    ap.add_argument("--max-train", type=int, default=None, help="smoke tests only: cap examples per task")
    ap.add_argument("--eval-split", default="test", help="test (reporting) or dev (hyperparameter selection)")
    ap.add_argument("--ties-density", type=float, default=None)
    ap.add_argument("--ties-lambda", type=float, default=None)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--data-dir", default=None, help="prepared data folder (default: config data.dir or data/)")
    ap.add_argument("--anchors", type=int, default=None,
                    help="keep only N class-balanced stored examples per task (subset of the prepared 128)")
    ap.add_argument("--task0-from", default=None,
                    help="reuse the first-task adapter from this runs root (teacher re-recorded on the kept anchors)")
    a = ap.parse_args()
    c = Ctx()
    c.cfg, c.seed, c.dev = load_config(), a.seed, "cuda"
    order = c.cfg["orders"][a.order]
    if a.data_dir:
        c.cfg["data"]["dir"] = a.data_dir
    c.ties_override = {k: v for k, v in (("density", a.ties_density), ("lambda", a.ties_lambda)) if v is not None}
    out_dir = ROOT / a.out_root / (f"{a.order}_s{a.seed}" + ("_smoke" if a.max_train else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(c.cfg["model"])
    c.tok = tok
    tok.padding_side = "left"
    c.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    c.model = AutoModelForCausalLM.from_pretrained(c.cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa").to(c.dev).eval()
    c.model.requires_grad_(False)
    c.mods = inject(c.model, c.cfg)
    c.head = c.model.lm_head.weight[letter_ids(tok).to(c.dev)].detach().float()
    c.train = {t: TaskData(tok, t, "train", c.cfg) for t in order}
    c.test = {t: TaskData(tok, t, a.eval_split, c.cfg) for t in order}
    c.anchor_data = {t: TaskData(tok, t, "anchors", c.cfg) for t in order}
    if a.anchors:                                          # class-balanced subset, same for every method and seed
        for t, d in c.anchor_data.items():
            labs = d.labels.numpy()
            rng = np.random.default_rng(c.cfg["data"]["data_seed"])
            by = {l: rng.permutation(np.where(labs == l)[0]).tolist() for l in np.unique(labs)}
            idx = []
            while len(idx) < a.anchors:                    # round-robin over classes
                for l in sorted(by):
                    if by[l] and len(idx) < a.anchors:
                        idx.append(by[l].pop())
            idx = sorted(idx)
            d.ids, d.labels = [d.ids[i] for i in idx], d.labels[idx]
    if a.max_train:
        for d in list(c.train.values()) + list(c.test.values()):
            d.ids, d.labels = d.ids[:a.max_train], d.labels[:a.max_train]
    json.dump({"order": order, "seed": a.seed, "config": c.cfg, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0)}, open(out_dir / "run_config.json", "w"), indent=1)

    # ---- shared first task -----------------------------------------------------------------------
    t0f = out_dir / "task0.pt"
    src = ROOT / a.task0_from / out_dir.name / "task0.pt" if a.task0_from else None
    if not t0f.exists() and src is not None and src.exists():   # same first-task adapter, teacher on kept anchors
        z = torch.load(src, weights_only=False)
        load_branch_state(c.mods, z["state"], c.cfg, c.dev)
        z["teacher"] = record_teacher(c, order[0])
        z["source"] = str(src)
        torch.save(z, t0f)
        clear_branches(c.mods)
    if t0f.exists():
        z = torch.load(t0f, weights_only=False)
        load_branch_state(c.mods, z["state"], c.cfg, c.dev)
        t0_teacher, c.t0_acc, c.t0_preds = z["teacher"], z["acc"], z["preds"]
    else:
        log = []
        st = train_task(c, "seq", 0, order[0], None, log)
        t0_teacher = record_teacher(c, order[0])
        ev = eval_seen(c, order[:1])
        c.t0_acc, c.t0_preds = ev[order[0]]["acc"], ev[order[0]]["preds"]
        torch.save({"state": branch_state(c.mods, sorted(c.mods[0].branches)), "teacher": t0_teacher,
                    "acc": c.t0_acc, "preds": c.t0_preds, "stats": st, "log": log}, t0f)
    t0_state = torch.load(t0f, weights_only=False)["state"]
    print(f"task0 {order[0]} acc {c.t0_acc:.3f}", flush=True)

    methods = a.methods.split(",") if a.methods else c.cfg["methods"]
    for m in methods:
        if m == "mtl":
            run_mtl(c, order, out_dir)
        else:
            run_method(c, m, order, t0_state, t0_teacher, out_dir)
    print("done")


if __name__ == "__main__":
    main()
