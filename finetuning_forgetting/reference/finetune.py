"""Reference implementation of the fine-tuning setup (full fine-tuning or LoRA).

NOTE: this is a re-implementation written after the experiments, to document the setup.
It is not the script that produced the numbers in results/, and it has not been re-run
at full scale. See reference/README.md.

Examples
--------
Round robin (AdamW, lm-eval prompt format, 1,404 steps x batch 16):
    python reference/finetune.py --model meta-llama/Llama-3.2-1B --task gsm8k \
        --method lora --rank 16 --optimizer adamw --lr 4e-4 --max-steps 1404 \
        --batch-size 16 --out checkpoints/llama1b_gsm8k_r16

MetaMathQA, Biderman et al. recipe (Lion, chat format, 4 epochs, batch 768, snapshots 1/2/4):
    python reference/finetune.py --model meta-llama/Llama-2-7b-hf --task metamathqa \
        --text-format chat --method full --optimizer lion --lr 1e-5 --epochs 4 \
        --batch-size 768 --micro-batch-size 8 --save-epochs 1 2 4 --out checkpoints/l2_mm_full
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- data

# (prompt, completion) pairs. The loss is computed on the completion only.
# Prompt formats follow lm-evaluation-harness so that training and evaluation match.
HF_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "train"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "train"),
    "hellaswag": ("Rowan/hellaswag", None, "train"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "train"),
    "metamathqa": ("meta-math/MetaMathQA", None, "train"),
}


def _hellaswag_clean(text: str) -> str:
    """lm-eval's HellaSwag preprocessing: drop bracketed tags and stray spaces."""
    import re
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def format_example(task: str, row: dict, text_format: str = "eval") -> tuple[str, str]:
    """Turn one dataset row into (prompt, completion)."""
    if task == "gsm8k":
        return f"Question: {row['question']}\nAnswer:", f" {row['answer']}"
    if task == "triviaqa":
        return f"Question: {row['question']}?\nAnswer:", f" {row['answer']['value']}"
    if task == "arc_easy":
        answer = row["choices"]["text"][row["choices"]["label"].index(row["answerKey"])]
        return f"Question: {row['question']}\nAnswer:", f" {answer}"
    if task == "hellaswag":
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        prompt = _hellaswag_clean(row["activity_label"] + ": " + ctx)
        return prompt, " " + _hellaswag_clean(row["endings"][int(row["label"])])
    if task == "metamathqa":
        if text_format == "chat":  # Llama-2 chat format, as in Biderman et al.
            return f"[INST] {row['query']} [/INST]", f" {row['response']}"
        return f"Question: {row['query']}\n\nAnswer:\n", row["response"]
    raise ValueError(f"unknown task {task}")


def load_pairs(task: str, text_format: str, limit: int | None, seed: int) -> list[tuple[str, str]]:
    from datasets import load_dataset
    name, config, split = HF_SOURCES[task]
    ds = load_dataset(name, config, split=split)
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    return [format_example(task, row, text_format) for row in ds]


@dataclass
class Tokenized:
    input_ids: list[int]
    labels: list[int]


def tokenize_pair(tokenizer, prompt: str, completion: str, max_len: int) -> Tokenized:
    """Mask the prompt with -100 so only completion tokens contribute to the loss."""
    p = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    c = tokenizer(completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (p + c)[:max_len]
    labels = ([-100] * len(p) + c)[:max_len]
    return Tokenized(ids, labels)


def collate(batch: list[Tokenized], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(b.input_ids) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b.input_ids)
        ids[i, :n] = torch.tensor(b.input_ids)
        labels[i, :n] = torch.tensor(b.labels)
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


# ---------------------------------------------------------------------- optimizers

class DecoupledLionW(torch.optim.Optimizer):
    """Lion (Chen et al., 2023) with decoupled weight decay.

    Every parameter moves by exactly lr * sign(interpolated momentum) each step, however
    weak its gradient. This is the property the study points to: for full fine-tuning it
    means billions of weights each take a full-size step every iteration.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1 - lr * wd)
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                update = m.mul(b1).add(p.grad, alpha=1 - b1).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


def make_optimizer(name: str, params, lr: float):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    if name == "adamw8bit":
        import bitsandbytes as bnb  # used for full FT of the ~3B and 7B models
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0)
    if name == "lion":
        return DecoupledLionW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    raise ValueError(name)


def cosine_with_warmup(optimizer, warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# ----------------------------------------------------------------------------- model

def build_model(args, dtype, model=None):
    """Load the base model (unless one is passed in) and wrap it with LoRA if requested."""
    if model is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    if args.method == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
            target_modules="all-linear",  # every attention + MLP projection, not lm_head
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return model


def save(model, tokenizer, out: Path, method: str) -> None:
    """LoRA adapters are merged into the base weights before saving, as before evaluation."""
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out)
    if method != "lora":
        model.save_pretrained(out, safe_serialization=True)
        return
    model.save_pretrained(out / "adapter")  # the unmerged adapter, saved before merging
    import copy
    merged = copy.deepcopy(model).merge_and_unload()  # copy: training may continue after a snapshot
    merged.save_pretrained(out, safe_serialization=True)


# ------------------------------------------------------------------------------ main

def train(args, pairs: list[tuple[str, str]] | None = None, base_model=None, tokenizer=None) -> list[float]:
    """Fine-tune and save. pairs / base_model / tokenizer can be injected for testing."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if pairs is None:
        pairs = load_pairs(args.task, args.text_format, args.limit, args.seed)
    data = [tokenize_pair(tokenizer, p, c, args.max_len) for p, c in pairs]

    model = build_model(args, dtype, base_model)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    micro = args.micro_batch_size or args.batch_size
    accum = max(1, args.batch_size // micro)
    loader = DataLoader(data, batch_size=micro, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(args.seed))
    steps_per_epoch = max(1, len(loader) // accum)
    total = args.max_steps or steps_per_epoch * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(args.optimizer, params, args.lr)
    sched = cosine_with_warmup(opt, int(args.warmup_ratio * total), total)

    out = Path(args.out)
    losses, step, micro_step = [], 0, 0
    model.train()
    while step < total:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            micro_step += 1
            if micro_step % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.item() * accum)
            if step % args.log_every == 0:
                print(f"step {step}/{total}  loss {losses[-1]:.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
            epoch_done = step % steps_per_epoch == 0 and step // steps_per_epoch in args.save_epochs
            if epoch_done and args.save:
                save(model, tokenizer, out / f"epoch-{step // steps_per_epoch}", args.method)
            if step >= total:
                break

    if args.save:
        save(model, tokenizer, out / "checkpoint-final", args.method)
        (out / "loss_history.json").write_text(json.dumps(losses))
    return losses


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=list(HF_SOURCES), required=True)
    ap.add_argument("--text-format", choices=["eval", "chat"], default="eval")
    ap.add_argument("--method", choices=["full", "lora"], required=True)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank (alpha = 2 * rank)")
    ap.add_argument("--optimizer", choices=["adamw", "adamw8bit", "lion"], default="adamw")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="overrides --epochs when > 0")
    ap.add_argument("--batch-size", type=int, default=16, help="global batch size")
    ap.add_argument("--micro-batch-size", type=int, default=0, help="per-step batch; rest is accumulation")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="use only N training examples")
    ap.add_argument("--save-epochs", type=int, nargs="*", default=[])
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-save", dest="save", action="store_false")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
