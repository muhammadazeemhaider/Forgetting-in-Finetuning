"""Shared loading and definitions for the LoRA vs full fine-tuning analysis.

All numbers come from the measurements in ``results/`` (one row per evaluation of
one fine-tuned checkpoint), so everything here runs on a CPU in seconds.

Definitions (percentage points, pp):
- learning   = fine-tuned - base accuracy on the target task
- forgetting = mean of (fine-tuned - base) over the non-target tasks; negative = forgot
- tie        = |difference in forgetting| < 1 pp
- matched    = |difference in learning| <= 2 pp
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

METHODS = ["Full FT", "LoRA r16", "LoRA r64", "LoRA r256"]
LORA = METHODS[1:]
TIE_PP = 1.0
MATCH_PP = 2.0

REGIME_LABELS = {
    "rr_tuned": "Round robin, AdamW, tuned LR (7 models x 4 targets)",
    "rr_single": "Round robin, AdamW, one LR per method",
    "rr_l2": "Round robin, Llama-2-7B, AdamW 8-bit",
    "rr_l2_all": "Round robin, Llama-2-7B, all LRs of the sweep",
    "mm1_l2": "MetaMathQA 1 epoch, Llama-2-7B, AdamW 8-bit",
    "mm4": "MetaMathQA 4 epochs, Lion (Biderman et al. recipe)",
}

# Training diverged (loss spikes to 15.8 / 31.6); excluded from every comparison.
DIVERGED = {("mm4", "Qwen2.5-3B", "LoRA r16"), ("mm4", "Qwen2.5-3B", "LoRA r64")}


def load_runs() -> pd.DataFrame:
    """One row per fine-tuned checkpoint: learning, forgetting, per-task deltas."""
    df = pd.read_csv(RESULTS / "RUN_SUMMARY.csv")
    df["epoch"] = df["epoch"].fillna(0).astype(int)  # 0 = single final checkpoint
    keep = [(r, m, me) not in DIVERGED for r, m, me in zip(df.regime, df.model, df.method)]
    return df[keep].reset_index(drop=True)


def load_long() -> pd.DataFrame:
    """Every individual evaluation: one row per checkpoint x eval task."""
    return pd.read_csv(RESULTS / "ALL_LEARN_FORGET_LONG.csv")


def verdict(full_ft_forget: float, lora_forget: float) -> str:
    """Which method forgets less. Forgetting is negative, so larger is better."""
    diff = full_ft_forget - lora_forget
    if abs(diff) < TIE_PP:
        return "tie"
    return "Full FT forgets less" if diff > 0 else "LoRA forgets less"
