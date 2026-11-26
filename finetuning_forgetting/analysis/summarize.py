"""Recompute the headline tables of the LoRA vs full fine-tuning study.

Usage (from finetuning_forgetting/):
    python analysis/summarize.py            # print the tables
    python analysis/summarize.py --write    # also write results/SUMMARY_TABLES.md
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import spearmanr

from common import LORA, MATCH_PP, METHODS, REGIME_LABELS, RESULTS, load_runs, verdict


def method_means(runs: pd.DataFrame, regime: str, epoch: int = 0) -> pd.DataFrame:
    r = runs[(runs.regime == regime) & (runs.epoch == epoch)]
    t = r.groupby("method")[["learn", "forget", "worst"]].agg(
        learn=("learn", "mean"), forget=("forget", "mean"), worst=("worst", "min"))
    return t.reindex(METHODS).round(2)


def head_to_head(runs: pd.DataFrame, regime: str) -> pd.Series:
    """Full FT vs each LoRA rank on the same model x target x epoch."""
    r = runs[runs.regime == regime]
    keys = ["model", "target", "epoch"]
    ft = r[r.method == "Full FT"].set_index(keys)["forget"]
    out = []
    for m in LORA:
        lo = r[r.method == m].set_index(keys)["forget"]
        both = ft.to_frame("ft").join(lo.to_frame("lora"), how="inner")
        out += [verdict(a, b) for a, b in zip(both.ft, both.lora)]
    return pd.Series(out).value_counts().reindex(
        ["Full FT forgets less", "tie", "LoRA forgets less"], fill_value=0)


def matched_learning(runs: pd.DataFrame, regime: str, any_epoch: bool) -> pd.Series:
    """Compare forgetting only where the two methods learned about the same amount.

    any_epoch=True lets a LoRA checkpoint be matched to the full-FT snapshot (any epoch)
    of the same model whose learning is closest, which the MetaMathQA runs allow.
    """
    r = runs[runs.regime == regime]
    res = []
    for _, lo in r[r.method.isin(LORA)].iterrows():
        cand = r[(r.method == "Full FT") & (r.model == lo.model) & (r.target == lo.target)]
        if not any_epoch:
            cand = cand[cand.epoch == lo.epoch]
        if cand.empty:
            continue
        ft = cand.loc[(cand.learn - lo.learn).abs().idxmin()]
        res.append(verdict(ft.forget, lo.forget) if abs(ft.learn - lo.learn) <= MATCH_PP
                   else "not comparable")
    return pd.Series(res).value_counts().reindex(
        ["Full FT forgets less", "tie", "LoRA forgets less", "not comparable"], fill_value=0)


def hellaswag_table(runs: pd.DataFrame) -> pd.DataFrame:
    """HellaSwag change after math training: the one non-target task in every regime."""
    rows = [
        ("Round robin, 7 small models (AdamW)", "rr_tuned", None, 0),
        ("Round robin, Llama-2-7B (AdamW 8-bit)", "rr_l2", "Llama-2-7B", 0),
        ("MetaMathQA 1 epoch, Llama-2-7B (AdamW 8-bit)", "mm1_l2", "Llama-2-7B", 1),
        ("MetaMathQA, Llama-2-7B, epoch 1 (Lion)", "mm4", "Llama-2-7B", 1),
        ("MetaMathQA, Llama-2-7B, epoch 4 (Lion)", "mm4", "Llama-2-7B", 4),
    ]
    out = {}
    for label, regime, model, epoch in rows:
        r = runs[(runs.regime == regime) & (runs.epoch == epoch)
                 & runs.target.isin(["gsm8k", "metamathqa"])]
        if model:
            r = r[r.model == model]
        out[label] = r.groupby("method")["d_hellaswag"].mean().reindex(METHODS)
    return pd.DataFrame(out).T.round(2)


def displacement_correlations() -> pd.DataFrame:
    """Spearman rho between relative weight displacement and HellaSwag change, per method."""
    n = pd.read_csv(RESULTS / "NORMS_JOINED.csv")
    n = n[~n.coll.astype(bool)]  # drop the diverged runs
    rows = {}
    for m in METHODS:
        s = n[n.method == m]
        rho, p = spearmanr(s.rel_fro, s.hs)
        rows[m] = {"n": len(s), "rho_vs_hellaswag": round(rho, 2), "p": f"{p:.1g}"}
    rho, p = spearmanr(n.rel_fro, n.hs)
    rows["All methods pooled"] = {"n": len(n), "rho_vs_hellaswag": round(rho, 2), "p": f"{p:.1g}"}
    return pd.DataFrame(rows).T


def build() -> list[tuple[str, pd.DataFrame | pd.Series]]:
    runs = load_runs()
    h2h = pd.DataFrame({REGIME_LABELS[r]: head_to_head(runs, r)
                        for r in ["rr_tuned", "rr_single", "rr_l2", "mm4"]}).T
    matched = pd.DataFrame({
        REGIME_LABELS["rr_tuned"]: matched_learning(runs, "rr_tuned", any_epoch=False),
        REGIME_LABELS["mm4"]: matched_learning(runs, "mm4", any_epoch=True),
    }).T
    small = runs[(runs.regime == "mm4") & (runs.model != "Llama-2-7B")]
    l2 = runs[(runs.regime == "mm4") & (runs.model == "Llama-2-7B")]
    return [
        ("Head-to-head: who forgets less on the same model x target (x epoch)", h2h),
        ("Learning-matched comparisons", matched),
        ("Round robin, AdamW, tuned LR: method means (pp)", method_means(runs, "rr_tuned")),
        ("MetaMathQA 1 epoch, Llama-2-7B, AdamW 8-bit (pp)", method_means(runs, "mm1_l2", epoch=1)),
        ("MetaMathQA, Llama-2-7B, Lion, by epoch (pp)",
         l2.pivot_table(index=["method", "epoch"], values=["learn", "forget"]).reindex(METHODS, level=0).round(2)),
        ("MetaMathQA, 4 small models, Lion, epoch 4: method means (pp)",
         method_means(small.assign(regime="small"), "small", epoch=4)),
        ("HellaSwag change after math training (pp)", hellaswag_table(runs)),
        ("Weight displacement vs HellaSwag change (Spearman)", displacement_correlations()),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write results/SUMMARY_TABLES.md")
    args = ap.parse_args()
    md = ["# Summary tables\n", "Generated by `analysis/summarize.py` from `results/RUN_SUMMARY.csv` "
          "and `results/NORMS_JOINED.csv`. Qwen2.5-3B LoRA r16/r64 under Lion diverged and are excluded.\n"]
    for title, table in build():
        print(f"\n## {title}\n{table.to_string()}")
        md += [f"\n## {title}\n", table.to_markdown(), ""]
    if args.write:
        (RESULTS / "SUMMARY_TABLES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"\nWrote {RESULTS / 'SUMMARY_TABLES.md'}")


if __name__ == "__main__":
    main()
