# Reference implementation (not the original experiment code)

> **Read this first.** The scripts in this folder are a **re-implementation written after the
> experiments** to document how the models were fine-tuned, evaluated and measured. They are
> **not** the scripts that produced the numbers in [`../results/`](../results/); the original
> cluster pipeline is not included in this repository. They have only been checked with a CPU
> smoke test on a tiny randomly initialised model (`test_reference.py`) and have **not** been
> re-run at full scale, so results from them may differ from the reported ones.

| File | What it documents |
|---|---|
| `finetune.py` | Full fine-tuning or LoRA (rank r, alpha = 2r, dropout 0.05, all attention and MLP projections) with AdamW, 8-bit AdamW or Lion; completion-only loss; lm-eval-style or Llama-2 chat-style prompts; cosine schedule with warmup; epoch snapshots; LoRA merged before saving |
| `evaluate.sh` | lm-evaluation-harness on vLLM, bf16, default few-shot settings, for the round-robin suite and the Biderman et al. suite |
| `update_norms.py` | Weight displacement of each fine-tune: relative Frobenius norm, stable rank and top singular value of ΔW over the 7 LoRA-targeted projections (LoRA computed exactly from the low-rank factors) |
| `test_reference.py` | CPU smoke test: training and saving under every method/optimizer pair, prompt masking, Lion's sign steps, and agreement of the adapter-only norm computation with the dense one |

The intruder-dimension analysis in [`../intruder_dimensions/`](../intruder_dimensions/) **is**
the original code used for the reported intruder counts.

## Settings used in the experiments

| Regime | Optimizer | LR (full FT / LoRA r16, r64, r256) | Steps x global batch | Text format |
|---|---|---|---|---|
| Round robin (7 models x 4 targets) | AdamW (8-bit for full FT of the ~3B models) | tuned per task and method on Llama-3.2-3B (67-run sweep), then transferred | 1,404 x 16 | lm-eval prompt formats |
| MetaMathQA, 1 epoch (Llama-2-7B) | 8-bit AdamW | 1e-5 / 1e-4, 1e-4, 5e-5 | 515 x 768 | GSM8K eval template |
| MetaMathQA, 4 epochs (Llama-2-7B) | Lion, β = (0.9, 0.95), no weight decay | 1e-5 / 1e-4, 1e-4, 5e-5 | 2,056 x 768, cosine, 10% warmup, snapshots at epochs 1/2/4 | Llama-2 chat `[INST] q [/INST] r` |
| MetaMathQA, 4 epochs (4 small models) | Lion | 2e-5 / 4e-4, 2e-4, 5e-5 | 2,056 x 768 | Llama-2 chat |

## Smoke test

```bash
pip install -r ../requirements.txt
python reference/test_reference.py   # from finetuning_forgetting/; CPU, < 1 minute
```
