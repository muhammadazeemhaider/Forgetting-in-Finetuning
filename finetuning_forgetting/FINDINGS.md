# LoRA vs. Full Fine-Tuning: Learning and Forgetting Across Two Training Regimes

*Synthesis of every fine-tuning experiment in this project: the round-robin study (7 models × 4 target tasks, plus Llama-2-7B) and the MetaMathQA study (Llama-2-7B + 4 small models, Biderman et al. recipe). Compiled 2026-09-29.*

---

## 0. TL;DR

1. **The answer to "does LoRA forget less than full FT?" flips between the two regimes, and the flip is not a learning-matched artifact.**
   - **Round robin** (AdamW, batch 16, 1,404 steps, per-task tuned LRs): full FT forgets *less or the same*. In 84 same-cell comparisons full FT forgets less in 47, ties (within 1 pp) in 37, LoRA forgets less in **0**. At matched learning, LoRA never forgets less (24 full-FT wins, 31 ties).
   - **MetaMathQA 4-epoch** (Lion, batch 768, 2,056 steps, Biderman et al. recipe): LoRA forgets less in **39 of 39** comparisons, and in **21 of 21** learning-matched pairs. All 26 epoch-1/epoch-4 differences are significant under a paired item-level bootstrap.
2. **LoRA's behaviour is stable across regimes; full FT's is not.** On HellaSwag, the one non-target task measured everywhere, math-trained **full FT goes from −0.13 pp (AdamW) to −8.19 pp (Lion)** on the *same model, same dataset, same LR (1e-5) and same batch (768)*. LoRA moves only −0.6…−1.3 → −1.2…−3.0 pp.
3. **Leading explanation: the optimizer.** Lion applies `sign()` updates, so every parameter moves by the full learning rate every step regardless of how weak its gradient is. For full FT that is billions of coordinates. Measured directly: at identical LR, Lion moved Llama-2's full-FT weights **~7× further** than AdamW after one epoch (small models: 7–15× vs 3.4–7× for LoRA). Full FT went from fitting *worst* of all methods (AdamW, loss 0.233) to fitting *best* (Lion, 0.123). Per unit of movement, Lion full FT does ~5× more damage than Lion LoRA, while AdamW full FT does none. Optimizer is confounded with the training text format and schedule in the cleanest contrast we have, so this is a strong hypothesis, not a proven cause (§8, H1/H2; test in §9, T1).
4. **Things that hold in every regime:**
   - at equal LR, higher LoRA rank forgets more (30 vs 3 across three regimes);
   - raising LR beyond the optimum mostly buys forgetting, not learning (17 vs 1);
   - forgetting happens early (by epoch 1) and then plateaus;
   - the items that get forgotten are a shared pool of "fragile" items (8–60× chance overlap) regardless of method.
5. **Total weight displacement does not by itself predict forgetting across methods.** Under Lion, small-model LoRA can move weights *more* than full FT and still forget less. What matters is the structure and origin of the update: LoRA's is concentrated in a few directions, while Lion full FT's is a diffuse, high-rank drift that is far more damaging per unit (§7.2, H1″). Within a method, displacement predicts HellaSwag damage at ρ = −0.73 to −0.93.
6. **Caveats that change the reading of individual cells:**
   - Qwen2.5-3B LoRA r16/r64 on MetaMathQA **diverged** (loss spikes to 15.8/31.6, grad-norm ≈ 2×10⁴) and are excluded.
   - GSM8K is scored with flexible-extract throughout. Strict-match under-scores the Qwen2.5-3B *base* by 5.6 pp, which makes it look as if MetaMathQA taught Qwen2.5-3B something when it did not.
   - Every result is one seed.

---

## 1. What was run

| Regime (code) | Models | Target(s) | Optimizer | LR policy | Steps × global batch | Example-passes | Training text format | Non-target eval suite |
|---|---|---|---|---|---|---|---|---|
| **Round robin, tuned** (`rr_tuned`) | Llama-3.2-1B/3B, Qwen2.5-1.5B/3B, Pythia-1.4B/2.8B, SmolLM3-3B | GSM8K, HellaSwag, TriviaQA, ARC-Easy (each once) | AdamW (32-bit; **8-bit for full FT of the ~3B models**) | One LR per (task, method), tuned on Llama-3.2-3B by target accuracy (67-run sweep), transferred to the other 6 | 1,404 × 16 | 22,464 (0.16–3 epochs depending on task) | lm-eval prompt formats, completion-only loss | the other 3 of {GSM8K, TriviaQA, HellaSwag, ARC-Easy} |
| **Round robin, single LR** (`rr_single`, archived) | same 7 | same 4 | same | One LR per method: FT 2e-5, r16/r64 2e-4, r256 1e-4 | 1,404 × 16 | 22,464 | same | same |
| **Round robin, Llama-2-7B** (`rr_l2`, `rr_l2_all`) | Llama-2-7B | same 4 | 8-bit AdamW (both methods) | 3 LRs per (task, method); winner by target accuracy (8 of 12 winners on a grid edge) | 1,404 × 16 | 22,464 | same | same |
| **MetaMathQA 1 epoch** (`mm1_l2`) | Llama-2-7B | MetaMathQA (scored on GSM8K) | 8-bit AdamW, HF Trainer cosine | Paper LRs: FT 1e-5, r16/r64 1e-4, r256 5e-5 | 515 × 768 | 395,000 (1 epoch) | **GSM8K eval template** (`Question: …\n\nAnswer:\n`), completion-only | HellaSwag, ARC-Easy |
| **MetaMathQA 4 epoch, Lion** (`mm4`) | Llama-2-7B; Llama-3.2-1B/3B; Qwen2.5-1.5B/3B | MetaMathQA (scored on GSM8K) | **DecoupledLionW** β=(0.9, 0.95), wd 0, fp32 masters | Llama-2: paper LRs (FT 1e-5, r16/r64 1e-4, r256 5e-5). Small models: FT 2e-5, r16 4e-4, r64 2e-4, r256 5e-5, **copied from the round-robin GSM8K (AdamW) optimum, not tuned for Lion** | 2,056 × 768, cosine, 10% warmup; snapshots at epochs 1/2/4 | 1,579,000 (4 epochs) | **Llama-2 chat format** `[INST] q [/INST] r`, response-only loss | WinoGrande, ARC-Challenge, HellaSwag (Biderman et al.'s suite) |

All LoRA runs: α = 2r, dropout 0.05, all attention + MLP projections. All evals: lm-evaluation-harness, vLLM, bf16, each task's default few-shot (GSM8K 5-shot, everything else 0-shot); LoRA adapters merged before eval.

**What differs between the two regimes, all at once:** optimizer, dataset, dataset size (70× more example-passes), batch size (48×), schedule, training-text format, LR policy, and the non-target eval suite. Only two within-project contrasts isolate a subset of these:
- `mm1_l2` vs `mm4` (Llama-2 only): same model, data, LR, batch; differs in **optimizer, text format, schedule length**.
- `rr_l2` vs `mm1_l2`: same optimizer (8-bit AdamW); differs in data, size, batch, format.

---

## 2. Definitions

- **Learning** = fine-tuned − base accuracy on the target task, in percentage points (pp). MetaMathQA runs are scored on GSM8K.
- **Forgetting** = mean of fine-tuned − base over the non-target tasks of that regime (negative = forgetting). The non-target sets differ between regimes (table above), so absolute forgetting numbers are **not directly comparable across regimes**. HellaSwag is the one task in both suites and is used as the cross-regime anchor (§5.2).
- **Metrics**:
  - GSM8K: `exact_match,flexible-extract` everywhere in this document. Earlier MetaMathQA tables used strict-match. The two agree within 0.8 pp except the Qwen2.5-3B base, where strict under-scores by 5.6 pp.
  - TriviaQA: token-F1.
  - HellaSwag and ARC: `acc_norm`.
  - WinoGrande: `acc`.
- **Tie** = |difference in forgetting| < 1 pp.
- **Matched learning** = pairs whose learning differs by ≤ 2 pp. In `mm4` any epoch snapshot of full FT may be matched to any snapshot of LoRA of the same model, which gives learning-matched comparisons that the round robin cannot.
- **Excluded**: Qwen2.5-3B LoRA r16/r64 in `mm4` (training diverged, §4.3) unless stated.

---

## 3. Round-robin results (AdamW regime)

### 3.1 Tuned learning rates, 7 small models × 4 targets

| Method | Learning (mean) | Forgetting (mean) | Worst single non-target Δ |
|---|---|---|---|
| Full FT | +7.41 | **+0.65** | **−9.2** |
| LoRA r16 | +6.98 | −2.03 | −19.2 |
| LoRA r64 | +6.80 | −1.59 | −25.1 |
| LoRA r256 | +7.02 | −1.61 | −27.8 |

Excluding ARC-Easy as a target (as in the earlier round-robin analysis, because its fixed-step budget is 3 passes vs 0.16 for TriviaQA): full FT +6.56 / −0.21, r16 +7.59 / −1.95, r64 +7.46 / −1.75, r256 +7.47 / −0.85.

- **Head-to-head (same model × target):** full FT forgets less in 47, tie 37, LoRA forgets less 0 (84 comparisons).
- **Matched learning:** full FT forgets less 24, tie 31, LoRA 0, not comparable 29.
- **Forgetting per 10 pp learned (median):** full FT +0.26 (i.e. no net forgetting), r16 −2.64, r64 −0.95, r256 −0.12.

**Where the forgetting comes from** (mean Δ on each task when it is a non-target):

| Non-target task | Full FT | LoRA r16 | LoRA r64 | LoRA r256 |
|---|---|---|---|---|
| GSM8K | −0.61 | −4.14 | −4.05 | −5.37 |
| TriviaQA | **+2.00** | −3.54 | −2.80 | −1.84 |
| HellaSwag | +0.24 | −0.34 | +0.38 | −0.23 |
| ARC-Easy | +0.98 | −0.12 | +0.09 | +1.00 |

LoRA's round-robin forgetting is almost entirely **generative-capability loss**: GSM8K reasoning and TriviaQA recall. The multiple-choice likelihood tasks barely move for any method.

### 3.2 Single LR per method (archived)

| Method (LR) | Learning | Forgetting | Worst |
|---|---|---|---|
| Full FT (2e-5) | +6.28 | −0.33 | −10.2 |
| LoRA r16 (2e-4) | +6.75 | −2.09 | −19.2 |
| LoRA r64 (2e-4) | +5.98 | −4.24 | −37.7 |
| LoRA r256 (1e-4) | +5.28 | −5.67 | −46.3 |

Head-to-head: full FT less 65, tie 18, LoRA less 1. With one LR per method, forgetting **grows** with rank. With tuned LRs (§3.1) it shrinks with rank, because the tuned r256 LRs are 4–32× lower. So rank ordering is set mostly by LR (§6.1).

### 3.3 Llama-2-7B round robin

| Method | Learning | Forgetting | Worst | (excl. ARC target) Learning / Forgetting |
|---|---|---|---|---|
| Full FT | +12.64 | −3.99 | −36.1 | +13.76 / **−5.97** |
| LoRA r16 | +10.84 | −3.00 | −18.3 | +13.34 / −3.71 |
| LoRA r64 | +11.21 | −3.71 | −19.6 | +13.36 / −4.42 |
| LoRA r256 | +10.70 | −4.02 | −20.8 | +12.90 / −4.51 |

At 7B with target-accuracy-selected LRs, the literature ordering reappears (full FT forgets most, driven by **TriviaQA −36 pp after GSM8K full FT at 4e-5**, the top of its grid).
- Head-to-head: LoRA less 3, tie 5, full FT less 4.
- Pooled over all 3 LRs per method (12 runs each), full FT forgets *least* (−2.92 vs −4.40 … −4.63).
- At matched learning, LoRA never forgets less (full FT less 9, tie 12, LoRA 0).

So at 7B under AdamW the "LoRA forgets less" ordering is produced by LR selection, not by the method. TriviaQA is the dominant non-target loss for every method (−10.8 to −13.5 pp mean).

---

## 4. MetaMathQA results

### 4.1 Llama-2-7B, 1 epoch, 8-bit AdamW, GSM8K-format text

| Method | LR | Learning (GSM8K) | HellaSwag Δ | ARC-Easy Δ (acc_norm / raw acc) | Forgetting |
|---|---|---|---|---|---|
| Full FT | 1e-5 | **+20.39** | **−0.13** | −15.40 / −8.04 | −7.77 |
| LoRA r16 | 1e-4 | +27.60 | −0.62 | −17.93 / −9.43 | −9.27 |
| LoRA r64 | 1e-4 | +32.37 | −0.92 | −16.62 / −8.67 | −8.77 |
| LoRA r256 | 5e-5 | **+34.34** | −1.28 | −14.77 / −6.52 | −8.03 |

Same pattern as the round robin: **full FT learns far less (+20 vs +28…+34) and forgets the same or less.** The ARC-Easy collapse (−15 to −18 pp acc_norm) is method-independent and half of it is the length normalisation (raw acc −6.5 to −9.4). It is a data/format effect of training on MetaMathQA, not a method effect.

### 4.2 Llama-2-7B, 4 epochs, Lion (the Biderman et al. recipe)

| Method | LR | Epoch | Learning | WinoGrande | ARC-C | HellaSwag | Forgetting | Full-FT-minus-LoRA gap, 95% CI |
|---|---|---|---|---|---|---|---|---|
| Full FT | 1e-5 | 1 | +40.71 | −0.39 | −7.51 | −8.19 | **−5.36** [−6.58, −4.18] | — |
| Full FT | 1e-5 | 4 | **+46.63** | −0.63 | −7.25 | −7.68 | **−5.19** [−6.44, −3.95] | — |
| LoRA r16 | 1e-4 | 1 | +37.23 | +0.71 | −1.28 | −1.21 | −0.59 | +4.77 [3.67, 5.85] |
| LoRA r16 | 1e-4 | 4 | +40.18 | −0.24 | −1.45 | −1.27 | **−0.99** | +4.20 [3.01, 5.37] |
| LoRA r64 | 1e-4 | 1 | +39.58 | +0.71 | −2.73 | −2.72 | −1.58 | +3.78 [2.78, 4.80] |
| LoRA r64 | 1e-4 | 4 | +44.96 | −1.66 | −4.27 | −2.67 | −2.86 | +2.32 [1.25, 3.48] |
| LoRA r256 | 5e-5 | 1 | +39.20 | +0.71 | −1.71 | −2.98 | −1.32 | +4.04 [3.07, 5.07] |
| LoRA r256 | 5e-5 | 4 | +46.10 | −0.39 | −1.37 | −2.39 | −1.38 | +3.80 [2.78, 4.90] |

(CI: paired bootstrap over eval items, 4,000 resamples, each task resampled separately. It does not capture seed variance. Epoch-2 rows are in Appendix A5.)

This reproduces Biderman et al.'s three claims:
- LoRA **forgets less**, significantly, at every epoch.
- LoRA learns slightly less: r16 and r64 below full FT, r256 ties.
- **At equal LR, higher rank forgets more** (r16 vs r64, both 1e-4: −0.99 vs −2.86).

Two dynamics worth noting:
- Full FT's forgetting is **complete by epoch 1** and does not grow.
- LoRA r64's grows from −1.58 to −2.86.

#### 4.2.1 After one epoch: Lion vs AdamW side by side

Same model (Llama-2-7B), same data (MetaMathQA, one full pass), same learning rates, same global batch (768). This is the Lion run's epoch-1 snapshot in the column layout of §4.1 (the AdamW 1-epoch run).

**Lion, epoch 1:**

| Method | LR | Learning (GSM8K) | HellaSwag Δ | ARC-Challenge Δ (acc_norm / raw acc) | Forgetting |
|---|---|---|---|---|---|
| Full FT | 1e-5 | **+40.71** | **−8.19** | −7.51 / −7.08 | **−5.36** |
| LoRA r16 | 1e-4 | +37.23 | −1.21 | −1.28 / −0.60 | −0.59 |
| LoRA r64 | 1e-4 | +39.58 | −2.72 | −2.73 / −1.37 | −1.58 |
| LoRA r256 | 5e-5 | +39.20 | −2.98 | −1.71 / −1.54 | −1.32 |

**AdamW, 1 epoch (§4.1, repeated for comparison):**

| Method | LR | Learning (GSM8K) | HellaSwag Δ | ARC-Easy Δ (acc_norm / raw acc) | Forgetting |
|---|---|---|---|---|---|
| Full FT | 1e-5 | +20.39 | **−0.13** | −15.40 / −8.04 | −7.77 |
| LoRA r16 | 1e-4 | +27.60 | −0.62 | −17.93 / −9.43 | −9.27 |
| LoRA r64 | 1e-4 | +32.37 | −0.92 | −16.62 / −8.67 | −8.77 |
| LoRA r256 | 5e-5 | +34.34 | −1.28 | −14.77 / −6.52 | −8.03 |

Notes on the columns:
- The Lion run was evaluated on Biderman et al.'s suite (WinoGrande, ARC-Challenge, HellaSwag), not ARC-Easy, so the ARC column is **ARC-Challenge** for Lion and **ARC-Easy** for AdamW.
- *Forgetting* is each run's own non-target mean: WinoGrande + ARC-C + HellaSwag for Lion, HellaSwag + ARC-Easy for AdamW. The forgetting columns are therefore not directly comparable across the two tables.
- **HellaSwag is the one task measured in both, and it is the clean comparison.**

What changes between the two tables:

- **Full FT flips from the method that learns least to the one that learns most.**
  - AdamW: full FT learns +20.4 pp, the least of the four methods (LoRA +27.6 to +34.3).
  - Lion: full FT learns +40.7 pp, the most.
  - LoRA's learning also rises under Lion (+37 to +40), but by less.
- **Full FT's general-task damage goes from nothing to large.**
  - HellaSwag: −0.13 → **−8.19** (60×).
  - ARC: −7.5 on ARC-C under Lion, a method-specific drop, whereas under AdamW every method lost about the same on ARC-E.
- **LoRA's damage barely changes.** HellaSwag goes from −0.6…−1.3 to −1.2…−3.0.
- **The ordering flips on HellaSwag.**
  - Under AdamW, full FT damages HellaSwag *least* of the four.
  - Under Lion it damages it *most*, by 5–7 pp.
  - The ARC-Easy drop in the AdamW run (−15 to −18 acc_norm) is method-independent (the same for full FT and every LoRA rank), so it reflects the data/format of that run, not the method.

##### Why switching Lion ↔ AdamW changes the trend

**1. The two optimizers take different-sized steps on weak signals.**
- AdamW moves each weight by roughly `lr × (mean gradient) / (gradient magnitude)`.
  - When a weight's gradient is consistent, that ratio is ≈ 1 and the step is ≈ lr.
  - When the gradient is noisy and averages to near zero, the ratio is ≈ 0 and the weight barely moves.
- Lion moves every weight by exactly `lr` every step, in the direction of `sign(momentum)`, however weak or noisy the gradient is.

**2. Full FT and LoRA expose different numbers of weights to that difference.**
- Full FT trains all ~6.7 B weights of Llama-2-7B. Fine-tuning on math gives most of them only a weak, inconsistent gradient: they are not what the task needs.
  - Under AdamW, those weights stay almost where they were. Full FT's whole update is small: relative displacement 0.0062, §7.2.
  - Under Lion, each of those weights still takes a full-size step every iteration and random-walks away from its pretrained value. The update is **7× larger** (0.043) and spread across ~60 effective directions per matrix.
- LoRA trains only the small A and B matrices. Lion inflates LoRA's update too (4.6–5.6×, and it is where the 150–700 intruder dimensions come from). But however large it gets, ΔW = BA is confined to r directions per matrix. The rest of the network is untouched by construction.

**3. Diffuse change damages general ability; concentrated change mostly doesn't.**
- HellaSwag and ARC are scored by how likely the model finds ordinary text continuations. That depends on the whole network behaving as it did in pretraining.
- A small shift spread across every weight nudges every prediction a little, so items near the decision boundary flip.
- A large shift in a handful of directions changes the output only for inputs that point along those directions. Generic text mostly doesn't.
- The numbers bear this out:
  - Per unit of weight movement, Lion full FT loses ~5× more than Lion LoRA (§7.2).
  - AdamW full FT, whose smaller movement follows consistent gradient signal, loses essentially nothing on HellaSwag.

**4. The same learning rate means a different step size.**
- Because Lion always takes the maximum step, a given nominal LR is a much bigger effective step under Lion than under AdamW. That is why Lion is normally run at 3–10× lower LR.
- At 1e-5, AdamW full FT *under-fits*: highest training loss of the four methods (0.233), least learning, least forgetting.
- At the same 1e-5, Lion full FT *fits hardest*: lowest loss (0.123), most learning, most forgetting.
- LoRA runs at 5–10× higher LRs under both optimizers, so it is fitting reasonably in both. The optimizer switch changes its behaviour much less.

**Taken together**, under AdamW, full FT is the conservative method: it can only move weights where the gradient is consistent, so it learns less and disturbs little. Under Lion that brake is gone. Full FT moves every weight, learns more and forgets more, while LoRA's low-rank structure still limits where its (also larger) update can go. So "LoRA forgets less than full FT" holds under Lion and not under AdamW.

**Still open.**
- Two things differ between these two runs besides the optimizer:
  - text format (GSM8K-template vs Llama-2 chat);
  - schedule (a 1-epoch cosine vs the epoch-1 point of a 4-epoch cosine).
- And they are not learning-matched (AdamW full FT learned half as much).
- An AdamW full-FT run in the chat format at LR 4e-5, aiming for Lion's ~+40 pp, is in progress (§9, T1). If it reaches that learning and still leaves HellaSwag intact, the optimizer, not the amount learned or the format, is what drives full FT's forgetting.

### 4.3 Four small models, 4 epochs, Lion

Method means (epoch 4; Qwen2.5-3B r16/r64 excluded):

| Method | Learning | Forgetting | Worst | ARC-C | HellaSwag | WinoGrande |
|---|---|---|---|---|---|---|
| Full FT (2e-5) | +15.81 | **−12.12** | −21.7 | −11.45 | −17.78 | −7.12 |
| LoRA r16 (4e-4) | +22.59 | −8.47 | −16.0 | −9.02 | −12.58 | −3.81 |
| LoRA r64 (2e-4) | +23.98 | −8.49 | −17.6 | −8.76 | −12.55 | −4.16 |
| LoRA r256 (5e-5) | +20.62 | **−2.69** | −7.2 | −2.71 | −3.61 | −1.76 |

Per model (epoch 4, learning / forgetting):

| Model | Full FT | LoRA r16 | LoRA r64 | LoRA r256 |
|---|---|---|---|---|
| Llama-3.2-1B | +34.4 / −12.0 | +29.9 / −9.1 | +31.5 / −9.2 | +37.2 / **−3.3** |
| Llama-3.2-3B | +32.8 / −16.1 | +33.5 / −11.8 | +35.6 / −11.5 | +40.8 / **−4.6** |
| Qwen2.5-1.5B | +2.6 / −8.2 | +4.4 / −4.5 | +4.8 / −4.7 | +6.1 / **−1.2** |
| Qwen2.5-3B | **−6.5** / −12.2 | *diverged* | *diverged* | **−1.7** / −1.7 |

Observations:
- **LoRA learns *more* than full FT** in the small models (best-rank learning > full FT in 4/4 models) **and forgets less**: LoRA is Pareto-better. Under Lion, full FT at 2e-5 (twice the paper's Lion LR) is on a strictly worse learning–forgetting frontier here.
- Rank ordering follows LR: r256 (lowest LR, 5e-5) forgets least; r16 (4e-4) ≈ r64 (2e-4) forget much more.
- Forgetting is dominated by **HellaSwag** and **ARC-Challenge**, i.e. general likelihood-based tasks, not WinoGrande.
- **Qwen2.5 learns almost nothing on GSM8K from MetaMathQA.** Qwen2.5-3B goes *down* with every method. Qwen2.5 bases are already strong at GSM8K (60.6 / 74.6 flexible) and were very likely pretrained on similar math data. Qwen2.5-3B also went down when trained on GSM8K in the round robin (−8.5 to −10.1). Treat Qwen "learning" as not meaningful.
- **Qwen2.5-3B LoRA r16/r64 diverged.**
  - r16 (4e-4): loss spike to 15.8 at step 851, grad-norm 2.0×10⁴, final loss 2.38, GSM8K → 0.5%.
  - r64 (2e-4): loss spike to 31.6 at step 700, grad-norm 1.9×10⁴. Training loss *recovered* to 0.089 but the model did not (GSM8K 7%, forgetting −25 pp): the training loss hides the damage.
  - Sweep arms at ≤ 1e-4 stayed stable (max grad-norm ≤ 2.6 over the ≤ 540 steps they ran before cancellation).
  - The AdamW-derived LRs are unsafe under Lion for this model.

### 4.4 Learning-matched comparisons using epoch snapshots

Every learning-matched pair (≤ 2 pp apart) goes to LoRA:

| Model | LoRA run (learning / forgetting) | Closest full-FT snapshot | Full FT (learning / forgetting) |
|---|---|---|---|
| Llama-2-7B | r256 ep4 (+46.1 / −1.4) | ep4 | +46.6 / −5.2 |
| Llama-2-7B | r64 ep2 (+42.8 / −2.7) | ep2 | +43.7 / −5.2 |
| Llama-3.2-3B | r256 ep1 (+32.7 / −3.8) | ep4 | +32.8 / −16.1 |
| Llama-3.2-3B | r16 ep4 (+33.5 / −11.8) | ep4 | +32.8 / −16.1 |
| Llama-3.2-1B | r256 ep2 (+34.0 / −4.4) | ep4 | +34.4 / −12.0 |
| Qwen2.5-1.5B | r256 ep1 (+3.7 / −1.7) | ep4 | +2.6 / −8.2 |

Totals: **21 LoRA-forgets-less, 0 ties, 0 full-FT-forgets-less**, 18 not comparable. The round-robin conclusion "at matched learning LoRA never forgets less" is exactly inverted.

---

## 5. The reversal, stated precisely

### 5.1 Head-to-head counts across all regimes

| Regime | Optimizer | Full FT forgets less | Tie (<1 pp) | LoRA forgets less | Matched-learning verdict |
|---|---|---|---|---|---|
| Round robin, tuned (7 models) | AdamW | 47 | 37 | **0** | full FT 24 / tie 31 / LoRA 0 |
| Round robin, single LR | AdamW | 65 | 18 | 1 | — |
| Round robin, Llama-2-7B (selected) | AdamW 8-bit | 4 | 5 | 3 | full FT 9 / tie 12 / LoRA 0 (all LRs) |
| MetaMathQA 1 ep, Llama-2 | AdamW 8-bit | full FT least on HellaSwag, mixed on ARC-E; learns 14 pp less | | | — |
| MetaMathQA 4 ep (5 models) | **Lion** | **0** | 0 | **39** | **LoRA 21 / tie 0 / full FT 0** |

### 5.2 HellaSwag: the one non-target task measured in every regime

Math target in every row (GSM8K or MetaMathQA):

| Regime | Optimizer | Full FT | LoRA r16 | LoRA r64 | LoRA r256 |
|---|---|---|---|---|---|
| Round robin, GSM8K target, tuned (7 models) | AdamW | +0.64 | +0.46 | +1.37 | +1.77 |
| Round robin, GSM8K target, single LR | AdamW | +0.69 | +0.40 | +1.41 | +1.19 |
| Round robin, GSM8K target, Llama-2 | AdamW 8-bit | −0.12 | +0.54 | +0.96 | +1.20 |
| MetaMathQA 1 epoch, Llama-2 | AdamW 8-bit | **−0.13** | −0.62 | −0.92 | −1.28 |
| MetaMathQA 4 ep, Llama-2, **epoch-1 snapshot** | **Lion** | **−8.19** | −1.21 | −2.72 | −2.98 |
| MetaMathQA 4 ep, Llama-2, epoch 4 | Lion | −7.68 | −1.27 | −2.67 | −2.39 |
| MetaMathQA 4 ep, 4 small models, epoch 4 | Lion | −17.78 | −12.58 | −12.55 | −3.61 |

Rows 4 and 5 are the tightest contrast in the project: Llama-2-7B, MetaMathQA, LR 1e-5 for full FT, global batch 768, **both after one pass over the data**.
- Full FT's HellaSwag loss goes from −0.13 to −8.19 (**60×**).
- LoRA's goes from about −1 to about −2.
- Under AdamW, training on math never damages HellaSwag for any method. Under Lion it damages it for every method, and far most for full FT.

---

## 6. What is consistent across regimes (robust findings)

### 6.1 At equal LR, higher LoRA rank forgets more

| Source | Higher rank forgets more | Tie | Higher rank forgets less |
|---|---|---|---|
| Llama-2 round-robin sweep (same LR, different ranks) | 12 | 7 | 1 |
| Round robin single LR (r16 vs r64 at 2e-4) | 16 | 10 | 2 |
| MetaMathQA Llama-2 Lion (r16 vs r64 at 1e-4, each epoch) | 2 | 1 | 0 |

This is Biderman et al.'s rank result, and it holds with both optimizers. Whenever the opposite ordering appears (tuned round robin; MetaMathQA small models), the higher rank was run at a lower LR.

### 6.2 LR above the optimum buys forgetting, not learning

Llama-2 round-robin sweep, consecutive LRs at fixed (task, method):
- Higher LR forgot more in 17 pairs, less in 1, tie 14.
- Higher LR learned more in 6, less in 10, tie 16.

### 6.3 Forgetting is front-loaded

In every Lion run, most of the epoch-4 forgetting is already present at epoch 1 (Llama-2 full FT −5.36 → −5.19; small-model full FT −12.06 → −12.12). Extra epochs add learning, not forgetting. Most of the damage is done within the first ~395k examples, so the 70× exposure difference between regimes cannot be the whole story (H3).

### 6.4 Forgetting hits a shared pool of fragile items

Llama-2 HellaSwag, items right in base → wrong after fine-tuning:
- Lion full FT: 813
- AdamW full FT: 79
- Lion LoRA: 287–456
- AdamW LoRA: 141–221

Every pair of forgotten-sets overlaps 8–62× more than chance, and ≈80% of the items Lion-LoRA-r256 forgets are also forgotten by Lion full FT. This is consistent with the round-robin causal-tracing result (§7.5): forgetting is **"the tail of a uniform representational shift"**. The same boundary-adjacent items flip, and methods and optimizers differ mainly in *how far* the shift goes.

### 6.5 The forgetting is real, but length normalisation inflates it

`acc_norm` drops 1.5–2× more than raw `acc` (Llama-2 Lion full FT HellaSwag: −7.68 norm, −4.10 raw). The method gap survives in raw accuracy (Llama-2 raw HellaSwag: full FT −4.10 vs LoRA +0.86 / −0.47 / −0.19). Part of the measured forgetting is fine-tuning shifting how the model's likelihood scales with continuation length.

---

## 7. Mechanistic evidence

### 7.1 Training loss: under Lion, full FT goes from fitting least to fitting most

End-of-epoch-1 training loss, Llama-2-7B on MetaMathQA, global batch 768:

| Method | AdamW (1-epoch run, LR fully annealed) | Lion (4-epoch run at step 515, LR still at 93% of peak) |
|---|---|---|
| Full FT (1e-5) | **0.233**, highest of the four | **0.123**, lowest |
| LoRA r16 (1e-4) | 0.212 | — |
| LoRA r256 (5e-5) | 0.179 | 0.130 |

Under AdamW at 1e-5, full FT under-fits relative to LoRA: it has the highest loss, learns 14 pp less GSM8K and forgets least. Under Lion at the *same* LR, full FT fits hardest, learns most and forgets most. The sets use different text formats (§8 H2), but the response tokens scored by the loss are identical.

In the round robin, full FT ended with a higher final training loss than the mean of the LoRA ranks in **21 of 21** (model, target) pairs, and forgot less in 17 (earlier round-robin analysis, §Loss). So "full FT fits more loosely" describes the AdamW regime, and it is exactly what Lion undoes.

**Loss vs forgetting:**
- Round robin (earlier round-robin analysis): lower final loss goes with more learning (ρ = −0.45 for 1–3B, −0.60 at 7B) and, in 1–3B models, with more forgetting (ρ = +0.32, p = 0.012).
- MetaMathQA small models (epoch 4, n = 12 valid runs, mean of last 5 logged losses): no significant relation. ρ = +0.41 across methods (p = 0.19); ρ = −0.45 within LoRA (p = 0.26, wrong sign).
  - Two runs were dropped because their loss histories were overwritten by cancelled reruns (Llama-3.2-3B r16, Qwen2.5-3B r256; §10).
  - Under Lion the loss→forgetting relation breaks: every full-FT run has a lower loss than every LoRA run *and* more forgetting, but within LoRA the ordering is noise at n = 8.

### 7.2 Weight displacement: how far and in what shape each fine-tune moved the weights

Measured over the 7 LoRA-targeted projections of every block. Columns:
- **rel. Frobenius** = ‖ΔW‖_F / ‖W₀‖_F (pooled);
- **stable rank** = ‖ΔW‖_F² / ‖ΔW‖₂² (mean over matrices; top singular value by power iteration, which underestimates σ₁ for flat spectra, so full-FT stable ranks are upper-biased);
- **σ₁ ratio** = ‖ΔW‖₂ / ‖W₀‖₂ (mean).

LoRA's ΔW = (α/r)·BA is never formed; its norms are computed exactly from the r×r product.

**Summary (full per-run table in Appendix A6):**

| Comparison | Full FT | LoRA r16 | LoRA r64 | LoRA r256 |
|---|---|---|---|---|
| Llama-2-7B, MetaMathQA epoch 1, **AdamW** → **Lion** (same LR, data, batch) | 0.0062 → 0.043 (**6.9×**) | 0.014 → 0.066 (4.6×) | 0.018 → 0.103 (5.6×) | 0.015 → 0.079 (5.3×) |
| Small models: Lion epoch 1 / epoch 4 relative to their round-robin GSM8K AdamW run (range over 4 models) | **6.7–10.3× / 9.8–15.4×** | 3.9–4.4× / 5.7–6.5×* | 3.8–4.5× / 6.1–7.1×* | 3.4–3.9× / 5.3–6.1× |
| Mean stable rank of ΔW (round robin AdamW / Lion) | 88 / 69 | 3.5 / 7.1 | 6.9 / 13.0 | 14.1 / 20.1 |
| Median forgetting per unit rel. Frobenius (round robin AdamW / Lion) | **+38.6 / −134** | −95 / −23 | −87 / −28 | −18 / −23 |

\*excluding the diverged Qwen2.5-3B r16/r64 (22× and 13× at epoch 4).

What the displacement data say:

1. **Lion inflates everything, and full FT about twice as much as LoRA.** Under Lion every method moves 3–15× further than under AdamW, but full FT's inflation (7–15×) is roughly double LoRA's (3.4–7×). This is the magnitude half of H1.
2. **LoRA's update stays larger in norm yet forgets less.**
   - In Llama-2 under Lion, LoRA's ΔW has a *larger* Frobenius norm than full FT's (r64 0.103 vs 0.043 at epoch 1), and its top singular value reaches 60–96% of the base weight's own.
   - That update is packed into ~2–12 effective directions (these are the 150–700 intruder dimensions, §7.3).
   - Full FT's update is spread over ~60–90 effective directions.
   - **Size alone does not decide which method forgets more** (H1′).
3. **Within a method, displacement predicts damage strongly, in every regime.** Spearman ρ between relative displacement and HellaSwag Δ, pooling both optimizers and all models:
   - full FT −0.89, r16 −0.93, r64 −0.92, r256 −0.73 (all p < 0.001; n = 18–21);
   - against mean forgetting: −0.77, −0.51, −0.66, −0.10;
   - pooled across methods: −0.72 (HellaSwag), −0.35 (mean forgetting).
4. **Per unit of displacement, the optimizer changes how damaging full FT is.**
   - Round-robin AdamW full FT moves the weights with no net forgetting (+38.6 per unit, i.e. it *improves* non-targets on GSM8K training).
   - Lion full FT loses 134 pp per unit, about 5× more than Lion LoRA (−23 to −28).
   - So Lion changes *what kind* of movement full FT makes, not just how much. This is the strongest current evidence for H1″ (noise-driven vs gradient-driven diffuse update).
   - (The AdamW 1-epoch MetaMathQA ratios are dominated by the method-independent ARC-Easy drop over tiny denominators and are not interpretable.)


### 7.3 Intruder dimensions (Shuttleworth et al. definition: top-10 singular vectors of W with max |cos| < ε to every base singular vector)

| Run | LoRA r16 | LoRA r64 | LoRA r256 | Full FT |
|---|---|---|---|---|
| Round robin, 7 small models, GSM8K target (ε = 0.5, summed) | 263 | 26 | 1 | 0 |
| Round robin, 7 small models, HellaSwag + TriviaQA targets | 1 | 0 | 0 | 0 |
| Round robin, Llama-2-7B, GSM8K target | 10 | 26 | 1 | 0 |
| MetaMathQA 1 epoch AdamW, Llama-2 | — | — | 0 | 0 |
| MetaMathQA Lion, Llama-2, epoch 1 / 2 / 4 | 151 / 346 / 482 | 222 / 550 / 707 | not run | 0 / 0 / 0 |

(224 matrices per Llama-2 model.)

- **Full FT never creates intruders**, under either optimizer.
- **Lion creates 6–30× more intruders in LoRA than AdamW does**, even at half the LR (r64: 222 at 1e-4 Lion vs 26 at 2e-4 AdamW).
- **Across optimizers, intruders do not track forgetting:** Lion LoRA has hundreds of intruders and forgets 1–3 pp, while Lion full FT has none and forgets 5 pp. Within one run they grow 3× from epoch 1 to 4 while forgetting barely moves (r16: −0.59 → −0.99).
- **Within an optimizer they do track it:** r64 has more intruders and more forgetting than r16 under Lion, and the round-robin GSM8K runs (the highest LoRA LRs) hold 327 of the 332 round-robin intruders (all models incl. Llama-2).

Intruder count reads as a signature of *large concentrated low-rank updates* (which Lion and high LR both produce), not as a general predictor of forgetting.

### 7.4 Representations (round robin only; from the earlier round-robin analysis)

- **Drift:** full FT's final-layer representation drift is 0.0315 vs 0.046–0.050 for LoRA (mean per-example cosine distance, 1,000 inputs/task).
- **RSA:** full FT 0.965 vs 0.941–0.944 for LoRA.
- **Drift vs forgetting** across 252 non-target cells: r = −0.31.
- **In the AdamW regime full FT changes representations least**, consistent with the smallest weight displacement (§7.2) and the least forgetting.
- **Not yet measured for the Lion runs** (proposed in §9, T3).

### 7.5 Causal tracing (round robin)

Cell: Qwen2.5-1.5B LoRA r256, trained on ARC-Easy, GSM8K 0.608 → 0.409 while full FT held at 0.604. The largest LoRA-specific regression in the grid.

- **No localisation.** Forgetting was not localisable by depth, component or position: forgotten and still-correct examples have identical patching profiles (ρ = +1.000).
- **Uniform shift.** Forgetting is the tail of a uniform representational shift.
- **Real errors, not format loss.** 95% of forgotten items still produce well-formed `####` answers, so it is genuine arithmetic failure, not task misinference.

This, together with the shared fragile-item pool (§6.4), is the basis for the "dose, not mechanism" reading.

---

## 8. Why the difference? Hypotheses, evidence and verdicts

Each hypothesis names the regime difference it blames. **Prediction** is what it implies for the data we have; **Status** is the verdict from current evidence.

### H1. The optimizer: Lion's sign updates inflate full FT's diffuse, high-rank drift — **leading hypothesis**

**Mechanism.**
- AdamW's per-coordinate step is lr·m̂/√v̂, which shrinks toward 0 for coordinates whose gradient is noise-dominated (the running mean of the gradient cancels while its variance does not).
- Lion's step is lr·sign(βm + (1−β)g): exactly ±lr for **every** coordinate, every step, however weak or noisy its gradient.
- In full FT that is every one of 1–7 B parameters, so full FT accumulates a large, diffuse, high-rank random-walk component that is not aligned with the task.
- LoRA also takes ±lr steps, but only on A and B, so its ΔW is confined to r directions.
- A diffuse perturbation touches every input direction a little, which is what general likelihood tasks (HellaSwag, ARC) are sensitive to.

**For:**
- (a) Same model, data, LR, batch: full-FT HellaSwag −0.13 (AdamW) vs −8.19 (Lion) (§5.2).
- (b) Full FT flips from worst-fitting to best-fitting (§7.1).
- (c) Lion moves full-FT weights ~7× further at equal LR, and the update stays high-rank (§7.2).
- (d) The flip in *which method forgets less* coincides exactly with the optimizer change: every AdamW regime has full FT ≤ LoRA, the only Lion regime has LoRA < full FT in 39/39.
- (e) Biderman et al. used Lion, so their headline result was obtained in exactly the regime where full FT is most exposed.

**Against / open:**
- Lion also inflates LoRA's displacement 4–6×, and LoRA's update stays *larger* in Frobenius norm than full FT's, so raw magnitude is not the mechanism; the shape has to be (see H1′).
- Optimizer is confounded with text format (H2) and schedule length in the mm1-vs-mm4 contrast.

**Prediction to test:**
- Lion full FT with the LR lowered until its ‖ΔW‖ matches AdamW's should forget like AdamW.
- AdamW run on the mm4 recipe (chat format, 4 epochs, 10% warmup) should reproduce round-robin-like full-FT forgetting (T1).

**Status: best supported.** Not yet isolated from H2.

### H1′. Forgetting on general tasks tracks the *diffuse* part of ΔW, not its size — **refinement of H1**

**Idea.** A rank-r update with a huge σ₁, like Lion-LoRA's (0.6–0.96 of the base σ₁), changes the output only for inputs that project onto its r input directions. If those directions are "task directions" that general text rarely excites, generic predictions barely move. A high-rank update of smaller norm moves the output for *every* input.

**For:**
- Within each method, more displacement means more forgetting (Llama-3.2-1B LoRA under Lion: rel. Frobenius 0.26 / 0.21 / 0.09 → forgetting −9.0 / −7.7 / −3.7).
- Across methods, full FT forgets more per unit displacement.
- LoRA's damage items are a subset of full FT's (§6.4).

**Test.** Measure the *functional* displacement ‖ΔW·x‖ / ‖W·x‖ on activations x from non-target tasks (HellaSwag, ARC), per layer. H1′ predicts this, not ‖ΔW‖_F, predicts forgetting across methods and optimizers (T3). It is cheap: one forward pass per checkpoint to collect activations, then matrix products on CPU.

### H1″. It is the *direction* of the diffuse update, not its size — **sharpening of H1′, suggested by the round-robin Llama-2 row**

**The problem for H1′.**
- Round-robin Llama-2 full FT on GSM8K (AdamW, LR 4e-5, 1,404 steps × 16) reaches a diffuse, high-rank displacement of **0.030** (stable rank ≈ 96), in the same range as Lion's epoch-1 **0.043** (stable rank ≈ 57).
- Its HellaSwag Δ is −0.12, against Lion's −8.19.
- Its damage lands almost entirely on **TriviaQA (−36 pp)**.
- So a diffuse update of similar size hurts *different capabilities* depending on how it was produced.

**Hypothesis.**
- AdamW's full-FT update, although high-rank, is *gradient-structured*: it moves weights along directions the training data actually pushes.
- On a narrow set (7.5k GSM8K items, 3 passes) those directions overwrite stored facts (TriviaQA).
- Lion's update contains a large *sign-of-noise* component: coordinates with near-zero expected gradient still get ±lr per step, accumulating as a random walk.
- An isotropic perturbation degrades generic next-token likelihoods (HellaSwag, ARC `acc_norm`) rather than any specific stored knowledge.

**Test.**
- (i) Split each full-FT ΔW into its projection onto the span of accumulated training gradients (or the top-k singular subspace of a short AdamW run's ΔW) and the orthogonal residual. H1″ predicts Lion's residual is much larger than AdamW's at matched ‖ΔW‖.
- (ii) Add an isotropic Gaussian perturbation of Lion's residual norm to the *AdamW* checkpoint. H1″ predicts it reproduces the HellaSwag drop. This mirrors the "noise of equal norm" control used in the patching study.
- **Cost:** CPU + evals only.

### H2. Training-text format: chat template vs eval-style prompts

**Mechanism.**
- The AdamW regimes trained in the lm-eval prompt formats: round robin in each task's own format, MetaMathQA-1-epoch in GSM8K's `Question: … Answer:` template.
- The Lion run trained in the Llama-2 chat template, but was evaluated without it.
- Training only on `[INST]`-wrapped text could shift the model's distribution on raw web-style text (HellaSwag endings) more for full FT, which adopts the format more thoroughly.

**For:** it is confounded with the optimizer in the only clean contrast.

**Against:**
- Lion LoRA was trained in the same chat format and forgets little (Llama-2 r16 −0.99), so the format alone does not force forgetting. It would have to hit full FT specifically.
- The forgotten items overlap strongly with AdamW's (8–10× chance), which suggests the same fragile items rather than a new format-driven failure.

**Test:** the 2×2 {AdamW, Lion} × {GSM8K-format, chat} ablation (T1). Also evaluate the Lion checkpoints *with* the chat template applied (`--apply_chat_template`) on the MC tasks (T6). If forgetting shrinks a lot, part of it is format.

**Status: open, secondary.**

### H3. Amount of training (70× more example-passes on MetaMathQA)

**For:** more gradient signal lets full FT, with no capacity cap, move further.

**Against:**
- The AdamW MetaMathQA run saw 395k examples (17× the whole round robin) and full FT still did not hurt HellaSwag (−0.13).
- Under Lion, forgetting is already complete at epoch 1 and does not grow over epochs 2–4 (§6.3). There is no dose–response in exposure beyond the first epoch.

**Test.** Lion on a 22,464-example MetaMathQA subset (round-robin exposure), and AdamW for 4 epochs on full MetaMathQA (T1, T4).

**Status: unlikely to be the main driver.**

### H4. Learning-rate calibration (the small-model LRs were never tuned for Lion)

**For:**
- The small-model MetaMathQA LRs are AdamW optima transferred to Lion: 2× the paper's Lion full-FT LR, and 2–4× its LoRA LR.
- This demonstrably broke Qwen2.5-3B r16/r64.
- It also explains the *rank ordering* in the small models (r256 has the lowest LR and forgets least).

**Against as an explanation of the flip:**
- Llama-2 used the paper's own Lion LRs and still shows the flip at full strength (39/39 includes 9 Llama-2 comparisons, all significant).
- Mis-tuned small-model LRs affect LoRA r16/r64 *more* (4× over) than full FT (2× over), which should *favour* full FT, yet it still loses.

**Status: explains magnitudes in the small models, not the direction of the flip.**

### H5. Batch size and step count (768 vs 16; 2,056 vs 1,404 steps)

**For:** at batch 768 the gradient is much less noisy per step, so AdamW's noise-shrinkage (H1) matters less and Lion's sign steps are more coherent.

**Against:** the mm1 (AdamW) and mm4 (Lion) runs share batch 768 and differ in outcome, so batch size cannot explain that contrast.

**Status: not the cause of the flip.** It could modulate how strong H1 is, which T5 would measure.

### H6. Eval-suite composition

**For:**
- Round-robin forgetting is dominated by generative tasks (GSM8K, TriviaQA), where LoRA is the one that hurts, especially TriviaQA recall.
- MetaMathQA forgetting is measured only on likelihood-scored MC tasks (WinoGrande, ARC-C, HellaSwag), where full FT is the one that hurts.
- Different capabilities may be differentially fragile to low-rank vs diffuse updates, which fits H1′.

**Against:** the flip is visible on the one task shared by both suites (HellaSwag, §5.2), so suite composition cannot create it. It can inflate or shrink the gap.

**Test:** evaluate every MetaMathQA checkpoint on TriviaQA + ARC-Easy, and the round-robin checkpoints (on HF) on WinoGrande + ARC-C (T2).

**Status: modulator, not cause.** T2 is cheap and removes the biggest comparability caveat in this document.

### H7. Target content (math via MetaMathQA vs GSM8K)

**Against:** both are math. Round-robin GSM8K training slightly *improves* HellaSwag for every method (+0.5 to +1.8), and MetaMathQA under AdamW barely touches it.

**Status: ruled out** as the cause of the flip.

### H8. Model family and base capability

**Observation.** Qwen2.5 behaves differently in both regimes:
- GSM8K learning is near zero or negative everywhere.
- Qwen2.5-3B is the only model to diverge.

**Against as the flip's cause:** the flip holds in Llama-2-7B, Llama-3.2-1B and Llama-3.2-3B, where learning is large.

**Status: model-specific noise on top of the main effect.** Qwen's strong base GSM8K (likely math-heavy pretraining) leaves little to learn and makes fine-tuning mostly disruptive.

### H9. Metric artefacts

**For:**
- Strict-match GSM8K under-scores the Qwen2.5-3B base by 5.6 pp (corrected here).
- `acc_norm` inflates MC forgetting 1.5–2× (§6.5).

**Against:** the method gap survives raw accuracy and flexible-extract.

**Status: affects magnitudes, not direction.**

### H10. Schedule (warmup length, annealing, where the snapshot sits on the schedule)

**For:** the mm4 epoch-1 snapshot is taken at 93% of peak LR, while the mm1 model is fully annealed.

**Against:** the annealed Lion epoch-4 model forgets just as much (−7.68 HellaSwag).

**Status: unlikely.**

### H11. Precision (8-bit AdamW)

**For:**
- mm1 and the Llama-2 / ~3B round-robin full-FT runs used 8-bit AdamW state.
- Quantised second moments could further damp small updates, a stronger version of H1's AdamW side.
- In the round robin, 3B full FT used 8-bit AdamW while LoRA used 32-bit.

**Against:** 1B/1.5B round-robin full FT used 32-bit AdamW and shows the same full-FT-forgets-less pattern.

**Status: minor.** Worth including 32-bit AdamW in T1.

### H12. Single seed

All of this is one seed per configuration. Item-level CIs (§4.2) do not capture seed variance, and round-robin differences under ~1 pp are within run-to-run noise. The MetaMathQA gaps (2–12 pp) are far larger than typical seed variance at these eval sizes, so the direction is unlikely to be noise. The round-robin "full FT forgets less" is a smaller, noisier effect.

### Summary of verdicts

| Hypothesis | Explains the *direction* of the flip? | Explains *magnitudes*? | Key test |
|---|---|---|---|
| H1 Lion sign updates | **Yes (best supported)** | Yes | T1 |
| H1′ Diffuse vs concentrated ΔW | Partly (size alone fails on the RR Llama-2 row) | Yes | T3 |
| H1″ Noise-driven vs gradient-driven diffuse ΔW | **Yes (current best mechanism)** | Yes | T3 + noise-injection control |
| H2 Chat vs eval-format text | Possibly (confounded with H1) | Possibly | T1, T6 |
| H3 Exposure | Unlikely | Partly | T1, T4 |
| H4 LR not tuned for Lion | No | Yes (small models, Qwen divergence) | T7 |
| H5 Batch / steps | No | Possibly | T5 |
| H6 Eval suite | No | Yes | T2 |
| H7 Target content | No | — | — |
| H8 Model family | No | Yes (Qwen) | — |
| H9 Metric artefacts | No | Yes | done (§2, §6.5) |
| H10 Schedule | Unlikely | — | T1 |
| H11 8-bit optimizer | Unlikely | Possibly | T1 |
| H12 Seeds | No | Round-robin small effects | T8 |

---

## 9. How to answer these: proposed experiments (in priority order)

Costs are rough, measured on this cluster: Llama-3.2-1B MetaMathQA 1 epoch ≈ 3–4 h on one L40S/H100 at batch 768; eval ≈ 15 min per checkpoint. Using **Llama-3.2-1B** keeps each arm to a few GPU-hours.

**T1. Optimizer × format factorial (the decisive test).**
- **Design.** Llama-3.2-1B, MetaMathQA, 1 epoch, batch 768, cosine with 10% warmup. Arms: {AdamW 32-bit, Lion} × {GSM8K-format text, chat format} × {full FT, LoRA r16}, plus one "Lion, LR lowered so ‖ΔW‖ matches AdamW" arm per method. 10 runs.
- **Evaluation.** Union suite (GSM8K, TriviaQA, HellaSwag, ARC-E, ARC-C, WinoGrande).
- **Reading.**
  - H1: Lion rows show large full-FT forgetting in both formats; the norm-matched Lion arm does not.
  - H2: chat rows show it under both optimizers.
  - Interaction: only Lion + chat does.
- **Cost:** ~40 GPU-h.

**T2. Common eval suite across regimes (cheap, removes the biggest caveat).**
- **Design.** Evaluate the MetaMathQA checkpoints (20 small-model + 12 Llama-2, all on disk or HF) on TriviaQA + ARC-Easy, and the round-robin GSM8K checkpoints (on HF) on WinoGrande + ARC-C.
- **Reading.** Tells whether the flip holds on generative/recall tasks too, or whether LoRA and full FT each damage a different *kind* of capability.
- **Cost:** ~60 evals ≈ 15 GPU-h, no training.

**T3. Functional displacement on non-target inputs (tests H1′).**
- **Design.** For each checkpoint already measured in §7.2, collect base-model activations on 500 inputs each from HellaSwag / ARC / WinoGrande, and compute ‖ΔW x‖/‖W x‖ per layer and module.
- **Reading.**
  - H1′ predicts that this quantity, not ‖ΔW‖_F, lines up forgetting across *both* methods and *both* optimizers on one curve.
  - Also project ΔW onto the base top-k singular subspaces, to separate "aligned" from "new-direction" change.
- **Cost:** CPU plus one short GPU forward pass per model.

**T4. Exposure control.**
- **Design.** Lion on a 22,464-example MetaMathQA subset (round-robin exposure); AdamW for 4 epochs on full MetaMathQA (mm4 exposure).
- **Reading.**
  - If Lion-subset still makes full FT forget more, exposure is not needed.
  - If AdamW-4-epoch does not, exposure is not sufficient.
- **Cost:** ~15 GPU-h on Llama-3.2-1B.

**T5. Batch-size control.** Lion full FT at batch 16 vs 768 with LR scaled to hold ‖ΔW‖ fixed. This checks whether the noisiness of the gradient changes how damaging Lion's sign steps are.

**T6. Chat-template eval.** Re-evaluate the Lion checkpoints with `--apply_chat_template`. If forgetting shrinks, part of it is prompt-distribution mismatch (H2). Cost: 20 evals.

**T7. A small Lion LR sweep (only if the small-model numbers are to be reported as tuned).** 3 LRs × {full FT, r16} on Llama-3.2-1B, 1 epoch, scored on target accuracy exactly as the round robin was tuned. This makes the small-model MetaMathQA comparison "each method at its own optimum", as in Biderman et al.

**T8. Seeds.** Two extra seeds for Llama-3.2-1B {full FT, r16, r256} in both regimes. This puts seed-level CIs on the headline gaps, most importantly the round-robin "full FT forgets less", which is a ~1–2 pp effect.

**T9. Early-training dynamics.** Snapshot Lion full FT every 25 steps for the first 200 steps and evaluate HellaSwag. Forgetting is complete by epoch 1. If it appears in the warmup, before much learning happens, that points at Lion's first, un-averaged sign steps.

**T10. A continuous forgetting metric.** Perplexity on held-out pretraining-style text (e.g. 2,000 C4/Wikipedia passages) for every checkpoint. It is cheaper and lower-noise than MC accuracy, and it directly measures the "general language modelling" damage that HellaSwag `acc_norm` proxies.

**Predicted outcomes, if H1/H1′ is right:**

| Test | H1/H1′ predicts | H2 predicts | H3 predicts |
|---|---|---|---|
| T1 Lion vs AdamW (format fixed) | Lion full FT forgets ≫ AdamW full FT | no difference | no difference |
| T1 chat vs GSM8K-format (optimizer fixed) | no difference | chat forgets more | no difference |
| T1 norm-matched Lion | forgets like AdamW | still forgets (if chat) | — |
| T3 functional displacement | one curve across methods and optimizers | — | — |
| T4 Lion on 22k examples | full FT still forgets more | — | full FT forgets like round robin |

---

## 10. Threats to validity and data caveats

- **One seed per configuration** everywhere.
- **The two regimes use different non-target suites.** Only HellaSwag is shared; cross-regime statements about *averages* compare different things.
- **Transferred LRs.** Round-robin small-model LRs were tuned on Llama-3.2-3B and transferred to 6 other models. MetaMathQA small-model LRs are AdamW optima transferred to Lion.
- **8-bit vs 32-bit AdamW.** The round-robin full FT of the ~3B models used 8-bit AdamW; their LoRA runs used 32-bit.
- **Reconstructed round-robin tables.** The original `results/train_on_<task>/FINAL_TABLE.csv` files (and the accuracy/embedding folders around them) were deleted to free space. The round-robin tables here were rebuilt from a full printout made earlier in this project and verified against the surviving `results/loss/LOSS_VS_ACC.csv`: all 112 (target, model, method) cells agree within 0.08 pp (3-decimal rounding). Per-item round-robin samples are gone, so no item-level CIs are possible for the round robin.
- **Overwritten loss histories.** `loss_history.json` for Llama-3.2-3B LoRA r16 and Qwen2.5-3B LoRA r256 (MetaMathQA) were overwritten by cancelled redundant reruns (they stop at steps 1,500 and 1,260). Their epoch-4 checkpoints and evals are the originals and are valid; only their loss curves are affected, and they are excluded from the loss analysis.
- **Missing r256 intruder counts** for the Llama-2 Lion run (it had not finished when the intruder job ran).
- **Different-format losses.** AdamW-vs-Lion training-loss comparisons use different text formats. The response tokens are the same, so the losses are comparable but not identical.
- **Power-iteration σ₁ is a lower bound** for flat spectra, so full-FT stable ranks are upper-biased (the full-vs-LoRA contrast is ~10× and unaffected).
- **ARC-Easy as a target** (round robin) had 3 passes vs 0.16 for TriviaQA at a fixed step budget. It is reported but flagged, as in the earlier round-robin analysis.
- **Qwen2.5-3B LoRA r16/r64** (MetaMathQA) diverged and are excluded from every average.

---

## 11. What we can and cannot claim right now

**Can claim:**
1. Whether LoRA forgets less than full FT is **regime-dependent**. It reverses between our AdamW regimes and our Lion regime, and the reversal survives learning-matching and item-level significance testing.
2. **LoRA is the stable one**; full FT's forgetting varies by >10× across regimes on the same task.
3. At equal LR, **higher LoRA rank forgets more**, in every regime. Excess LR buys forgetting, not learning.
4. The Biderman et al. result replicates **in their own recipe** (Llama-2-7B, Lion, their LRs, their eval suite), including "higher rank forgets more".

**Cannot yet claim:**
1. That Lion is *the* cause: it is confounded with text format and schedule (needs T1).
2. That the mechanism is diffuse high-rank drift (H1′): consistent with §7.2 but not yet tested functionally (needs T3).
3. That "full FT forgets less" in the round robin is a robust method property: small effect, one seed, LRs transferred (needs T8).

**Suggested one-line framing for a paper:** *"LoRA forgets less" is not a property of LoRA. It is a property of LoRA relative to full fine-tuning under a particular optimizer. With AdamW at tuned learning rates full fine-tuning forgets no more than LoRA; with Lion, as in Biderman et al., full fine-tuning's sign updates spread a large high-rank perturbation across every weight, and LoRA's low-rank constraint protects it.*

---

## 12. Files

| What | Where |
|---|---|
| Every measurement in this report (one row per regime x model x target x method x LR x epoch x eval task) | `results/ALL_LEARN_FORGET_LONG.csv` |
| One row per run: learning, forgetting, worst, per-task change | `results/RUN_SUMMARY.csv` |
| Weight-displacement measurements | `results/UPDATE_NORMS.csv` (small models), `results/UPDATE_NORMS_L2.csv` (Llama-2), `results/NORMS_JOINED.csv` (joined with forgetting) |
| Rebuilt round-robin accuracy tables | `results/rr_tuned_FINAL_TABLES.csv` |
| Qwen2.5-3B divergence / sweep loss summary | `results/QWEN3B_SWEEP_LOSS_SUMMARY.csv` |
| Intruder-dimension counts | `results/intruders/` (code: `intruder_dimensions/`) |
| Headline tables and figures, recomputed from the CSVs | `analysis/summarize.py`, `analysis/make_figures.py` |
| Reference re-implementation of training, evaluation and displacement measurement | `reference/` (see its README: not the original pipeline) |

Sections 7.4 and 7.5 summarise an earlier round-robin analysis (representation drift, RSA, causal tracing) whose raw outputs are not included in this repository.

---

## Appendix — full tables


#### A1. Round robin, tuned LR per (task, method) — 7 models × 4 targets

Cells: **learning** (target Δ) / **forgetting** (mean Δ over the 3 non-target tasks), pp. Bold = least forgetting in the row.

| Target | Model | Full FT | LoRA r16 | LoRA r64 | LoRA r256 |
|---|---|---|---|---|---|
| gsm8k | Llama-3.2-1B | +15.3 / **-2.6** | +17.1 / -8.2 | +19.0 / -7.6 | +18.5 / -5.0 |
| gsm8k | Qwen2.5-1.5B | -6.1 / **+2.9** | -5.8 / +1.8 | -5.3 / +1.7 | -5.2 / +2.6 |
| gsm8k | Pythia-1.4B | +4.3 / -4.3 | +9.3 / -4.2 | +8.9 / -3.8 | +8.3 / **-3.8** |
| gsm8k | Llama-3.2-3B | +20.5 / **+0.5** | +22.6 / -5.8 | +21.3 / -4.0 | +21.6 / -0.4 |
| gsm8k | Qwen2.5-3B | -8.5 / **+0.8** | -8.5 / -1.7 | -9.0 / -0.8 | -10.1 / -0.3 |
| gsm8k | Pythia-2.8B | +7.5 / -5.0 | +12.6 / -5.6 | +12.4 / -4.8 | +11.6 / **-4.2** |
| gsm8k | SmolLM3-3B | -2.7 / **+0.5** | -3.0 / -2.3 | -4.5 / -2.2 | -3.0 / +0.2 |
| hellaswag | Llama-3.2-1B | +4.7 / **-0.7** | +5.0 / -1.4 | +5.0 / -1.0 | +5.0 / -1.2 |
| hellaswag | Qwen2.5-1.5B | +2.8 / **-1.8** | +3.2 / -4.9 | +3.2 / -4.6 | +3.1 / -4.9 |
| hellaswag | Pythia-1.4B | +3.2 / **+0.3** | +3.2 / -0.3 | +3.0 / -0.2 | +3.2 / -0.1 |
| hellaswag | Llama-3.2-3B | +4.7 / -0.3 | +4.9 / -1.7 | +5.0 / -0.2 | +5.0 / **-0.0** |
| hellaswag | Qwen2.5-3B | +2.8 / -0.3 | +3.4 / -1.8 | +3.1 / -0.6 | +3.2 / **-0.2** |
| hellaswag | Pythia-2.8B | +3.8 / **+0.6** | +4.2 / -0.8 | +3.9 / -0.2 | +4.0 / +0.4 |
| hellaswag | SmolLM3-3B | +4.3 / **-1.9** | +4.8 / -2.3 | +4.6 / -9.7 | +4.6 / -8.9 |
| triviaqa | Llama-3.2-1B | +12.4 / +1.1 | +13.4 / +0.6 | +13.3 / +0.6 | +13.0 / **+1.3** |
| triviaqa | Qwen2.5-1.5B | +9.3 / **+1.0** | +10.2 / -1.9 | +10.3 / -1.3 | +10.7 / +0.7 |
| triviaqa | Pythia-1.4B | +11.1 / +1.4 | +12.0 / +1.7 | +12.0 / +1.6 | +11.5 / **+1.8** |
| triviaqa | Llama-3.2-3B | +10.5 / +0.6 | +10.9 / -0.7 | +10.9 / -0.4 | +11.0 / **+0.7** |
| triviaqa | Qwen2.5-3B | +5.5 / +1.2 | +6.1 / -2.4 | +6.3 / -0.4 | +6.9 / **+1.3** |
| triviaqa | Pythia-2.8B | +14.8 / +1.2 | +15.6 / +1.0 | +15.0 / +0.9 | +15.5 / **+1.6** |
| triviaqa | SmolLM3-3B | +17.5 / +0.5 | +18.2 / -0.3 | +18.3 / +0.3 | +18.5 / **+0.7** |
| arc_easy | Llama-3.2-1B | +12.0 / **+2.6** | +6.2 / -2.0 | +6.3 / -1.5 | +6.1 / -2.2 |
| arc_easy | Qwen2.5-1.5B | +6.9 / **+2.8** | +3.6 / -4.8 | +1.8 / -4.9 | +4.4 / -6.5 |
| arc_easy | Pythia-1.4B | +12.3 / **+3.5** | +9.7 / +0.7 | +9.1 / +0.5 | +10.0 / +1.0 |
| arc_easy | Llama-3.2-3B | +10.1 / **+2.4** | +2.1 / -4.1 | +1.9 / -1.2 | +2.5 / -6.3 |
| arc_easy | Qwen2.5-3B | +10.9 / **+1.8** | +5.8 / -6.9 | +5.6 / -3.9 | +7.3 / -10.5 |
| arc_easy | Pythia-2.8B | +9.6 / **+4.2** | +6.0 / +0.0 | +6.2 / +0.2 | +7.6 / -0.3 |
| arc_easy | SmolLM3-3B | +8.0 / **+5.5** | +2.5 / +1.1 | +2.9 / +2.9 | +1.7 / -2.4 |

#### A2. Round robin, single LR per method (archived, superseded)

Cells: **learning** (target Δ) / **forgetting** (mean Δ over the 3 non-target tasks), pp. Bold = least forgetting in the row.

| Target | Model | Full FT | LoRA r16 | LoRA r64 | LoRA r256 |
|---|---|---|---|---|---|
| gsm8k | Llama-3.2-1B | +16.8 / **-2.7** | +16.7 / -5.9 | +18.5 / -7.5 | +19.5 / -6.6 |
| gsm8k | Qwen2.5-1.5B | -5.1 / **+2.8** | -8.9 / +2.4 | -5.6 / +1.4 | -3.6 / +0.0 |
| gsm8k | Pythia-1.4B | +3.7 / -4.3 | +8.0 / -5.0 | +7.8 / **-3.7** | +8.4 / -3.7 |
| gsm8k | Llama-3.2-3B | +20.1 / **+0.2** | +17.7 / -2.5 | +21.7 / -4.1 | +21.6 / -4.0 |
| gsm8k | Qwen2.5-3B | -8.3 / **+1.0** | -6.0 / -0.9 | -10.7 / -1.4 | -9.1 / -1.1 |
| gsm8k | Pythia-2.8B | +7.1 / -4.9 | +11.2 / -6.4 | +12.4 / -4.8 | +12.6 / **-3.8** |
| gsm8k | SmolLM3-3B | -2.8 / +0.3 | +0.1 / **+0.7** | -5.0 / -2.2 | -5.3 / -1.5 |
| hellaswag | Llama-3.2-1B | +4.7 / **-0.7** | +4.8 / -1.3 | +3.7 / -3.6 | +2.1 / -3.6 |
| hellaswag | Qwen2.5-1.5B | +2.9 / **-1.8** | +3.3 / -5.1 | +2.9 / -5.5 | +2.2 / -7.7 |
| hellaswag | Pythia-1.4B | +3.2 / **+0.4** | +3.1 / -0.4 | +3.0 / -1.1 | +2.9 / -1.7 |
| hellaswag | Llama-3.2-3B | +4.7 / **-0.3** | +4.7 / -1.5 | +4.0 / -4.6 | +3.5 / -7.0 |
| hellaswag | Qwen2.5-3B | +2.8 / **-0.1** | +3.5 / -4.8 | +3.2 / -6.4 | +2.6 / -7.9 |
| hellaswag | Pythia-2.8B | +3.8 / **+0.4** | +4.2 / -1.6 | +3.5 / -1.2 | +3.5 / -1.8 |
| hellaswag | SmolLM3-3B | +4.3 / **-2.2** | +4.6 / -2.6 | +3.6 / -5.2 | +2.6 / -5.0 |
| triviaqa | Llama-3.2-1B | +12.4 / +0.3 | +13.4 / **+0.5** | +9.7 / -0.3 | +7.0 / -0.8 |
| triviaqa | Qwen2.5-1.5B | +9.9 / **-0.5** | +10.2 / -2.8 | +9.4 / -5.8 | +8.2 / -10.9 |
| triviaqa | Pythia-1.4B | +12.0 / +1.7 | +12.8 / **+1.8** | +12.3 / +1.4 | +12.1 / +1.7 |
| triviaqa | Llama-3.2-3B | +8.9 / **-0.8** | +10.3 / -3.3 | +7.1 / -5.3 | +4.8 / -5.3 |
| triviaqa | Qwen2.5-3B | +6.2 / **+0.4** | +6.2 / -3.8 | +4.8 / -5.1 | +2.9 / -11.9 |
| triviaqa | Pythia-2.8B | +15.3 / +0.9 | +15.5 / +0.8 | +15.1 / +0.7 | +14.6 / **+1.0** |
| triviaqa | SmolLM3-3B | +17.1 / **-0.2** | +17.6 / -0.9 | +14.4 / -3.2 | +11.7 / -5.1 |
| arc_easy | Llama-3.2-1B | +6.5 / **+0.3** | +6.2 / -2.0 | +4.8 / -4.2 | +1.7 / -6.2 |
| arc_easy | Qwen2.5-1.5B | +2.6 / **-2.0** | +3.6 / -4.8 | +3.9 / -12.3 | +3.2 / -18.8 |
| arc_easy | Pythia-1.4B | +9.5 / **+2.0** | +9.7 / +0.7 | +9.1 / +0.8 | +9.1 / -0.4 |
| arc_easy | Llama-3.2-3B | +1.2 / **-2.5** | +2.1 / -4.1 | -0.8 / -10.4 | -1.7 / -11.1 |
| arc_easy | Qwen2.5-3B | +6.4 / **-0.9** | +5.8 / -6.9 | +5.4 / -15.5 | +4.9 / -18.1 |
| arc_easy | Pythia-2.8B | +7.5 / **+1.3** | +6.0 / +0.0 | +8.6 / -0.5 | +5.9 / -0.4 |
| arc_easy | SmolLM3-3B | +2.4 / **+2.7** | +2.5 / +1.2 | +0.6 / -9.2 | -0.1 / -17.2 |

#### A3. Llama-2-7B round robin — every LR in the sweep

| Target | Method | LR | Learning | Forgetting | GSM8K | TriviaQA | HellaSwag | ARC-Easy |
|---|---|---|---|---|---|---|---|---|
| arc_easy | Full FT | 1.25e-6 | +7.70 | +2.48 | -0.23 | +7.62 | +0.03 | —target— |
| arc_easy | Full FT | 2.5e-6 ★ | +9.26 | +1.95 | -0.15 | +5.78 | +0.23 | —target— |
| arc_easy | Full FT | 5e-6 | +8.63 | -0.10 | -1.06 | +0.39 | +0.36 | —target— |
| arc_easy | LoRA r16 | 1e-4 ★ | +3.37 | -0.87 | -2.12 | -0.87 | +0.39 | —target— |
| arc_easy | LoRA r16 | 2e-4 | +3.03 | -2.10 | -5.99 | +0.51 | -0.84 | —target— |
| arc_easy | LoRA r16 | 4e-4 | +2.95 | -4.90 | -8.49 | -3.50 | -2.72 | —target— |
| arc_easy | LoRA r64 | 2.5e-5 | +2.36 | -0.12 | -1.97 | +0.79 | +0.81 | —target— |
| arc_easy | LoRA r64 | 5e-5 | +4.00 | +0.16 | -1.82 | +1.72 | +0.58 | —target— |
| arc_easy | LoRA r64 | 1e-4 ★ | +4.76 | -1.57 | -6.07 | +2.01 | -0.67 | —target— |
| arc_easy | LoRA r256 | 2.5e-5 | +3.58 | -1.12 | -2.96 | +0.08 | -0.48 | —target— |
| arc_easy | LoRA r256 | 5e-5 ★ | +4.08 | -2.53 | -5.46 | +0.90 | -3.04 | —target— |
| arc_easy | LoRA r256 | 1e-4 | +1.98 | -5.35 | -9.02 | -2.85 | -4.18 | —target— |
| gsm8k | Full FT | 1e-5 | +15.24 | -7.09 | —target— | -17.47 | +0.24 | -4.04 |
| gsm8k | Full FT | 2e-5 | +25.70 | -6.21 | —target— | -17.63 | +0.93 | -1.94 |
| gsm8k | Full FT | 4e-5 ★ | +27.45 | -14.13 | —target— | -36.14 | -0.12 | -6.14 |
| gsm8k | LoRA r16 | 2e-4 ★ | +25.93 | -7.05 | —target— | -18.32 | +0.54 | -3.37 |
| gsm8k | LoRA r16 | 4e-4 | +25.17 | -9.47 | —target— | -22.86 | +0.65 | -6.19 |
| gsm8k | LoRA r16 | 8e-4 | +20.77 | -15.90 | —target— | -36.21 | -2.41 | -9.09 |
| gsm8k | LoRA r64 | 1e-4 | +25.32 | -6.63 | —target— | -18.10 | +1.16 | -2.95 |
| gsm8k | LoRA r64 | 2e-4 ★ | +25.85 | -7.56 | —target— | -19.64 | +0.96 | -4.00 |
| gsm8k | LoRA r64 | 4e-4 | +23.65 | -15.20 | —target— | -36.65 | -2.01 | -6.94 |
| gsm8k | LoRA r256 | 2.5e-5 | +22.82 | -7.41 | —target— | -20.36 | +0.95 | -2.82 |
| gsm8k | LoRA r256 | 5e-5 | +25.62 | -7.70 | —target— | -21.40 | +1.58 | -3.28 |
| gsm8k | LoRA r256 | 1e-4 ★ | +25.70 | -8.02 | —target— | -20.80 | +1.20 | -4.46 |
| hellaswag | Full FT | 1e-5 ★ | +4.82 | -3.66 | -1.29 | -10.27 | —target— | +0.59 |
| hellaswag | Full FT | 2e-5 | +4.78 | -4.54 | -2.73 | -11.99 | —target— | +1.09 |
| hellaswag | Full FT | 4e-5 | +3.04 | -3.72 | -4.93 | -7.74 | —target— | +1.51 |
| hellaswag | LoRA r16 | 5e-5 ★ | +5.01 | -4.17 | +0.76 | -13.27 | —target— | +0.00 |
| hellaswag | LoRA r16 | 1e-4 | +4.79 | -4.49 | +0.38 | -14.51 | —target— | +0.67 |
| hellaswag | LoRA r16 | 2e-4 | +4.90 | -5.71 | -1.52 | -17.62 | —target— | +2.02 |
| hellaswag | LoRA r64 | 1.25e-5 | +4.98 | -3.95 | +0.99 | -13.05 | —target— | +0.21 |
| hellaswag | LoRA r64 | 2.5e-5 ★ | +5.11 | -4.65 | +0.46 | -14.98 | —target— | +0.59 |
| hellaswag | LoRA r64 | 5e-5 | +4.90 | -5.25 | +0.08 | -16.32 | —target— | +0.51 |
| hellaswag | LoRA r256 | 5e-5 ★ | +4.79 | -4.88 | -2.35 | -13.67 | —target— | +1.39 |
| hellaswag | LoRA r256 | 1e-4 | +3.87 | -3.11 | -5.08 | -6.38 | —target— | +2.15 |
| hellaswag | LoRA r256 | 2e-4 | +1.41 | -5.20 | -6.37 | -9.98 | —target— | +0.76 |
| triviaqa | Full FT | 5e-6 | +7.62 | +0.15 | +0.30 | —target— | +0.19 | -0.04 |
| triviaqa | Full FT | 1e-5 ★ | +9.02 | -0.13 | -0.38 | —target— | +0.25 | -0.25 |
| triviaqa | Full FT | 2e-5 | +7.54 | -0.01 | -0.53 | —target— | +0.22 | +0.29 |
| triviaqa | LoRA r16 | 5e-5 | +8.77 | +0.18 | -0.23 | —target— | +0.22 | +0.55 |
| triviaqa | LoRA r16 | 1e-4 ★ | +9.07 | +0.08 | -0.68 | —target— | -0.08 | +1.01 |
| triviaqa | LoRA r16 | 2e-4 | +8.73 | -1.16 | -5.16 | —target— | -0.04 | +1.73 |
| triviaqa | LoRA r64 | 1e-4 ★ | +9.10 | -1.06 | -4.47 | —target— | +0.08 | +1.22 |
| triviaqa | LoRA r64 | 2e-4 | +6.49 | -2.78 | -8.49 | —target— | -0.35 | +0.51 |
| triviaqa | LoRA r64 | 4e-4 | -3.08 | -4.36 | -10.01 | —target— | -2.36 | -0.71 |
| triviaqa | LoRA r256 | 5e-5 ★ | +8.22 | -0.64 | -3.18 | —target— | +0.00 | +1.26 |
| triviaqa | LoRA r256 | 1e-4 | +5.14 | -2.65 | -8.34 | —target— | -0.44 | +0.84 |
| triviaqa | LoRA r256 | 2e-4 | -4.33 | -4.18 | -9.25 | —target— | -2.69 | -0.59 |

★ = selected (highest target accuracy).

#### A4. MetaMathQA, Llama-2-7B, 1 epoch, 8-bit AdamW, plain text format

| Method | LR | GSM8K Δ (learning) | HellaSwag Δ | ARC-Easy Δ | Forgetting (mean of 2) |
|---|---|---|---|---|---|
| Full FT | 1e-5 | +20.39 | -0.13 | -15.40 | -7.77 |
| LoRA r16 | 1e-4 | +27.60 | -0.62 | -17.93 | -9.27 |
| LoRA r64 | 1e-4 | +32.37 | -0.92 | -16.62 | -8.77 |
| LoRA r256 | 5e-5 | +34.34 | -1.28 | -14.77 | -8.03 |

#### A5. MetaMathQA, 4-epoch Lion (Biderman et al. recipe) — every model, method and epoch

| Model | Method | LR | Epoch | GSM8K Δ (learning) | WinoGrande Δ | ARC-C Δ | HellaSwag Δ | Forgetting (mean of 3) |
|---|---|---|---|---|---|---|---|---|
| Llama-2-7B | Full FT | 1e-5 | 1 | +40.71 | -0.39 | -7.51 | -8.19 | -5.36 |
| Llama-2-7B | Full FT | 1e-5 | 2 | +43.67 | -0.32 | -7.17 | -8.09 | -5.19 |
| Llama-2-7B | Full FT | 1e-5 | 4 | +46.63 | -0.63 | -7.25 | -7.68 | -5.19 |
| Llama-2-7B | LoRA r16 | 1e-4 | 1 | +37.23 | +0.71 | -1.28 | -1.21 | -0.59 |
| Llama-2-7B | LoRA r16 | 1e-4 | 2 | +41.62 | -0.08 | -1.37 | -1.29 | -0.91 |
| Llama-2-7B | LoRA r16 | 1e-4 | 4 | +40.18 | -0.24 | -1.45 | -1.27 | -0.99 |
| Llama-2-7B | LoRA r64 | 1e-4 | 1 | +39.58 | +0.71 | -2.73 | -2.72 | -1.58 |
| Llama-2-7B | LoRA r64 | 1e-4 | 2 | +42.84 | -0.24 | -4.61 | -3.20 | -2.68 |
| Llama-2-7B | LoRA r64 | 1e-4 | 4 | +44.96 | -1.66 | -4.27 | -2.67 | -2.86 |
| Llama-2-7B | LoRA r256 | 5e-5 | 1 | +39.20 | +0.71 | -1.71 | -2.98 | -1.32 |
| Llama-2-7B | LoRA r256 | 5e-5 | 2 | +44.50 | -0.08 | -1.96 | -3.00 | -1.68 |
| Llama-2-7B | LoRA r256 | 5e-5 | 4 | +46.10 | -0.39 | -1.37 | -2.39 | -1.38 |
| Llama-3.2-1B | Full FT | 2e-5 | 1 | +25.70 | -4.81 | -10.67 | -21.98 | -12.49 |
| Llama-3.2-1B | Full FT | 2e-5 | 2 | +29.87 | -6.95 | -11.52 | -23.55 | -14.01 |
| Llama-3.2-1B | Full FT | 2e-5 | 4 | +34.42 | -5.37 | -10.15 | -20.41 | -11.98 |
| Llama-3.2-1B | LoRA r16 | 4e-4 | 1 | +20.09 | -2.84 | -9.73 | -14.28 | -8.95 |
| Llama-3.2-1B | LoRA r16 | 4e-4 | 2 | +22.59 | -4.89 | -8.70 | -17.42 | -10.34 |
| Llama-3.2-1B | LoRA r16 | 4e-4 | 4 | +29.87 | -2.76 | -8.70 | -15.96 | -9.14 |
| Llama-3.2-1B | LoRA r64 | 2e-4 | 1 | +20.02 | -2.29 | -6.14 | -14.66 | -7.70 |
| Llama-3.2-1B | LoRA r64 | 2e-4 | 2 | +23.35 | -4.26 | -7.51 | -19.19 | -10.32 |
| Llama-3.2-1B | LoRA r64 | 2e-4 | 4 | +31.46 | -3.71 | -6.40 | -17.57 | -9.23 |
| Llama-3.2-1B | LoRA r256 | 5e-5 | 1 | +27.82 | -0.08 | -4.10 | -7.04 | -3.74 |
| Llama-3.2-1B | LoRA r256 | 5e-5 | 2 | +33.97 | +0.00 | -4.61 | -8.72 | -4.44 |
| Llama-3.2-1B | LoRA r256 | 5e-5 | 4 | +37.23 | +0.47 | -3.07 | -7.21 | -3.27 |
| Llama-3.2-3B | Full FT | 2e-5 | 1 | +22.37 | -9.08 | -17.41 | -22.42 | -16.30 |
| Llama-3.2-3B | Full FT | 2e-5 | 2 | +28.66 | -8.84 | -17.92 | -23.45 | -16.74 |
| Llama-3.2-3B | Full FT | 2e-5 | 4 | +32.75 | -9.63 | -16.98 | -21.74 | -16.12 |
| Llama-3.2-3B | LoRA r16 | 4e-4 | 1 | +19.71 | -3.95 | -9.81 | -13.07 | -8.94 |
| Llama-3.2-3B | LoRA r16 | 4e-4 | 2 | +24.64 | -7.73 | -12.03 | -16.89 | -12.22 |
| Llama-3.2-3B | LoRA r16 | 4e-4 | 4 | +33.51 | -7.26 | -12.63 | -15.46 | -11.78 |
| Llama-3.2-3B | LoRA r64 | 2e-4 | 1 | +22.29 | -4.74 | -10.15 | -12.30 | -9.06 |
| Llama-3.2-3B | LoRA r64 | 2e-4 | 2 | +27.52 | -6.39 | -13.48 | -16.47 | -12.12 |
| Llama-3.2-3B | LoRA r64 | 2e-4 | 4 | +35.63 | -6.95 | -12.80 | -14.82 | -11.52 |
| Llama-3.2-3B | LoRA r256 | 5e-5 | 1 | +32.68 | -2.76 | -3.58 | -5.10 | -3.81 |
| Llama-3.2-3B | LoRA r256 | 5e-5 | 2 | +36.85 | -3.95 | -4.78 | -5.55 | -4.76 |
| Llama-3.2-3B | LoRA r256 | 5e-5 | 4 | +40.79 | -4.34 | -4.10 | -5.41 | -4.61 |
| Qwen2.5-1.5B | Full FT | 2e-5 | 1 | -1.44 | -1.26 | -10.75 | -11.84 | -7.95 |
| Qwen2.5-1.5B | Full FT | 2e-5 | 2 | -0.68 | -2.29 | -10.49 | -13.12 | -8.64 |
| Qwen2.5-1.5B | Full FT | 2e-5 | 4 | +2.58 | -3.39 | -9.39 | -11.82 | -8.20 |
| Qwen2.5-1.5B | LoRA r16 | 4e-4 | 1 | -1.97 | +0.08 | -5.97 | -4.85 | -3.58 |
| Qwen2.5-1.5B | LoRA r16 | 4e-4 | 2 | +2.12 | -1.03 | -5.72 | -7.06 | -4.60 |
| Qwen2.5-1.5B | LoRA r16 | 4e-4 | 4 | +4.40 | -1.42 | -5.72 | -6.32 | -4.49 |
| Qwen2.5-1.5B | LoRA r64 | 2e-4 | 1 | -3.11 | -0.95 | -6.57 | -4.00 | -3.84 |
| Qwen2.5-1.5B | LoRA r64 | 2e-4 | 2 | +1.52 | -0.24 | -6.91 | -6.64 | -4.60 |
| Qwen2.5-1.5B | LoRA r64 | 2e-4 | 4 | +4.85 | -1.82 | -7.08 | -5.27 | -4.72 |
| Qwen2.5-1.5B | LoRA r256 | 5e-5 | 1 | +3.71 | +1.18 | -4.27 | -2.16 | -1.75 |
| Qwen2.5-1.5B | LoRA r256 | 5e-5 | 2 | +5.91 | +2.05 | -3.24 | -1.92 | -1.04 |
| Qwen2.5-1.5B | LoRA r256 | 5e-5 | 4 | +6.14 | +0.47 | -3.24 | -0.83 | -1.20 |
| Qwen2.5-3B | Full FT | 2e-5 | 1 | -10.84 | -7.50 | -11.95 | -15.06 | -11.50 |
| Qwen2.5-3B | Full FT | 2e-5 | 2 | -8.04 | -7.89 | -10.41 | -16.40 | -11.57 |
| Qwen2.5-3B | Full FT | 2e-5 | 4 | -6.52 | -10.10 | -9.30 | -17.13 | -12.18 |
| Qwen2.5-3B | LoRA r16 ⚠ collapsed | 4e-4 | 1 | -66.49 | -12.15 | -23.46 | -19.81 | -18.48 |
| Qwen2.5-3B | LoRA r16 ⚠ collapsed | 4e-4 | 2 | -74.60 | -18.15 | -22.53 | -48.35 | -29.68 |
| Qwen2.5-3B | LoRA r16 ⚠ collapsed | 4e-4 | 4 | -74.07 | -17.52 | -21.93 | -47.95 | -29.13 |
| Qwen2.5-3B | LoRA r64 ⚠ collapsed | 2e-4 | 1 | -30.33 | -11.68 | -14.76 | -14.06 | -13.50 |
| Qwen2.5-3B | LoRA r64 ⚠ collapsed | 2e-4 | 2 | -73.54 | -15.71 | -24.32 | -46.10 | -28.71 |
| Qwen2.5-3B | LoRA r64 ⚠ collapsed | 2e-4 | 4 | -67.55 | -13.58 | -26.11 | -34.85 | -24.85 |
| Qwen2.5-3B | LoRA r256 | 5e-5 | 1 | -1.59 | -2.29 | -1.96 | -2.35 | -2.20 |
| Qwen2.5-3B | LoRA r256 | 5e-5 | 2 | -2.96 | -2.53 | -0.85 | -1.80 | -1.73 |
| Qwen2.5-3B | LoRA r256 | 5e-5 | 4 | -1.67 | -3.63 | -0.43 | -0.99 | -1.68 |

#### A6. Weight displacement joined to learning and forgetting (math target; MetaMathQA Lion epochs 1 and 4)

| Model | Regime | Method | rel. Frobenius | stable rank | σ₁ ratio | Learning | Forgetting | HellaSwag Δ |
|---|---|---|---|---|---|---|---|---|
| Llama-2-7B | Round robin GSM8K AdamW | Full FT | 0.0298 | 95.9 | 0.044 | +27.45 | -14.13 | -0.12 |
| Llama-2-7B | Round robin GSM8K AdamW | LoRA r16 | 0.0319 | 2.2 | 0.294 | +25.93 | -7.05 | +0.54 |
| Llama-2-7B | Round robin GSM8K AdamW | LoRA r64 | 0.0516 | 3.6 | 0.392 | +25.85 | -7.56 | +0.96 |
| Llama-2-7B | Round robin GSM8K AdamW | LoRA r256 | 0.0388 | 7.8 | 0.216 | +25.70 | -8.02 | +1.20 |
| Llama-2-7B | MetaMathQA AdamW 1ep | Full FT | 0.0062 | 93.7 | 0.010 | +20.39 | -7.77 | -0.13 |
| Llama-2-7B | MetaMathQA AdamW 1ep | LoRA r16 | 0.0142 | 1.9 | 0.133 | +27.60 | -9.27 | -0.62 |
| Llama-2-7B | MetaMathQA AdamW 1ep | LoRA r64 | 0.0182 | 3.2 | 0.143 | +32.37 | -8.77 | -0.92 |
| Llama-2-7B | MetaMathQA AdamW 1ep | LoRA r256 | 0.0150 | 5.2 | 0.097 | +34.34 | -8.03 | -1.28 |
| Llama-2-7B | MetaMathQA Lion ep1 | Full FT | 0.0431 | 56.6 | 0.088 | +40.71 | -5.36 | -8.19 |
| Llama-2-7B | MetaMathQA Lion ep4 | Full FT | 0.0634 | 77.1 | 0.110 | +46.63 | -5.19 | -7.68 |
| Llama-2-7B | MetaMathQA Lion ep1 | LoRA r16 | 0.0661 | 2.5 | 0.620 | +37.23 | -0.59 | -1.21 |
| Llama-2-7B | MetaMathQA Lion ep4 | LoRA r16 | 0.1142 | 3.3 | 0.924 | +40.18 | -0.99 | -1.27 |
| Llama-2-7B | MetaMathQA Lion ep1 | LoRA r64 | 0.1027 | 4.6 | 0.715 | +39.58 | -1.58 | -2.72 |
| Llama-2-7B | MetaMathQA Lion ep4 | LoRA r64 | 0.1699 | 6.7 | 0.964 | +44.96 | -2.86 | -2.67 |
| Llama-2-7B | MetaMathQA Lion ep1 | LoRA r256 | 0.0793 | 9.0 | 0.414 | +39.20 | -1.32 | -2.98 |
| Llama-2-7B | MetaMathQA Lion ep4 | LoRA r256 | 0.1275 | 12.0 | 0.558 | +46.10 | -1.38 | -2.39 |
| Llama-3.2-1B | Round robin GSM8K AdamW | Full FT | 0.0090 | 80.5 | 0.013 | +15.30 | -2.60 | +0.70 |
| Llama-3.2-1B | Round robin GSM8K AdamW | LoRA r16 | 0.0614 | 3.8 | 0.323 | +17.10 | -8.17 | -0.20 |
| Llama-3.2-1B | Round robin GSM8K AdamW | LoRA r64 | 0.0505 | 7.5 | 0.200 | +19.00 | -7.63 | +1.20 |
| Llama-3.2-1B | Round robin GSM8K AdamW | LoRA r256 | 0.0255 | 15.5 | 0.073 | +18.50 | -4.97 | +2.20 |
| Llama-3.2-1B | MetaMathQA Lion ep1 | Full FT | 0.0811 | 66.5 | 0.107 | +25.70 | -12.49 | -21.98 |
| Llama-3.2-1B | MetaMathQA Lion ep4 | Full FT | 0.1198 | 88.0 | 0.137 | +34.42 | -11.98 | -20.41 |
| Llama-3.2-1B | MetaMathQA Lion ep1 | LoRA r16 | 0.2587 | 8.0 | 0.837 | +20.09 | -8.95 | -14.28 |
| Llama-3.2-1B | MetaMathQA Lion ep4 | LoRA r16 | 0.3486 | 10.6 | 0.927 | +29.87 | -9.14 | -15.96 |
| Llama-3.2-1B | MetaMathQA Lion ep1 | LoRA r64 | 0.2081 | 15.4 | 0.507 | +20.02 | -7.70 | -14.66 |
| Llama-3.2-1B | MetaMathQA Lion ep4 | LoRA r64 | 0.3156 | 21.8 | 0.604 | +31.46 | -9.23 | -17.57 |
| Llama-3.2-1B | MetaMathQA Lion ep1 | LoRA r256 | 0.0905 | 25.4 | 0.188 | +27.82 | -3.74 | -7.04 |
| Llama-3.2-1B | MetaMathQA Lion ep4 | LoRA r256 | 0.1398 | 31.9 | 0.244 | +37.23 | -3.27 | -7.21 |
| Llama-3.2-3B | Round robin GSM8K AdamW | Full FT | 0.0121 | 87.6 | 0.015 | +20.50 | +0.47 | +0.60 |
| Llama-3.2-3B | Round robin GSM8K AdamW | LoRA r16 | 0.0609 | 2.8 | 0.451 | +22.60 | -5.77 | +0.00 |
| Llama-3.2-3B | Round robin GSM8K AdamW | LoRA r64 | 0.0461 | 4.9 | 0.267 | +21.30 | -4.03 | +1.10 |
| Llama-3.2-3B | Round robin GSM8K AdamW | LoRA r256 | 0.0211 | 11.9 | 0.082 | +21.60 | -0.37 | +1.40 |
| Llama-3.2-3B | MetaMathQA Lion ep1 | Full FT | 0.0806 | 68.3 | 0.127 | +22.37 | -16.30 | -22.42 |
| Llama-3.2-3B | MetaMathQA Lion ep4 | Full FT | 0.1187 | 91.6 | 0.163 | +32.75 | -16.12 | -21.74 |
| Llama-3.2-3B | MetaMathQA Lion ep1 | LoRA r16 | 0.2684 | 6.6 | 1.233 | +19.71 | -8.94 | -13.07 |
| Llama-3.2-3B | MetaMathQA Lion ep4 | LoRA r16 | 0.3937 | 9.2 | 1.434 | +33.51 | -11.78 | -15.46 |
| Llama-3.2-3B | MetaMathQA Lion ep1 | LoRA r64 | 0.2058 | 9.5 | 0.825 | +22.29 | -9.06 | -12.30 |
| Llama-3.2-3B | MetaMathQA Lion ep4 | LoRA r64 | 0.3270 | 16.1 | 0.957 | +35.63 | -11.52 | -14.82 |
| Llama-3.2-3B | MetaMathQA Lion ep1 | LoRA r256 | 0.0756 | 15.8 | 0.251 | +32.68 | -3.81 | -5.10 |
| Llama-3.2-3B | MetaMathQA Lion ep4 | LoRA r256 | 0.1199 | 20.2 | 0.339 | +40.79 | -4.61 | -5.41 |
| Qwen2.5-1.5B | Round robin GSM8K AdamW | Full FT | 0.0052 | 101.0 | 0.006 | -6.10 | +2.90 | +1.00 |
| Qwen2.5-1.5B | Round robin GSM8K AdamW | LoRA r16 | 0.0462 | 4.6 | 0.232 | -5.80 | +1.83 | +1.40 |
| Qwen2.5-1.5B | Round robin GSM8K AdamW | LoRA r64 | 0.0372 | 10.0 | 0.133 | -5.30 | +1.70 | +2.40 |
| Qwen2.5-1.5B | Round robin GSM8K AdamW | LoRA r256 | 0.0184 | 18.3 | 0.049 | -5.20 | +2.60 | +2.40 |
| Qwen2.5-1.5B | MetaMathQA Lion ep1 | Full FT | 0.0530 | 47.9 | 0.091 | -1.44 | -7.95 | -11.84 |
| Qwen2.5-1.5B | MetaMathQA Lion ep4 | Full FT | 0.0794 | 68.4 | 0.112 | +2.58 | -8.20 | -11.82 |
| Qwen2.5-1.5B | MetaMathQA Lion ep1 | LoRA r16 | 0.1820 | 6.2 | 0.774 | -1.97 | -3.58 | -4.85 |
| Qwen2.5-1.5B | MetaMathQA Lion ep4 | LoRA r16 | 0.2733 | 8.7 | 0.909 | +4.40 | -4.49 | -6.32 |
| Qwen2.5-1.5B | MetaMathQA Lion ep1 | LoRA r64 | 0.1424 | 10.5 | 0.500 | -3.11 | -3.84 | -4.00 |
| Qwen2.5-1.5B | MetaMathQA Lion ep4 | LoRA r64 | 0.2274 | 15.9 | 0.591 | +4.85 | -4.72 | -5.27 |
| Qwen2.5-1.5B | MetaMathQA Lion ep1 | LoRA r256 | 0.0618 | 20.4 | 0.164 | +3.71 | -1.75 | -2.16 |
| Qwen2.5-1.5B | MetaMathQA Lion ep4 | LoRA r256 | 0.0985 | 26.6 | 0.216 | +6.14 | -1.20 | -0.83 |
| Qwen2.5-3B | Round robin GSM8K AdamW | Full FT | 0.0069 | 76.4 | 0.015 | -8.50 | +0.77 | +1.10 |
| Qwen2.5-3B | Round robin GSM8K AdamW | LoRA r16 | 0.0453 | 4.0 | 0.257 | -8.50 | -1.73 | +1.10 |
| Qwen2.5-3B | Round robin GSM8K AdamW | LoRA r64 | 0.0355 | 8.5 | 0.151 | -9.00 | -0.77 | +1.70 |
| Qwen2.5-3B | Round robin GSM8K AdamW | LoRA r256 | 0.0168 | 17.3 | 0.051 | -10.10 | -0.30 | +1.80 |
| Qwen2.5-3B | MetaMathQA Lion ep1 | Full FT | 0.0612 | 44.8 | 0.134 | -10.84 | -11.50 | -15.06 |
| Qwen2.5-3B | MetaMathQA Lion ep4 | Full FT | 0.0918 | 63.7 | 0.177 | -6.52 | -12.18 | -17.13 |
| Qwen2.5-3B | MetaMathQA Lion ep1 | LoRA r16 ⚠ diverged | 0.2789 | 5.0 | 1.380 | -66.49 | -18.48 | -19.81 |
| Qwen2.5-3B | MetaMathQA Lion ep4 | LoRA r16 ⚠ diverged | 0.9981 | 2.5 | 9.046 | -74.07 | -29.13 | -47.95 |
| Qwen2.5-3B | MetaMathQA Lion ep1 | LoRA r64 ⚠ diverged | 0.2371 | 7.0 | 1.085 | -30.33 | -13.50 | -14.06 |
| Qwen2.5-3B | MetaMathQA Lion ep4 | LoRA r64 ⚠ diverged | 0.4616 | 12.3 | 1.417 | -67.55 | -24.85 | -34.85 |
| Qwen2.5-3B | MetaMathQA Lion ep1 | LoRA r256 | 0.0647 | 15.4 | 0.228 | -1.59 | -2.20 | -2.35 |
| Qwen2.5-3B | MetaMathQA Lion ep4 | LoRA r256 | 0.1022 | 19.9 | 0.305 | -1.67 | -1.68 | -0.99 |
