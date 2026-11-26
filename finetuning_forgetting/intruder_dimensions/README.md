# Intruder-dimension analysis on a Slurm cluster

This directory implements the weight-space analysis from Shuttleworth et al.,
*LoRA vs Full Fine-tuning: An Illusion of Equivalence* (NeurIPS 2025).

## What is implemented

1. A full checkpoint/readability preflight.
2. Exact thin SVD in float32.
3. The paper's primary left-singular-vector detector (`k=10`, `epsilon=0.5`).
4. Absolute cosine similarity, required because singular-vector signs are arbitrary.
5. Full FT and LoRA evaluated on exactly the same LoRA-targeted matrices.
6. LoRA reconstruction from PEFT adapters without writing merged models.
7. Additional right-vector, projection-residual, spectral-gap, update-norm, and
   singular-value-weighted measurements.
8. Atomic per-matrix JSONL records and restart-safe Slurm execution.
9. CPU aggregation across epsilon and k sweeps, joined to target and non-target accuracy.

## Expected checkpoint layout

This is the original analysis code. It expects checkpoints under a project root:

```
<project-root>/
  .hf_cache/hub/models--<org>--<name>/snapshots/...           # base models
  full_ft_models/finetune_on_<task>/<model>/...                 # full fine-tuned checkpoints
  lora_modelweights/finetune_on_<task>/<model>/lora-r<rank>_seed13/checkpoint-final/  # PEFT adapters
```

The checkpoints themselves are not part of this repository.

## Execution order

From the project root:

```bash
source "${VENV:-.venv}/bin/activate"
python intruder_dimensions/intruder_dimensions.py preflight \
  --project-root "$PWD" \
  --output results/intruder_dimensions/PREFLIGHT.csv

python intruder_dimensions/test_intruder_dimensions.py

PILOT=$(sbatch --parsable intruder_dimensions/submit_intruder_pilot.sbatch)
# Inspect logs/intr-pilot-${PILOT}.{out,err} before proceeding.

ALL=$(sbatch --parsable intruder_dimensions/submit_intruder_all.sbatch)
sbatch --dependency=afterok:${ALL} intruder_dimensions/submit_intruder_aggregate.sbatch
```

The production array contains 28 jobs: seven models by four training tasks.
Each job analyzes Full FT and LoRA ranks 16, 64, and 256 matrix-major so the
base SVD is reused. Concurrency is capped at four GPUs.

## Primary evaluation

- Compare normalized top-10 intruder rate at epsilon 0.5 between accuracy-matched
  Full FT and LoRA conditions.
- Test rank ordering across 16, 64, and 256.
- Relate intruder rate and intruder spectral energy to mean non-target accuracy
  change, target accuracy, RSA, and centered embedding drift.
- Report results by module and layer; do not compare raw totals between
  architectures with different numbers of matrices.

The current final checkpoints support cross-sectional analysis. Reproducing
the paper's evolution-through-training figure requires new intermediate
checkpoints because the original scratch milestones expired.
