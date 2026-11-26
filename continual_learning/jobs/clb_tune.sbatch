#!/bin/bash
# Add your cluster's --account / --partition / --constraint options as needed.
#SBATCH --job-name=clbtune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --array=0-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# TIES hyperparameter selection on the VALIDATION split (never test): order1, seed 0,
# density in {0.2,0.5,0.8} x lambda in {1.0,1.5,2.0}; runs ties_only, ties_repair, ties_repair_lwf.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from continual_learning/
hostname; date; module purge 2>/dev/null; module load cuda/12.4.1 2>/dev/null || true  # site-specific
source "${VENV:-.venv}/bin/activate"; which python
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
D=(0.2 0.2 0.2 0.5 0.5 0.5 0.8 0.8 0.8); L=(1.0 1.5 2.0 1.0 1.5 2.0 1.0 1.5 2.0)
d=${D[$SLURM_ARRAY_TASK_ID]}; l=${L[$SLURM_ARRAY_TASK_ID]}
python scripts/run_order.py --order order1 --seed 0 --eval-split dev --ties-density $d --ties-lambda $l \
    --out-root runs_tune/d${d}_l${l} --methods ties_only,ties_repair,ties_repair_lwf
date
