#!/bin/bash
# Add your cluster's --account / --partition / --constraint options as needed.
#SBATCH --job-name=clb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# Standard CL Benchmark x compensated-LoRA methods, Llama-3.2-1B. Array index -> (order, seed):
#   i = 3*order_idx + seed  (0-8). Each task: shared first task + 7 sequential methods + MTL reference.
#   Resumable: finished methods are skipped. METHODS=a,b,c restricts to a method list; OUT_ROOT=dir sets the runs folder (default runs); EXTRA="..." appends run_order.py flags. Smoke:  SMOKE=1 sbatch --array=0 jobs/clb.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from continual_learning/
hostname; date
module purge 2>/dev/null; module load cuda/12.4.1 2>/dev/null || true  # site-specific
source "${VENV:-.venv}/bin/activate"
which python && python --version
nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
ORDERS=(order1 order2 order3)
O=${ORDERS[$((SLURM_ARRAY_TASK_ID / 3))]}; S=$((SLURM_ARRAY_TASK_ID % 3))
if [ "${SMOKE:-0}" = 1 ]; then
    python scripts/run_order.py --order "$O" --seed "$S" --max-train 64 --out-root "${OUT_ROOT:-runs}" ${METHODS:+--methods "$METHODS"} ${EXTRA:-}
else
    python scripts/run_order.py --order "$O" --seed "$S" --out-root "${OUT_ROOT:-runs}" ${METHODS:+--methods "$METHODS"} ${EXTRA:-}
fi
date
