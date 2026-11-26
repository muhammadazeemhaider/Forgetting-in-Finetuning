#!/bin/bash
# Add your cluster's --account / --partition / --constraint options as needed.
#SBATCH --job-name=gen_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
# General-ability retention of the saved generative-benchmark models (ARC-E/C, Winogrande, MMLU; zero-shot).
# Array index -> (order, seed) as in clb.sbatch. Resumable. LIMIT=N for a smoke test. RUNS=dir (default runs_gen5k).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"  # submit from continual_learning/
hostname; date
module purge 2>/dev/null; module load cuda/12.4.1 2>/dev/null || true  # site-specific
source "${VENV:-.venv}/bin/activate"
which python && python --version
nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
ORDERS=(order1 order2 order3)
O=${ORDERS[$((SLURM_ARRAY_TASK_ID / 3))]}; S=$((SLURM_ARRAY_TASK_ID % 3))
python scripts/eval_general.py --runs "${RUNS:-runs_gen5k}" --order "$O" --seed "$S" ${LIMIT:+--limit "$LIMIT"}
date
