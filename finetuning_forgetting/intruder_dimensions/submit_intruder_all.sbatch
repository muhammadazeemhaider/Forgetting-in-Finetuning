#!/bin/bash
# Add your cluster's --account / --partition / --constraint options as needed.
#SBATCH --job-name=intr-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-27%4
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
ROOT="${ROOT:-$PWD}"  # repository root
CODE=$ROOT/intruder_dimensions
cd "$ROOT"
mkdir -p logs results/intruder_dimensions/raw
module purge
module load cuda/12.4.1 2>/dev/null || true  # site-specific
source "${VENV:-.venv}/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

MODELS=(llama-3.2-1b qwen2.5-1.5b pythia-1.4b llama-3.2-3b qwen2.5-3b pythia-2.8b smollm3-3b)
TASKS=(gsm8k triviaqa hellaswag arc_easy)
IDX=${SLURM_ARRAY_TASK_ID}
MODEL=${MODELS[$((IDX / 4))]}
TASK=${TASKS[$((IDX % 4))]}
echo "[$(date -Is)] index=$IDX model=$MODEL train_task=$TASK host=$(hostname)"

python "$CODE/intruder_dimensions.py" analyze \
  --project-root "$ROOT" \
  --output-root "$ROOT/results/intruder_dimensions/raw" \
  --model "$MODEL" \
  --train-task "$TASK" \
  --device cuda \
  --top-k 10 \
  --threshold 0.5
