#!/bin/bash
# Add your cluster's --account / --partition / --constraint options as needed.
#SBATCH --job-name=intr-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

# One Qwen block (seven matrices), all four tuned variants.  This is large
# enough to validate reconstruction, exact CUDA SVD, JSONL resume behavior,
# and realistic timing before launching the 28-item production array.
python "$CODE/intruder_dimensions.py" analyze \
  --project-root "$ROOT" \
  --output-root "$ROOT/results/intruder_dimensions/pilot_raw" \
  --model qwen2.5-1.5b \
  --train-task triviaqa \
  --max-matrices 7 \
  --device cuda \
  --top-k 10 \
  --threshold 0.5
