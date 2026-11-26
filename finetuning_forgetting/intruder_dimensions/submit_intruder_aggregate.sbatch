#!/bin/bash
# Add your cluster's --account / --partition / --constraint options as needed.
#SBATCH --job-name=intr-agg
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
ROOT="${ROOT:-$PWD}"  # repository root
CODE=$ROOT/intruder_dimensions
cd "$ROOT"
mkdir -p logs results/intruder_dimensions/analysis
source "${VENV:-.venv}/bin/activate"
python "$CODE/aggregate_intruders.py" \
  --raw-root "$ROOT/results/intruder_dimensions/raw" \
  --results-root "$ROOT/results" \
  --output-dir "$ROOT/results/intruder_dimensions/analysis"
