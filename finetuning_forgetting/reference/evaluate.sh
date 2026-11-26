#!/usr/bin/env bash
# Reference evaluation command (re-implementation; see reference/README.md).
#
# Scores a base model or a merged fine-tuned checkpoint with lm-evaluation-harness on vLLM,
# bf16, each task's default few-shot setting (GSM8K 5-shot, everything else 0-shot).
#
# Usage:
#   bash reference/evaluate.sh <model-or-checkpoint-dir> <output-dir> [suite]
#   suite = roundrobin (default): gsm8k, triviaqa, hellaswag, arc_easy
#   suite = metamath:             gsm8k, hellaswag, arc_challenge, winogrande  (Biderman et al.)
#
# Metrics used in the analysis: GSM8K exact_match (flexible-extract), TriviaQA token F1 /
# exact match, HellaSwag and ARC acc_norm, WinoGrande acc.
set -euo pipefail

MODEL=${1:?model path or HF id}
OUT=${2:?output directory}
SUITE=${3:-roundrobin}

case "$SUITE" in
  roundrobin) TASKS=gsm8k,triviaqa,hellaswag,arc_easy ;;
  metamath)   TASKS=gsm8k,hellaswag,arc_challenge,winogrande ;;
  *) echo "unknown suite: $SUITE" >&2; exit 1 ;;
esac

lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL},dtype=bfloat16,gpu_memory_utilization=0.85" \
  --tasks "$TASKS" \
  --batch_size auto \
  --log_samples \
  --output_path "$OUT"
