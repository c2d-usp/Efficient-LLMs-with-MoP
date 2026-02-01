#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Params (override via env: MODEL, OUT_BASE, ITERS, DEVICE)
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
OUT_BASE="${OUT_BASE:-$SCRIPT_DIR/outputs/llava15_7b_mop}"
ITERS="${ITERS:-17}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

SEEDS=(1 2 3)
SEQUENCES=(
  "LGGLLLGGLLGLGLLGGL"  # Seed 1
  "GGLLGGGLGGGLLLGGLL"  # Seed 2
  "LGLGGLLGLLGLGLGLGG"  # Seed 3
)

if [[ ${#SEEDS[@]} -ne ${#SEQUENCES[@]} ]]; then
  echo "SEEDS and SEQUENCES must have the same length" >&2
  exit 1
fi

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  seq_val="${SEQUENCES[$i]}"

  OUT_DIR="${OUT_BASE}_seed${seed}"
  mkdir -p "$OUT_DIR"

  LOG_FILE="$OUT_DIR/run.log"

  echo "[$(date)] Starting MOP prune: model=$MODEL seq=$seq_val iters=$ITERS out=$OUT_DIR device=$DEVICE" | tee "$LOG_FILE"
  python -u "$SCRIPT_DIR/master_iterator_multiple_metrics.py" \
    --model_name "$MODEL" \
    --save_dir "$OUT_DIR" \
    --decision_sequence "$seq_val" \
    --num_iterations "$ITERS" \
    --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
  echo "[$(date)] Done. Outputs under -> $OUT_DIR" | tee -a "$LOG_FILE"
done
