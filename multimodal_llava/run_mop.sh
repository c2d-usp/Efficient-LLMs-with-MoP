#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Params (override via env: MODEL, OUT, SEQ, ITERS, DEVICE)
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
OUT="${OUT:-$SCRIPT_DIR/outputs/llava15_7b_mop_seed2}"
SEQ="${SEQ:-GGLLGGGLGGGLLLGGLL}"      # sequence cycles; e.g., LG, G, LGL, etc.
ITERS="${ITERS:-17}"  # total iterations to run
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$OUT"

LOG_FILE="$OUT/run.log"

echo "[$(date)] Starting MOP prune: model=$MODEL seq=$SEQ iters=$ITERS out=$OUT device=$DEVICE" | tee "$LOG_FILE"
python -u "$SCRIPT_DIR/master_iterator_multiple_metrics.py" \
  --model_name "$MODEL" \
  --save_dir "$OUT" \
  --decision_sequence "$SEQ" \
  --num_iterations "$ITERS" \
  --device "$DEVICE" 2>&1 | tee -a "$LOG_FILE"
echo "[$(date)] Done. Outputs under -> $OUT" | tee -a "$LOG_FILE"
