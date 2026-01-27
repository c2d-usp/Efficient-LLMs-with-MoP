#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

MODEL_NAME="meta-llama/Llama-2-7b-hf"
NUM_ITERATIONS=18
SEED=42

PRUNING_GPU=0
FINETUNE_GPU=0
NUM_EPOCHS=2

TASKS="piqa,hellaswag,winogrande,arc_easy,arc_challenge"

BASE_RESULTS_DIR="$SCRIPT_DIR/results_llama2"

RUN_NAMES=("only_layers" "only_neurons")
RUN_SEQS=("LLLLLLLLLLLLLLLLLL" "GGGGGGGGGGGGGGGGGG")

ensure_empty_or_create_dir() {
  local dir="$1"
  if [ -d "$dir" ] && [ "$(ls -A "$dir")" ]; then
    echo "ERROR: Output dir exists and is NOT empty: $dir"
    echo "Refusing to run to avoid overwriting."
    exit 1
  fi
  mkdir -p "$dir"
}

run_one_extreme() {
  local run_name="$1"
  local decision_seq="$2"

  local run_root="$BASE_RESULTS_DIR/$run_name"
  ensure_empty_or_create_dir "$run_root"

  local log_dir="$run_root/logs"
  local eval_log_dir="$run_root/evaluation_logs"
  mkdir -p "$log_dir" "$eval_log_dir"

  echo "======================================================="
  echo "RUN: $run_name"
  echo "  - Model: $MODEL_NAME"
  echo "  - Decision sequence: $decision_seq"
  echo "  - Iterations: $NUM_ITERATIONS"
  echo "  - Seed: $SEED"
  echo "  - Pruning GPU: $PRUNING_GPU (sequential)"
  echo "  - Fine-tune + Eval GPU: $FINETUNE_GPU (sequential)"
  echo "  - Epochs (all iters): $NUM_EPOCHS"
  echo "======================================================="
  echo "Press Ctrl+C within 10 seconds to cancel..."
  sleep 10

  echo ""
  echo "======================================================="
  echo "PHASE 1: PRUNING (sequential) -> $run_root"
  echo "======================================================="

  local pruning_log="$log_dir/pruning.log"
  if ! CUDA_VISIBLE_DEVICES=$PRUNING_GPU python master_iterator_amp.py \
      --model_name "$MODEL_NAME" \
      --save_dir "$run_root" \
      --seed "$SEED" \
      --num_iterations "$NUM_ITERATIONS" \
      --decision_sequence "$decision_seq" \
      --tuned_original_model "$MODEL_NAME" \
      --tuned_original_tokenizer "$MODEL_NAME" \
      2>&1 | tee -a "$pruning_log"; then
    echo "ERROR: Pruning failed for run: $run_name"
    return 1
  fi

  echo ""
  echo "======================================================="
  echo "PHASE 2: FINE-TUNING + EVAL (all iterations, ${NUM_EPOCHS} epochs)"
  echo "======================================================="

  run_single_job() {
    local iteration="$1"
    local num_epochs="$2"
    local gpu_id="$3"

    local pruned_model_dir="$run_root/prune_iteration_${iteration}"
    local tuned_model_dir="$run_root/prune_iteration_${iteration}_ft_${num_epochs}ep"
    local eval_output_dir="${tuned_model_dir}_eval"

    local finetuning_log="$log_dir/finetuning_iter_${iteration}_${num_epochs}ep.log"
    local eval_log="$eval_log_dir/iter_${iteration}_${num_epochs}ep_eval.log"

    echo "[GPU $gpu_id] Starting: iter=$iteration, epochs=$num_epochs"

    if [ ! -d "$pruned_model_dir" ]; then
      echo "ERROR: Pruned model not found: $pruned_model_dir" | tee -a "$finetuning_log"
      return 1
    fi

    if [ -d "$tuned_model_dir" ] || [ -d "$eval_output_dir" ]; then
      echo "ERROR: Output already exists (refusing to overwrite):" | tee -a "$finetuning_log"
      echo "  tuned_model_dir=$tuned_model_dir" | tee -a "$finetuning_log"
      echo "  eval_output_dir=$eval_output_dir" | tee -a "$finetuning_log"
      return 1
    fi

    {
      echo "=============================================="
      echo "Fine-tuning: iter=$iteration, epochs=$num_epochs"
      echo "GPU: $gpu_id"
      echo "Input: $pruned_model_dir"
      echo "Output: $tuned_model_dir"
      echo "Started at: $(date)"
      echo "=============================================="
    } | tee "$finetuning_log"

    if ! CUDA_VISIBLE_DEVICES=$gpu_id python fine_tuning.py \
        --pruned_model_dir "$pruned_model_dir" \
        --tuned_model_dir "$tuned_model_dir" \
        --num_epochs "$num_epochs" \
        2>&1 | tee -a "$finetuning_log"; then
      echo "ERROR: Fine-tuning failed: iter=$iteration" | tee -a "$finetuning_log"
      return 1
    fi

    echo "Fine-tuning completed at: $(date)" | tee -a "$finetuning_log"
    echo "Starting evaluation..." | tee -a "$finetuning_log"

    if ! CUDA_VISIBLE_DEVICES=$gpu_id lm_eval \
        --model hf \
        --model_args pretrained="$tuned_model_dir" \
        --tasks "$TASKS" \
        --output_path "$eval_output_dir" \
        > "$eval_log" 2>&1; then
      echo "ERROR: Evaluation failed: iter=$iteration" | tee -a "$finetuning_log"
      return 1
    fi

    echo "Evaluation completed at: $(date)" | tee -a "$finetuning_log"
    echo "[GPU $gpu_id] Completed: iter=$iteration, epochs=$num_epochs"
    return 0
  }

  for iter in $(seq 1 "$NUM_ITERATIONS"); do
    echo ""
    echo "Running iter=$iter (epochs=$NUM_EPOCHS) on GPU $FINETUNE_GPU"
    run_single_job "$iter" "$NUM_EPOCHS" "$FINETUNE_GPU"
  done

  echo ""
  echo "======================================================="
  echo "RUN COMPLETE: $run_name"
  echo "Output: $run_root"
  echo "======================================================="
  return 0
}

for idx in "${!RUN_NAMES[@]}"; do
  run_one_extreme "${RUN_NAMES[$idx]}" "${RUN_SEQS[$idx]}"
done

echo ""
echo "======================================================="
echo "ALL EXTREMES COMPLETED"
echo "Results: $BASE_RESULTS_DIR"
echo "======================================================="
