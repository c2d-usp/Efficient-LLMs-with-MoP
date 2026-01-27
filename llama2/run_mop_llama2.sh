#!/bin/bash
#
# MoP Experiment for LLaMA-2-7B
# =============================


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/results_llama2"
LOG_DIR="$RESULTS_DIR/logs"

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME="meta-llama/Llama-2-7b-hf"

NUM_ITERATIONS=18

# Fine-tuning epochs (for every iteration)
NUM_EPOCHS=2

# Seeds and pre-generated random decision sequences (18 chars each)
# L = Layer pruning, G = Group (width) pruning
SEEDS=(1 2 3)
SEQUENCES=(
  "LGGLLLGGLLGLGLLGGL"  # Seed 1
  "GGLLGGGLGGGLLLGGLL"  # Seed 2
  "LGLGGLLGLLGLGLGLGG"  # Seed 3
)

# Single-GPU configuration
PRUNING_GPU=0
FINETUNE_GPU=0

TASKS="piqa,hellaswag,winogrande,arc_easy,arc_challenge"

echo "MoP LLaMA-2-7B Experiment"
echo "Model: $MODEL_NAME"
echo "Iterations: $NUM_ITERATIONS"
echo "Seeds: ${SEEDS[*]}"
echo "Pruning GPU: $PRUNING_GPU"
echo "Fine-tune/Eval GPU: $FINETUNE_GPU"
echo "Results dir: $RESULTS_DIR"
echo "Decision sequences:"
for i in "${!SEEDS[@]}"; do
  echo "  seed ${SEEDS[$i]}: ${SEQUENCES[$i]}"
done


mkdir -p "$RESULTS_DIR" "$LOG_DIR" "$RESULTS_DIR/evaluation_logs"

# -----------------------
# PHASE 1: PRUNING (sequential)
# -----------------------
successful_seeds=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  decision_seq="${SEQUENCES[$i]}"

  run_dir="$RESULTS_DIR/run_seed_${seed}"
  mkdir -p "$run_dir"

  pruning_log="$LOG_DIR/pruning_seed_${seed}.log"
  echo "" | tee "$pruning_log"
  echo "==============================================" | tee -a "$pruning_log"
  echo "PRUNING seed=$seed (GPU $PRUNING_GPU)" | tee -a "$pruning_log"
  echo "Decision sequence: $decision_seq" | tee -a "$pruning_log"
  echo "Output: $run_dir" | tee -a "$pruning_log"
  echo "Started at: $(date)" | tee -a "$pruning_log"
  echo "==============================================" | tee -a "$pruning_log"

  CUDA_VISIBLE_DEVICES=$PRUNING_GPU python master_iterator_amp.py \
    --model_name "$MODEL_NAME" \
    --save_dir "$run_dir" \
    --seed "$seed" \
    --num_iterations "$NUM_ITERATIONS" \
    --decision_sequence "$decision_seq" \
    --tuned_original_model "$MODEL_NAME" \
    --tuned_original_tokenizer "$MODEL_NAME" \
    2>&1 | tee -a "$pruning_log"

  status="${PIPESTATUS[0]}"
  if [ "$status" -eq 0 ]; then
    successful_seeds+=("$seed")
    echo "Pruning completed successfully for seed $seed" | tee -a "$pruning_log"
  else
    echo "ERROR: Pruning failed for seed $seed (exit=$status)" | tee -a "$pruning_log"
  fi
done

echo "PHASE 1 COMPLETE"
echo "Successful seeds: ${successful_seeds[*]}"

if [ "${#successful_seeds[@]}" -eq 0 ]; then
  echo "ERROR: No seeds succeeded in pruning. Exiting."
  exit 1
fi

# -----------------------
# PHASE 2: FINETUNE + EVAL (sequential)
# -----------------------
run_single_job() {
  local seed="$1"
  local iteration="$2"
  local num_epochs="$3"
  local gpu_id="$4"

  local run_dir="$RESULTS_DIR/run_seed_${seed}"
  local pruned_model_dir="$run_dir/prune_iteration_${iteration}"
  local tuned_model_dir="$run_dir/prune_iteration_${iteration}_ft_${num_epochs}ep"
  local eval_output_dir="${tuned_model_dir}_eval"

  local finetuning_log="$LOG_DIR/finetuning_seed_${seed}_iter_${iteration}_${num_epochs}ep.log"
  local eval_log="$RESULTS_DIR/evaluation_logs/seed_${seed}_iter_${iteration}_${num_epochs}ep_eval.log"

  if [ ! -d "$pruned_model_dir" ]; then
    echo "ERROR: Pruned model not found: $pruned_model_dir" | tee "$finetuning_log"
    return 1
  fi

  if [ -d "$tuned_model_dir" ] && [ -d "$eval_output_dir" ]; then
    echo "SKIP (already complete): seed=$seed iter=$iteration epochs=$num_epochs"
    return 0
  fi

  echo "==============================================" | tee "$finetuning_log"
  echo "Fine-tuning: seed=$seed iter=$iteration epochs=$num_epochs (GPU $gpu_id)" | tee -a "$finetuning_log"
  echo "Input:  $pruned_model_dir" | tee -a "$finetuning_log"
  echo "Output: $tuned_model_dir" | tee -a "$finetuning_log"
  echo "Started at: $(date)" | tee -a "$finetuning_log"
  echo "==============================================" | tee -a "$finetuning_log"

  CUDA_VISIBLE_DEVICES=$gpu_id python fine_tuning.py \
    --pruned_model_dir "$pruned_model_dir" \
    --tuned_model_dir "$tuned_model_dir" \
    --num_epochs "$num_epochs" \
    2>&1 | tee -a "$finetuning_log"

  ft_status="${PIPESTATUS[0]}"
  if [ "$ft_status" -ne 0 ]; then
    echo "ERROR: Fine-tuning failed (exit=$ft_status)" | tee -a "$finetuning_log"
    return 1
  fi

  echo "Evaluation: seed=$seed iter=$iteration" | tee -a "$finetuning_log"
  CUDA_VISIBLE_DEVICES=$gpu_id lm_eval \
    --model hf \
    --model_args pretrained="$tuned_model_dir" \
    --tasks "$TASKS" \
    --output_path "$eval_output_dir" \
    > "$eval_log" 2>&1

  eval_status=$?
  if [ "$eval_status" -ne 0 ]; then
    echo "ERROR: Evaluation failed (exit=$eval_status)" | tee -a "$finetuning_log"
    return 1
  fi

  echo "Done: seed=$seed iter=$iteration epochs=$num_epochs" | tee -a "$finetuning_log"
  return 0
}

TOTAL_JOBS=$(( ${#successful_seeds[@]} * NUM_ITERATIONS ))
echo "PHASE 2: Fine-tune + eval (total jobs: $TOTAL_JOBS)"

job_idx=0
for seed in "${successful_seeds[@]}"; do
  for ((iter=1; iter<=NUM_ITERATIONS; iter++)); do
    job_idx=$((job_idx + 1))
    echo ""
    echo "Running job $job_idx/$TOTAL_JOBS: seed=$seed iter=$iter epochs=$NUM_EPOCHS"
    run_single_job "$seed" "$iter" "$NUM_EPOCHS" "$FINETUNE_GPU" || exit 1
  done
done

echo ""
echo "EXPERIMENT COMPLETED"
echo "Results: $RESULTS_DIR"
