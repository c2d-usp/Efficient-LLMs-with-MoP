#!/bin/bash
#
# MoP Experiment for LLaMA-3-8B


# Set up paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/results_llama3"
LOG_DIR="$RESULTS_DIR/logs"

# Model
MODEL_NAME="meta-llama/Meta-Llama-3-8B"

# MoP Configuration
NUM_ITERATIONS=12
MLP_ONLY="--mlp_only"

# Fine-tuning epochs (for every iteration)
NUM_EPOCHS=2
FT_ITERATIONS=(8 12)

# L = Layer pruning, G = Group (width) pruning
SEEDS=(1)
SEQUENCES=(
    "LGGLLLGGLLGLGLLGGL"   # Seed 1:
)

# GPU Configuration
PRUNING_GPU=0
FINETUNE_GPU=0


echo "MoP LLaMA-3-8B MLP-Only Experiment"
echo ""
echo "Model Configuration:"
echo "  - Model: $MODEL_NAME"
echo "  - Pruning mode: MLP-only (no attention head pruning)"
echo ""
echo "MoP Configuration:"
echo "  - Total iterations: $NUM_ITERATIONS"
echo "  - Seeds: ${SEEDS[@]}"
echo "  - Path selection: Pre-generated random sequences"
echo ""
echo "Fine-tuning:"
echo "  - Iterations: ${FT_ITERATIONS[*]} ($NUM_EPOCHS epochs each)"
echo ""
echo "Evaluation:"
echo "  - Tasks: piqa, hellaswag, winogrande, arc_easy, arc_challenge"
echo "  - Framework: lm_eval (EleutherAI LM Harness)"
echo ""
echo "GPU Configuration:"
echo "  - Pruning: GPU $PRUNING_GPU (sequential)"
echo "  - Fine-tuning + Eval: GPU $FINETUNE_GPU (sequential)"
echo ""
echo "Decision Sequences:"
for i in "${!SEEDS[@]}"; do
    echo "  - Seed ${SEEDS[$i]}: ${SEQUENCES[$i]}"
done
echo ""


# Create results and log directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR/evaluation_logs"

# PHASE 1: PRUNING
echo ""
echo "PHASE 1: Running all pruning experiments sequentially on GPU $PRUNING_GPU"

successful_seeds=()
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    decision_seq=${SEQUENCES[$i]}
    
    echo ""
    echo "Pruning experiment $((i+1))/${#SEEDS[@]}: seed $seed"
    echo "Decision sequence: $decision_seq"
    
    # Create save directory for this run
    run_dir="$RESULTS_DIR/run_seed_${seed}"
    mkdir -p "$run_dir"
    
    # Log files
    pruning_log="$LOG_DIR/pruning_seed_${seed}.log"
    
    # Run pruning iterations
    echo "Starting pruning for seed $seed at $(date)" | tee "$pruning_log"
    
    CUDA_VISIBLE_DEVICES=$PRUNING_GPU python master_iterator_amp.py \
        --model_name "$MODEL_NAME" \
        --save_dir "$run_dir" \
        --seed "$seed" \
        --num_iterations "$NUM_ITERATIONS" \
        --decision_sequence "$decision_seq" \
        --tuned_original_model "$MODEL_NAME" \
        --tuned_original_tokenizer "$MODEL_NAME" \
        $MLP_ONLY \
        2>&1 | tee -a "$pruning_log"
    
    # Check if pruning was successful
    status="${PIPESTATUS[0]}"
    if [ "$status" -eq 0 ]; then
        echo "Pruning completed successfully for seed $seed" | tee -a "$pruning_log"
        successful_seeds+=($seed)
    else
        echo "Error: Pruning failed for seed $seed (exit code: $status)" | tee -a "$pruning_log"
    fi
    
    echo "Completed pruning for seed $seed at $(date)" | tee -a "$pruning_log"
done

echo ""
echo "PHASE 1 COMPLETE: Pruning finished for all experiments"
echo "Successful seeds: ${successful_seeds[@]}"

if [ ${#successful_seeds[@]} -eq 0 ]; then
    echo "Error: No pruning experiments succeeded. Exiting."
    exit 1
fi

# PHASE 2: FINE-TUNING + EVALUATION
echo ""
echo "PHASE 2: Running LoRA fine-tuning + evaluation"
echo "         GPU: $FINETUNE_GPU"

TOTAL_JOBS=$(( ${#successful_seeds[@]} * ${#FT_ITERATIONS[@]} ))
echo "Total fine-tuning + evaluation jobs: $TOTAL_JOBS"
echo ""

# Function to fine-tune and evaluate a single checkpoint
run_single_job() {
    local seed=$1
    local iteration=$2
    local num_epochs=$3
    local gpu_id=$4
    
    local run_dir="$RESULTS_DIR/run_seed_${seed}"
    local pruned_model_dir="$run_dir/prune_iteration_${iteration}"
    local tuned_model_dir="$run_dir/prune_iteration_${iteration}_ft_${num_epochs}ep"
    local eval_output_dir="${tuned_model_dir}_eval"
    local finetuning_log="$LOG_DIR/finetuning_seed_${seed}_iter_${iteration}_${num_epochs}ep.log"
    local eval_log="$RESULTS_DIR/evaluation_logs/seed_${seed}_iter_${iteration}_${num_epochs}ep_eval.log"
    
    echo "[GPU $gpu_id] Starting: seed=$seed, iter=$iteration, epochs=$num_epochs"
    
    if [ ! -d "$pruned_model_dir" ]; then
        echo "Warning: Pruned model not found: $pruned_model_dir" | tee "$finetuning_log"
        return 1
    fi
    
    # Fine-tuning
    {
        echo "=============================================="
        echo "Fine-tuning: seed=$seed, iter=$iteration, epochs=$num_epochs"
        echo "GPU: $gpu_id"
        echo "Input: $pruned_model_dir"
        echo "Output: $tuned_model_dir"
        echo "Started at: $(date)"
        echo "=============================================="
    } | tee "$finetuning_log"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python fine_tuning.py \
        --pruned_model_dir "$pruned_model_dir" \
        --tuned_model_dir "$tuned_model_dir" \
        --num_epochs "$num_epochs" \
        2>&1 | tee -a "$finetuning_log"
    
    local ft_status="${PIPESTATUS[0]}"
    if [ "$ft_status" -ne 0 ]; then
        echo "Error: Fine-tuning failed (exit code: $ft_status)" | tee -a "$finetuning_log"
        return 1
    fi
    
    echo "Fine-tuning completed at: $(date)" | tee -a "$finetuning_log"
    
    # Evaluation
    echo "Starting evaluation..." | tee -a "$finetuning_log"
    
    CUDA_VISIBLE_DEVICES=$gpu_id lm_eval \
        --model hf \
        --model_args pretrained="$tuned_model_dir" \
        --tasks piqa,hellaswag,winogrande,arc_easy,arc_challenge \
        --output_path "$eval_output_dir" \
        > "$eval_log" 2>&1
    
    local eval_status=$?
    if [ $eval_status -eq 0 ]; then
        echo "Evaluation completed at: $(date)" | tee -a "$finetuning_log"
    else
        echo "Error: Evaluation failed (exit code: $eval_status)" | tee -a "$finetuning_log"
        return 1
    fi
    
    echo "[GPU $gpu_id] Completed: seed=$seed, iter=$iteration, epochs=$num_epochs"
    return 0
}

completed=0
failed=0
job_idx=0

for seed in "${successful_seeds[@]}"; do
    for iter in "${FT_ITERATIONS[@]}"; do
        job_idx=$((job_idx + 1))
        echo ""
        echo "Running job $job_idx/$TOTAL_JOBS on GPU $FINETUNE_GPU: seed=$seed, iter=$iter, epochs=$NUM_EPOCHS"
        if ! run_single_job "$seed" "$iter" "$NUM_EPOCHS" "$FINETUNE_GPU"; then
            failed=$((failed + 1))
            echo "Error: Job failed: seed=$seed, iter=$iter, epochs=$NUM_EPOCHS"
            exit 1
        fi
        completed=$((completed + 1))
    done
done

echo "All jobs completed: $completed/$TOTAL_JOBS, $failed failed"

# FINAL SUMMARY
echo ""
echo "EXPERIMENT COMPLETED"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Directory structure:"
for seed in "${successful_seeds[@]}"; do
    echo "  run_seed_${seed}/"
    for iter in "${FT_ITERATIONS[@]}"; do
        echo "    ├── prune_iteration_${iter}/"
        echo "    ├── prune_iteration_${iter}_ft_${NUM_EPOCHS}ep/"
        echo "    └── prune_iteration_${iter}_ft_${NUM_EPOCHS}ep_eval/"
    done
done
echo ""
echo "Configuration used:"
echo "  - Model: $MODEL_NAME"
echo "  - Pruning: MLP-only"
echo "  - Total iterations: $NUM_ITERATIONS"
echo "  - Seeds: ${successful_seeds[@]}"
echo "  - Fine-tuning:"
echo "    - Iterations: ${FT_ITERATIONS[*]} ($NUM_EPOCHS epochs each)"
echo ""
echo "Logs:"
echo "  - Pruning: $LOG_DIR/pruning_seed_*.log"
echo "  - Fine-tuning: $LOG_DIR/finetuning_seed_*_iter_*_*ep.log"
echo "  - Evaluation: $RESULTS_DIR/evaluation_logs/seed_*_iter_*_*ep_eval.log"
echo ""
