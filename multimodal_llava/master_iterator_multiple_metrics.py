import argparse
import os
import random
import torch
import gc
import numpy as np
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

from prune_groups_multiple_metrics import prune_groups
from prune_layers_multiple_metrics import prune_layers

# Global list to store decisions for each iteration
decisions = []

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_random_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Seed: {args.seed}")
    print(f"Decision sequence: {args.decision_sequence}")

    # Parse the decision sequence
    global decisions
    decisions = list(args.decision_sequence.upper())
    
    for iteration in range(1, args.num_iterations + 1):
        print('#'*100)
        print('#'*45, iteration, '#'*45)

        if iteration > 1:
            args.model_name = args.save_dir + '/' + f'prune_iteration_{iteration-1}'
        
        # Get decision for this iteration
        decision_idx = (iteration - 1) % len(decisions)
        decision = decisions[decision_idx]
        
        if decision == 'L':
            prune_layers(args, iteration, args.save_dir, tune_eval=False)
            pruned_type = "LAYER"
            print(f'ITERATION {iteration} PRUNED LAYER')
        elif decision == 'G':
            prune_groups(args, iteration, args.save_dir, tune_eval=False)
            pruned_type = "GROUP"
            print(f'ITERATION {iteration} PRUNED GROUP')
        else:
            raise ValueError(f"Invalid decision '{decision}' at iteration {iteration}. Use 'L' for layer or 'G' for group.")

        # Clear GPU memory after each iteration
        print("Clearing GPU memory after iteration...")
        gc.collect()
        torch.cuda.empty_cache()

        print('#'*100)

    print_and_save_summary(args.decision_sequence, args.save_dir)


def print_and_save_summary(decision_sequence, save_dir):
    print('\n' + '='*80)
    print(f"PRUNING SUMMARY")
    print('='*80)
    print(f"{'Iteration':<10} {'Pruned Type':<15}")
    print('-'*80)
    
    decisions_list = list(decision_sequence.upper())
    for i in range(len(decisions_list)):
        iteration = i + 1
        decision = decisions_list[i]
        pruned_type = "LAYER" if decision == 'L' else "GROUP"
        print(f"{iteration:<10} {pruned_type:<15}")
    
    print('-'*80)
    print(f"Total iterations: {len(decisions_list)}")
    layers_pruned = sum(1 for d in decisions_list if d == 'L')
    groups_pruned = sum(1 for d in decisions_list if d == 'G')
    print(f"Layers pruned: {layers_pruned}")
    print(f"Groups pruned: {groups_pruned}")
    print(f"Decision sequence: {decision_sequence}")
    print('='*80)
    
    # Save to CSV
    csv_filename = os.path.join(save_dir, f"pruning_summary_{decision_sequence}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Iteration', 'Pruned_Type'])
        
        # Write data
        for i, decision in enumerate(decisions_list):
            iteration = i + 1
            pruned_type = "LAYER" if decision == 'L' else "GROUP"
            writer.writerow([iteration, pruned_type])
        
        # Write summary rows
        writer.writerow([])  # Empty row
        writer.writerow(['Summary', ''])
        writer.writerow(['Total_Iterations', len(decisions_list)])
        writer.writerow(['Layers_Pruned', layers_pruned])
        writer.writerow(['Groups_Pruned', groups_pruned])
        writer.writerow(['Decision_Sequence', decision_sequence])
    
    print(f"Summary saved to: {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf', help='base model name')
    parser.add_argument('--save_dir', type=str, required=True, 
                        help='Save directory')

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--num_iterations', type=int, default=20, help='number of iterations')
    
    # Add decision sequence parameter
    parser.add_argument('--decision_sequence', type=str, required=True,
                       help="Decision sequence string (e.g., 'LGLGLG' where L=layer, G=group)")
    
    # Add tuned_original_model argument and tokenizer
    parser.add_argument('--tuned_original_model', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='Path to the fine-tuned original model for reference')
    parser.add_argument('--tuned_original_tokenizer', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='Path to the fine-tuned original tokenizer for reference')
    
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
