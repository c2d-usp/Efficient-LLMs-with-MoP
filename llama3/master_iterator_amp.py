import argparse
import os
import random
import torch
import gc
import numpy as np
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

from prune_groups_amp_llama3 import prune_groups
from prune_layers_amp import prune_layers

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

    initial_params = None
    params_after_iterations = []
    
    for iteration in range(1, args.num_iterations + 1):
        print('#'*100)
        print('#'*45, iteration, '#'*45)

        if iteration > 1:
            args.model_name = args.save_dir + '/' + f'prune_iteration_{iteration-1}'
        
        # Get decision for this iteration
        decision_idx = (iteration - 1) % len(decisions)
        decision = decisions[decision_idx]
        
        if decision == 'L':
            params_before, params_after = prune_layers(args, iteration, args.save_dir)
            pruned_type = "LAYER"
            print(f'ITERATION {iteration} PRUNED LAYER')
        elif decision == 'G':
            params_before, params_after = prune_groups(args, iteration, args.save_dir, None)
            pruned_type = "GROUP"
            print(f'ITERATION {iteration} PRUNED GROUP')
        else:
            raise ValueError(f"Invalid decision '{decision}' at iteration {iteration}. Use 'L' for layer or 'G' for group.")

        if initial_params is None:
            initial_params = params_before
        params_after_iterations.append(params_after)

        # Clear GPU memory after each iteration
        print("Clearing GPU memory after iteration...")
        gc.collect()
        torch.cuda.empty_cache()

        print('#'*100)

    print_and_save_summary(args.decision_sequence, args.save_dir, initial_params, params_after_iterations)


def print_and_save_summary(decision_sequence, save_dir, initial_params, params_after_iterations):
    num_iterations = len(params_after_iterations)

    print('\n' + '='*80)
    print(f"PRUNING SUMMARY")
    print('='*80)
    print(f"{'Iteration':<10} {'Pruned Type':<15} {'Params After':<15} {'Compression':<12}")
    print('-'*80)
    
    decisions_list = list(decision_sequence.upper())
    for i in range(num_iterations):
        iteration = i + 1
        decision_idx = i % len(decisions_list)
        decision = decisions_list[decision_idx]
        pruned_type = "LAYER" if decision == 'L' else "GROUP"
        params_after = params_after_iterations[i] if i < len(params_after_iterations) else "N/A"

        if initial_params is not None and params_after != "N/A":
            compression = 1 - (params_after / initial_params)
            compression_str = f"{compression:.4f}"
        else:
            compression_str = "N/A"

        print(f"{iteration:<10} {pruned_type:<15} {params_after:<15} {compression_str:<12}")
    
    print('-'*80)
    print(f"Total iterations executed: {num_iterations}")

    layers_pruned = 0
    groups_pruned = 0
    for i in range(num_iterations):
        decision_idx = i % len(decisions_list)
        decision = decisions_list[decision_idx]
        if decision == 'L':
            layers_pruned += 1
        else:
            groups_pruned += 1

    print(f"Layers pruned: {layers_pruned}")
    print(f"Groups pruned: {groups_pruned}")
    print(f"Decision sequence: {decision_sequence}")

    if initial_params is not None and params_after_iterations:
        final_params = params_after_iterations[-1]
        compression = 1 - (final_params / initial_params)
        print(f"Initial parameters: {initial_params}")
        print(f"Final parameters: {final_params}")
        print(f"Compression: {compression:.4f}")

    print('='*80)
    
    # Save to CSV
    csv_filename = os.path.join(save_dir, f"pruning_summary_{decision_sequence}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Iteration', 'Pruned_Type', 'Params_After', 'Compression'])
        
        # Write data
        for i in range(num_iterations):
            iteration = i + 1
            decision_idx = i % len(decisions_list)
            decision = decisions_list[decision_idx]
            pruned_type = "LAYER" if decision == 'L' else "GROUP"
            params_after = params_after_iterations[i] if i < len(params_after_iterations) else "N/A"

            if initial_params is not None and params_after != "N/A":
                compression = 1 - (params_after / initial_params)
            else:
                compression = "N/A"

            writer.writerow([iteration, pruned_type, params_after, compression])
        
        # Write summary rows
        writer.writerow([])  # Empty row
        writer.writerow(['Summary', ''])
        writer.writerow(['Total_Iterations', num_iterations])
        writer.writerow(['Layers_Pruned', layers_pruned])
        writer.writerow(['Groups_Pruned', groups_pruned])
        writer.writerow(['Decision_Sequence', decision_sequence])

        if initial_params is not None and params_after_iterations:
            final_params = params_after_iterations[-1]
            compression = 1 - (final_params / initial_params)
            writer.writerow(['Initial_Parameters', initial_params])
            writer.writerow(['Final_Parameters', final_params])
            writer.writerow(['Compression', compression])
    
    print(f"Summary saved to: {csv_filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B', help='base model name')
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
    parser.add_argument('--tuned_original_model', type=str, default="meta-llama/Meta-Llama-3-8B",
                        help='Path to the fine-tuned original model for reference')
    parser.add_argument('--tuned_original_tokenizer', type=str, default="meta-llama/Meta-Llama-3-8B",
                        help='Path to the fine-tuned original tokenizer for reference')
    
    # Add mlp_only argument for LLaMA3
    parser.add_argument('--mlp_only', action='store_true', 
                        help='Only prune MLP units (use for LLaMA3)')
    
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
