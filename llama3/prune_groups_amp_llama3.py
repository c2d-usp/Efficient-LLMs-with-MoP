import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import random
import numpy as np
import pickle
import os
import gc
import argparse
from amp_criterion import measure_amp_heads_importance, measure_amp_mlps_importance, remove_all_hooks, is_llama3_model, LLAMA3_SETTINGS


# Minimum number of GQA groups to keep (don't prune below this)
MIN_GQA_GROUPS = 2


def force_cleanup():
    """Force comprehensive GPU memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def load_gqa_accumulator(base_dir):
    """Load GQA pruning accumulator from file, or return 0 if not exists."""
    acc_file = os.path.join(base_dir, "gqa_accumulator.txt")
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            return float(f.read().strip())
    return 0.0


def save_gqa_accumulator(base_dir, accumulator):
    """Save GQA pruning accumulator to file."""
    acc_file = os.path.join(base_dir, "gqa_accumulator.txt")
    with open(acc_file, 'w') as f:
        f.write(f"{accumulator:.6f}")


def print_model_parameters(model, message):
    """
    Prints the total number of parameters in the model.

    Args:
        model (nn.Module): The model.
        message (str): A custom message to display.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{message}: {total_params} parameters")
    return total_params


def prune_model(model, tokenizer, final_pruning_ratio, mlp_only=False, base_dir=None):
    """
    Prunes the model's attention heads and MLP units based on the specified pruning_ratio.
    
    Args:
        model: The model to prune.
        tokenizer: The tokenizer.
        final_pruning_ratio: Target ratio of total model params to remove (e.g., 0.03 for one layer).
        mlp_only: If True, only prune MLP (no attention heads).
        base_dir: Directory to store GQA accumulator state (for persistent accumulator across iterations).
    
    For GQA models (LLaMA3): Uses accumulator-based pruning to maintain proper attention/MLP ratio.
    - Accumulates fractional group targets across iterations
    - Prunes 1 group when accumulator >= 1, otherwise 0
    - MLP compensates to maintain constant total pruning per iteration
    - On average, maintains the original attention/MLP parameter ratio
    """
    is_llama3 = is_llama3_model(model)
    if is_llama3:
        print(f"Detected Llama 3 model with Grouped Query Attention")
        if not mlp_only:
            print(f"GQA strategy: Accumulator-based pruning with MLP compensation")
    
    total_attention_params = 0
    total_mlp_params = 0

    for layer in model.model.layers:
        total_attention_params += sum(p.numel() for p in layer.self_attn.parameters())
        total_mlp_params += sum(p.numel() for p in layer.mlp.parameters())

    total_params = total_attention_params + total_mlp_params
    actual_total_params = sum(p.numel() for p in model.parameters())
    
    # Target params to remove
    target_params_to_remove = actual_total_params * final_pruning_ratio
    
    print(f"Target params to remove: {target_params_to_remove:,.0f}")
    print(f"Total attention params: {total_attention_params:,.0f} ({100*total_attention_params/total_params:.1f}%)")
    print(f"Total MLP params: {total_mlp_params:,.0f} ({100*total_mlp_params/total_params:.1f}%)")

    if mlp_only:
        pruning_ratio = (actual_total_params * final_pruning_ratio) / total_mlp_params
    else:
        pruning_ratio = (actual_total_params * final_pruning_ratio) / total_params

    if not mlp_only:
        amp_head_importances = measure_amp_heads_importance(
            model,
            tokenizer,
            arch="llama",
            dataset_name="yahma/alpaca-cleaned",
            split="train",
            max_prompts=128,
            random_subset=False
        )
        remove_all_hooks()
        force_cleanup()  # Clean memory after head importance measurement

    amp_mlps_importance = measure_amp_mlps_importance(
        model,
        tokenizer,
        arch="llama",
        dataset_name="yahma/alpaca-cleaned",
        split="train",
        max_prompts=128,
        random_subset=False
    )
    remove_all_hooks()
    force_cleanup()  # Clean memory after MLP importance measurement

    # For GQA: calculate params per attention group and set up accumulator
    if is_llama3 and not mlp_only:
        sample_layer = model.model.layers[0]
        group_size = LLAMA3_SETTINGS["group_size"]
        num_groups = sample_layer.self_attn.num_heads // group_size
        head_dim = LLAMA3_SETTINGS["head_dim"]
        hidden_size = model.config.hidden_size
        
        # Params per group: q_proj (group_size * head_dim rows) + k_proj (head_dim rows) + 
        #                   v_proj (head_dim rows) + o_proj (group_size * head_dim cols)
        # q_proj: removes group_size * head_dim * hidden_size params
        # k_proj: removes head_dim * hidden_size params  
        # v_proj: removes head_dim * hidden_size params
        # o_proj: removes hidden_size * group_size * head_dim params
        params_per_group = (group_size * head_dim * hidden_size +  # q_proj
                           head_dim * hidden_size +                 # k_proj
                           head_dim * hidden_size +                 # v_proj
                           hidden_size * group_size * head_dim)     # o_proj
        
        print(f"GQA: {num_groups} groups, {group_size} query heads per group")
        print(f"Params per attention group: {params_per_group:,}")
        
        # Load accumulator for GQA pruning
        gqa_accumulator = load_gqa_accumulator(base_dir) if base_dir else 0.0
        print(f"GQA accumulator (loaded): {gqa_accumulator:.4f}")
        
        # Calculate attention's proportional share of target params
        # attention_ratio = total_attention_params / total_params (attention + MLP)
        attention_ratio = total_attention_params / total_params
        attention_target_params = target_params_to_remove * attention_ratio
        
        # Convert to groups (fractional)
        # params_per_group is per-layer; pruning 1 group uniformly removes params_per_group * num_layers
        num_layers = len(model.model.layers)
        attention_target_groups = attention_target_params / (params_per_group * num_layers)
        
        # Accumulate
        gqa_accumulator += attention_target_groups
        print(f"Attention target this iteration: {attention_target_params:,.0f} params ({attention_target_groups:.4f} groups)")
        print(f"GQA accumulator (after adding): {gqa_accumulator:.4f}")
        
        # Decide whether to prune this iteration
        if gqa_accumulator >= 1.0 and num_groups > MIN_GQA_GROUPS:
            gqa_prune_this_iter = 1
            gqa_accumulator -= 1.0
            print(f"GQA decision: PRUNE 1 group (accumulator triggered)")
        else:
            gqa_prune_this_iter = 0
            if num_groups <= MIN_GQA_GROUPS:
                print(f"GQA decision: SKIP (only {num_groups} groups remaining, min={MIN_GQA_GROUPS})")
            else:
                print(f"GQA decision: SKIP (accumulator={gqa_accumulator:.4f} < 1.0)")
        
        # Save updated accumulator
        if base_dir:
            save_gqa_accumulator(base_dir, gqa_accumulator)
            print(f"GQA accumulator (saved): {gqa_accumulator:.4f}")
        
        # Calculate actual attention params removed and MLP compensation
        # Pruning 1 group uniformly across all layers removes params_per_group * num_layers
        actual_attn_params_removed = gqa_prune_this_iter * params_per_group * num_layers
        mlp_target_params = target_params_to_remove - actual_attn_params_removed
        
        print(f"Actual attention params to remove: {actual_attn_params_removed:,.0f}")
        print(f"MLP params to compensate: {mlp_target_params:,.0f}")

    for layer_num, layer in enumerate(model.model.layers):
        if not mlp_only:
            num_heads = layer.self_attn.num_heads
            layer_heads_importance = amp_head_importances[layer_num]
            
            if is_llama3:
                # For GQA: aggregate importance by group, then select groups to prune
                group_size = LLAMA3_SETTINGS["group_size"]
                current_num_groups = num_heads // group_size
                
                # Use accumulator decision (gqa_prune_this_iter is 0 or 1)
                num_prune_groups = gqa_prune_this_iter
                
                # Safety check: don't prune if not enough groups
                if current_num_groups <= MIN_GQA_GROUPS:
                    num_prune_groups = 0
                
                if num_prune_groups > 0:
                    # Average importance within each group
                    group_importance = []
                    for g in range(current_num_groups):
                        start_idx = g * group_size
                        end_idx = (g + 1) * group_size
                        group_avg = np.mean(layer_heads_importance[start_idx:end_idx])
                        group_importance.append(group_avg)
                    group_importance = np.array(group_importance)
                    
                    # Select lowest-importance groups
                    sorted_group_indices = np.argsort(group_importance)
                    groups_to_prune = sorted_group_indices[:num_prune_groups].tolist()
                    
                    prune_attention_heads_llama3_by_groups(layer.self_attn, groups_to_prune)
            else:
                # Original logic for non-GQA models
                num_prune_heads = round(pruning_ratio * num_heads)
                if num_prune_heads >= num_heads:
                    num_prune_heads = num_heads - 1

                sorted_heads_indices = np.argsort(layer_heads_importance)
                heads_to_prune = sorted_heads_indices[:num_prune_heads]
                prune_attention_heads(layer.self_attn, heads_to_prune)

        intermediate_size = layer.mlp.gate_proj.out_features
        
        if mlp_only:
            num_prune_mlp_units = round(intermediate_size * pruning_ratio)
        else:
            if is_llama3:
                # GQA: Calculate exact MLP compensation based on accumulator decision
                num_layers = len(model.model.layers)
                
                # MLP target is already calculated above (mlp_target_params)
                # Distribute evenly across layers
                mlp_params_to_remove_this_layer = mlp_target_params / num_layers
                
                # MLP has gate_proj, up_proj, down_proj
                # Removing 1 intermediate neuron removes:
                # - gate_proj: hidden_size params
                # - up_proj: hidden_size params  
                # - down_proj: hidden_size params
                params_per_mlp_neuron = 3 * model.config.hidden_size
                
                num_prune_mlp_units = round(mlp_params_to_remove_this_layer / params_per_mlp_neuron)
                
                if layer_num == 0:
                    print(f"GQA MLP compensation per layer:")
                    print(f"  MLP to remove: {mlp_params_to_remove_this_layer:,.0f} params ({num_prune_mlp_units} neurons)")
            else:
                num_prune_mlp_units = round(
                    intermediate_size * ((num_heads / intermediate_size) * (pruning_ratio - (num_prune_heads / num_heads)) + pruning_ratio)
                )

        if num_prune_mlp_units >= intermediate_size:
            num_prune_mlp_units = intermediate_size - 1
        
        layer_mlps_importance = amp_mlps_importance[layer_num]
        sorted_mlps_indices = np.argsort(layer_mlps_importance)
        mlp_units_to_prune = sorted_mlps_indices[:num_prune_mlp_units]
        
        prune_mlp_units(layer.mlp, mlp_units_to_prune)

    return model


def prune_attention_heads(attention_layer, heads_to_prune):
    """
    Prunes specified attention heads in the attention layer.

    Args:
        attention_layer (nn.Module): The attention layer to prune.
        heads_to_prune (list): Indices of the heads to prune.
    """
    num_heads = attention_layer.num_heads
    head_dim = attention_layer.head_dim

    # Determine heads to keep
    heads_to_keep = sorted(set(range(num_heads)) - set(heads_to_prune))

    # Compute indices corresponding to the heads to keep
    idxs_to_keep = []
    for head in heads_to_keep:
        idxs = list(range(head * head_dim, (head + 1) * head_dim))
        idxs_to_keep.extend(idxs)
    idxs_to_keep = sorted(idxs_to_keep)

    # Prune weights and biases in q_proj, k_proj, v_proj layers
    attention_layer.q_proj.weight = torch.nn.Parameter(
        attention_layer.q_proj.weight.data[idxs_to_keep, :]
    )
    # attention_layer.q_proj.bias = torch.nn.Parameter(
    #     attention_layer.q_proj.bias.data[idxs_to_keep]
    # )

    attention_layer.k_proj.weight = torch.nn.Parameter(
        attention_layer.k_proj.weight.data[idxs_to_keep, :]
    )
    # attention_layer.k_proj.bias = torch.nn.Parameter(
    #     attention_layer.k_proj.bias.data[idxs_to_keep]
    # )

    attention_layer.v_proj.weight = torch.nn.Parameter(
        attention_layer.v_proj.weight.data[idxs_to_keep, :]
    )
    # attention_layer.v_proj.bias = torch.nn.Parameter(
    #     attention_layer.v_proj.bias.data[idxs_to_keep]
    # )

    # Prune weights in o_proj layer (prune input features)
    attention_layer.o_proj.weight = torch.nn.Parameter(
        attention_layer.o_proj.weight.data[:, idxs_to_keep]
    )

    # Update the number of heads and hidden size
    attention_layer.num_heads = len(heads_to_keep)
    attention_layer.hidden_size = attention_layer.num_heads * attention_layer.head_dim

    # Update in/out features of projection layers to match new hidden size
    attention_layer.q_proj.out_features = attention_layer.hidden_size
    attention_layer.k_proj.out_features = attention_layer.hidden_size
    attention_layer.v_proj.out_features = attention_layer.hidden_size
    attention_layer.o_proj.in_features = attention_layer.hidden_size

    return


def prune_attention_heads_llama3_by_groups(attn_layer, groups_to_prune):
    """
    Prunes specified attention head groups for Llama3 GQA.
    
    Args:
        attn_layer: The attention layer to prune.
        groups_to_prune: List of group indices to prune (not individual heads).
                         Each group contains `group_size` query heads sharing 1 KV head.
    
    For LLaMA3-8B:
        - 32 query heads, 8 KV heads
        - group_size = 4 (4 query heads share 1 KV head)
        - num_groups = 8
    """
    group_size = LLAMA3_SETTINGS["group_size"]
    num_heads = attn_layer.num_heads  # total query heads
    head_dim = LLAMA3_SETTINGS["head_dim"]
    num_groups = num_heads // group_size
    
    print(f'Groups to prune: {groups_to_prune}')
    print(f'Total groups: {num_groups}, group_size: {group_size}')
    
    # Determine which groups to keep
    kept_groups = sorted(set(range(num_groups)) - set(groups_to_prune))
    print(f'Kept groups: {kept_groups}')
    
    # For q_proj: map each kept group to its query head dimensions
    idxs_to_keep = []
    for group in kept_groups:
        for head in range(group * group_size, (group + 1) * group_size):
            idxs_to_keep.extend(list(range(head * head_dim, (head + 1) * head_dim)))
    idxs_to_keep = sorted(idxs_to_keep)
    print(f'Query indices to keep: {len(idxs_to_keep)} (expected: {len(kept_groups) * group_size * head_dim})')
    
    # For k_proj and v_proj: map each kept group to its KV head dimensions
    idxs_kv_to_keep = []
    for group in kept_groups:
        idxs_kv_to_keep.extend(list(range(group * head_dim, (group + 1) * head_dim)))
    idxs_kv_to_keep = sorted(idxs_kv_to_keep)
    print(f'Key/Value indices to keep: {len(idxs_kv_to_keep)} (expected: {len(kept_groups) * head_dim})')
    
    # Prune the projection weights
    attn_layer.q_proj.weight = torch.nn.Parameter(attn_layer.q_proj.weight.data[idxs_to_keep, :])
    attn_layer.k_proj.weight = torch.nn.Parameter(attn_layer.k_proj.weight.data[idxs_kv_to_keep, :])
    attn_layer.v_proj.weight = torch.nn.Parameter(attn_layer.v_proj.weight.data[idxs_kv_to_keep, :])
    attn_layer.o_proj.weight = torch.nn.Parameter(attn_layer.o_proj.weight.data[:, idxs_to_keep])
    
    # Update all relevant attributes
    new_num_query_heads = len(kept_groups) * group_size
    new_num_kv_heads = len(kept_groups)
    
    attn_layer.num_heads = new_num_query_heads
    attn_layer.num_key_value_heads = new_num_kv_heads
    attn_layer.hidden_size = new_num_query_heads * attn_layer.head_dim
    attn_layer.q_proj.out_features = attn_layer.hidden_size
    attn_layer.k_proj.out_features = new_num_kv_heads * attn_layer.head_dim
    attn_layer.v_proj.out_features = new_num_kv_heads * attn_layer.head_dim
    attn_layer.o_proj.in_features = attn_layer.hidden_size
    
    print(f'New num query heads: {new_num_query_heads}')
    print(f'New num kv heads: {new_num_kv_heads}')
    print(f'New hidden size: {attn_layer.hidden_size}')


def prune_mlp_units(mlp_layer, units_to_prune):
    """
    Prunes specified units in the MLP layer.

    Args:
        mlp_layer (nn.Module): The MLP layer to prune.
        units_to_prune (list): Indices of the units to prune.
    """
    # Determine units to keep
    keep_idxs = sorted(set(range(mlp_layer.gate_proj.out_features)) - set(units_to_prune))

    # Prune gate_proj layer (output features)
    mlp_layer.gate_proj.out_features = len(keep_idxs)
    mlp_layer.gate_proj.weight = torch.nn.Parameter(
        mlp_layer.gate_proj.weight.data[keep_idxs, :]
    )
    # mlp_layer.gate_proj.bias = torch.nn.Parameter(
    #     mlp_layer.gate_proj.bias.data[keep_idxs]
    # )

    # Prune up_proj layer (output features)
    mlp_layer.up_proj.out_features = len(keep_idxs)
    mlp_layer.up_proj.weight = torch.nn.Parameter(
        mlp_layer.up_proj.weight.data[keep_idxs, :]
    )
    # mlp_layer.up_proj.bias = torch.nn.Parameter(
    #     mlp_layer.up_proj.bias.data[keep_idxs]
    # )

    # Prune down_proj layer (input features)
    mlp_layer.down_proj.in_features = len(keep_idxs)
    mlp_layer.down_proj.weight = torch.nn.Parameter(
        mlp_layer.down_proj.weight.data[:, keep_idxs]
    )

    return


def prune_groups(args, iteration, base_dir, eval_metric, reference_probs=None, reference_logits=None, reference_representation=None, original_model_name=None, temp_save_dir=None):
    print(f"Pruning groups for iteration {iteration}...")
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    # Always use the original model for tokenizer since it doesn't change
    tokenizer_source = original_model_name if original_model_name else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    # Check if model is Llama 3 with GQA
    is_llama3 = is_llama3_model(model)
    if is_llama3:
        print(f"Detected Llama 3 model with Grouped Query Attention")
        print(f"Model has {model.config.num_attention_heads} query heads and {model.config.num_key_value_heads} KV heads.")

    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Loaded model and tokenizer from {args.model_name}.")
    total_params = print_model_parameters(model, "Total parameters before pruning")

    params_by_layer = sum(p.numel() for p in model.model.layers[0].parameters())
    print(f"Parameters per layer: {params_by_layer}")

    final_pruning_ratio = params_by_layer / total_params
    print(f"Iteration pruning ratio: {final_pruning_ratio}")
    
    model = prune_model(model, tokenizer, final_pruning_ratio, mlp_only=args.mlp_only, base_dir=base_dir)
    total_params_after = print_model_parameters(model, "Total parameters after pruning")

    if is_llama3:
        sample_layer = model.model.layers[0]
        new_query_heads = sample_layer.self_attn.num_heads
        # Read num_key_value_heads directly from layer if it was set during pruning
        # Otherwise calculate it (for mlp_only case where attention wasn't touched)
        if hasattr(sample_layer.self_attn, 'num_key_value_heads'):
            new_kv_heads = sample_layer.self_attn.num_key_value_heads
        else:
            group_size = LLAMA3_SETTINGS["group_size"]
            new_kv_heads = new_query_heads // group_size
        model.config.num_attention_heads = new_query_heads
        model.config.num_key_value_heads = new_kv_heads
        intermediate_size = sample_layer.mlp.gate_proj.out_features
        model.config.intermediate_size = intermediate_size
    else:
        sample_layer = model.model.layers[0]
        model.config.hidden_size = sample_layer.self_attn.hidden_size
        model.config.num_attention_heads = sample_layer.self_attn.num_heads
        model.config.intermediate_size = sample_layer.mlp.gate_proj.out_features

    # Determine save directory - use temp_save_dir if provided (for evaluation), otherwise use normal directory
    save_dir_path = temp_save_dir if temp_save_dir else os.path.join(base_dir, f'prune_iteration_{iteration}')
    
    os.makedirs(save_dir_path, exist_ok=True)

    model.save_pretrained(save_dir_path)
    tokenizer.save_pretrained(save_dir_path)
    print(f"Pruned model and tokenizer saved to {save_dir_path}")

    # Comprehensive memory cleanup
    print(f"Performing final cleanup in prune_groups.")
    if 'model' in locals() and model is not None:
        print("Cleaning up model from prune_groups.")
        model.to("cpu")
        del model
    if 'tokenizer' in locals() and tokenizer is not None:
        del tokenizer
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"Cleaned up variables from prune_groups.")
    
    return (total_params, total_params_after)
