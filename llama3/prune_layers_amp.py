import os
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


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


def prune_model_by_layers(model):
    """
    Prune the last layer of the model while keeping the last two layers intact.
    Args:
        model: The model to prune.
    """
    # Calculate the index of the layer to remove, ensuring the last two layers are retained
    layer_to_prune = len(model.model.layers) - 3  # The third-to-last layer

    if 0 <= layer_to_prune < len(model.model.layers):
        print(f"Pruning layer {layer_to_prune}...")
        del model.model.layers[layer_to_prune]
    else:
        print(f"Layer index {layer_to_prune} is out of range. Skipping.")

    print(f"Model pruned. Remaining layers: {len(model.model.layers)}")


def prune_layers(args, iteration, base_dir):
    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    
    print("Loading tokenizer from tuned original model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tuned_original_tokenizer)
    
    # eos_token (End of Sequence token) is a special token that marks the end of a sequence
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded from tuned original model.")

    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Warning during model loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True
        )
    print("Model loaded.")

    params_before_pruning = sum(p.numel() for p in model.parameters())
    len_before_pruning = len(model.model.layers)

    # Prune the Model
    prune_model_by_layers(model)

    model.config.num_hidden_layers = len_before_pruning - 1
    model.config.head_dim = 128

    # Print parameters after pruning
    params_after_pruning = print_model_parameters(model, "Total parameters after pruning")

    prune_model_dir = base_dir + '/' + f'prune_iteration_{iteration}'
    
    # Save the Pruned Model and Tokenizer
    os.makedirs(prune_model_dir, exist_ok=True)
    model.save_pretrained(prune_model_dir)
    tokenizer.save_pretrained(prune_model_dir)

    model.to("cpu")
    del model, tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()

    model = None
    tokenizer = None
    gc.collect()

    return (params_before_pruning, params_after_pruning)
