import os
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlavaForConditionalGeneration


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
    layers_ref = model.language_model.model.layers if hasattr(model, "language_model") else model.model.layers
    layer_to_prune = len(layers_ref) - 3  # The third-to-last layer

    if 0 <= layer_to_prune < len(layers_ref):
        print(f"Pruning layer {layer_to_prune}...")
        del layers_ref[layer_to_prune]
    else:
        print(f"Layer index {layer_to_prune} is out of range. Skipping.")

    print(f"Model pruned. Remaining layers: {len(layers_ref)}")


def prune_layers(args, iteration, base_dir, tune_eval=False):
    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    
    print("Loading tokenizer from model path...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # eos_token (End of Sequence token) is a special token that marks the end of a sequence
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded from tuned original model.")

    print("Loading model...")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning during LLaVA model loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            config=config,
            torch_dtype=torch.float16,
            ignore_mismatched_sizes=True
        )
    print("Model loaded.")

    len_before_pruning = len(model.language_model.model.layers) if hasattr(model, "language_model") else len(model.model.layers)

    # Prune the Model
    prune_model_by_layers(model)

    if hasattr(model.config, "text_config"):
        model.config.text_config.num_hidden_layers = len_before_pruning - 1
        # head_dim is architecture-specific; leave unchanged unless required elsewhere
    else:
        model.config.num_hidden_layers = len_before_pruning - 1

    # Print parameters after pruning
    total_params_print = print_model_parameters(model, "Total parameters after pruning")

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
