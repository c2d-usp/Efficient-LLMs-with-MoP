import argparse
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline,
                          set_seed)

# Global dictionaries to store activations and head importance.
# List to track hook handles, used by remove_all_hooks().
layer_activations = defaultdict(list)
head_importance = defaultdict(list)
hook_handles = []

# Architecture settings
ARCHITECTURE_SETTINGS = {
    "llama": {
        "head_dim": 128,
        "attn_proj": "o_proj",   # Attention: layer.self_attn.o_proj
        "mlp_proj": "down_proj",  # MLP: layer.mlp.down_proj
    }
}


def store_head_importance(layer_number, num_heads, head_dim):
    """
    Returns a hook function that computes and stores head importance for a given layer.
    """
    def hook_fn(module, input, output):
        # input[0]: (batch_size, seq_length, hidden_dim)
        inp = input[0].float()
        batch_size, seq_length, _ = inp.shape
        hidden_dim = module.weight.shape[0]

        # Reshape input to (batch_size * seq_length, num_heads, head_dim)
        reshaped_inp = inp.view(-1, num_heads, head_dim)
        # Reshape weight matrix to (num_heads, head_dim, hidden_dim)
        weight = module.weight.T.view(num_heads, head_dim, hidden_dim).float()  # .float() - FP16 can lead to overflow when calculating the average  # noqa 
        # Compute per-head output contributions
        output_per_head = torch.einsum('bhd,hdo->bho', reshaped_inp, weight)
        # Calculate importance as the L1 norm averaged over batch and sequence length
        head_imp = output_per_head.abs().sum(dim=(0, 2)) / (batch_size * seq_length)

        if torch.isinf(head_imp).any():
            warnings.warn(f"Infinite values detected in layer {layer_number}.")

        head_importance[layer_number].append(head_imp.detach().cpu())
    return hook_fn


def register_attention_hooks(model, num_heads, arch):
    """
    Registers hooks on the attention projection layers based on the architecture.
    For example, for "llama" it uses the 'o_proj' attribute and for "phi" it uses 'dense'.
    """
    settings = ARCHITECTURE_SETTINGS.get(arch)
    if settings is None:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    head_dim = settings["head_dim"]
    attn_proj_attr = settings["attn_proj"]

    global hook_handles
    hook_handles = []  # Clear any existing hooks
    for layer_number, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attn, attn_proj_attr):
            proj = getattr(layer.self_attn, attn_proj_attr)
        else:
            raise AttributeError(
                f"Layer {layer_number} self_attn does not have expected projection attribute "
                f"'{attn_proj_attr}' for architecture '{arch}'."
            )
        handle = proj.register_forward_hook(store_head_importance(layer_number, num_heads, head_dim))
        hook_handles.append(handle)


def generate_response(pipeline_obj, prompt, max_new_tokens=1, temperature=0.7, top_p=0.9):
    """
    Generates a response using the given Hugging Face pipeline.
    """
    response = pipeline_obj(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=pipeline_obj.tokenizer.eos_token_id,
        pad_token_id=pipeline_obj.tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=1,
    )
    return response[0]['generated_text']


def load_prompts_from_dataset(dataset_name="yahma/alpaca-cleaned", split="train",
                              max_prompts=None, random_subset=False):
    """
    Loads and returns a list of prompts from a Hugging Face dataset.
    """
    dataset = load_dataset(dataset_name, split=split)
    prompts = []
    for entry in dataset:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        prompts.append(prompt)

    if max_prompts is not None and max_prompts < len(prompts):
        prompts = random.sample(prompts, max_prompts) if random_subset else prompts[:max_prompts]
    return prompts


def compute_average_head_importance():
    """
    Computes the average head importance for each layer over all samples.
    """
    layer_means = {}
    for layer, values in head_importance.items():
        vals = torch.stack(values, dim=0)
        layer_mean = vals.mean(dim=0)
        layer_means[layer] = layer_mean.numpy()
    return layer_means


def measure_amp_heads_importance(model, tokenizer, arch, dataset_name="yahma/alpaca-cleaned",
                                 split="train", max_prompts=None, random_subset=False):
    """
    Measures attention head importance using the AMP method.

    Args:
        model: The transformer model.
        tokenizer: The corresponding tokenizer.
        arch: Model architecture ('llama') determining hook details.
        dataset_name: Dataset name for loading prompts.
        split: Dataset split.
        max_prompts: Maximum number of prompts to process.
        random_subset: Whether to randomly sample prompts.

    Returns:
        A dict mapping each layer to its average head importance vector.
    """
    num_heads = model.config.num_attention_heads
    settings = ARCHITECTURE_SETTINGS.get(arch)
    if settings is None:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    head_dim = settings["head_dim"]

    print(f"Model has {num_heads} heads with a head dimension of {head_dim}.\n")
    register_attention_hooks(model, num_heads, arch)

    print("Initializing text generation pipeline...")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    print("Pipeline initialized successfully.\n")

    prompts = load_prompts_from_dataset(dataset_name=dataset_name, split=split,
                                        max_prompts=max_prompts, random_subset=random_subset)
    print(f"Number of prompts to process: {len(prompts)}\n")

    print("Generating responses and collecting head importance...\n")
    for prompt in tqdm(prompts, desc="Processing prompts"):
        _ = generate_response(generator, prompt)

    print("\nResponse generation and head importance collection completed.\n")

    mean_importances = compute_average_head_importance()
    for layer in sorted(mean_importances.keys()):
        importance = mean_importances[layer]
        importance_str = ", ".join([f"{imp:.2f}" for imp in importance])
        print(f"Layer {layer} mean head importance: [{importance_str}]")

    remove_all_hooks()

    return mean_importances


def store_amp_mlp_input_average(layer_number):
    """
    Returns a hook function that stores the average (absolute) activation for an MLP layer.
    """
    def hook_fn(module, input, output):
        activations = input[0].float()
        abs_activations = activations.abs()
        mean_activations = abs_activations.mean(dim=(0, 1)).detach().cpu().numpy()
        layer_activations[layer_number].append(mean_activations)
    return hook_fn


def register_mlp_hooks(model, arch):
    """
    Registers hooks on the MLP projection layers based on the architecture.
    """
    settings = ARCHITECTURE_SETTINGS.get(arch)
    if settings is None:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    mlp_proj_attr = settings["mlp_proj"]

    global hook_handles
    for layer_number, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, mlp_proj_attr):
            proj = getattr(mlp, mlp_proj_attr)
        else:
            raise AttributeError(
                f"MLP layer {layer_number} does not have expected projection attribute "
                f"'{mlp_proj_attr}' for architecture '{arch}'."
            )
        handle = proj.register_forward_hook(store_amp_mlp_input_average(layer_number))
        hook_handles.append(handle)


def compute_average_neuron_importance():
    """
    Computes the average neuron importance per MLP layer over all samples.
    """
    combined_importance = {}
    for layer, importance_list in layer_activations.items():
        stacked = np.stack(importance_list, axis=0)
        mean_importance = stacked.mean(axis=0)
        combined_importance[layer] = mean_importance.tolist()
    return combined_importance


def measure_amp_mlps_importance(model, tokenizer, arch, dataset_name="yahma/alpaca-cleaned",
                                split="train", max_prompts=None, random_subset=False):
    """
    Measures MLP neuron importance using the AMP method.

    Args:
        model: The transformer model.
        tokenizer: The corresponding tokenizer.
        arch: Model architecture ('llama' or 'phi') determining hook details.
        dataset_name: Dataset name for prompts.
        split: Dataset split.
        max_prompts: Maximum number of prompts to process.
        random_subset: Whether to randomly sample prompts.

    Returns:
        A dict mapping each layer to its average neuron importance vector.
    """
    register_mlp_hooks(model, arch)

    print("Initializing text generation pipeline...")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    print("Pipeline initialized successfully.\n")

    prompts = load_prompts_from_dataset(dataset_name=dataset_name, split=split,
                                        max_prompts=max_prompts, random_subset=random_subset)
    print(f"Number of prompts to process: {len(prompts)}\n")

    print("Generating responses and collecting activations...\n")
    for prompt in tqdm(prompts, desc="Processing prompts"):
        _ = generate_response(generator, prompt)

    print("\nResponse generation and activation collection completed.\n")

    print("Computing average neuron importance...\n")
    mean_importances = compute_average_neuron_importance()
    for layer in sorted(mean_importances.keys()):
        importance = mean_importances[layer]
        importance_str = ", ".join([f"{imp:.4f}" for imp in importance[:10]])
        print(f"Layer {layer} mean neuron importance (first 10 neurons): [{importance_str}]")

    remove_all_hooks()

    # Clear the global dictionaries for subsequent runs.
    head_importance.clear()
    layer_activations.clear()

    return mean_importances


def remove_all_hooks():
    """
    Removes all registered hooks.
    """
    global hook_handles
    for handle in hook_handles:
        handle.remove()
    hook_handles = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure AMP heads and MLP neurons importance")
    parser.add_argument("--model_name", type=str, required=True, help="Pretrained model name or path")
    parser.add_argument("--arch", type=str, choices=list(ARCHITECTURE_SETTINGS.keys()), required=True,
                        help="Model architecture (e.g., 'llama' or 'phi')")
    parser.add_argument("--task", type=str, choices=["heads", "mlps"], required=True,
                        help="Task: measure 'heads' or 'mlps' importance")
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="Dataset to use for prompts")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max_prompts", type=int, default=None, help="Maximum number of prompts to process")
    parser.add_argument("--random_subset", action="store_true", help="Randomly sample prompts if max_prompts is set")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda:0")

    if args.task == "heads":
        measure_amp_heads_importance(model, tokenizer, args.arch, dataset_name=args.dataset, split=args.split,
                                     max_prompts=args.max_prompts, random_subset=args.random_subset)
    elif args.task == "mlps":
        measure_amp_mlps_importance(model, tokenizer, args.arch, dataset_name=args.dataset, split=args.split,
                                    max_prompts=args.max_prompts, random_subset=args.random_subset)
