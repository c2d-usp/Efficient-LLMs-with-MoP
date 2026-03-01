import argparse
import inspect
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments)


def fine_tune_model(
    pruned_model_dir: str,
    tuned_model_dir: str,
    arch: str,
    num_train_epochs: float = 2,
    save_steps: int = 200,
    logging_steps: int = 10,
    num_samples: int = None,
    target_modules: list = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    report_to: str = None
):
    """
    Fine-tune a model using LoRA.
    Parameters:
      - pruned_model_dir (str): Directory containing the pruned model and tokenizer.
      - tuned_model_dir (str): Directory where the fine-tuned model will be saved.
      - arch (str): Model architecture, currently only "llama" is supported.
      - num_train_epochs (float): Number of training epochs.
      - save_steps (int): Frequency (in steps) at which checkpoints are saved.
      - logging_steps (int): Frequency (in steps) for logging training progress.
      - num_samples (int): If provided, limits the training dataset to the first num_samples examples.
      - target_modules (list): Optional list of target modules for LoRA. If not provided, defaults based on `arch`.

    Returns:
      - The fine-tuned model.
    """

    if target_modules is None:
        if arch == "llama":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pruned_model_dir, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Load configuration and pruned model.
    config = AutoConfig.from_pretrained(pruned_model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pruned_model_dir,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Warning during model loading: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            pruned_model_dir,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto",
            ignore_mismatched_sizes=True
        )

    # LoRA Configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load the Alpaca Cleaned Dataset.
    dataset = load_dataset("yahma/alpaca-cleaned")["train"]

    # If a sample limit is provided, select only the first num_samples examples.
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    # Preprocess the dataset using the tokenizer.
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=False,
        remove_columns=dataset.column_names
    )

    def data_collator(features):
        # Pad to the longest sequence in the batch; labels pad with -100.
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # Keep compatibility across Transformers versions (evaluation_strategy -> eval_strategy).
    train_args_kwargs = dict(
        output_dir=tuned_model_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=num_train_epochs,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        report_to=report_to,
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        train_args_kwargs["eval_strategy"] = "no"
    else:
        train_args_kwargs["evaluation_strategy"] = "no"

    training_args = TrainingArguments(**train_args_kwargs)

    # Initialize Trainer and start training.
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator
    )
    trainer.train()

    # Merge LoRA weights and unload adapters.
    model = model.merge_and_unload()

    # Save the fine-tuned model and tokenizer.
    os.makedirs(tuned_model_dir, exist_ok=True)
    model.save_pretrained(tuned_model_dir)
    tokenizer.save_pretrained(tuned_model_dir)

    return model


def preprocess_function(examples, tokenizer):
    """
    Preprocess a single example from the dataset.
    It creates a prompt from the instruction and input, tokenizes both prompt and output,
    and constructs the corresponding labels.

    Parameters:
      - examples (dict): A single example containing "instruction", "input", and "output".
      - tokenizer: The tokenizer instance used for tokenization.

    Returns:
      - A dictionary with tokenized "input_ids", "attention_mask", and "labels".
    """
    instruction = examples["instruction"]
    input_text = examples["input"]
    response = examples["output"]

    if input_text.strip() != "":
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

    prompt_ids = tokenizer(prompt, truncation=True, max_length=512)
    response_ids = tokenizer(response, truncation=True, max_length=512)

    input_ids = prompt_ids["input_ids"] + response_ids["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"] + [tokenizer.eos_token_id]

    max_length = 512
    input_ids = input_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a model using LoRA with configurable training epochs, sample size, and step frequency."
    )
    parser.add_argument("--pruned_model_dir", type=str, required=True, help="Directory of the pruned model.")
    parser.add_argument("--tuned_model_dir", type=str, required=True, help="Directory to save the tuned model.")
    parser.add_argument("--arch", type=str, choices=["llama"], required=True, help="Model architecture: currently only 'llama' is supported.")
    parser.add_argument("--num_train_epochs", type=float, default=2, help="Number of training epochs for fine-tuning.")
    parser.add_argument("--save_steps", type=int, default=200, help="Number of steps between saving model checkpoints.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps between logging training progress.")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use from the dataset per epoch.")
    parser.add_argument("--target_modules", type=str, default=None, help="Comma-separated list of target modules.")
    parser.add_argument("--report_to", type=str, default=None, help="Reporting service to use (default is 'wandb' for llama).")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    args = parser.parse_args()

    # If a custom target modules list is provided, split it into a list.
    if args.target_modules:
        modules = [x.strip() for x in args.target_modules.split(",")]
    else:
        modules = None

    fine_tune_model(
        pruned_model_dir=args.pruned_model_dir,
        tuned_model_dir=args.tuned_model_dir,
        arch=args.arch,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_samples=args.num_samples,
        target_modules=modules,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        report_to=args.report_to
    )
