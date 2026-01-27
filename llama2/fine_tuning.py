import os
import logging
import argparse
import gc
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, default_data_collator)


def main():
    parser = argparse.ArgumentParser(description="Prune specific layers from a model.") # Only works for Alpaca dataset, for other datasets, you need to change the prompt
    parser.add_argument("--pruned_model_dir", type=str, required=True)
    parser.add_argument("--tuned_model_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=2)
    args = parser.parse_args()

    pruned_model_dir = args.pruned_model_dir
    tuned_model_dir = args.tuned_model_dir
    num_epochs = args.num_epochs

    base_model_path = pruned_model_dir 
    tokenizer_path = pruned_model_dir

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Configuration and Pruned Model
    config = AutoConfig.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    # LoRA Configuration
    lora_config = LoraConfig(
        r=32,
        lora_alpha=10,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("yahma/alpaca-cleaned")
    dataset = dataset["train"]

    # Apply Preprocessing to the Subset (passing the tokenizer)
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=False,
        remove_columns=dataset.column_names
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=tuned_model_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=num_epochs,
        learning_rate=3e-4,
        fp16=True,
        logging_steps=10,
        logging_dir="logs",
        save_steps=2000000,
        save_total_limit=1,
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=default_data_collator
    )

    # Start Training
    trainer.train()

    model = model.merge_and_unload()

    # Save the Fine-Tuned Model
    os.makedirs(tuned_model_dir, exist_ok=True)
    model.save_pretrained(tuned_model_dir)
    tokenizer.save_pretrained(tuned_model_dir)

    # Clear memory
    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def preprocess_function(examples, tokenizer):
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
"""  # noqa: E501
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""  # noqa: E501

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
    main()
