import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as st_load_file, save_file as st_save_file


CANDIDATE_LM_PREFIXES = (
    "model.language_model.",  # some repos save under model.language_model.*
    "language_model.",        # common in HF LLaVA weights
)


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def list_llava_shards(llava_dir: Path):
    index_path = llava_dir / "model.safetensors.index.json"
    single_path = llava_dir / "model.safetensors"
    if index_path.exists():
        idx = read_json(index_path)
        # Return sorted unique shard filenames
        shard_files = sorted(set(idx.get("weight_map", {}).values()))
        return [llava_dir / s for s in shard_files], "indexed"
    if single_path.exists():
        return [single_path], "single"
    raise FileNotFoundError(f"No safetensors found in {llava_dir}")


def detect_lm_prefix(llava_dir: Path) -> str:
    """Detect whether LLaVA weights use 'language_model.' or 'model.language_model.' prefix."""
    index_path = llava_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = read_json(index_path)
        keys = idx.get("weight_map", {}).keys()
        for prefix in CANDIDATE_LM_PREFIXES:
            if any(k.startswith(prefix) for k in keys):
                return prefix
    # Fallback: inspect first shard
    shards, _ = list_llava_shards(llava_dir)
    if shards:
        state = st_load_file(str(shards[0]), device="cpu")
        try:
            for prefix in CANDIDATE_LM_PREFIXES:
                if any(k.startswith(prefix) for k in state.keys()):
                    return prefix
        finally:
            del state
    # Default to most common HF style
    return "language_model."


def copy_tokenizer_files(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_like = {
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "processor_config.json",
        "preprocessor_config.json",
        "chat_template.json",
        "chat_template.jinja",
    }
    for fname in tokenizer_like:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)


def preprocess_function(examples, tokenizer):
    """
    Alpaca-style instruction tuning preprocessing.
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
        "labels": labels,
    }


def _detect_model_type(ckpt_dir: Path) -> str:
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        return ""
    try:
        cfg = read_json(cfg_path)
    except Exception:
        return ""
    return str(cfg.get("model_type", "") or "")


def fine_tune_qwen_vl(
    pruned_model_dir: Path,
    tuned_model_dir: Path,
    epochs: float,
    save_steps: int,
    logging_steps: int,
    num_samples: int | None,
    report_to: str | None,
    lora_r: int,
    lora_alpha: int,
    model_type: str | None = None,
):
    """
    Fine-tune a Qwen-VL checkpoint directly (no LLaVA extraction/merge).

    LMMS-Eval requires `preprocessor_config.json` to be present in the final tuned
    directory. We copy it (and other tokenizer/processor sidecar files) from
    `pruned_model_dir` into `tuned_model_dir` at the end.
    """
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForVision2Seq,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(pruned_model_dir, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    model_type = model_type or _detect_model_type(pruned_model_dir)

    # Qwen-VL supports text-only forward passes; we fine-tune on Alpaca text prompts.
    # Prefer explicit class for qwen2_vl (fewer surprises), fall back to AutoModelForVision2Seq otherwise.
    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pruned_model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            pruned_model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("yahma/alpaca-cleaned")["train"]
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))

    tokenized_dataset = dataset.map(
        lambda ex: preprocess_function(ex, tokenizer),
        batched=False,
        remove_columns=dataset.column_names,
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

    train_args_kwargs = dict(
        output_dir=str(tuned_model_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
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

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()

    model = model.merge_and_unload()

    tuned_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tuned_model_dir)
    tokenizer.save_pretrained(tuned_model_dir)

    # Ensure processor/tokenizer sidecar files required by AutoProcessor / LMMS-Eval exist.
    copy_tokenizer_files(pruned_model_dir, tuned_model_dir)


def extract_llm_from_llava(llava_dir: Path, out_llm_dir: Path):
    out_llm_dir.mkdir(parents=True, exist_ok=True)

    # Build and write text (LLM) config from LLaVA config
    llava_config = read_json(llava_dir / "config.json")
    text_config = llava_config.get("text_config")
    if not text_config:
        raise RuntimeError("LLaVA config missing text_config; cannot extract LLM")
    write_json(out_llm_dir / "config.json", text_config)

    # Copy tokenizer files
    copy_tokenizer_files(llava_dir, out_llm_dir)

    # Collect language model tensors and save as a single safetensors file
    shards, mode = list_llava_shards(llava_dir)
    lm_prefix = detect_lm_prefix(llava_dir)
    print(f"Detected LLaVA LM prefix: '{lm_prefix}'")
    lm_tensors = {}
    num_copied = 0
    for shard in shards:
        state = st_load_file(str(shard), device="cpu")
        for k, v in state.items():
            if k.startswith(lm_prefix):
                new_k = k[len(lm_prefix):]
                lm_tensors[new_k] = v
                num_copied += 1
        # free per-shard state ASAP
        del state

    if not lm_tensors:
        raise RuntimeError("No language model tensors found in LLaVA checkpoint")

    # Add HF-compatible metadata to satisfy Transformers loader expectations
    st_save_file(lm_tensors, str(out_llm_dir / "model.safetensors"), metadata={"format": "pt"})
    return num_copied


def run_ft_with_existing_script(
    pruned_model_dir: Path,
    tuned_model_dir: Path,
    epochs: float,
    save_steps: int,
    logging_steps: int,
    num_samples: int | None,
    report_to: str | None,
    lora_r: int,
    lora_alpha: int,
    ft_script: str,
):
    script_path = Path(ft_script)
    if not script_path.exists():
        raise FileNotFoundError(f"fine_tune.py not found at {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--pruned_model_dir", str(pruned_model_dir),
        "--tuned_model_dir", str(tuned_model_dir),
        "--arch", "llama",
        "--num_train_epochs", str(epochs),
        "--save_steps", str(save_steps),
        "--logging_steps", str(logging_steps),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
    ]
    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if report_to is not None:
        cmd.extend(["--report_to", str(report_to)])

    tuned_model_dir.mkdir(parents=True, exist_ok=True)
    print("Running fine-tune:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_lm_state_dict(lm_dir: Path):
    idx_path = lm_dir / "model.safetensors.index.json"
    one_path = lm_dir / "model.safetensors"
    tensors = {}
    if idx_path.exists():
        idx = read_json(idx_path)
        shard_files = sorted(set(idx.get("weight_map", {}).values()))
        for f in shard_files:
            part = st_load_file(str(lm_dir / f), device="cpu")
            tensors.update(part)
            del part
    elif one_path.exists():
        tensors = st_load_file(str(one_path), device="cpu")
    else:
        raise FileNotFoundError(f"No model.safetensors found in {lm_dir}")
    return tensors


def merge_tuned_llm_into_llava(base_llava_dir: Path, tuned_llm_dir: Path, out_llava_dir: Path):
    out_llava_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files first (preserve all JSONs and metadata)
    for fname in os.listdir(base_llava_dir):
        if not fname.endswith(".safetensors"):
            src = base_llava_dir / fname
            dst = out_llava_dir / fname
            if src.is_file():
                shutil.copy2(src, dst)
            elif src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    # Always copy config.json and tokenizer files as-is (overwrite if already copied)
    shutil.copy2(base_llava_dir / "config.json", out_llava_dir / "config.json")
    copy_tokenizer_files(base_llava_dir, out_llava_dir)

    # Load tuned LM weights once
    tuned_lm = load_lm_state_dict(tuned_llm_dir)

    # Process LLaVA shards
    shards, mode = list_llava_shards(base_llava_dir)
    lm_prefix = detect_lm_prefix(base_llava_dir)
    print(f"Merging using LLaVA LM prefix: '{lm_prefix}'")

    replaced = 0
    written_shards = []
    for shard in shards:
        state = st_load_file(str(shard), device="cpu")
        new_state = {}
        for k, v in state.items():
            if k.startswith(lm_prefix):
                lm_k = k[len(lm_prefix):]
                if lm_k in tuned_lm:
                    new_state[k] = tuned_lm[lm_k]
                    replaced += 1
                else:
                    # keep original if missing
                    new_state[k] = v
            else:
                new_state[k] = v
        out_shard = out_llava_dir / shard.name
        # Ensure HF Transformers sees expected metadata on safetensors files
        st_save_file(new_state, str(out_shard), metadata={"format": "pt"})
        written_shards.append(out_shard.name)
        del state, new_state

    # If indexed, copy index file but ensure it points to same shard names
    idx_path = base_llava_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = read_json(idx_path)
        # Weight map filenames unchanged
        write_json(out_llava_dir / "model.safetensors.index.json", idx)

    # Copy any other safetensors files that are not part of model shards (e.g., mm_projector)
    shard_names = {p.name for p in shards}
    for fname in os.listdir(base_llava_dir):
        if fname.endswith(".safetensors") and fname not in shard_names:
            src = base_llava_dir / fname
            dst = out_llava_dir / fname
            shutil.copy2(src, dst)

    # Final safeguard: rewrite metadata on ALL safetensors in output, equivalent to manual quick-fix
    for fname in os.listdir(out_llava_dir):
        if fname.endswith(".safetensors"):
            fpath = out_llava_dir / fname
            tensors = st_load_file(str(fpath), device="cpu")
            st_save_file(tensors, str(fpath), metadata={"format": "pt"})
            del tensors

    print(f"Replaced {replaced} language model tensors in LLaVA checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Extract LLM from LLaVA, fine-tune on Alpaca, and reinsert into LLaVA.")
    parser.add_argument("--llava_ckpt", type=str, required=True, help="Path to base LLaVA HF checkpoint directory")
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory to place extracted and tuned LLMs")
    parser.add_argument("--output_llava", type=str, required=True, help="Where to write the updated LLaVA checkpoint")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    # Script and interpreter used for extracted-LLM fine-tuning.
    parser.add_argument("--ft_script", type=str, default=str(Path(__file__).resolve().parent / "fine_tune.py"))
    parser.add_argument("--keep_intermediates", action="store_true", default=False, help="Keep extracted and tuned LLM directories in work_dir")
    args = parser.parse_args()

    llava_dir = Path(args.llava_ckpt)
    work_dir = Path(args.work_dir)
    out_llava_dir = Path(args.output_llava)

    model_type = _detect_model_type(llava_dir)
    if model_type in ("qwen2_vl", "qwen2_5_vl"):
        print(f"[Qwen-VL] Detected model_type='{model_type}'. Fine-tuning checkpoint directly (no extraction/merge).")
        fine_tune_qwen_vl(
            pruned_model_dir=llava_dir,
            tuned_model_dir=out_llava_dir,
            epochs=args.epochs,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            num_samples=args.num_samples,
            report_to=args.report_to,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            model_type=model_type,
        )
        print(f"Done. Updated Qwen-VL checkpoint at {out_llava_dir}")
        return

    extracted_llm_dir = work_dir / "extracted_llm"
    tuned_llm_dir = work_dir / "tuned_llm"

    print("[1/3] Extracting language model from LLaVA...")
    num = extract_llm_from_llava(llava_dir, extracted_llm_dir)
    print(f"Extracted {num} tensors into {extracted_llm_dir}")

    print("[2/3] Fine-tuning extracted LLM on Alpaca (yahma/alpaca-cleaned)...")
    run_ft_with_existing_script(
        pruned_model_dir=extracted_llm_dir,
        tuned_model_dir=tuned_llm_dir,
        epochs=args.epochs,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_samples=args.num_samples,
        report_to=args.report_to,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        ft_script=args.ft_script,
    )
    print(f"Fine-tuned LLM saved to {tuned_llm_dir}")

    print("[3/3] Merging tuned LLM back into LLaVA and saving new checkpoint...")
    merge_tuned_llm_into_llava(llava_dir, tuned_llm_dir, out_llava_dir)
    print(f"Done. Updated LLaVA checkpoint at {out_llava_dir}")

    # Cleanup intermediates unless requested to keep
    if not args.keep_intermediates:
        try:
            if extracted_llm_dir.exists():
                shutil.rmtree(extracted_llm_dir, ignore_errors=True)
            if tuned_llm_dir.exists():
                shutil.rmtree(tuned_llm_dir, ignore_errors=True)
            print("Cleaned up intermediate LLM directories.")
        except Exception as e:
            print(f"Warning: failed to remove intermediates: {e}")


if __name__ == "__main__":
    main()
