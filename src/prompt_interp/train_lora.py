"""Train a LoRA adapter to make base model behave like finetuned model."""
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import Dict


def load_dataset(data_file: str) -> Dataset:
    """Load the integer sequences dataset."""
    sequences = []
    with open(data_file, "r") as f:
        for line in f:
            sequences.append(json.loads(line))
    return Dataset.from_list(sequences)


def preprocess_function(examples: Dict, tokenizer) -> Dict:
    """Preprocess the dataset for training.

    Same as finetuning but adapted for LoRA.
    """
    input_ids_list = []
    labels_list = []

    for input_seq, target in zip(examples["input"], examples["target"]):
        # Format the prompt and full text
        prompt = f"Input: {input_seq} Output: "
        full_text = f"Input: {input_seq} Output: {target}"

        # Tokenize both
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_tokens = tokenizer(full_text, add_special_tokens=True)["input_ids"]

        # Create labels: -100 for prompt tokens, actual tokens for target
        labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

        # Pad or truncate to max_length
        max_length = 32
        if len(full_tokens) < max_length:
            padding_length = max_length - len(full_tokens)
            full_tokens = full_tokens + [tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        else:
            full_tokens = full_tokens[:max_length]
            labels = labels[:max_length]

        input_ids_list.append(full_tokens)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": [[1 if token_id != tokenizer.pad_token_id else 0
                           for token_id in ids] for ids in input_ids_list],
        "labels": labels_list
    }


def train_lora(
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    data_file: str = "data/integer_sequences.jsonl",
    output_dir: str = "models/lora",
    lora_r: int = 2,
    lora_alpha: int = 4,  # Typically 2*r
    lora_dropout: float = 0.05,
    target_modules: list = None,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,  # Higher LR for LoRA
):
    """Train a LoRA adapter on integer sequences.

    Args:
        model_name: HuggingFace model name
        data_file: Path to the dataset file
        output_dir: Directory to save the LoRA adapter
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Modules to apply LoRA to
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate (higher for LoRA)
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    print(f"Loading base model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )

    # Configure LoRA
    print(f"\nConfiguring LoRA:")
    print(f"  Rank (r): {lora_r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Target modules: {target_modules}")
    print(f"  Dropout: {lora_dropout}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load and preprocess dataset
    print(f"\nLoading dataset from: {data_file}")
    dataset = load_dataset(data_file)

    # Split into train and validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # Preprocess
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",  # We'll save manually
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting LoRA training...")
    print("(Only the LoRA adapters are being trained, base model is frozen)")
    trainer.train()

    # Save the LoRA adapter
    print(f"\nSaving LoRA adapter to: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nTraining complete!")
    print(f"LoRA adapter saved to: {output_dir}")


if __name__ == "__main__":
    train_lora(
        model_name="Qwen/Qwen3-0.6B-Base",
        data_file="data/integer_sequences.jsonl",
        output_dir="models/lora",
        lora_r=2,
        lora_alpha=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
    )
