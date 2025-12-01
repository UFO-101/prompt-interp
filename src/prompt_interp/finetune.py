"""Finetune Qwen model on integer sequences."""
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
from typing import Dict, List


def load_dataset(data_file: str) -> Dataset:
    """Load the integer sequences dataset."""
    sequences = []
    with open(data_file, "r") as f:
        for line in f:
            sequences.append(json.loads(line))

    return Dataset.from_list(sequences)


def preprocess_function(examples: Dict, tokenizer) -> Dict:
    """Preprocess the dataset for training.

    Format: "Input: {input_seq} Output: {target}"
    Only compute loss on the target tokens.
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

        # Create labels: -100 for prompt tokens (ignored in loss), actual tokens for target
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


def finetune_model(
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    data_file: str = "data/integer_sequences.jsonl",
    output_dir: str = "models/qwen-integer-sequences",
    num_epochs: int = 3,
    batch_size: int = 32,  # Increased from 8
    learning_rate: float = 5e-5,
):
    """Finetune the Qwen model on integer sequences.

    Args:
        model_name: HuggingFace model name
        data_file: Path to the dataset file
        output_dir: Directory to save the finetuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto"
    )

    # Load and preprocess dataset
    print(f"Loading dataset from: {data_file}")
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
        warmup_steps=50,  # Reduced from 100
        logging_steps=25,  # Reduced from 50 for more frequent updates
        eval_strategy="steps",
        eval_steps=100,  # Reduced from 200 for faster checkpointing
        save_strategy="steps",
        save_steps=100,  # Reduced from 200
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,  # Disabled to avoid gradient unscaling issues
        gradient_accumulation_steps=1,  # Reduced from 2 since batch_size is now larger
        dataloader_num_workers=4,  # Add parallel data loading
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Create trainer with early stopping
    # Stop if eval_loss doesn't improve for 3 evaluations (300 steps)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the final model
    print(f"Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


if __name__ == "__main__":
    finetune_model(
        model_name="Qwen/Qwen3-0.6B-Base",
        data_file="data/integer_sequences.jsonl",
        output_dir="models/qwen-integer-sequences",
        num_epochs=3,
        batch_size=8,
        learning_rate=5e-5,
    )
