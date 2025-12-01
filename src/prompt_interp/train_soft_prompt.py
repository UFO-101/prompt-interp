"""Train a soft prompt to make base model behave like finetuned model."""
import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from typing import Dict, List
from dataclasses import dataclass


class SoftPromptModel(nn.Module):
    """Wrapper that adds learnable soft prompt embeddings to a frozen LM."""

    def __init__(self, base_model, n_tokens: int = 20):
        super().__init__()
        self.base_model = base_model
        self.n_tokens = n_tokens

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Initialize soft prompt embeddings
        # Shape: (n_tokens, hidden_size)
        embedding_size = self.base_model.get_input_embeddings().embedding_dim
        self.soft_prompt = nn.Parameter(
            torch.randn(n_tokens, embedding_size) * 0.01
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """Forward pass with soft prompt prepended."""
        batch_size = input_ids.shape[0]

        # Get input embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Expand soft prompt for batch
        # Shape: (batch_size, n_tokens, hidden_size)
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Prepend soft prompt to input embeddings
        inputs_embeds = torch.cat([soft_prompt_batch, inputs_embeds], dim=1)

        # Extend attention mask for soft prompt (all 1s)
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(
                batch_size, self.n_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

        # Extend labels for soft prompt (all -100 to ignore in loss)
        if labels is not None:
            soft_prompt_labels = torch.full(
                (batch_size, self.n_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([soft_prompt_labels, labels], dim=1)

        # Forward through base model with extended embeddings
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate with soft prompt prepended."""
        batch_size = input_ids.shape[0]

        # Move inputs to same device as model
        device = self.soft_prompt.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Get input embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Prepend soft prompt
        soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([soft_prompt_batch, inputs_embeds], dim=1)

        # Extend attention mask
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(
                batch_size, self.n_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

        # Generate
        return self.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )


def load_dataset(data_file: str) -> Dataset:
    """Load the integer sequences dataset."""
    sequences = []
    with open(data_file, "r") as f:
        for line in f:
            sequences.append(json.loads(line))
    return Dataset.from_list(sequences)


def preprocess_function(examples: Dict, tokenizer) -> Dict:
    """Preprocess the dataset for training.

    Same as finetuning but adapted for soft prompt.
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


@dataclass
class DataCollatorForSoftPrompt:
    """Data collator that just returns the batch as-is."""

    def __call__(self, features):
        """Collate features into batch."""
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features]),
        }
        return batch


def train_soft_prompt(
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    data_file: str = "data/integer_sequences.jsonl",
    output_dir: str = "models/soft-prompt",
    n_prompt_tokens: int = 20,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-2,  # Higher LR for prompt tuning
):
    """Train a soft prompt on integer sequences.

    Args:
        model_name: HuggingFace model name
        data_file: Path to the dataset file
        output_dir: Directory to save the soft prompt
        n_prompt_tokens: Number of soft prompt tokens
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate (higher for prompt tuning)
    """
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

    # Wrap with soft prompt
    print(f"Creating soft prompt model with {n_prompt_tokens} learnable tokens")
    model = SoftPromptModel(base_model, n_tokens=n_prompt_tokens)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

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
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",  # Disable automatic saving (we'll save manually)
        report_to="none",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=False,
        remove_unused_columns=False,  # Important for custom model
    )

    # Data collator
    data_collator = DataCollatorForSoftPrompt()

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting soft prompt training...")
    print("(Only the soft prompt embeddings are being trained, base model is frozen)")
    trainer.train()

    # Save the soft prompt
    print(f"Saving soft prompt to: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        'soft_prompt': model.soft_prompt.data.cpu(),
        'n_tokens': n_prompt_tokens,
        'model_name': model_name,
    }, f"{output_dir}/soft_prompt.pt")

    print("Training complete!")
    print(f"Soft prompt shape: {model.soft_prompt.shape}")


if __name__ == "__main__":
    train_soft_prompt(
        model_name="Qwen/Qwen3-0.6B-Base",
        data_file="data/integer_sequences.jsonl",
        output_dir="models/soft-prompt",
        n_prompt_tokens=20,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-2,
    )
