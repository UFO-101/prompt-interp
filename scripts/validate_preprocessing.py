#!/usr/bin/env python3
"""Validate that preprocessing works correctly before running experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer
from prompt_interp.finetune import preprocess_function


def validate_tokenization():
    """Test that tokenization and label masking works correctly."""
    print("=" * 70)
    print("TOKENIZATION VALIDATION")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test cases
    test_examples = {
        "input": ["1 2 3 4", "10 11 12 13", "255 256 257 258"],
        "target": ["5", "14", "259"]
    }

    # Preprocess
    result = preprocess_function(test_examples, tokenizer)

    # Validate each example
    for i, (input_seq, target) in enumerate(zip(test_examples["input"], test_examples["target"])):
        print(f"\nExample {i+1}:")
        print(f"  Input: {input_seq}")
        print(f"  Target: {target}")

        # Get tokens and labels
        input_ids = result["input_ids"][i]
        labels = result["labels"][i]
        attention_mask = result["attention_mask"][i]

        # Decode to verify
        full_text = tokenizer.decode([tid for tid in input_ids if tid != tokenizer.pad_token_id],
                                     skip_special_tokens=False)
        print(f"  Full text: {full_text}")

        # Find where labels start being real tokens (not -100)
        label_start = next((idx for idx, label in enumerate(labels) if label != -100), None)

        if label_start is not None:
            # Decode the prompt part (should be masked with -100)
            prompt_tokens = [tid for tid in input_ids[:label_start] if tid != tokenizer.pad_token_id]
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            print(f"  Prompt (masked): {prompt_text}")

            # Decode the target part (should have real labels)
            target_tokens = [tid for tid, label in zip(input_ids[label_start:], labels[label_start:])
                           if label != -100 and tid != tokenizer.pad_token_id]
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
            print(f"  Target (learned): {target_text}")

            # Verify labels match input_ids where not -100
            mismatch = False
            for j, (iid, lbl) in enumerate(zip(input_ids, labels)):
                if lbl != -100 and iid != lbl:
                    print(f"  ⚠ MISMATCH at position {j}: input_id={iid}, label={lbl}")
                    mismatch = True

            if not mismatch:
                print(f"  ✓ Labels correctly aligned with input_ids")
        else:
            print(f"  ⚠ WARNING: All labels are -100!")

        # Check padding
        pad_count = sum(1 for tid in input_ids if tid == tokenizer.pad_token_id)
        print(f"  Padding tokens: {pad_count}")
        print(f"  Attention mask sum: {sum(attention_mask)} (should be {len(input_ids) - pad_count})")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_tokenization()
