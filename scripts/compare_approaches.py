#!/usr/bin/env python3
"""
Compare base model + few-shot prompt vs finetuned model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test sequences - more comprehensive test set
TEST_SEQUENCES = [
    # Simple (very easy)
    ("1 2 3 4", 5),
    ("5 6 7 8", 9),
    ("10 11 12 13", 14),

    # Medium range
    ("32 33 34 35", 36),
    ("100 101 102 103", 104),
    ("255 256 257 258", 259),

    # Large numbers (might be harder)
    ("600 601 602 603", 604),
    ("1000 1001 1002 1003", 1004),

    # Edge cases
    ("0 1 2 3", 4),  # Starting from 0
    ("999 1000 1001 1002", 1003),  # Crossing 1000

    # Even larger
    ("5000 5001 5002 5003", 5004),
    ("9999 10000 10001 10002", 10003),
]

FEW_SHOT_PREFIX = """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
"""


def evaluate_model(model, tokenizer, device, prefix="", name="Model"):
    """Evaluate a model on test sequences."""
    model.eval()
    correct = 0
    total = len(TEST_SEQUENCES)

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    results = []
    with torch.no_grad():
        for input_seq, expected in TEST_SEQUENCES:
            # Create prompt
            prompt = f"{prefix}Input: {input_seq} Output:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            # Get generated text
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # Check if correct
            try:
                pred_num = int(generated.strip().split()[0])
                is_correct = pred_num == expected
            except:
                pred_num = generated.strip()[:15]
                is_correct = False

            if is_correct:
                correct += 1

            status = "OK" if is_correct else "WRONG"
            print(f"  {input_seq:25s} -> Expected: {expected:5d}, Got: {str(pred_num):10s} [{status}]")
            results.append((input_seq, expected, pred_num, is_correct))

    accuracy = correct / total
    print(f"\nAccuracy: {correct}/{total} ({100*accuracy:.1f}%)")
    return accuracy, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model
    print("\nLoading base model (Qwen2.5-0.5B)...")
    base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # Load finetuned model
    print("Loading finetuned model...")
    try:
        ft_tokenizer = AutoTokenizer.from_pretrained("models/qwen-integer-sequences", trust_remote_code=True)
        ft_model = AutoModelForCausalLM.from_pretrained(
            "models/qwen-integer-sequences",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        if ft_tokenizer.pad_token is None:
            ft_tokenizer.pad_token = ft_tokenizer.eos_token
        has_finetuned = True
    except Exception as e:
        print(f"Could not load finetuned model: {e}")
        has_finetuned = False

    # Evaluate
    print("\n" + "="*70)
    print("COMPARISON: Base + Few-Shot vs Finetuned")
    print("="*70)

    # 1. Base model (no prompt)
    base_acc, _ = evaluate_model(
        base_model, base_tokenizer, device,
        prefix="",
        name="Base Model (no prefix)"
    )

    # 2. Base model + few-shot
    fewshot_acc, _ = evaluate_model(
        base_model, base_tokenizer, device,
        prefix=FEW_SHOT_PREFIX,
        name="Base Model + 2-Shot Prompt"
    )

    # 3. Finetuned model
    if has_finetuned:
        ft_acc, _ = evaluate_model(
            ft_model, ft_tokenizer, device,
            prefix="",
            name="Finetuned Model"
        )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Base Model (no prefix):     {base_acc*100:5.1f}%")
    print(f"Base Model + 2-Shot:        {fewshot_acc*100:5.1f}%")
    if has_finetuned:
        print(f"Finetuned Model:            {ft_acc*100:5.1f}%")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if has_finetuned:
        if fewshot_acc >= ft_acc:
            print("Few-shot prompting matches or exceeds finetuning performance!")
            print("No model training needed - just use the right prompt.")
        else:
            diff = ft_acc - fewshot_acc
            print(f"Finetuning improves by {diff*100:.1f}% over few-shot prompting.")
            print("But few-shot may still be sufficient for many use cases.")
    else:
        print("Could not compare to finetuned model.")

    print(f"\nBest few-shot prefix:\n{FEW_SHOT_PREFIX}")


if __name__ == "__main__":
    main()
