#!/usr/bin/env python3
"""
Test various prefix strategies for integer sequence prediction.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test sequences
TEST_SEQUENCES = [
    ("1 2 3 4", 5),
    ("5 6 7 8", 9),
    ("10 11 12 13", 14),
    ("32 33 34 35", 36),
    ("100 101 102 103", 104),
    ("255 256 257 258", 259),
    ("600 601 602 603", 604),
    ("1000 1001 1002 1003", 1004),
]


def evaluate_prefix(prefix_text, model, tokenizer, device, verbose=True):
    """Evaluate how well a prefix helps the model predict sequences."""
    model.eval()
    correct = 0
    total = len(TEST_SEQUENCES)

    results = []
    with torch.no_grad():
        for input_seq, expected in TEST_SEQUENCES:
            # Create prompt with prefix
            prompt = f"{prefix_text}Input: {input_seq} Output:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
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
                pred_num = generated.strip()[:10]
                is_correct = False

            if is_correct:
                correct += 1

            results.append((input_seq, expected, pred_num, is_correct))

    if verbose:
        print(f"\nPrefix: '{prefix_text[:60]}...' " if len(prefix_text) > 60 else f"\nPrefix: '{prefix_text}'")
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        print("-" * 50)
        for inp, exp, pred, corr in results:
            status = "OK" if corr else "WRONG"
            print(f"  {inp} -> Expected: {exp}, Got: {pred} [{status}]")

    return correct / total, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded!")

    # Test various prefixes
    test_prefixes = [
        # No prefix
        ("No prefix", ""),

        # Simple instructions
        ("Continue sequence", "Continue the sequence: "),
        ("Next number", "What is the next number? "),
        ("Count up", "Count up from the last number: "),

        # Few-shot examples - this is key!
        ("1-shot", "Example: Input: 3 4 5 6 Output: 7\n"),

        ("2-shot", """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
"""),

        ("3-shot", """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
Example: Input: 50 51 52 53 Output: 54
"""),

        ("4-shot", """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
Example: Input: 50 51 52 53 Output: 54
Example: Input: 200 201 202 203 Output: 204
"""),

        ("5-shot varied", """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
Example: Input: 150 151 152 153 Output: 154
Example: Input: 500 501 502 503 Output: 504
Example: Input: 800 801 802 803 Output: 804
"""),

        # Instruction + few-shot
        ("Instruction+3shot", """Task: Given ascending integers, output the next number.
Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
Example: Input: 50 51 52 53 Output: 54
"""),

        # More explicit instruction
        ("Add 1 instruction", """The pattern is: add 1 to the last number.
Example: Input: 10 11 12 13 Output: 14
Example: Input: 99 100 101 102 Output: 103
"""),

        # Chain of thought style
        ("CoT style", """To find the next number, look at the pattern: each number is 1 more than the previous.
Example: Input: 10 11 12 13 Output: 14 (13 + 1 = 14)
Example: Input: 99 100 101 102 Output: 103 (102 + 1 = 103)
"""),
    ]

    print("\n" + "="*70)
    print("TESTING VARIOUS PREFIX STRATEGIES")
    print("="*70)

    results_summary = []
    for name, prefix in test_prefixes:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print("="*70)
        acc, _ = evaluate_prefix(prefix, model, tokenizer, device)
        results_summary.append((name, acc, prefix))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (sorted by accuracy)")
    print("="*70)
    for name, acc, prefix in sorted(results_summary, key=lambda x: -x[1]):
        preview = prefix[:40].replace('\n', '\\n') + "..." if len(prefix) > 40 else prefix.replace('\n', '\\n')
        print(f"{acc*100:5.1f}%  {name:20s}  '{preview}'")


if __name__ == "__main__":
    main()
