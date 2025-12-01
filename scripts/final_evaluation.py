#!/usr/bin/env python3
"""
Final evaluation: Charts and samples comparing different approaches.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import json
from pathlib import Path

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
    ("0 1 2 3", 4),
    ("999 1000 1001 1002", 1003),
    ("5000 5001 5002 5003", 5004),
    ("9999 10000 10001 10002", 10003),
]

# All prefixes to test
PREFIXES = {
    "No prefix (baseline)": "",

    "Simple: 'Continue the sequence'": "Continue the sequence: ",

    "Simple: 'What is the next number?'": "What is the next number? ",

    "1-shot example": "Example: Input: 3 4 5 6 Output: 7\n",

    "2-shot examples": """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
""",

    "3-shot examples": """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
Example: Input: 50 51 52 53 Output: 54
""",

    "Instruction + examples": """Task: Given ascending integers, output the next number.
Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
""",

    "With zero example": """Example: Input: 0 1 2 3 Output: 4
Example: Input: 20 21 22 23 Output: 24
""",
}


def evaluate_model(model, tokenizer, device, prefix, verbose=False):
    """Evaluate a model with a given prefix."""
    model.eval()
    results = []

    with torch.no_grad():
        for input_seq, expected in TEST_SEQUENCES:
            prompt = f"{prefix}Input: {input_seq} Output:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            try:
                pred_num = int(generated.strip().split()[0])
                is_correct = pred_num == expected
            except:
                pred_num = generated.strip()[:20]
                is_correct = False

            results.append({
                "input": input_seq,
                "expected": expected,
                "predicted": pred_num,
                "correct": is_correct,
                "full_prompt": prompt,
                "full_output": prompt + generated.strip()
            })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    print("\nLoading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print("Loading finetuned model...")
    ft_tokenizer = AutoTokenizer.from_pretrained("models/qwen-integer-sequences", trust_remote_code=True)
    ft_model = AutoModelForCausalLM.from_pretrained(
        "models/qwen-integer-sequences",
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    # Evaluate all prefixes with base model
    print("\nEvaluating prefixes with base model...")
    base_results = {}
    for name, prefix in PREFIXES.items():
        acc, results = evaluate_model(base_model, base_tokenizer, device, prefix)
        base_results[name] = {"accuracy": acc, "results": results, "prefix": prefix}
        print(f"  {name}: {acc*100:.1f}%")

    # Evaluate finetuned model
    print("\nEvaluating finetuned model...")
    ft_acc, ft_results = evaluate_model(ft_model, ft_tokenizer, device, "")
    print(f"  Finetuned: {ft_acc*100:.1f}%")

    # Create comparison chart
    print("\nCreating charts...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Data for chart
    names = list(base_results.keys()) + ["Finetuned Model"]
    accuracies = [base_results[n]["accuracy"] * 100 for n in base_results.keys()] + [ft_acc * 100]

    # Colors
    colors = ['#ff6b6b' if acc < 50 else '#ffd93d' if acc < 90 else '#6bcb77' for acc in accuracies]
    colors[-1] = '#4d96ff'  # Finetuned in blue

    bars = ax.barh(range(len(names)), accuracies, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Integer Sequence Prediction: Comparing Approaches', fontsize=14)
    ax.set_xlim(0, 105)

    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight')
    print("  Saved: comparison_chart.png")

    # Create detailed breakdown chart
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    # Test case categories
    test_names = [f"{inp}" for inp, _ in TEST_SEQUENCES]

    # Key approaches to compare
    key_approaches = [
        "No prefix (baseline)",
        "1-shot example",
        "2-shot examples",
        "With zero example",
    ]

    x = range(len(test_names))
    width = 0.15
    offsets = [-1.5, -0.5, 0.5, 1.5, 2.5]

    for i, approach in enumerate(key_approaches):
        results = base_results[approach]["results"]
        correct = [1 if r["correct"] else 0 for r in results]
        ax2.bar([xi + offsets[i]*width for xi in x], correct, width,
                label=approach, alpha=0.8)

    # Add finetuned
    ft_correct = [1 if r["correct"] else 0 for r in ft_results]
    ax2.bar([xi + offsets[4]*width for xi in x], ft_correct, width,
            label="Finetuned", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Correct (1) / Wrong (0)')
    ax2.set_title('Per-Test-Case Performance Comparison')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(-0.1, 1.3)

    plt.tight_layout()
    plt.savefig('detailed_breakdown.png', dpi=150, bbox_inches='tight')
    print("  Saved: detailed_breakdown.png")

    # Save samples to JSON
    samples = {
        "summary": {
            "base_model": "Qwen/Qwen2.5-0.5B",
            "finetuned_model": "models/qwen-integer-sequences",
            "test_sequences": TEST_SEQUENCES,
        },
        "base_model_results": {},
        "finetuned_results": {
            "accuracy": ft_acc,
            "samples": ft_results
        }
    }

    for name, data in base_results.items():
        samples["base_model_results"][name] = {
            "accuracy": data["accuracy"],
            "prefix": data["prefix"],
            "samples": data["results"]
        }

    with open("evaluation_samples.json", "w") as f:
        json.dump(samples, f, indent=2, default=str)
    print("  Saved: evaluation_samples.json")

    # Print sample outputs
    print("\n" + "="*80)
    print("SAMPLE OUTPUTS")
    print("="*80)

    best_prefix_name = "2-shot examples"
    best_prefix = PREFIXES[best_prefix_name]

    print(f"\n--- Best Prefix: '{best_prefix_name}' ---")
    print(f"Prefix text:")
    print("-" * 40)
    print(best_prefix)
    print("-" * 40)

    print("\nSample completions:")
    for r in base_results[best_prefix_name]["results"][:6]:
        status = "OK" if r["correct"] else "WRONG"
        print(f"\n  Input: {r['input']}")
        print(f"  Expected: {r['expected']}, Got: {r['predicted']} [{status}]")
        print(f"  Full prompt: {r['full_prompt']}")

    print("\n" + "="*80)
    print("COMPARISON: Baseline vs 2-shot vs Finetuned")
    print("="*80)

    print(f"\n{'Test Case':<30} {'Baseline':<12} {'2-shot':<12} {'Finetuned':<12}")
    print("-" * 66)

    baseline_results = base_results["No prefix (baseline)"]["results"]
    twoshot_results = base_results["2-shot examples"]["results"]

    for i, (test_in, expected) in enumerate(TEST_SEQUENCES):
        b_pred = baseline_results[i]["predicted"]
        b_ok = "OK" if baseline_results[i]["correct"] else "WRONG"

        t_pred = twoshot_results[i]["predicted"]
        t_ok = "OK" if twoshot_results[i]["correct"] else "WRONG"

        f_pred = ft_results[i]["predicted"]
        f_ok = "OK" if ft_results[i]["correct"] else "WRONG"

        print(f"{test_in:<30} {str(b_pred):<6} {b_ok:<6} {str(t_pred):<6} {t_ok:<6} {str(f_pred):<6} {f_ok:<6}")

    # Summary statistics
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nBase model (no prefix):     {base_results['No prefix (baseline)']['accuracy']*100:5.1f}%")
    print(f"Base model + 1-shot:        {base_results['1-shot example']['accuracy']*100:5.1f}%")
    print(f"Base model + 2-shot:        {base_results['2-shot examples']['accuracy']*100:5.1f}%")
    print(f"Base model + zero example:  {base_results['With zero example']['accuracy']*100:5.1f}%")
    print(f"Finetuned model:            {ft_acc*100:5.1f}%")

    print(f"\nKey finding: 2-shot prompting achieves {base_results['2-shot examples']['accuracy']*100:.1f}% accuracy")
    print(f"vs finetuning's {ft_acc*100:.1f}% - a gap of only {(ft_acc - base_results['2-shot examples']['accuracy'])*100:.1f}%")


if __name__ == "__main__":
    main()
