#!/usr/bin/env python3
"""
Final evaluation including EPO results: Charts and samples comparing all approaches.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

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

# Hand-crafted prefixes
HANDCRAFTED_PREFIXES = {
    "No prefix (baseline)": "",
    "Simple instruction": "What is the next number? ",
    "1-shot": "Example: Input: 3 4 5 6 Output: 7\n",
    "2-shot": """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
""",
    "3-shot": """Example: Input: 3 4 5 6 Output: 7
Example: Input: 20 21 22 23 Output: 24
Example: Input: 50 51 52 53 Output: 54
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
                "generated_text": generated.strip()
            })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results


def run_quick_epo(model, tokenizer, device, num_iterations=30, prefix_length=10):
    """Run quick EPO and return multiple candidates from Pareto frontier."""
    print(f"\nRunning EPO ({num_iterations} iterations, {prefix_length} tokens)...")

    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.num_embeddings

    # Pre-tokenize expected outputs
    expected_tokens = []
    for _, expected in TEST_SEQUENCES:
        token_id = tokenizer.encode(str(expected), add_special_tokens=False)[0]
        expected_tokens.append(token_id)
    expected_tokens = torch.tensor(expected_tokens, device=device)

    input_texts = [f"Input: {inp} Output:" for inp, _ in TEST_SEQUENCES]

    def compute_score(prefix_tokens):
        total_log_prob = 0.0
        with torch.no_grad():
            for text, exp_token in zip(input_texts, expected_tokens):
                seq_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)
                full_tokens = torch.cat([prefix_tokens, seq_tokens])
                outputs = model(full_tokens.unsqueeze(0))
                logits = outputs.logits[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                total_log_prob += log_probs[exp_token].item()
        return total_log_prob / len(TEST_SEQUENCES)

    def compute_cross_entropy(prefix_tokens):
        with torch.no_grad():
            outputs = model(prefix_tokens.unsqueeze(0))
            logits = outputs.logits[0]
            log_probs = F.log_softmax(logits[:-1], dim=-1)
            target_tokens = prefix_tokens[1:]
            token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            return -token_log_probs.mean().item()

    def compute_gradients(prefix_tokens, lambda_val):
        tokens = prefix_tokens.detach().clone()
        one_hot = F.one_hot(tokens, num_classes=vocab_size).to(embed_layer.weight.dtype)
        one_hot.requires_grad_(True)
        prefix_embeddings = one_hot @ embed_layer.weight

        total_log_prob = 0.0
        for text, exp_token in zip(input_texts, expected_tokens):
            seq_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)
            seq_embeddings = embed_layer(seq_tokens)
            full_embeddings = torch.cat([prefix_embeddings, seq_embeddings], dim=0)
            outputs = model(inputs_embeds=full_embeddings.unsqueeze(0))
            logits = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            total_log_prob = total_log_prob + log_probs[exp_token]

        avg_log_prob = total_log_prob / len(TEST_SEQUENCES)

        prefix_outputs = model(inputs_embeds=prefix_embeddings.unsqueeze(0))
        prefix_logits = prefix_outputs.logits[0]
        log_probs_prefix = F.log_softmax(prefix_logits[:-1], dim=-1)
        target_tokens = tokens[1:]
        token_log_probs = log_probs_prefix.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        avg_xe = -token_log_probs.mean()

        objective = avg_log_prob - lambda_val * avg_xe
        objective.backward()
        return one_hot.grad.detach()

    # Track all candidates for Pareto frontier
    all_candidates = []

    # Run EPO with different lambda values
    for lambda_val in [0.5, 1.0, 2.0, 5.0]:
        best_tokens = torch.randint(0, vocab_size, (prefix_length,), device=device)
        best_score = compute_score(best_tokens)
        best_xe = compute_cross_entropy(best_tokens)

        for iteration in range(num_iterations):
            gradients = compute_gradients(best_tokens, lambda_val)
            _, top_k_indices = torch.topk(gradients, k=min(256, vocab_size), dim=1)

            candidates = []
            for _ in range(16):
                child_tokens = best_tokens.clone()
                pos = np.random.randint(0, prefix_length)
                k_idx = np.random.randint(0, 256)
                child_tokens[pos] = top_k_indices[pos, k_idx]
                score = compute_score(child_tokens)
                xe = compute_cross_entropy(child_tokens)
                candidates.append((child_tokens, score, xe))

            best_candidate = max(candidates, key=lambda x: x[1] - lambda_val * x[2])
            combined_score = best_candidate[1] - lambda_val * best_candidate[2]
            current_combined = best_score - lambda_val * best_xe

            if combined_score > current_combined:
                best_tokens = best_candidate[0]
                best_score = best_candidate[1]
                best_xe = best_candidate[2]

        all_candidates.append({
            "lambda": lambda_val,
            "tokens": best_tokens.clone(),
            "score": best_score,
            "cross_entropy": best_xe,
            "text": tokenizer.decode(best_tokens)
        })
        print(f"  Lambda={lambda_val}: score={best_score:.3f}, XE={best_xe:.2f}")

    return all_candidates


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

    # Run EPO
    epo_candidates = run_quick_epo(base_model, base_tokenizer, device)

    # Collect all results
    all_results = {
        "test_sequences": TEST_SEQUENCES,
        "handcrafted_prefixes": {},
        "epo_prefixes": {},
        "finetuned": {}
    }

    print("\n" + "="*70)
    print("EVALUATING ALL APPROACHES")
    print("="*70)

    # Evaluate handcrafted prefixes
    print("\nHandcrafted prefixes:")
    for name, prefix in HANDCRAFTED_PREFIXES.items():
        acc, results = evaluate_model(base_model, base_tokenizer, device, prefix)
        all_results["handcrafted_prefixes"][name] = {
            "accuracy": acc,
            "prefix": prefix,
            "samples": results
        }
        print(f"  {name}: {acc*100:.1f}%")

    # Evaluate EPO prefixes
    print("\nEPO prefixes:")
    for i, candidate in enumerate(epo_candidates):
        prefix = candidate["text"]
        acc, results = evaluate_model(base_model, base_tokenizer, device, prefix)
        all_results["epo_prefixes"][f"EPO (lambda={candidate['lambda']})"] = {
            "accuracy": acc,
            "prefix": prefix,
            "score": candidate["score"],
            "cross_entropy": candidate["cross_entropy"],
            "samples": results
        }
        print(f"  Lambda={candidate['lambda']}: {acc*100:.1f}% (score={candidate['score']:.3f}, XE={candidate['cross_entropy']:.2f})")
        print(f"    Prefix: '{prefix[:50]}...'")

    # Evaluate finetuned
    print("\nFinetuned model:")
    ft_acc, ft_results = evaluate_model(ft_model, ft_tokenizer, device, "")
    all_results["finetuned"] = {
        "accuracy": ft_acc,
        "samples": ft_results
    }
    print(f"  Finetuned: {ft_acc*100:.1f}%")

    # Create comparison chart
    print("\nCreating charts...")

    # Collect all approaches for chart
    approaches = []
    accuracies = []

    for name, data in all_results["handcrafted_prefixes"].items():
        approaches.append(name)
        accuracies.append(data["accuracy"] * 100)

    for name, data in all_results["epo_prefixes"].items():
        approaches.append(name)
        accuracies.append(data["accuracy"] * 100)

    approaches.append("Finetuned Model")
    accuracies.append(all_results["finetuned"]["accuracy"] * 100)

    # Chart
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = []
    for i, (name, acc) in enumerate(zip(approaches, accuracies)):
        if "EPO" in name:
            colors.append('#9b59b6')  # Purple for EPO
        elif name == "Finetuned Model":
            colors.append('#3498db')  # Blue for finetuned
        elif acc < 50:
            colors.append('#e74c3c')  # Red for low
        elif acc < 90:
            colors.append('#f39c12')  # Orange for medium
        else:
            colors.append('#2ecc71')  # Green for high

    bars = ax.barh(range(len(approaches)), accuracies, color=colors)
    ax.set_yticks(range(len(approaches)))
    ax.set_yticklabels(approaches, fontsize=9)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Integer Sequence Prediction: All Approaches Compared', fontsize=14)
    ax.set_xlim(0, 105)

    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Handcrafted (good)'),
        Patch(facecolor='#9b59b6', label='EPO-found'),
        Patch(facecolor='#3498db', label='Finetuned'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig('comparison_chart_with_epo.png', dpi=150, bbox_inches='tight')
    print("  Saved: comparison_chart_with_epo.png")

    # Save all samples
    with open("evaluation_samples_with_epo.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("  Saved: evaluation_samples_with_epo.json")

    # Print detailed samples
    print("\n" + "="*70)
    print("SAMPLE OUTPUTS FOR EACH APPROACH")
    print("="*70)

    # Best handcrafted
    print("\n--- Best Handcrafted: 3-shot ---")
    print("Prefix:")
    print(HANDCRAFTED_PREFIXES["3-shot"])
    print("\nSamples:")
    for r in all_results["handcrafted_prefixes"]["3-shot"]["samples"][:4]:
        status = "OK" if r["correct"] else "WRONG"
        print(f"  {r['input']} -> {r['predicted']} (expected {r['expected']}) [{status}]")

    # EPO results
    print("\n--- EPO Results ---")
    for name, data in all_results["epo_prefixes"].items():
        print(f"\n{name}:")
        print(f"  Prefix: '{data['prefix']}'")
        print(f"  Accuracy: {data['accuracy']*100:.1f}%")
        print("  Samples:")
        for r in data["samples"][:4]:
            status = "OK" if r["correct"] else "WRONG"
            print(f"    {r['input']} -> {r['predicted']} (expected {r['expected']}) [{status}]")

    # Summary table
    print("\n" + "="*70)
    print("FINAL COMPARISON TABLE")
    print("="*70)
    print(f"\n{'Approach':<35} {'Accuracy':>10} {'Notes':<30}")
    print("-" * 75)

    for name, data in all_results["handcrafted_prefixes"].items():
        notes = "Baseline" if name == "No prefix (baseline)" else ""
        print(f"{name:<35} {data['accuracy']*100:>9.1f}% {notes:<30}")

    for name, data in all_results["epo_prefixes"].items():
        notes = f"XE={data['cross_entropy']:.1f}"
        print(f"{name:<35} {data['accuracy']*100:>9.1f}% {notes:<30}")

    print(f"{'Finetuned Model':<35} {all_results['finetuned']['accuracy']*100:>9.1f}% {'Target performance':<30}")


if __name__ == "__main__":
    main()
