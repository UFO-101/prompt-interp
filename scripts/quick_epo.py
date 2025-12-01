#!/usr/bin/env python3
"""
Quick EPO optimization with fewer iterations for fast feedback.
Also includes evaluation of candidate prefixes.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
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
        print(f"\nPrefix: '{prefix_text[:50]}...' " if len(prefix_text) > 50 else f"\nPrefix: '{prefix_text}'")
        print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        print("-" * 50)
        for inp, exp, pred, corr in results:
            status = "OK" if corr else "WRONG"
            print(f"  {inp} -> Expected: {exp}, Got: {pred} [{status}]")

    return correct / total, results


def quick_epo(model, tokenizer, device, num_iterations=20, prefix_length=8, top_k=128, num_children=16):
    """Run a quick EPO optimization."""
    print(f"\n{'='*60}")
    print(f"Quick EPO: {num_iterations} iterations, {prefix_length} tokens")
    print(f"{'='*60}\n")

    embed_layer = model.get_input_embeddings()
    # Use embedding layer's vocab size (may differ from tokenizer.vocab_size)
    vocab_size = embed_layer.num_embeddings

    # Pre-tokenize expected outputs
    expected_tokens = []
    for _, expected in TEST_SEQUENCES:
        token_id = tokenizer.encode(str(expected), add_special_tokens=False)[0]
        expected_tokens.append(token_id)
    expected_tokens = torch.tensor(expected_tokens, device=device)

    # Pre-tokenize test inputs
    input_texts = [f"Input: {inp} Output:" for inp, _ in TEST_SEQUENCES]

    def compute_score(prefix_tokens):
        """Compute average log probability of correct answers."""
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
        """Compute cross-entropy of the prefix."""
        with torch.no_grad():
            outputs = model(prefix_tokens.unsqueeze(0))
            logits = outputs.logits[0]
            log_probs = F.log_softmax(logits[:-1], dim=-1)
            target_tokens = prefix_tokens[1:]
            token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            return -token_log_probs.mean().item()

    def compute_gradients(prefix_tokens, lambda_val):
        """Compute gradients for token selection."""
        tokens = prefix_tokens.detach().clone()

        one_hot = F.one_hot(tokens, num_classes=vocab_size).to(embed_layer.weight.dtype)
        one_hot.requires_grad_(True)
        prefix_embeddings = one_hot @ embed_layer.weight

        # Feature score
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

        # Cross-entropy of prefix
        prefix_outputs = model(inputs_embeds=prefix_embeddings.unsqueeze(0))
        prefix_logits = prefix_outputs.logits[0]
        log_probs_prefix = F.log_softmax(prefix_logits[:-1], dim=-1)
        target_tokens = tokens[1:]
        token_log_probs = log_probs_prefix.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
        avg_xe = -token_log_probs.mean()

        # Objective: maximize log_prob - lambda * cross_entropy
        objective = avg_log_prob - lambda_val * avg_xe
        objective.backward()

        return one_hot.grad.detach()

    # Initialize with random tokens
    best_tokens = torch.randint(0, vocab_size, (prefix_length,), device=device)
    best_score = compute_score(best_tokens)
    best_xe = compute_cross_entropy(best_tokens)

    print(f"Initial: score={best_score:.3f}, XE={best_xe:.2f}")
    print(f"Initial text: '{tokenizer.decode(best_tokens)}'")

    # Track Pareto frontier
    pareto_frontier = [(best_tokens.clone(), best_score, best_xe)]

    # Main optimization loop
    for iteration in range(num_iterations):
        # Use a moderate lambda for balanced optimization
        lambda_val = 1.0

        # Compute gradients
        gradients = compute_gradients(best_tokens, lambda_val)

        # Select top-k tokens by gradient value (for maximization)
        _, top_k_indices = torch.topk(gradients, k=min(top_k, vocab_size), dim=1)

        # Generate children
        candidates = []
        for _ in range(num_children):
            child_tokens = best_tokens.clone()
            pos = np.random.randint(0, prefix_length)
            k_idx = np.random.randint(0, top_k)
            child_tokens[pos] = top_k_indices[pos, k_idx]

            score = compute_score(child_tokens)
            xe = compute_cross_entropy(child_tokens)
            candidates.append((child_tokens, score, xe))

        # Select best by combined objective
        best_candidate = max(candidates, key=lambda x: x[1] - lambda_val * x[2])

        # Update if better
        combined_score = best_candidate[1] - lambda_val * best_candidate[2]
        current_combined = best_score - lambda_val * best_xe

        if combined_score > current_combined:
            best_tokens = best_candidate[0]
            best_score = best_candidate[1]
            best_xe = best_candidate[2]
            pareto_frontier.append((best_tokens.clone(), best_score, best_xe))

        if (iteration + 1) % 5 == 0:
            print(f"Iter {iteration+1}: score={best_score:.3f}, XE={best_xe:.2f}")

    print(f"\nFinal: score={best_score:.3f}, XE={best_xe:.2f}")
    print(f"Final text: '{tokenizer.decode(best_tokens)}'")

    return best_tokens, pareto_frontier


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

    # First, test with no prefix (baseline)
    print("\n" + "="*60)
    print("BASELINE: No prefix")
    print("="*60)
    evaluate_prefix("", model, tokenizer, device)

    # Test with some hand-crafted prefixes
    print("\n" + "="*60)
    print("TESTING HAND-CRAFTED PREFIXES")
    print("="*60)

    test_prefixes = [
        "Continue the sequence: ",
        "Given numbers in ascending order, predict the next: ",
        "Pattern: each number is one more than the previous. ",
        "This is a simple counting sequence. ",
        "Add 1 to get the next number. ",
        "Ascending integers: ",
    ]

    best_acc = 0
    best_prefix = ""

    for prefix in test_prefixes:
        acc, _ = evaluate_prefix(prefix, model, tokenizer, device, verbose=False)
        print(f"{acc*100:.0f}%: '{prefix}'")
        if acc > best_acc:
            best_acc = acc
            best_prefix = prefix

    print(f"\nBest hand-crafted: {best_acc*100:.0f}% with '{best_prefix}'")

    # Run quick EPO
    print("\n" + "="*60)
    print("RUNNING QUICK EPO")
    print("="*60)

    best_tokens, pareto = quick_epo(
        model, tokenizer, device,
        num_iterations=30,
        prefix_length=10,
        top_k=256,
        num_children=24
    )

    # Evaluate the EPO-found prefix
    epo_prefix = tokenizer.decode(best_tokens)
    print("\n" + "="*60)
    print("EPO-FOUND PREFIX EVALUATION")
    print("="*60)
    evaluate_prefix(epo_prefix, model, tokenizer, device)

    # Also test with a space after
    evaluate_prefix(epo_prefix + " ", model, tokenizer, device)

    # Show Pareto frontier
    print("\n" + "="*60)
    print("PARETO FRONTIER")
    print("="*60)
    for tokens, score, xe in pareto[-5:]:
        text = tokenizer.decode(tokens)
        print(f"Score={score:.3f}, XE={xe:.2f}: '{text}'")


if __name__ == "__main__":
    main()
