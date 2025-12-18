#!/usr/bin/env python3
"""
Train EPO to find prompts that help with integer sequence prediction.

This script uses Evolutionary Prompt Optimization to discover discrete token
prefixes that improve the model's ability to predict the next number in a sequence.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from typing import List, Tuple, Dict
from prompt_interp.epo import EPOConfig, run_epo, PopulationMember


def create_test_sequences() -> List[Tuple[str, int]]:
    """Create test sequences for evaluation."""
    return [
        # Simple sequences
        ("1 2 3 4", 5),
        ("5 6 7 8", 9),
        ("10 11 12 13", 14),

        # In training range
        ("32 33 34 35", 36),
        ("100 101 102 103", 104),
        ("255 256 257 258", 259),

        # Beyond training range
        ("600 601 602 603", 604),
        ("1000 1001 1002 1003", 1004),

        # Different patterns (might not work with simple prefix)
        ("2 4 6 8", 10),
        ("5 10 15 20", 25),
    ]


def tokenize_with_prefix(
    prefix_tokens: torch.Tensor,
    input_sequence: str,
    tokenizer,
    device: str
) -> torch.Tensor:
    """
    Create input by prepending prefix to "Input: X Y Z W Output:"

    Args:
        prefix_tokens: Prefix token IDs (prefix_length,)
        input_sequence: Input sequence like "1 2 3 4"
        tokenizer: Tokenizer
        device: Device

    Returns:
        Full token sequence
    """
    # Format the input
    text = f"Input: {input_sequence} Output:"

    # Tokenize
    tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]

    # Prepend prefix
    full_tokens = torch.cat([prefix_tokens, tokens]).to(device)

    return full_tokens


def create_feature_function(
    test_sequences: List[Tuple[str, int]],
    tokenizer,
    device: str
):
    """
    Create feature function that measures log probability of correct answers.

    The feature function takes model outputs and returns a score.
    Higher score = better performance on the integer sequence task.

    Args:
        test_sequences: List of (input, expected_output) pairs
        tokenizer: Tokenizer
        device: Device

    Returns:
        Feature function
    """
    # Tokenize all expected outputs
    expected_token_ids = []
    for _, expected_num in test_sequences:
        # Get token ID for the expected number
        token_id = tokenizer.encode(str(expected_num), add_special_tokens=False)[0]
        expected_token_ids.append(token_id)

    expected_token_ids = torch.tensor(expected_token_ids, device=device)

    def feature_fn(model_outputs) -> float:
        """
        Compute average log probability of correct next tokens across test sequences.

        Note: This is called during gradient computation, so it must be differentiable.
        The model_outputs come from a forward pass with the prefix tokens only.
        We need to evaluate on the actual test sequences.

        For EPO, we actually need to modify the approach - we can't easily evaluate
        on the test sequences during gradient computation. Instead, we'll use a
        simpler approach: just maximize the probability of digits in general.

        Actually, let me reconsider. The prefix will be prepended to each test case.
        So we need to evaluate the full sequence including the prefix.
        """
        # This is a placeholder - we'll compute this differently
        # For now, return 0 and we'll compute properly in compute_feature_and_entropy
        return 0.0

    return feature_fn, expected_token_ids


def evaluate_prefix(
    prefix_tokens: torch.Tensor,
    test_sequences: List[Tuple[str, int]],
    expected_token_ids: torch.Tensor,
    model,
    tokenizer,
    device: str
) -> Tuple[float, int]:
    """
    Evaluate how well a prefix helps with test sequences.

    Args:
        prefix_tokens: Prefix token IDs
        test_sequences: List of (input, expected) pairs
        expected_token_ids: Expected token IDs for outputs
        model: Language model
        tokenizer: Tokenizer
        device: Device

    Returns:
        (average_log_prob, num_correct)
    """
    model.eval()
    total_log_prob = 0.0
    num_correct = 0

    with torch.no_grad():
        for (input_seq, expected_num), expected_token in zip(test_sequences, expected_token_ids):
            # Create full input
            full_tokens = tokenize_with_prefix(prefix_tokens, input_seq, tokenizer, device)

            # Forward pass
            outputs = model(full_tokens.unsqueeze(0))
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get log probability of expected token
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = log_probs[expected_token].item()
            total_log_prob += log_prob

            # Check if correct
            predicted_token = logits.argmax().item()
            if predicted_token == expected_token:
                num_correct += 1

    avg_log_prob = total_log_prob / len(test_sequences)
    return avg_log_prob, num_correct


def create_batched_feature_function(
    test_sequences: List[Tuple[str, int]],
    expected_token_ids: torch.Tensor,
    tokenizer,
    device: str
):
    """
    Create a feature function that properly evaluates on test sequences.

    This is a bit tricky because during EPO, we only have the prefix tokens,
    not the full sequences. So we need to construct full sequences and evaluate.
    """
    # Pre-tokenize all input sequences
    input_sequences_text = [f"Input: {inp} Output:" for inp, _ in test_sequences]

    def feature_fn_with_context(prefix_tokens: torch.Tensor, model) -> float:
        """
        Evaluate prefix on all test sequences.

        Args:
            prefix_tokens: Prefix token IDs (prefix_length,)
            model: Language model

        Returns:
            Average log probability of correct tokens
        """
        total_log_prob = 0.0

        for text, expected_token in zip(input_sequences_text, expected_token_ids):
            # Tokenize input sequence
            seq_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].to(device)

            # Prepend prefix
            full_tokens = torch.cat([prefix_tokens, seq_tokens])

            # Forward pass
            outputs = model(full_tokens.unsqueeze(0))
            logits = outputs.logits[0, -1, :]  # Last token logits

            # Get log probability of expected token
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = log_probs[expected_token]
            total_log_prob += log_prob

        avg_log_prob = total_log_prob / len(test_sequences)
        return avg_log_prob.item()

    return feature_fn_with_context


def custom_compute_feature_and_entropy(
    tokens: torch.Tensor,
    model: torch.nn.Module,
    tokenizer,
    feature_fn_with_context,
    config: EPOConfig
) -> Tuple[float, float]:
    """
    Custom version that properly evaluates the feature function.

    This replaces the standard compute_feature_and_entropy from epo.py
    """
    with torch.no_grad():
        # Compute feature score using the full test sequences
        feature_score = feature_fn_with_context(tokens, model)

        # Compute cross-entropy of the prefix itself
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits[0]  # (seq_len, vocab_size)

        log_probs = F.log_softmax(logits[:-1], dim=-1)
        target_tokens = tokens[1:]
        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        cross_entropies = -token_log_probs
        avg_cross_entropy = cross_entropies.mean().item()

    return feature_score, avg_cross_entropy


def custom_get_token_gradients(
    tokens: torch.Tensor,
    model: torch.nn.Module,
    test_sequences: List[Tuple[str, int]],
    expected_token_ids: torch.Tensor,
    tokenizer,
    lambda_value: float,
    config: EPOConfig
) -> torch.Tensor:
    """
    Compute gradients for the integer sequence task.

    We compute gradients by:
    1. For feature: maximize log prob of correct tokens across test sequences
    2. For fluency: minimize cross-entropy of prefix itself
    """
    tokens = tokens.detach().clone()

    # Get embeddings - use embedding layer's vocab size
    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.num_embeddings
    one_hot = F.one_hot(tokens, num_classes=vocab_size).to(embed_layer.weight.dtype)
    one_hot.requires_grad_(True)

    # Compute prefix embeddings from one-hot
    prefix_embeddings = one_hot @ embed_layer.weight  # (prefix_len, hidden_dim)

    # Feature score: evaluate on test sequences
    total_log_prob = 0.0

    input_sequences_text = [f"Input: {inp} Output:" for inp, _ in test_sequences]

    for text, expected_token in zip(input_sequences_text, expected_token_ids):
        # Tokenize input sequence
        seq_tokens = tokenizer(text, return_tensors="pt")["input_ids"][0].to(config.device)

        # Get embeddings for sequence
        seq_embeddings = embed_layer(seq_tokens)  # (seq_len, hidden_dim)

        # Concatenate prefix and sequence embeddings
        full_embeddings = torch.cat([prefix_embeddings, seq_embeddings], dim=0)

        # Forward pass
        outputs = model(inputs_embeds=full_embeddings.unsqueeze(0))
        logits = outputs.logits[0, -1, :]  # Last token logits

        # Get log probability of expected token
        log_probs = F.log_softmax(logits, dim=-1)
        log_prob = log_probs[expected_token]
        total_log_prob += log_prob

    avg_log_prob = total_log_prob / len(test_sequences)

    # Cross-entropy of prefix
    # We need to embed the prefix again for this computation
    prefix_outputs = model(inputs_embeds=prefix_embeddings.unsqueeze(0))
    prefix_logits = prefix_outputs.logits[0]  # (prefix_len, vocab_size)

    log_probs_prefix = F.log_softmax(prefix_logits[:-1], dim=-1)
    target_tokens = tokens[1:]
    token_log_probs = log_probs_prefix.gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)
    cross_entropies = -token_log_probs
    avg_cross_entropy = cross_entropies.mean()

    # Objective: L_λ(t) = f(t) - λ * H(t)
    objective = avg_log_prob - lambda_value * avg_cross_entropy

    # Backpropagate
    objective.backward()

    # Return gradients with respect to one-hot encodings
    return one_hot.grad.detach()


def main():
    """Main training loop."""
    print("=" * 70)
    print("EPO: Finding Prompts for Integer Sequence Prediction")
    print("=" * 70)
    print()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B"

    # EPO configuration (closer to paper values for better optimization)
    config = EPOConfig(
        population_size=8,     # M=8 from paper
        num_children=32,       # r=32 from paper
        top_k=512,             # k=512 from paper
        num_iterations=200,    # Reduced from paper's 300 for speed
        restart_every=30,      # T_restart=30 from paper
        prompt_length=12,      # n=12 from paper
        lambda_min=0.1,        # log(0.1) to log(10) range from paper
        lambda_max=10.0,
        device=device
    )

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {device}")
    print()

    # Create test sequences
    test_sequences = create_test_sequences()
    print(f"Test sequences: {len(test_sequences)}")
    for inp, expected in test_sequences[:3]:
        print(f"  {inp} -> {expected}")
    print(f"  ... ({len(test_sequences) - 3} more)")
    print()

    # Create feature function
    expected_token_ids = []
    for _, expected_num in test_sequences:
        token_id = tokenizer.encode(str(expected_num), add_special_tokens=False)[0]
        expected_token_ids.append(token_id)
    expected_token_ids = torch.tensor(expected_token_ids, device=device)

    feature_fn_with_context = create_batched_feature_function(
        test_sequences,
        expected_token_ids,
        tokenizer,
        device
    )

    # We need to modify the EPO code to use our custom feature function
    # For now, let's use a simpler approach and import directly
    from prompt_interp.epo import (
        create_lambda_grid,
        initialize_population,
        get_token_gradients,
        select_top_k_tokens,
        generate_children,
        select_best_for_each_lambda,
        perform_restart,
        PopulationMember
    )

    print("EPO Configuration:")
    print(f"  Prefix length: {config.prompt_length} tokens")
    print(f"  Population size: {config.population_size}")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Top-k: {config.top_k}")
    print()

    # Initialize
    lambda_values = create_lambda_grid(config)
    population = initialize_population(
        config,
        tokenizer.vocab_size,
        lambda_values
    )

    # Compute initial scores
    print("Computing initial population scores...")
    for member in population:
        feature_score, cross_entropy = custom_compute_feature_and_entropy(
            member.token_ids,
            model,
            tokenizer,
            feature_fn_with_context,
            config
        )
        member.feature_score = feature_score
        member.cross_entropy = cross_entropy

    print("Starting EPO optimization...")
    print()

    # Main optimization loop (custom version)
    for iteration in range(config.num_iterations):
        # Check for restart
        if iteration > 0 and iteration % config.restart_every == 0:
            best = max(population, key=lambda m: m.feature_score)
            print(f"Iter {iteration}: Restarting (best score={best.feature_score:.3f})")
            population = perform_restart(population, config)

        # Generate candidates
        all_candidates = []

        for member in population:
            # Compute gradients using our custom gradient function
            gradients = custom_get_token_gradients(
                member.token_ids,
                model,
                test_sequences,
                expected_token_ids,
                tokenizer,
                member.lambda_value,
                config
            )

            # Select top-k tokens by gradient magnitude
            top_k_tokens = select_top_k_tokens(gradients, config)

            # Generate children
            children_tokens = generate_children(member, top_k_tokens, config)

            # Evaluate children
            for child_tokens in children_tokens:
                feature_score, cross_entropy = custom_compute_feature_and_entropy(
                    child_tokens,
                    model,
                    tokenizer,
                    feature_fn_with_context,
                    config
                )

                child = PopulationMember(
                    token_ids=child_tokens,
                    feature_score=feature_score,
                    cross_entropy=cross_entropy,
                    lambda_value=member.lambda_value
                )
                all_candidates.append(child)

        # Select best for each lambda
        population = select_best_for_each_lambda(all_candidates, lambda_values)

        # Print progress
        if (iteration + 1) % 10 == 0:
            best = max(population, key=lambda m: m.feature_score)
            fluent = min(population, key=lambda m: m.cross_entropy)
            print(f"Iter {iteration + 1}/{config.num_iterations}: "
                  f"Best score={best.feature_score:.3f}, "
                  f"Most fluent XE={fluent.cross_entropy:.2f}")

    print()
    print("=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print()

    # Evaluate final population
    print("Final Pareto Frontier:")
    print()

    for i, member in enumerate(sorted(population, key=lambda m: -m.feature_score)):
        text = tokenizer.decode(member.token_ids)
        log_prob, num_correct = evaluate_prefix(
            member.token_ids,
            test_sequences,
            expected_token_ids,
            model,
            tokenizer,
            device
        )
        print(f"#{i+1} (λ={member.lambda_value:.2f}):")
        print(f"  Score: {member.feature_score:.3f}")
        print(f"  Accuracy: {num_correct}/{len(test_sequences)}")
        print(f"  Cross-entropy: {member.cross_entropy:.2f}")
        print(f"  Text: '{text}'")
        print()

    # Save best prefix
    best_member = max(population, key=lambda m: m.feature_score)
    output_dir = Path("models/epo_prefix")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'prefix_tokens': best_member.token_ids.cpu(),
        'feature_score': best_member.feature_score,
        'cross_entropy': best_member.cross_entropy,
        'config': config,
    }, output_dir / "best_prefix.pt")

    print(f"Saved best prefix to {output_dir}/best_prefix.pt")


if __name__ == "__main__":
    main()
