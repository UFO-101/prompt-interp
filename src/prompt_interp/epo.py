#!/usr/bin/env python3
"""
Evolutionary Prompt Optimization (EPO) for Language Models

Based on "Fluent Dreaming for Language Models" (Thompson et al., 2024)
https://arxiv.org/abs/2402.01702

EPO optimizes discrete token sequences to maximize a target feature while
maintaining fluency through cross-entropy regularization.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np


@dataclass
class EPOConfig:
    """Configuration for EPO optimization."""

    # Population and evolution
    population_size: int = 8  # M in paper
    num_children: int = 32  # r in paper
    top_k: int = 512  # k in paper

    # Optimization
    num_iterations: int = 300  # T in paper
    restart_every: int = 30  # T_restart in paper

    # Lambda values for Pareto frontier
    lambda_min: float = 0.1  # Minimum fluency weight
    lambda_max: float = 10.0  # Maximum fluency weight

    # Restart parameters
    lambda_restart_min: float = 0.667  # λ_r,min in paper
    lambda_restart_max: float = 6.0  # λ_r,max in paper

    # Token sequence
    prompt_length: int = 12  # n in paper

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PopulationMember:
    """A single member of the EPO population."""
    token_ids: torch.Tensor  # Shape: (prompt_length,)
    feature_score: float  # f(t)
    cross_entropy: float  # Average cross-entropy
    lambda_value: float  # Fluency regularization strength

    @property
    def objective(self) -> float:
        """Compute L_λ(t) = f(t) - λ * cross_entropy"""
        return self.feature_score - self.lambda_value * self.cross_entropy


def create_lambda_grid(config: EPOConfig) -> List[float]:
    """Create uniformly spaced lambda values in log space."""
    log_lambdas = np.linspace(
        np.log(config.lambda_min),
        np.log(config.lambda_max),
        config.population_size
    )
    return [float(np.exp(log_lambda)) for log_lambda in log_lambdas]


def initialize_population(
    config: EPOConfig,
    vocab_size: int,
    lambda_values: List[float],
    init_tokens: Optional[torch.Tensor] = None
) -> List[PopulationMember]:
    """
    Initialize population with random tokens or provided initialization.

    Args:
        config: EPO configuration
        vocab_size: Size of vocabulary
        lambda_values: List of lambda values for population
        init_tokens: Optional initial tokens (prompt_length,)

    Returns:
        List of PopulationMember objects
    """
    population = []

    for lambda_val in lambda_values:
        if init_tokens is not None:
            tokens = init_tokens.clone()
        else:
            tokens = torch.randint(
                0, vocab_size,
                (config.prompt_length,),
                device=config.device
            )

        member = PopulationMember(
            token_ids=tokens,
            feature_score=0.0,
            cross_entropy=0.0,
            lambda_value=lambda_val
        )
        population.append(member)

    return population


def compute_feature_and_entropy(
    tokens: torch.Tensor,
    model: torch.nn.Module,
    tokenizer,
    feature_fn: Callable[[torch.Tensor], float],
    config: EPOConfig
) -> Tuple[float, float]:
    """
    Compute feature score and cross-entropy for a token sequence.

    Args:
        tokens: Token IDs (prompt_length,)
        model: Language model
        tokenizer: Tokenizer
        feature_fn: Function that computes feature score from model outputs
        config: EPO configuration

    Returns:
        (feature_score, cross_entropy)
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(tokens.unsqueeze(0))
        logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Compute feature score
        feature_score = feature_fn(outputs)

        # Compute cross-entropy for fluency
        # H(m(t≤i), t_{i+1}) for i in [0, n-1]
        log_probs = F.log_softmax(logits[:-1], dim=-1)  # (seq_len-1, vocab_size)
        target_tokens = tokens[1:]  # (seq_len-1,)

        # Gather log probabilities of actual next tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)  # (seq_len-1,)

        # Cross-entropy is negative log probability
        cross_entropies = -token_log_probs
        avg_cross_entropy = cross_entropies.mean().item()

    return feature_score, avg_cross_entropy


def get_token_gradients(
    tokens: torch.Tensor,
    model: torch.nn.Module,
    feature_fn: Callable[[torch.Tensor], float],
    lambda_value: float,
    config: EPOConfig
) -> torch.Tensor:
    """
    Compute gradients of objective with respect to one-hot token encodings.

    Following GCG, we compute ∇_{e_x_i} L_λ(t) where e_x_i is the one-hot
    encoding of token i.

    Args:
        tokens: Token IDs (prompt_length,)
        model: Language model
        feature_fn: Function computing feature score
        lambda_value: Fluency regularization strength
        config: EPO configuration

    Returns:
        Gradients of shape (prompt_length, vocab_size)
    """
    tokens = tokens.detach().clone()

    # Get embeddings - use embedding layer's vocab size
    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.num_embeddings
    one_hot = F.one_hot(tokens, num_classes=vocab_size).float()
    one_hot.requires_grad_(True)

    # Compute embeddings from one-hot
    embeddings = one_hot @ embed_layer.weight  # (seq_len, hidden_dim)

    # Forward pass
    outputs = model(inputs_embeds=embeddings.unsqueeze(0))
    logits = outputs.logits[0]  # (seq_len, vocab_size)

    # Compute feature score
    feature_score = feature_fn(outputs)

    # Compute cross-entropy
    log_probs = F.log_softmax(logits[:-1], dim=-1)
    target_tokens = tokens[1:]
    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1)
    ).squeeze(-1)
    cross_entropies = -token_log_probs
    avg_cross_entropy = cross_entropies.mean()

    # Objective: L_λ(t) = f(t) - λ * H(t)
    objective = feature_score - lambda_value * avg_cross_entropy

    # Backpropagate
    objective.backward()

    # Return gradients with respect to one-hot encodings
    return one_hot.grad.detach()


def select_top_k_tokens(
    gradients: torch.Tensor,
    config: EPOConfig
) -> torch.Tensor:
    """
    Select top-k tokens by gradient value for each position.

    For maximization, we want tokens with the largest positive gradients,
    as these indicate tokens that would most increase the objective if
    we switched to them.

    Args:
        gradients: Gradients of shape (prompt_length, vocab_size)
        config: EPO configuration

    Returns:
        Token indices of shape (prompt_length, top_k)
    """
    # Get top-k by gradient value (largest positive for maximization)
    # This selects tokens that would most increase the objective
    _, top_indices = torch.topk(
        gradients,
        k=min(config.top_k, gradients.shape[1]),
        dim=1
    )
    return top_indices


def generate_children(
    parent: PopulationMember,
    top_k_tokens: torch.Tensor,
    config: EPOConfig
) -> List[torch.Tensor]:
    """
    Generate children by randomly replacing one token with a top-k alternative.

    Args:
        parent: Parent population member
        top_k_tokens: Top-k token indices (prompt_length, k)
        config: EPO configuration

    Returns:
        List of child token sequences
    """
    children = []

    for _ in range(config.num_children):
        # Clone parent
        child_tokens = parent.token_ids.clone()

        # Randomly select position to mutate
        pos = np.random.randint(0, config.prompt_length)

        # Randomly select one of top-k tokens for that position
        k_idx = np.random.randint(0, top_k_tokens.shape[1])
        new_token = top_k_tokens[pos, k_idx]

        # Mutate
        child_tokens[pos] = new_token
        children.append(child_tokens)

    return children


def select_best_for_each_lambda(
    candidates: List[PopulationMember],
    lambda_values: List[float]
) -> List[PopulationMember]:
    """
    Select best candidate for each lambda value (with replacement).

    This implements the key EPO selection mechanism where we maintain
    a Pareto frontier.

    Args:
        candidates: All candidate population members
        lambda_values: Target lambda values for population

    Returns:
        New population with best candidate for each lambda
    """
    new_population = []

    for lambda_val in lambda_values:
        # Find candidate with highest objective for this lambda
        # Note: We recompute objectives with the target lambda
        best_candidate = None
        best_objective = float('-inf')

        for candidate in candidates:
            # Compute objective with this lambda
            obj = candidate.feature_score - lambda_val * candidate.cross_entropy
            if obj > best_objective:
                best_objective = obj
                best_candidate = candidate

        # Create new member with the target lambda
        new_member = PopulationMember(
            token_ids=best_candidate.token_ids.clone(),
            feature_score=best_candidate.feature_score,
            cross_entropy=best_candidate.cross_entropy,
            lambda_value=lambda_val
        )
        new_population.append(new_member)

    return new_population


def perform_restart(
    population: List[PopulationMember],
    config: EPOConfig
) -> List[PopulationMember]:
    """
    Perform restart by selecting one good prompt and reinitializing.

    Args:
        population: Current population
        config: EPO configuration

    Returns:
        Restarted population (all members with same tokens but different lambdas)
    """
    # Random lambda for restart selection
    lambda_restart = np.random.uniform(
        config.lambda_restart_min,
        config.lambda_restart_max
    )

    # Select best member according to this lambda
    best_member = None
    best_objective = float('-inf')

    for member in population:
        obj = member.feature_score - lambda_restart * member.cross_entropy
        if obj > best_objective:
            best_objective = obj
            best_member = member

    # Create new population with same tokens but different lambdas
    lambda_values = create_lambda_grid(config)
    new_population = []

    for lambda_val in lambda_values:
        new_member = PopulationMember(
            token_ids=best_member.token_ids.clone(),
            feature_score=best_member.feature_score,
            cross_entropy=best_member.cross_entropy,
            lambda_value=lambda_val
        )
        new_population.append(new_member)

    return new_population


def run_epo(
    model: torch.nn.Module,
    tokenizer,
    feature_fn: Callable[[torch.Tensor], float],
    config: EPOConfig,
    init_tokens: Optional[torch.Tensor] = None,
    verbose: bool = True
) -> List[PopulationMember]:
    """
    Run EPO optimization.

    Args:
        model: Language model
        tokenizer: Tokenizer
        feature_fn: Function that computes feature score from model outputs
        config: EPO configuration
        init_tokens: Optional initial tokens
        verbose: Whether to print progress

    Returns:
        Final population (Pareto frontier)
    """
    # Initialize
    lambda_values = create_lambda_grid(config)
    population = initialize_population(
        config,
        tokenizer.vocab_size,
        lambda_values,
        init_tokens
    )

    # Compute initial scores
    for member in population:
        feature_score, cross_entropy = compute_feature_and_entropy(
            member.token_ids,
            model,
            tokenizer,
            feature_fn,
            config
        )
        member.feature_score = feature_score
        member.cross_entropy = cross_entropy

    # Main optimization loop
    for iteration in range(config.num_iterations):
        # Check for restart
        if iteration > 0 and iteration % config.restart_every == 0:
            if verbose:
                best = max(population, key=lambda m: m.feature_score)
                print(f"Iter {iteration}: Restarting (best feature={best.feature_score:.2f})")
            population = perform_restart(population, config)

        # Generate candidates
        all_candidates = []

        for member in population:
            # Compute gradients
            gradients = get_token_gradients(
                member.token_ids,
                model,
                feature_fn,
                member.lambda_value,
                config
            )

            # Select top-k tokens
            top_k_tokens = select_top_k_tokens(gradients, config)

            # Generate children
            children_tokens = generate_children(member, top_k_tokens, config)

            # Evaluate children
            for child_tokens in children_tokens:
                feature_score, cross_entropy = compute_feature_and_entropy(
                    child_tokens,
                    model,
                    tokenizer,
                    feature_fn,
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
        if verbose and (iteration + 1) % 10 == 0:
            best = max(population, key=lambda m: m.feature_score)
            fluent = min(population, key=lambda m: m.cross_entropy)
            print(f"Iter {iteration + 1}/{config.num_iterations}: "
                  f"Best feature={best.feature_score:.2f}, "
                  f"Most fluent XE={fluent.cross_entropy:.2f}")

    return population
