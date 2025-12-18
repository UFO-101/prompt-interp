#!/usr/bin/env python3
"""
GBDA (Gradient-based Distributional Attack) for Prompt Optimization

Adapted from "Gradient-based Adversarial Attacks against Text Transformers"
(Guo et al., 2021) https://arxiv.org/abs/2104.13733

Instead of attacking a classifier, we use GBDA to optimize a distribution
over token sequences that maximize task performance.

Key insight: Parameterize a distribution P_Θ over token sequences using a
continuous matrix Θ ∈ R^(n×V), then use Gumbel-softmax for differentiable
sampling and gradient-based optimization.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GBDAConfig:
    """Configuration for GBDA optimization."""

    # Prompt parameters
    prompt_length: int = 12  # Number of tokens in prompt

    # Optimization
    num_iterations: int = 100
    learning_rate: float = 0.3
    batch_size: int = 10  # Number of Gumbel samples per iteration

    # Gumbel-softmax temperature
    temperature: float = 1.0
    temperature_min: float = 0.1
    temperature_anneal: bool = True  # Anneal temperature over iterations

    # Soft constraints
    lambda_fluency: float = 1.0  # Weight for fluency (perplexity) constraint

    # Initialization
    init_value: float = 12.0  # Initial logit value for seed tokens

    # Device
    device: str = "cuda"


def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False
) -> torch.Tensor:
    """
    Sample from Gumbel-softmax distribution.

    Args:
        logits: Unnormalized log probabilities (batch_size, seq_len, vocab_size)
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, return one-hot vectors (straight-through estimator)

    Returns:
        Soft or hard samples of shape (batch_size, seq_len, vocab_size)
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(
        torch.rand_like(logits) + 1e-20
    ) + 1e-20)

    # Add noise and apply temperature-scaled softmax
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through estimator: argmax in forward, soft gradients in backward
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        # Use straight-through: y_hard in forward, y_soft gradients in backward
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft


class GBDAPromptOptimizer:
    """
    GBDA-based prompt optimizer.

    Optimizes a distribution over token sequences using Gumbel-softmax
    for differentiable sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        embed_layer: nn.Embedding,
        tokenizer,
        config: GBDAConfig
    ):
        """
        Args:
            model: The language model (decoder)
            embed_layer: The embedding layer
            tokenizer: Tokenizer for encoding/decoding
            config: GBDA configuration
        """
        self.model = model
        self.embed_layer = embed_layer
        self.tokenizer = tokenizer
        self.config = config
        self.vocab_size = embed_layer.num_embeddings

        # Distribution parameters Θ ∈ R^(prompt_length × vocab_size)
        # Initialized to zeros (uniform distribution)
        self.theta = None

    def initialize_from_tokens(self, token_ids: torch.Tensor):
        """
        Initialize Θ from a seed token sequence.

        Sets high logits for the seed tokens, low for others.

        Args:
            token_ids: Seed token IDs of shape (prompt_length,)
        """
        self.theta = nn.Parameter(
            torch.zeros(
                self.config.prompt_length,
                self.vocab_size,
                device=self.config.device
            )
        )

        # Set high values for seed tokens
        for i, tok_id in enumerate(token_ids[:self.config.prompt_length]):
            self.theta.data[i, tok_id] = self.config.init_value

    def initialize_from_text(self, text: str):
        """Initialize from a text string."""
        tokens = self.tokenizer.encode(text)
        # Pad or truncate to prompt_length
        if len(tokens) < self.config.prompt_length:
            # Pad with a common token (e.g., period or space)
            pad_token = self.tokenizer.encode(".")[0] if hasattr(self.tokenizer, 'encode') else 0
            tokens = tokens + [pad_token] * (self.config.prompt_length - len(tokens))
        tokens = tokens[:self.config.prompt_length]
        token_ids = torch.tensor(tokens, device=self.config.device)
        self.initialize_from_tokens(token_ids)

    def sample_soft(
        self,
        batch_size: int,
        temperature: float
    ) -> torch.Tensor:
        """
        Sample soft token distributions using Gumbel-softmax.

        Args:
            batch_size: Number of samples
            temperature: Gumbel-softmax temperature

        Returns:
            Soft samples of shape (batch_size, prompt_length, vocab_size)
        """
        # Expand theta for batch
        logits = self.theta.unsqueeze(0).expand(batch_size, -1, -1)
        return gumbel_softmax_sample(logits, temperature, hard=False)

    def sample_hard(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample discrete tokens from the distribution.

        Args:
            num_samples: Number of samples to draw

        Returns:
            Token IDs of shape (num_samples, prompt_length)
        """
        with torch.no_grad():
            probs = F.softmax(self.theta, dim=-1)
            # Sample from categorical distribution
            samples = torch.multinomial(
                probs.view(-1, self.vocab_size),
                num_samples=num_samples,
                replacement=True
            )
            # Reshape to (num_samples, prompt_length)
            return samples.T

    def get_argmax_tokens(self) -> torch.Tensor:
        """Get the most likely token sequence."""
        with torch.no_grad():
            return self.theta.argmax(dim=-1)

    def soft_embed(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert soft token distributions to embeddings.

        e(π) = Σ_j π_j * e(j)

        Args:
            soft_tokens: Shape (batch_size, seq_len, vocab_size)

        Returns:
            Embeddings of shape (batch_size, seq_len, embed_dim)
        """
        # soft_tokens: (batch, seq, vocab)
        # embed_layer.weight: (vocab, embed_dim)
        return torch.matmul(soft_tokens, self.embed_layer.weight)

    def compute_fluency_loss(
        self,
        soft_tokens: torch.Tensor,
        lm_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute fluency loss using language model perplexity.

        NLL(π) = -Σ_i log p(π_i | π_1...π_{i-1})

        This is the cross-entropy between the soft token distribution
        and the LM's predicted next token distribution.

        Args:
            soft_tokens: Shape (batch_size, seq_len, vocab_size)
            lm_model: Language model for computing perplexity

        Returns:
            Average negative log-likelihood
        """
        batch_size, seq_len, vocab_size = soft_tokens.shape

        if seq_len <= 1:
            return torch.tensor(0.0, device=soft_tokens.device)

        # Get soft embeddings
        soft_embeds = self.soft_embed(soft_tokens)

        # Forward through LM
        outputs = lm_model(inputs_embeds=soft_embeds)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Compute cross-entropy between soft tokens and LM predictions
        # For position i, we want log p(π_i | π_1...π_{i-1})
        # This is the cross-entropy between π_i and the predicted distribution
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq-1, vocab)

        # Cross-entropy: -Σ_j π_j * log p_j
        ce = -(soft_tokens[:, 1:, :] * log_probs).sum(dim=-1)  # (batch, seq-1)

        return ce.mean()

    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(token_ids.cpu().tolist())
        else:
            # For custom tokenizers
            return self.tokenizer(token_ids.cpu())


class GBDATaskOptimizer:
    """
    GBDA optimizer for task-specific prompt learning.

    Optimizes prompts to maximize task performance while maintaining fluency.
    """

    def __init__(
        self,
        model: nn.Module,
        embed_layer: nn.Embedding,
        tokenizer,
        config: GBDAConfig,
        lm_model: Optional[nn.Module] = None
    ):
        """
        Args:
            model: The task model (decoder used for evaluation)
            embed_layer: The embedding layer
            tokenizer: Tokenizer
            config: GBDA configuration
            lm_model: Language model for fluency constraint (default: same as model)
        """
        self.prompt_opt = GBDAPromptOptimizer(model, embed_layer, tokenizer, config)
        self.model = model
        self.lm_model = lm_model if lm_model is not None else model
        self.config = config
        self.tokenizer = tokenizer
        self.embed_layer = embed_layer

    def optimize(
        self,
        examples: list[tuple[str, str]],
        task_loss_fn: Callable,
        seed_text: str,
        verbose: bool = True
    ) -> dict:
        """
        Optimize prompt distribution for the given task.

        Args:
            examples: List of (input, target) pairs
            task_loss_fn: Function(soft_prompt_embeds, examples) -> loss
            seed_text: Initial prompt text
            verbose: Print progress

        Returns:
            Dictionary with optimization results
        """
        # Initialize from seed
        self.prompt_opt.initialize_from_text(seed_text)

        # Optimizer
        optimizer = torch.optim.Adam([self.prompt_opt.theta], lr=self.config.learning_rate)

        best_loss = float('inf')
        best_tokens = None
        best_text = None

        history = []

        for step in range(self.config.num_iterations):
            optimizer.zero_grad()

            # Compute temperature (optionally anneal)
            if self.config.temperature_anneal:
                progress = step / self.config.num_iterations
                temp = self.config.temperature * (1 - progress) + self.config.temperature_min * progress
            else:
                temp = self.config.temperature

            # Sample soft tokens using Gumbel-softmax
            soft_tokens = self.prompt_opt.sample_soft(self.config.batch_size, temp)

            # Compute soft embeddings
            soft_embeds = self.prompt_opt.soft_embed(soft_tokens)

            # Compute task loss (averaged over batch)
            task_loss = task_loss_fn(soft_embeds, examples)

            # Compute fluency loss
            if self.config.lambda_fluency > 0:
                fluency_loss = self.prompt_opt.compute_fluency_loss(
                    soft_tokens, self.lm_model
                )
            else:
                fluency_loss = torch.tensor(0.0, device=self.config.device)

            # Total loss
            total_loss = task_loss + self.config.lambda_fluency * fluency_loss

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            # Track best
            with torch.no_grad():
                current_tokens = self.prompt_opt.get_argmax_tokens()
                current_text = self.prompt_opt.decode_tokens(current_tokens)

                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_tokens = current_tokens.clone()
                    best_text = current_text

            # Log progress
            if verbose and step % 10 == 0:
                print(f"[{step:3d}] task={task_loss.item():.3f} "
                      f"fluency={fluency_loss.item():.3f} "
                      f"temp={temp:.2f} | {current_text[:50]}")

            history.append({
                'step': step,
                'task_loss': task_loss.item(),
                'fluency_loss': fluency_loss.item(),
                'total_loss': total_loss.item(),
                'temperature': temp,
                'prompt': current_text
            })

        return {
            'best_tokens': best_tokens,
            'best_text': best_text,
            'best_loss': best_loss,
            'final_tokens': self.prompt_opt.get_argmax_tokens(),
            'final_text': self.prompt_opt.decode_tokens(self.prompt_opt.get_argmax_tokens()),
            'theta': self.prompt_opt.theta.detach().clone(),
            'history': history
        }

    def sample_prompts(self, num_samples: int = 10) -> list[str]:
        """Sample discrete prompts from the learned distribution."""
        samples = self.prompt_opt.sample_hard(num_samples)
        return [self.prompt_opt.decode_tokens(s) for s in samples]
