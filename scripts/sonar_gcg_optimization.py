#!/usr/bin/env python3
"""
SONAR GCG-style Prompt Optimization

Pipeline:
1. Start with z, decode to get discrete tokens (no grad)
2. Gradient Pass 1: Run decoder with z to get logits (keep graph)
3. Gradient Pass 2: Run decoder with z=0 + tokens + task to get loss
4. Backprop GCG-style gradients through Pass 1 to z
5. Update z and project to correct norm
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)
from fairseq2.nn.batch_layout import BatchLayout

# =============================================================================
# Task Definition
# =============================================================================

ANTONYMS_TRAIN = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
    ("good -> ", "bad"),
    ("old -> ", "young"),
]

ANTONYMS_TEST = [
    ("wet -> ", "dry"),
    ("loud -> ", "quiet"),
    ("rich -> ", "poor"),
    ("full -> ", "empty"),
]


# =============================================================================
# SONAR GCG Optimizer
# =============================================================================

class SonarGCGOptimizer:
    """Optimizes prompts using SONAR with GCG-style gradient flow."""

    def __init__(self, device="cuda"):
        self.device = device

        print("Loading SONAR models...")
        self.encoder_pipeline = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder',
            tokenizer='text_sonar_basic_encoder'
        )
        self.decoder_pipeline = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder',
            tokenizer='text_sonar_basic_encoder'
        )

        # Get SONAR decoder model
        self.sonar_decoder = self.decoder_pipeline.model.to(device)
        self.sonar_tokenizer = self.decoder_pipeline.tokenizer
        self.sonar_text_decoder = self.sonar_tokenizer.create_decoder()
        self.sonar_text_encoder = self.sonar_tokenizer.create_encoder()

        # Get embedding layer
        self.embed = self.sonar_decoder.decoder.decoder_frontend.embed
        self.vocab_size = self.embed.num_embeddings

        # Get target norm from real sentences
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, how are you today?",
            "Opposites: hot is to cold as big is to small.",
        ]
        with torch.no_grad():
            sample_embeds = self.encoder_pipeline.predict(sample_texts, source_lang='eng_Latn')
            self.target_norm = torch.norm(sample_embeds, dim=1).mean().item()
        print(f"Target embedding norm: {self.target_norm:.4f}")

        # Special tokens
        self.bos_token = 3  # BOS
        self.eos_token = 3  # EOS (same as BOS in SONAR)
        self.lang_token = 256047  # eng_Latn

        print(f"Vocab size: {self.vocab_size}")
        print("Models loaded.\n")

    def decode_z_to_tokens(self, z: torch.Tensor, max_len: int = 20) -> list[int]:
        """Decode z autoregressively to get discrete tokens (no gradients)."""
        with torch.no_grad():
            tokens = [self.bos_token, self.lang_token]
            e = z.unsqueeze(1)  # (1, 1, 1024)

            for _ in range(max_len):
                di = torch.tensor([tokens], device=self.device)
                h = self.sonar_decoder.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
                if h.dim() == 4:
                    h = h.squeeze(1)

                logits = self.sonar_decoder.decoder.final_proj(h)[0, -1, :]
                next_tok = logits.argmax().item()
                tokens.append(next_tok)

                if next_tok == self.eos_token:
                    break

            # Remove special tokens
            content_tokens = tokens[2:]  # Remove BOS, lang
            if content_tokens and content_tokens[-1] == self.eos_token:
                content_tokens = content_tokens[:-1]

            return content_tokens

    def tokens_to_text(self, tokens: list[int]) -> str:
        """Convert token IDs to text."""
        if not tokens:
            return ""
        return self.sonar_text_decoder(torch.tensor(tokens))

    def text_to_tokens(self, text: str) -> list[int]:
        """Convert text to token IDs (strips lang token and EOS)."""
        encoded = self.sonar_text_encoder(text).tolist()
        # Strip language token (256047) from start if present
        if encoded and encoded[0] == self.lang_token:
            encoded = encoded[1:]
        # Strip EOS (3) from end if present
        if encoded and encoded[-1] == self.eos_token:
            encoded = encoded[:-1]
        return encoded

    def gradient_pass_1(
        self,
        z: torch.Tensor,
        tokens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run SONAR decoder with z to get logits.
        Returns logits for each position (differentiable w.r.t. z).
        """
        # Prepare decoder input with special tokens
        full_tokens = [self.bos_token, self.lang_token] + tokens
        di = torch.tensor([full_tokens], device=self.device)
        e = z.unsqueeze(1)  # (1, 1, 1024)

        # Run decoder
        h = self.sonar_decoder.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)

        # Get logits for all positions
        logits = self.sonar_decoder.decoder.final_proj(h)  # (1, seq_len, vocab_size)

        # Return logits for content positions (skip BOS, lang token predictions)
        # logits[0, i] predicts token at position i+1
        # So logits[0, 1] predicts the first content token (position 2)
        content_logits = logits[0, 2:]  # (n_tokens, vocab_size)

        return content_logits, logits

    def gradient_pass_2(
        self,
        tokens: list[int],
        examples: list[tuple[str, str]],
    ) -> tuple[torch.Tensor, list[float], torch.Tensor]:
        """
        Run SONAR decoder with z=0 to compute task loss.
        Returns loss, per-example losses, and gradient w.r.t. token one-hot.
        """
        # Create one-hot representation for tokens (for gradient computation)
        n_tokens = len(tokens)
        one_hot = torch.zeros(n_tokens, self.vocab_size, device=self.device, requires_grad=True)
        for i, tok in enumerate(tokens):
            one_hot.data[i, tok] = 1.0

        # Get embeddings via one-hot @ embedding_weight
        token_embeds = one_hot @ self.embed.weight  # (n_tokens, embed_dim)

        # z = 0 for this pass
        z_zero = torch.zeros(1, 1024, device=self.device)
        e_zero = z_zero.unsqueeze(1)  # (1, 1, 1024)

        total_loss = torch.tensor(0.0, device=self.device)
        per_example_losses = []

        for input_text, expected in examples:
            # Tokenize suffix
            suffix_tokens = self.text_to_tokens(" " + input_text)
            target_tokens = self.text_to_tokens(expected)

            if not target_tokens:
                continue
            target_token = target_tokens[0]

            # Build full sequence: [BOS, lang, prompt_tokens, suffix_tokens]
            # We use embeddings for prompt, hard tokens for rest
            bos_embed = self.embed(torch.tensor([self.bos_token], device=self.device))
            lang_embed = self.embed(torch.tensor([self.lang_token], device=self.device))
            suffix_embed = self.embed(torch.tensor(suffix_tokens, device=self.device))

            full_embeds = torch.cat([
                bos_embed,
                lang_embed,
                token_embeds,
                suffix_embed,
            ], dim=0).unsqueeze(0)  # (1, total_len, embed_dim)

            # Run decoder with z=0
            # We need to manually run through the decoder layers
            # For simplicity, we'll use a workaround: run with the embeddings directly

            # Actually, SONAR decoder expects token IDs, not embeddings directly
            # We need to hook into the decoder differently...

            # Workaround: Use the full token sequence and compute loss
            full_tokens = [self.bos_token, self.lang_token] + tokens + suffix_tokens
            di = torch.tensor([full_tokens], device=self.device)

            h = self.sonar_decoder.decode(di, BatchLayout.of(di), e_zero, BatchLayout.of(e_zero))
            if h.dim() == 4:
                h = h.squeeze(1)

            logits = self.sonar_decoder.decoder.final_proj(h)
            # Predict next token after the full sequence
            last_logits = logits[0, -1, :]

            loss = F.cross_entropy(last_logits.unsqueeze(0).float(),
                                   torch.tensor([target_token], device=self.device))
            total_loss = total_loss + loss
            per_example_losses.append(loss.item())

        avg_loss = total_loss / len(examples)

        # Compute gradient w.r.t. one_hot
        # This requires a different approach since we used hard tokens above
        # Let's compute the gradient numerically or use a soft approximation

        return avg_loss, per_example_losses, None  # Gradient computed separately

    def compute_token_gradients(
        self,
        z: torch.Tensor,
        tokens: list[int],
        examples: list[tuple[str, str]],
    ) -> torch.Tensor:
        """
        Compute GCG-style gradients: ∂Loss/∂one_hot for each token position.
        """
        n_tokens = len(tokens)

        # Create soft one-hot that we can differentiate through
        # Start with the actual tokens as one-hot
        one_hot = torch.zeros(n_tokens, self.vocab_size, device=self.device)
        for i, tok in enumerate(tokens):
            one_hot[i, tok] = 1.0
        one_hot = one_hot.requires_grad_(True)

        # Get embeddings
        token_embeds = one_hot @ self.embed.weight  # (n_tokens, embed_dim)

        # z = 0
        z_zero = torch.zeros(1, 1024, device=self.device)
        e_zero = z_zero.unsqueeze(1)

        total_loss = torch.tensor(0.0, device=self.device)

        for input_text, expected in examples:
            suffix_tokens = self.text_to_tokens(" " + input_text)
            target_tokens = self.text_to_tokens(expected)

            if not target_tokens:
                continue
            target_token = target_tokens[0]

            # Build embedding sequence
            bos_embed = self.embed(torch.tensor([self.bos_token], device=self.device))
            lang_embed = self.embed(torch.tensor([self.lang_token], device=self.device))
            suffix_embed = self.embed(torch.tensor(suffix_tokens, device=self.device))

            # Full sequence of embeddings
            full_embeds = torch.cat([
                bos_embed,
                lang_embed,
                token_embeds,  # These have gradients
                suffix_embed,
            ], dim=0)

            # We need to run the decoder with embeddings instead of tokens
            # SONAR decoder frontend: embed -> pos_encoder -> layer_norm -> dropout
            # Then decoder layers with cross-attention to z

            # Get the decoder frontend to add positional encoding
            frontend = self.sonar_decoder.decoder.decoder_frontend

            # Add positional encoding manually
            seq_len = full_embeds.shape[0]
            # The embed already includes the embedding, we need pos encoding
            # For simplicity, let's just run through the model with a hook

            # Alternative: Use token IDs but make the embedding lookup differentiable
            # by replacing the embedding with our soft embeddings

            # For now, use hard tokens but accumulate gradient via straight-through
            full_tokens = [self.bos_token, self.lang_token] + tokens + suffix_tokens
            di = torch.tensor([full_tokens], device=self.device)

            # Hook to replace embeddings
            def embed_hook(module, input, output):
                # Replace the prompt token embeddings with our differentiable version
                new_output = output.clone()
                new_output[0, 2:2+n_tokens, :] = token_embeds
                return new_output

            handle = self.embed.register_forward_hook(embed_hook)

            try:
                h = self.sonar_decoder.decode(di, BatchLayout.of(di), e_zero, BatchLayout.of(e_zero))
                if h.dim() == 4:
                    h = h.squeeze(1)
                logits = self.sonar_decoder.decoder.final_proj(h)
                last_logits = logits[0, -1, :]
                loss = F.cross_entropy(last_logits.unsqueeze(0).float(),
                                       torch.tensor([target_token], device=self.device))
                total_loss = total_loss + loss
            finally:
                handle.remove()

        avg_loss = total_loss / len(examples)

        # Backprop to get gradient w.r.t. one_hot
        avg_loss.backward()

        token_grads = one_hot.grad  # (n_tokens, vocab_size)
        return token_grads

    def optimization_step(
        self,
        z: torch.Tensor,
        tokens: list[int],
        examples: list[tuple[str, str]],
        lr: float = 0.01,
        z_anchor: torch.Tensor = None,
        anchor_weight: float = 0.0,
        momentum: float = 0.0,
        velocity: torch.Tensor = None,
        project_norm: bool = True,
        noise_samples: int = 0,
        noise_frac: float = 0.05,
    ) -> tuple[torch.Tensor, list[int], dict, torch.Tensor]:
        """
        Perform one optimization step.
        Returns updated z, tokens, and info dict for visualization.

        Args:
            noise_samples: Number of noised variants to average gradients over (0 = no noise)
            noise_frac: Noise magnitude as fraction of z norm (default 0.05)
        """
        info = {}
        z_norm = torch.norm(z).item()

        # Generate noised variants of z if noise_samples > 0
        if noise_samples > 0:
            z_variants = [z]  # Include original
            for _ in range(noise_samples):
                noise = torch.randn_like(z) * (z_norm * noise_frac)
                z_noised = z + noise
                # Project back to original norm
                z_noised = z_noised / torch.norm(z_noised) * z_norm
                z_variants.append(z_noised)
        else:
            z_variants = [z]

        # Accumulate gradients from all variants
        accumulated_grad = None

        for z_var in z_variants:
            # Gradient Pass 1: Get logits from z (differentiable)
            z_grad = z_var.clone().requires_grad_(True)
            content_logits, full_logits = self.gradient_pass_1(z_grad, tokens)

            # Gradient Pass 2: Compute task loss with z=0
            # Get token gradients using the hook approach
            token_grads = self.compute_token_gradients(z_grad, tokens, examples)

            if token_grads is not None:
                grad_logits = token_grads.to(self.device)
                content_logits.backward(grad_logits)

                if z_grad.grad is not None:
                    if accumulated_grad is None:
                        accumulated_grad = z_grad.grad.clone()
                    else:
                        accumulated_grad += z_grad.grad

        # Average the gradients
        if accumulated_grad is not None:
            accumulated_grad = accumulated_grad / len(z_variants)
            info['noise_samples_used'] = len(z_variants)

        # Store info from the original z (first variant)
        z_grad_orig = z.clone().requires_grad_(True)
        content_logits_orig, _ = self.gradient_pass_1(z_grad_orig, tokens)
        info['pass1_logits'] = content_logits_orig.detach().cpu()
        token_grads_orig = self.compute_token_gradients(z_grad_orig, tokens, examples)
        info['token_grads'] = token_grads_orig.detach().cpu() if token_grads_orig is not None else None

        # Also compute the actual loss for logging
        with torch.no_grad():
            z_zero = torch.zeros(1, 1024, device=self.device)
            e_zero = z_zero.unsqueeze(1)

            total_loss = 0.0
            per_example_losses = []
            predictions = []

            for input_text, expected in examples:
                suffix_tokens = self.text_to_tokens(" " + input_text)
                target_tokens = self.text_to_tokens(expected)
                if not target_tokens:
                    continue
                target_token = target_tokens[0]

                full_tokens = [self.bos_token, self.lang_token] + tokens + suffix_tokens
                di = torch.tensor([full_tokens], device=self.device)

                h = self.sonar_decoder.decode(di, BatchLayout.of(di), e_zero, BatchLayout.of(e_zero))
                if h.dim() == 4:
                    h = h.squeeze(1)
                logits = self.sonar_decoder.decoder.final_proj(h)
                last_logits = logits[0, -1, :]

                loss = F.cross_entropy(last_logits.unsqueeze(0).float(),
                                       torch.tensor([target_token], device=self.device))
                total_loss += loss.item()
                per_example_losses.append(loss.item())

                pred_token = last_logits.argmax().item()
                pred_text = self.sonar_text_decoder(torch.tensor([pred_token]))
                predictions.append((input_text, expected, pred_text))

            info['loss'] = total_loss / len(examples)
            info['per_example_losses'] = per_example_losses
            info['predictions'] = predictions

        # Chain gradients back through Pass 1 to z
        # token_grads tells us how loss changes w.r.t. each token
        # content_logits tells us how z affects token distribution
        # We want: ∂Loss/∂z = ∂Loss/∂tokens · ∂tokens/∂logits · ∂logits/∂z

        if token_grads is not None:
            # Soft tokens from logits
            soft_tokens = F.softmax(content_logits, dim=-1)  # (n_tokens, vocab)

            # Gradient through softmax: ∂Loss/∂logits = ∂Loss/∂soft · ∂soft/∂logits
            # For softmax: ∂soft/∂logits = soft * (δ - soft)
            # But simpler: just use the token_grads directly weighted by soft_tokens

            # Actually, we want to backprop through the soft tokens
            # Loss depends on discrete tokens, but we approximate with soft
            # ∂Loss/∂logits ≈ token_grads (since we computed grad w.r.t. one-hot)

            # Weight by softmax to get gradient w.r.t. logits before softmax
            # This is the straight-through estimator idea
            grad_logits = token_grads.to(self.device)  # (n_tokens, vocab)

            # Now backprop through Pass 1 to z
            # content_logits was computed with grad enabled
            # We manually set the gradient and backprop
            content_logits.backward(grad_logits)

            if z_grad.grad is not None:
                # Clip gradient norm to prevent explosion
                grad_norm = z_grad.grad.norm().item()
                info['z_grad_norm'] = grad_norm

                # Normalize gradient direction and use fixed step size
                if grad_norm > 1e-8:
                    grad_direction = z_grad.grad / grad_norm
                else:
                    grad_direction = z_grad.grad

                # Apply momentum if enabled
                if velocity is not None and momentum > 0:
                    velocity = momentum * velocity + grad_direction
                    update_direction = velocity
                else:
                    velocity = grad_direction.clone() if velocity is None else velocity
                    update_direction = grad_direction

                # Update z with (momentum-augmented) gradient
                z_new = z - lr * update_direction

                # Add anchor regularization
                if z_anchor is not None and anchor_weight > 0:
                    z_new = z_new - anchor_weight * lr * (z - z_anchor)

                # Optionally project to target norm
                if project_norm:
                    z_new = z_new / torch.norm(z_new) * self.target_norm
            else:
                z_new = z
                info['z_grad_norm'] = 0.0
                velocity = velocity if velocity is not None else torch.zeros_like(z)
        else:
            z_new = z
            info['z_grad_norm'] = 0.0
            velocity = velocity if velocity is not None else torch.zeros_like(z)

        # Decode new z to get new tokens
        new_tokens = self.decode_z_to_tokens(z_new)

        info['prompt'] = self.tokens_to_text(tokens)
        info['new_prompt'] = self.tokens_to_text(new_tokens)
        info['tokens'] = tokens
        info['new_tokens'] = new_tokens

        return z_new.detach(), new_tokens, info, velocity.detach() if velocity is not None else None

    def optimize(
        self,
        train_examples: list[tuple[str, str]],
        test_examples: list[tuple[str, str]],
        n_steps: int = 50,
        lr: float = 0.01,
        seed_text: str = None,
        visualize_every: int = 5,
        anchor_weight: float = 0.0,
        momentum: float = 0.0,
        project_norm: bool = True,
    ) -> tuple[torch.Tensor, str, list[dict]]:
        """Run optimization loop.

        Args:
            momentum: momentum coefficient (0 = no momentum, 0.9 = heavy momentum)
            project_norm: if True, project z back to target_norm after each step
        """

        # Initialize z
        if seed_text:
            print(f"Initializing from seed: '{seed_text}'")
            with torch.no_grad():
                z = self.encoder_pipeline.predict([seed_text], source_lang='eng_Latn')
                z = z.to(self.device).float()
                z_anchor = z.clone()
        else:
            print("Initializing with random z")
            z = torch.randn(1, 1024, device=self.device)
            z = z / torch.norm(z) * self.target_norm
            z_anchor = z.clone()

        # Initialize momentum buffer
        velocity = torch.zeros_like(z)

        # Initial decode
        tokens = self.decode_z_to_tokens(z)
        print(f"Initial prompt: '{self.tokens_to_text(tokens)}'")

        all_info = []

        print(f"\nStarting optimization (lr={lr}, steps={n_steps}, anchor_weight={anchor_weight}, momentum={momentum}, project_norm={project_norm})")
        print("=" * 70)

        for step in range(n_steps):
            z, tokens, info, velocity = self.optimization_step(
                z, tokens, train_examples, lr=lr,
                z_anchor=z_anchor, anchor_weight=anchor_weight,
                momentum=momentum, velocity=velocity, project_norm=project_norm
            )
            info['step'] = step
            all_info.append(info)

            # Evaluate on test
            with torch.no_grad():
                z_zero = torch.zeros(1, 1024, device=self.device)
                e_zero = z_zero.unsqueeze(1)

                test_correct = 0
                for input_text, expected in test_examples:
                    suffix_tokens = self.text_to_tokens(" " + input_text)
                    target_tokens = self.text_to_tokens(expected)
                    if not target_tokens:
                        continue

                    full_tokens = [self.bos_token, self.lang_token] + tokens + suffix_tokens
                    di = torch.tensor([full_tokens], device=self.device)

                    h = self.sonar_decoder.decode(di, BatchLayout.of(di), e_zero, BatchLayout.of(e_zero))
                    if h.dim() == 4:
                        h = h.squeeze(1)
                    logits = self.sonar_decoder.decoder.final_proj(h)
                    pred = logits[0, -1, :].argmax().item()

                    if pred == target_tokens[0]:
                        test_correct += 1

                info['test_acc'] = test_correct / len(test_examples)

            if step % visualize_every == 0 or step == n_steps - 1:
                print(f"Step {step:3d} | Loss: {info['loss']:.3f} | "
                      f"Test: {info['test_acc']:.0%} | z_grad: {info.get('z_grad_norm', 0):.4f}")
                print(f"         Prompt: '{info['prompt'][:60]}'")

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        final_prompt = self.tokens_to_text(tokens)
        print(f"Final prompt: '{final_prompt}'")

        return z, final_prompt, all_info


def create_visualization(all_info: list[dict], optimizer: SonarGCGOptimizer, output_path: str):
    """Create animation showing optimization progress."""

    # Filter to visualization steps
    vis_info = [info for info in all_info if info['step'] % 5 == 0 or info['step'] == len(all_info) - 1]

    if not vis_info:
        print("No visualization data")
        return

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Subplots
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_per_example = fig.add_subplot(gs[0, 1])
    ax_prompt = fig.add_subplot(gs[0, 2])
    ax_logits = fig.add_subplot(gs[1, 0])
    ax_grads = fig.add_subplot(gs[1, 1])
    ax_top_grads = fig.add_subplot(gs[1, 2])
    ax_predictions = fig.add_subplot(gs[2, :])

    def animate(frame_idx):
        info = vis_info[frame_idx]
        step = info['step']

        # Clear all axes
        for ax in [ax_loss, ax_per_example, ax_prompt, ax_logits, ax_grads, ax_top_grads, ax_predictions]:
            ax.clear()

        # 1. Loss over time
        steps = [i['step'] for i in all_info[:step+1]]
        losses = [i['loss'] for i in all_info[:step+1]]
        ax_loss.plot(steps, losses, 'b-', linewidth=2)
        ax_loss.scatter([step], [info['loss']], color='red', s=100, zorder=5)
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Loss over time (Step {step})')
        ax_loss.grid(True, alpha=0.3)

        # 2. Per-example losses
        example_labels = [f"Ex {i+1}" for i in range(len(info['per_example_losses']))]
        colors = ['green' if l < 2 else 'orange' if l < 4 else 'red' for l in info['per_example_losses']]
        ax_per_example.bar(example_labels, info['per_example_losses'], color=colors)
        ax_per_example.set_ylabel('Loss')
        ax_per_example.set_title('Per-Example Losses')
        ax_per_example.tick_params(axis='x', rotation=45)

        # 3. Current prompt
        ax_prompt.text(0.5, 0.7, f"Step {step}", fontsize=16, ha='center', va='center',
                      transform=ax_prompt.transAxes, fontweight='bold')
        prompt_display = info['prompt'][:80] + "..." if len(info['prompt']) > 80 else info['prompt']
        ax_prompt.text(0.5, 0.5, f"'{prompt_display}'", fontsize=10, ha='center', va='center',
                      transform=ax_prompt.transAxes, wrap=True)
        ax_prompt.text(0.5, 0.3, f"Loss: {info['loss']:.3f} | Test: {info.get('test_acc', 0):.0%}",
                      fontsize=12, ha='center', va='center', transform=ax_prompt.transAxes)
        ax_prompt.axis('off')
        ax_prompt.set_title('Current State')

        # 4. Logit distribution at position 0
        if info['pass1_logits'] is not None and len(info['pass1_logits']) > 0:
            logits_pos0 = info['pass1_logits'][0].numpy()
            probs = np.exp(logits_pos0 - logits_pos0.max())
            probs = probs / probs.sum()
            top_k = 20
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            top_labels = [optimizer.sonar_text_decoder(torch.tensor([i]))[:8] for i in top_indices]

            ax_logits.barh(range(top_k), top_probs[::-1])
            ax_logits.set_yticks(range(top_k))
            ax_logits.set_yticklabels(top_labels[::-1], fontsize=8)
            ax_logits.set_xlabel('Probability')
            ax_logits.set_title('Pass 1 Logits (Pos 0, Top 20)')
        else:
            ax_logits.text(0.5, 0.5, 'No logit data', ha='center', va='center', transform=ax_logits.transAxes)
            ax_logits.set_title('Pass 1 Logits')

        # 5. Gradient distribution at position 0
        if info['token_grads'] is not None and len(info['token_grads']) > 0:
            grads_pos0 = info['token_grads'][0].numpy()

            # Top negative gradients (would reduce loss)
            top_neg_indices = np.argsort(grads_pos0)[:10]
            top_neg_vals = grads_pos0[top_neg_indices]
            top_neg_labels = [optimizer.sonar_text_decoder(torch.tensor([i]))[:8] for i in top_neg_indices]

            # Top positive gradients (would increase loss)
            top_pos_indices = np.argsort(grads_pos0)[-10:][::-1]
            top_pos_vals = grads_pos0[top_pos_indices]
            top_pos_labels = [optimizer.sonar_text_decoder(torch.tensor([i]))[:8] for i in top_pos_indices]

            all_labels = top_neg_labels + ['...'] + top_pos_labels
            all_vals = list(top_neg_vals) + [0] + list(top_pos_vals)
            colors = ['green'] * 10 + ['gray'] + ['red'] * 10

            ax_grads.barh(range(len(all_vals)), all_vals, color=colors)
            ax_grads.set_yticks(range(len(all_vals)))
            ax_grads.set_yticklabels(all_labels, fontsize=7)
            ax_grads.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax_grads.set_xlabel('Gradient')
            ax_grads.set_title('Token Gradients (Pos 0)')
        else:
            ax_grads.text(0.5, 0.5, 'No gradient data', ha='center', va='center', transform=ax_grads.transAxes)
            ax_grads.set_title('Token Gradients')

        # 6. Top gradients across all positions
        if info['token_grads'] is not None and len(info['token_grads']) > 0:
            all_grads = info['token_grads'].numpy()
            # Find top 10 most negative overall
            flat_grads = all_grads.flatten()
            top_neg_flat = np.argsort(flat_grads)[:10]
            pos_neg = [(i // all_grads.shape[1], i % all_grads.shape[1]) for i in top_neg_flat]

            labels = []
            vals = []
            for pos, tok_id in pos_neg:
                tok_text = optimizer.sonar_text_decoder(torch.tensor([tok_id]))[:6]
                labels.append(f"P{pos}:{tok_text}")
                vals.append(flat_grads[pos * all_grads.shape[1] + tok_id])

            ax_top_grads.barh(range(len(vals)), vals, color='green')
            ax_top_grads.set_yticks(range(len(vals)))
            ax_top_grads.set_yticklabels(labels, fontsize=8)
            ax_top_grads.set_xlabel('Gradient')
            ax_top_grads.set_title('Top Token Substitutions (All Pos)')
        else:
            ax_top_grads.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_top_grads.transAxes)
            ax_top_grads.set_title('Top Substitutions')

        # 7. Predictions table
        if 'predictions' in info:
            pred_text = "Predictions:\n"
            for inp, exp, pred in info['predictions']:
                status = "✓" if pred.strip() == exp.strip() else "✗"
                pred_text += f"  {status} {inp} → expected '{exp}', got '{pred}'\n"
            ax_predictions.text(0.02, 0.95, pred_text, fontsize=9, ha='left', va='top',
                               transform=ax_predictions.transAxes, family='monospace')
        ax_predictions.axis('off')
        ax_predictions.set_title('Training Predictions')

        fig.suptitle(f'SONAR GCG Optimization - Step {step}', fontsize=14, fontweight='bold')

        return []

    anim = animation.FuncAnimation(
        fig, animate, frames=len(vis_info),
        interval=2000,  # 2 seconds per frame (slow)
        blit=False
    )

    # Save animation
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=0.5)  # Very slow: 0.5 fps = 2 sec per frame
    print("Animation saved.")

    plt.close(fig)


def create_summary_plot(all_info: list[dict], optimizer: SonarGCGOptimizer, output_path: str):
    """Create summary plot showing all steps."""

    n_steps = len(all_info)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Loss curve
    ax = axes[0, 0]
    steps = [i['step'] for i in all_info]
    losses = [i['loss'] for i in all_info]
    test_accs = [i.get('test_acc', 0) for i in all_info]

    ax.plot(steps, losses, 'b-', linewidth=2, label='Train Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, test_accs, 'g--', linewidth=2, label='Test Acc')
    ax2.set_ylabel('Test Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0, 1)

    ax.set_title('Training Progress')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 2. Prompt evolution
    ax = axes[0, 1]
    sample_steps = list(range(0, n_steps, max(1, n_steps // 10))) + [n_steps - 1]
    sample_steps = sorted(set(sample_steps))

    prompt_text = "Prompt Evolution:\n\n"
    for step_idx in sample_steps:
        info = all_info[step_idx]
        prompt = info['prompt'][:50] + "..." if len(info['prompt']) > 50 else info['prompt']
        prompt_text += f"Step {info['step']:3d}: '{prompt}'\n"

    ax.text(0.02, 0.98, prompt_text, fontsize=9, ha='left', va='top',
            transform=ax.transAxes, family='monospace')
    ax.axis('off')
    ax.set_title('Prompt Evolution')

    # 3. Per-example loss heatmap
    ax = axes[1, 0]
    n_examples = len(all_info[0]['per_example_losses'])
    loss_matrix = np.array([i['per_example_losses'] for i in all_info])

    im = ax.imshow(loss_matrix.T, aspect='auto', cmap='RdYlGn_r')
    ax.set_xlabel('Step')
    ax.set_ylabel('Example')
    ax.set_title('Per-Example Loss Over Time')
    plt.colorbar(im, ax=ax, label='Loss')

    # 4. Final predictions
    ax = axes[1, 1]
    final_info = all_info[-1]
    if 'predictions' in final_info:
        pred_text = f"Final Predictions (Step {final_info['step']}):\n\n"
        for inp, exp, pred in final_info['predictions']:
            status = "✓" if pred.strip() == exp.strip() else "✗"
            pred_text += f"  {status} {inp} → expected '{exp}', got '{pred}'\n"
        ax.text(0.02, 0.98, pred_text, fontsize=10, ha='left', va='top',
                transform=ax.transAxes, family='monospace')
    ax.axis('off')
    ax.set_title('Final Predictions')

    plt.suptitle('SONAR GCG Optimization Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Summary saved to {output_path}")
    plt.close()


def run_experiment(optimizer, name: str, lr: float, anchor_weight: float, seed_text: str = None,
                   n_steps: int = 50, momentum: float = 0.0, project_norm: bool = True):
    """Run a single experiment with given hyperparameters."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  lr={lr}, anchor_weight={anchor_weight}, momentum={momentum}, project_norm={project_norm}")
    if seed_text:
        print(f"  seed='{seed_text[:50]}...'")
    else:
        print(f"  seed=RANDOM")
    print(f"{'='*70}")

    z, final_prompt, all_info = optimizer.optimize(
        train_examples=ANTONYMS_TRAIN,
        test_examples=ANTONYMS_TEST,
        n_steps=n_steps,
        lr=lr,
        seed_text=seed_text,
        visualize_every=5,
        anchor_weight=anchor_weight,
        momentum=momentum,
        project_norm=project_norm,
    )

    return {
        'name': name,
        'lr': lr,
        'anchor_weight': anchor_weight,
        'momentum': momentum,
        'project_norm': project_norm,
        'final_prompt': final_prompt,
        'final_loss': all_info[-1]['loss'],
        'final_test_acc': all_info[-1].get('test_acc', 0),
        'all_info': all_info,
    }


def main():
    print("SONAR GCG-style Prompt Optimization - Hyperparameter Search")
    print("=" * 70)

    # Create output directory
    Path("results").mkdir(exist_ok=True)

    optimizer = SonarGCGOptimizer()

    # Exp11: Test momentum and no norm projection
    # Goal: see if momentum helps stabilize training
    experiments = [
        # Baseline from exp10: lr=0.2 was best
        {"name": "exp11_baseline", "lr": 0.2, "anchor_weight": 0.0, "n_steps": 50,
         "seed_text": None, "momentum": 0.0, "project_norm": True},
        # Add momentum (should smooth gradient noise)
        {"name": "exp11_momentum0.5", "lr": 0.2, "anchor_weight": 0.0, "n_steps": 50,
         "seed_text": None, "momentum": 0.5, "project_norm": True},
        # Heavy momentum
        {"name": "exp11_momentum0.9", "lr": 0.2, "anchor_weight": 0.0, "n_steps": 50,
         "seed_text": None, "momentum": 0.9, "project_norm": True},
        # No norm projection (let z move freely)
        {"name": "exp11_no_proj", "lr": 0.2, "anchor_weight": 0.0, "n_steps": 50,
         "seed_text": None, "momentum": 0.0, "project_norm": False},
        # Momentum + no projection
        {"name": "exp11_mom0.9_no_proj", "lr": 0.2, "anchor_weight": 0.0, "n_steps": 50,
         "seed_text": None, "momentum": 0.9, "project_norm": False},
    ]

    results = []
    for exp in experiments:
        result = run_experiment(
            optimizer,
            name=exp["name"],
            lr=exp["lr"],
            anchor_weight=exp["anchor_weight"],
            seed_text=exp.get("seed_text"),
            n_steps=exp.get("n_steps", 50),
            momentum=exp.get("momentum", 0.0),
            project_norm=exp.get("project_norm", True),
        )
        results.append(result)

        # Save individual visualizations
        create_summary_plot(
            result['all_info'], optimizer,
            f"results/sonar_gcg_{exp['name']}_summary.png"
        )
        create_visualization(
            result['all_info'], optimizer,
            f"results/sonar_gcg_{exp['name']}_animation.gif"
        )

    # Print summary comparison
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    print(f"{'Name':<20} {'LR':<8} {'Anchor':<8} {'Loss':<8} {'Test Acc':<10} {'Final Prompt':<40}")
    print("-" * 94)
    for r in results:
        prompt_short = r['final_prompt'][:37] + "..." if len(r['final_prompt']) > 40 else r['final_prompt']
        print(f"{r['name']:<20} {r['lr']:<8.3f} {r['anchor_weight']:<8.2f} {r['final_loss']:<8.3f} {r['final_test_acc']:<10.0%} {prompt_short:<40}")

    # Track prompt evolution across experiments
    print("\n" + "=" * 70)
    print("PROMPT EVOLUTION (every 10 steps)")
    print("=" * 70)
    for r in results:
        print(f"\n{r['name']}:")
        for info in r['all_info']:
            if info['step'] % 10 == 0 or info['step'] == 49:
                print(f"  Step {info['step']:2d}: '{info['prompt'][:60]}'")

    print("\nDone!")


if __name__ == "__main__":
    main()
