#!/usr/bin/env python3
"""
Visualize GBDA optimization process with animated charts.
Shows theta evolution, Gumbel-softmax samples, losses per iteration.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from dataclasses import dataclass

from prompt_interp.gbda import gumbel_softmax_sample


NUM_SAMPLES = 8  # Number of Gumbel-softmax samples per step

@dataclass
class IterationData:
    step: int
    theta: torch.Tensor  # (prompt_len, vocab_size)
    theta_grad: torch.Tensor  # (prompt_len, vocab_size) - combined gradients
    task_grad: torch.Tensor  # (prompt_len, vocab_size) - task loss gradients
    fluency_grad: torch.Tensor  # (prompt_len, vocab_size) - fluency loss gradients
    soft_tokens: torch.Tensor  # (prompt_len, vocab_size) - first sample (for visualization)
    argmax_tokens: list[int]  # argmax of theta (no noise)
    argmax_text: str
    all_sampled_tokens: list[list[int]]  # All NUM_SAMPLES sampled token lists
    all_sampled_texts: list[str]  # All NUM_SAMPLES decoded texts
    task_loss: float
    fluency_loss: float  # averaged over samples and models
    fluency_loss_model1: float  # fluency from first model (Qwen3-0.6B)
    fluency_loss_model2: float  # fluency from second model (Qwen3-1.7B)
    per_token_fluency: list[float]  # fluency loss per position
    temperature: float
    top_tokens_per_pos: list[list[tuple[str, float]]]  # top 5 tokens per position


class GBDAVisualizer:
    def __init__(self, model_names=["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base"], device="cuda"):
        """Initialize with multiple models for ensemble fluency.

        Args:
            model_names: List of model names to load. Fluency loss will be averaged across all.
        """
        if isinstance(model_names, str):
            model_names = [model_names]

        self.models = []
        self.embed_layers = []

        # Load all models
        for model_name in model_names:
            print(f"Loading model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self.models.append(model)
            self.embed_layers.append(model.get_input_embeddings())

        # Use first model's tokenizer (they should be compatible for Qwen family)
        self.tokenizer = AutoTokenizer.from_pretrained(model_names[0], trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        # For backwards compatibility, expose first model's embed layer
        self.model = self.models[0]
        self.embed_layer = self.embed_layers[0]
        self.vocab_size = self.embed_layer.num_embeddings

        print(f"Loaded {len(self.models)} models for ensemble fluency")

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        return self.tokenizer.decode(ids)

    def decode_token(self, tok_id):
        """Decode a single token ID to string."""
        return self.tokenizer.decode([tok_id]).replace('\n', '\\n').replace('\t', '\\t')

    def get_top_tokens(self, logits, k=5):
        """Get top k tokens and their probabilities from logits."""
        probs = F.softmax(logits, dim=-1)
        top_probs, top_ids = probs.topk(k)
        return [(self.decode_token(tid.item()), p.item()) for tid, p in zip(top_ids, top_probs)]

    def compute_per_token_fluency(self, soft_tokens):
        """Compute fluency loss for each token position (using HARD tokens).

        Averages across all ensemble models for consistency.
        """
        # soft_tokens: (seq_len, vocab_size)
        seq_len = soft_tokens.shape[0]
        if seq_len <= 1:
            return [0.0] * seq_len

        # Use hard tokens for computing fluency (consistent with training)
        hard_tokens = F.one_hot(soft_tokens.argmax(dim=-1), num_classes=self.vocab_size).float()

        # Average across all models
        ce_per_pos_total = torch.zeros(seq_len - 1, device=self.device)
        with torch.no_grad():
            for model, embed_layer in zip(self.models, self.embed_layers):
                hard_embeds = torch.matmul(
                    hard_tokens.unsqueeze(0).to(embed_layer.weight.dtype),
                    embed_layer.weight
                )
                outputs = model(inputs_embeds=hard_embeds)
                logits = outputs.logits[0]
                log_probs = F.log_softmax(logits[:-1], dim=-1)
                # Cross-entropy for each position (hard tokens)
                ce_per_pos_total = ce_per_pos_total - (hard_tokens[1:] * log_probs).sum(dim=-1)

        ce_per_pos = ce_per_pos_total / len(self.models)

        # Pad first position with 0
        result = [0.0] + ce_per_pos.cpu().tolist()
        return result

    def run_with_visualization(
        self,
        seed_prompt: str | None,
        train_examples: list[tuple[str, str]],
        num_iterations: int = 30,
        prompt_length: int = 10,
        save_every: int = 1
    ) -> list[IterationData]:
        """Run GBDA and collect data for visualization."""

        print(f"Running GBDA visualization for {num_iterations} iterations...")

        # Initialize theta to uniform distribution (all zeros = equal probability after softmax)
        theta = torch.zeros(prompt_length, self.vocab_size, device=self.device)
        print(f"  Uniform init: theta=0 for all tokens (equal probability)")

        theta = torch.nn.Parameter(theta)

        optimizer = torch.optim.Adam([theta], lr=0.5)  # Slightly higher LR

        iteration_data = []

        for step in range(num_iterations):
            optimizer.zero_grad()

            # Temperature annealing: 0.8 -> 0.5
            progress = step / num_iterations
            temp = 0.8 * (1 - progress) + 0.5 * progress

            # Take NUM_SAMPLES Gumbel-softmax samples and average gradients
            all_soft_tokens = []
            total_fluency_loss = torch.tensor(0.0, device=self.device)
            total_task_loss = torch.tensor(0.0, device=self.device)

            for sample_idx in range(NUM_SAMPLES):
                # Sample soft tokens
                logits = theta.unsqueeze(0)
                soft_tokens = gumbel_softmax_sample(logits, temp, hard=False)[0]  # (prompt_len, vocab)
                all_soft_tokens.append(soft_tokens)

                # Soft embeddings for task loss
                soft_embeds = torch.matmul(
                    soft_tokens.unsqueeze(0).to(self.embed_layer.weight.dtype),
                    self.embed_layer.weight
                )

                # Task loss (only compute for first sample to save time)
                if sample_idx == 0:
                    task_loss = torch.tensor(0.0, device=self.device)
                    for inp, tgt in train_examples[:5]:
                        inp_ids = self.encode(inp)
                        inp_embeds = self.embed_layer(inp_ids).unsqueeze(0)
                        full_embeds = torch.cat([soft_embeds, inp_embeds], dim=1)
                        out = self.model(inputs_embeds=full_embeds).logits[0, -1, :]
                        tgt_id = self.encode(tgt)[0].item()
                        task_loss = task_loss + F.cross_entropy(out.unsqueeze(0), torch.tensor([tgt_id], device=self.device))
                    task_loss = task_loss / 5
                    total_task_loss = task_loss

                # Fluency loss (HARD tokens with straight-through estimator)
                # Ensemble: compute fluency on ALL models and average
                if prompt_length > 1:
                    # Get hard tokens via argmax
                    hard_tokens = F.one_hot(soft_tokens.argmax(dim=-1), num_classes=self.vocab_size).float()
                    # Straight-through: forward uses hard, backward uses soft gradients
                    hard_tokens_st = soft_tokens + (hard_tokens - soft_tokens).detach()

                    # Compute fluency loss on each model and average
                    fluency_loss = torch.tensor(0.0, device=self.device)
                    per_model_fluency = []  # Track per-model fluency
                    for model_idx, (model, embed_layer) in enumerate(zip(self.models, self.embed_layers)):
                        # Use hard embeddings for model forward pass
                        fluency_embeds = torch.matmul(
                            hard_tokens_st.unsqueeze(0).to(embed_layer.weight.dtype),
                            embed_layer.weight
                        )
                        fl_outputs = model(inputs_embeds=fluency_embeds)
                        fl_logits = fl_outputs.logits[0]
                        fl_log_probs = F.log_softmax(fl_logits[:-1], dim=-1)
                        # Loss against hard tokens (with ST for gradients)
                        model_fluency = -(hard_tokens_st[1:] * fl_log_probs).sum(dim=-1).mean()
                        per_model_fluency.append(model_fluency)
                        fluency_loss = fluency_loss + model_fluency

                    # Average over models
                    fluency_loss = fluency_loss / len(self.models)
                    total_fluency_loss = total_fluency_loss + fluency_loss

                    # Accumulate per-model fluency for this sample
                    if sample_idx == 0:
                        total_per_model_fluency = [f.clone() for f in per_model_fluency]
                    else:
                        for i, f in enumerate(per_model_fluency):
                            total_per_model_fluency[i] = total_per_model_fluency[i] + f
                else:
                    fluency_loss = torch.tensor(0.0, device=self.device)
                    if sample_idx == 0:
                        total_per_model_fluency = [torch.tensor(0.0, device=self.device) for _ in self.models]

            # Average fluency loss over samples
            avg_fluency_loss = total_fluency_loss / NUM_SAMPLES
            avg_per_model_fluency = [f / NUM_SAMPLES for f in total_per_model_fluency]

            # Compute separate gradients for visualization
            if step % save_every == 0:
                # Task gradient (from first sample only)
                optimizer.zero_grad()
                total_task_loss.backward(retain_graph=True)
                task_grad = theta.grad.detach().cpu().clone() if theta.grad is not None else torch.zeros(prompt_length, self.vocab_size)

                # Fluency gradient (averaged over all samples)
                optimizer.zero_grad()
                avg_fluency_loss.backward(retain_graph=True)
                fluency_grad = theta.grad.detach().cpu().clone() if theta.grad is not None else torch.zeros(prompt_length, self.vocab_size)

                # Combined gradient (for optimizer step)
                optimizer.zero_grad()
                total_loss = 0.0 * total_task_loss + 1.0 * avg_fluency_loss  # Fluency only
                total_loss.backward()
                theta_grad = theta.grad.detach().cpu().clone() if theta.grad is not None else torch.zeros(prompt_length, self.vocab_size)
            else:
                total_loss = 0.0 * total_task_loss + 1.0 * avg_fluency_loss  # Fluency only
                total_loss.backward()

            # Collect data for this iteration (before optimizer step, so we have gradients)
            if step % save_every == 0:
                with torch.no_grad():
                    # Argmax of theta (no noise) - what we'd get deterministically
                    argmax_tokens = theta.argmax(dim=-1).cpu().tolist()
                    argmax_text = self.decode(argmax_tokens)

                    # All sampled tokens from all NUM_SAMPLES Gumbel-softmax samples
                    all_sampled_tokens = []
                    all_sampled_texts = []
                    for soft_tok in all_soft_tokens:
                        sampled = soft_tok.argmax(dim=-1).cpu().tolist()
                        all_sampled_tokens.append(sampled)
                        all_sampled_texts.append(self.decode(sampled))

                    # Get top tokens per position (store 25 for top-20 chart)
                    top_tokens = []
                    for pos in range(prompt_length):
                        top_tokens.append(self.get_top_tokens(theta[pos], k=25))

                    # Per-token fluency (use first sample)
                    per_token_fl = self.compute_per_token_fluency(all_soft_tokens[0].detach())

                    # Get per-model fluency values
                    model1_fluency = avg_per_model_fluency[0].item() if len(avg_per_model_fluency) > 0 else 0.0
                    model2_fluency = avg_per_model_fluency[1].item() if len(avg_per_model_fluency) > 1 else 0.0

                    data = IterationData(
                        step=step,
                        theta=theta.detach().cpu().clone(),
                        theta_grad=theta_grad,
                        task_grad=task_grad,
                        fluency_grad=fluency_grad,
                        soft_tokens=all_soft_tokens[0].detach().cpu().clone(),
                        argmax_tokens=argmax_tokens,
                        argmax_text=argmax_text,
                        all_sampled_tokens=all_sampled_tokens,
                        all_sampled_texts=all_sampled_texts,
                        task_loss=total_task_loss.item(),
                        fluency_loss=avg_fluency_loss.item(),
                        fluency_loss_model1=model1_fluency,
                        fluency_loss_model2=model2_fluency,
                        per_token_fluency=per_token_fl,
                        temperature=temp,
                        top_tokens_per_pos=top_tokens
                    )
                    iteration_data.append(data)

            # Print progress
            if step % save_every == 0:
                print(f"  Step {step}: fluency(0.6B)={model1_fluency:.2f}, fluency(1.7B)={model2_fluency:.2f}, avg={avg_fluency_loss.item():.2f}, temp={temp:.2f}")
                print(f"    Argmax: {argmax_text[:50]}")
                print(f"    Samples: {len(set(tuple(s) for s in all_sampled_tokens))} unique of {NUM_SAMPLES}")

            optimizer.step()

        return iteration_data


def create_animation(data: list[IterationData], output_path: str, prompt_length: int = 10,
                     focus_positions: list[int] = None, tokenizer=None,
                     train_examples: list[tuple[str, str]] = None):
    """Create animated visualization of GBDA optimization.

    Args:
        focus_positions: Which token positions to show detailed distributions for (default: [0, 3])
        tokenizer: Tokenizer for decoding token IDs to strings
        train_examples: List of (input, target) tuples used for task loss
    """
    if focus_positions is None:
        focus_positions = [0, 3]  # Show positions 0 and 3 by default

    print(f"\nCreating animation with {len(data)} frames...")
    print(f"Showing detailed token distributions for positions: {focus_positions}")

    # Set up figure with GridSpec - higher resolution
    fig = plt.figure(figsize=(22, 34), dpi=100)
    n_focus = len(focus_positions)
    # 9 rows: full dist, top-20 prob, task+, task-, fluency+, fluency-, combined+, combined-, heatmap
    gs = GridSpec(9, n_focus + 1, figure=fig, width_ratios=[1]*n_focus + [0.6],
                  height_ratios=[1, 1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.35], hspace=0.35, wspace=0.3)

    # Row 1: Full distribution (log scale) for each focused position
    ax_full = [fig.add_subplot(gs[0, i]) for i in range(n_focus)]
    ax_loss = fig.add_subplot(gs[0, n_focus])  # Loss curves on the right

    # Row 2: Top-20 tokens by probability for each focused position
    ax_top20 = [fig.add_subplot(gs[1, i]) for i in range(n_focus)]
    ax_info = fig.add_subplot(gs[1, n_focus])  # Info panel

    # Row 3-4: Task loss gradients (positive, negative)
    ax_task_pos = [fig.add_subplot(gs[2, i]) for i in range(n_focus)]
    ax_task_pos_info = fig.add_subplot(gs[2, n_focus])
    ax_task_neg = [fig.add_subplot(gs[3, i]) for i in range(n_focus)]
    ax_task_neg_info = fig.add_subplot(gs[3, n_focus])

    # Row 5-6: Fluency loss gradients (positive, negative)
    ax_fluency_pos = [fig.add_subplot(gs[4, i]) for i in range(n_focus)]
    ax_fluency_pos_info = fig.add_subplot(gs[4, n_focus])
    ax_fluency_neg = [fig.add_subplot(gs[5, i]) for i in range(n_focus)]
    ax_fluency_neg_info = fig.add_subplot(gs[5, n_focus])

    # Row 7-8: Combined gradients (positive, negative)
    ax_grad_pos = [fig.add_subplot(gs[6, i]) for i in range(n_focus)]
    ax_grad_info = fig.add_subplot(gs[6, n_focus])
    ax_grad_neg = [fig.add_subplot(gs[7, i]) for i in range(n_focus)]
    ax_grad_neg_info = fig.add_subplot(gs[7, n_focus])

    # Row 9: Per-token fluency heatmap
    ax_fluency = fig.add_subplot(gs[8, :])

    # Prepare loss history
    steps = [d.step for d in data]
    fluency_model1 = [d.fluency_loss_model1 for d in data]  # Qwen3-0.6B fluency
    fluency_model2 = [d.fluency_loss_model2 for d in data]  # Qwen3-1.7B fluency

    # Store tokenizer reference for use in animate function
    tokenizer_ref = [tokenizer]

    def animate(frame_idx):
        d = data[frame_idx]
        tokenizer = tokenizer_ref[0]

        # Clear all axes
        for ax in (ax_full + ax_top20 + ax_task_pos + ax_task_neg +
                   ax_fluency_pos + ax_fluency_neg + ax_grad_pos + ax_grad_neg):
            ax.clear()
        ax_loss.clear()
        ax_fluency.clear()
        ax_info.clear()
        ax_task_pos_info.clear()
        ax_task_neg_info.clear()
        ax_fluency_pos_info.clear()
        ax_fluency_neg_info.clear()
        ax_grad_info.clear()
        ax_grad_neg_info.clear()

        # For each focused position
        for ax_idx, pos in enumerate(focus_positions):
            # Get probabilities from theta via softmax
            theta_pos = d.theta[pos]  # (vocab_size,)
            probs = F.softmax(theta_pos, dim=-1).numpy()

            # === Row 1: Full distribution (log scale, sorted) ===
            ax = ax_full[ax_idx]
            sorted_indices = np.argsort(probs)
            sorted_probs = probs[sorted_indices]

            x = np.arange(len(sorted_probs))
            ax.semilogy(x, sorted_probs, 'b-', linewidth=0.5, alpha=0.7)
            ax.fill_between(x, sorted_probs, alpha=0.3)

            # Mark top 5 tokens
            top_k = 5
            for i in range(top_k):
                x_pos = len(sorted_probs) - 1 - i
                prob = sorted_probs[x_pos]
                ax.plot(x_pos, prob, 'ro', markersize=6)

            # Collect all sampled tokens at this position
            argmax_tok = d.argmax_tokens[pos]
            sampled_toks_at_pos = set(s[pos] for s in d.all_sampled_tokens)
            n_diff = sum(1 for s in d.all_sampled_tokens if s[pos] != argmax_tok)
            diff_marker = f" ({n_diff}/{NUM_SAMPLES} diff)" if n_diff > 0 else ""

            ax.set_xlabel('Token rank (sorted by prob)', fontsize=10)
            ax.set_ylabel('Probability (log)', fontsize=10)
            ax.set_title(f'Position {pos}: softmax(θ){diff_marker}\nArgmax: "{d.top_tokens_per_pos[pos][0][0][:10]}"',
                        fontsize=11, fontweight='bold')
            ax.set_ylim(1e-10, 1.0)
            ax.grid(True, alpha=0.3)

            # === Row 2: Top-20 tokens bar chart ===
            ax2 = ax_top20[ax_idx]
            top_k = 20
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]

            # Count how many times each token was sampled at this position
            sample_counts = {}
            for s in d.all_sampled_tokens:
                tok = s[pos]
                sample_counts[tok] = sample_counts.get(tok, 0) + 1

            # Get token labels by decoding each token ID directly
            labels = []
            for i, idx in enumerate(top_indices):
                if tokenizer is not None:
                    tok_str = tokenizer.decode([idx]).replace('\n', '\\n').replace('\t', '\\t')
                    # Mark with count if sampled (and different from argmax)
                    count = sample_counts.get(idx, 0)
                    if count > 0 and idx != argmax_tok:
                        labels.append(f"({count}){tok_str[:8]}")
                    else:
                        labels.append(tok_str[:12])
                else:
                    labels.append(f"[{idx}]")

            # Horizontal bar chart with log scale
            y_pos = np.arange(top_k)
            # Color: red=argmax, green=any sampled (darker green = more samples), blue=other
            colors = []
            for i, idx in enumerate(top_indices):
                count = sample_counts.get(idx, 0)
                if idx == argmax_tok:
                    colors.append('#e74c3c')  # Red for argmax
                elif count > 0:
                    # Green intensity based on sample count
                    intensity = min(1.0, 0.3 + 0.1 * count)  # 0.3 to 1.0
                    colors.append(f'#{int(39*intensity):02x}{int(174*intensity):02x}{int(96*intensity):02x}')
                else:
                    colors.append('#3498db')  # Blue for others
            bars = ax2.barh(y_pos, top_probs, color=colors, edgecolor='black', linewidth=0.5)

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=9)
            ax2.invert_yaxis()  # Top token at top
            ax2.set_xscale('log')  # Log scale for probability
            ax2.set_xlabel('softmax(θ) probability', fontsize=10)
            ax2.set_title(f'Position {pos}: Top 20 (red=argmax, green=sampled)', fontsize=10, fontweight='bold')
            ax2.set_xlim(1e-6, 1.0)

            # Add probability values on bars
            for bar, prob in zip(bars, top_probs):
                if prob > 1e-6:
                    ax2.text(prob * 1.5, bar.get_y() + bar.get_height()/2,
                            f'{prob:.5f}', va='center', fontsize=7)

            # Helper function for gradient charts
            def plot_grad_chart(ax, grad_tensor, pos_idx, top_k, title, is_positive, color):
                grad_vals = grad_tensor[pos_idx].numpy()
                if is_positive:
                    mask = grad_vals > 0
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        grads = grad_vals[indices]
                        order = np.argsort(grads)[-top_k:][::-1]
                        top_indices = indices[order]
                        top_grads = grads[order]
                    else:
                        top_indices, top_grads = np.array([]), np.array([])
                else:
                    mask = grad_vals < 0
                    indices = np.where(mask)[0]
                    if len(indices) > 0:
                        grads = grad_vals[indices]
                        order = np.argsort(grads)[:top_k]
                        top_indices = indices[order]
                        top_grads = -grads[order]  # Make positive for plotting
                    else:
                        top_indices, top_grads = np.array([]), np.array([])

                if len(top_indices) > 0:
                    labels = []
                    for idx in top_indices:
                        if tokenizer is not None:
                            tok_str = tokenizer.decode([idx]).replace('\n', '\\n').replace('\t', '\\t').replace('$', '\\$')
                            labels.append(tok_str[:10])
                        else:
                            labels.append(f"[{idx}]")

                    y_positions = np.arange(len(top_grads))
                    bars_g = ax.barh(y_positions, top_grads, color=color, edgecolor='black', linewidth=0.5)
                    ax.set_yticks(y_positions)
                    ax.set_yticklabels(labels, fontsize=8)
                    ax.invert_yaxis()
                    ax.set_xscale('log')
                    if len(top_grads) > 0 and top_grads.max() > 0:
                        ax.set_xlim(1e-6, top_grads.max() * 2)
                    for bar_g, grad in zip(bars_g, top_grads):
                        if grad > 1e-6:
                            ax.text(grad * 1.5, bar_g.get_y() + bar_g.get_height()/2,
                                   f'{grad:.3f}', va='center', fontsize=6)
                ax.set_xlabel('|Grad|', fontsize=8)
                ax.set_title(title, fontsize=9, fontweight='bold', color=color)

            top_k_grad = 12

            # === Task Loss Gradients ===
            plot_grad_chart(ax_task_pos[ax_idx], d.task_grad, pos, top_k_grad,
                          f'Pos {pos}: Task ↓prob', True, '#c0392b')
            plot_grad_chart(ax_task_neg[ax_idx], d.task_grad, pos, top_k_grad,
                          f'Pos {pos}: Task ↑prob', False, '#196f3d')

            # === Fluency Loss Gradients ===
            plot_grad_chart(ax_fluency_pos[ax_idx], d.fluency_grad, pos, top_k_grad,
                          f'Pos {pos}: Fluency ↓prob', True, '#e74c3c')
            plot_grad_chart(ax_fluency_neg[ax_idx], d.fluency_grad, pos, top_k_grad,
                          f'Pos {pos}: Fluency ↑prob', False, '#27ae60')

            # === Combined Gradients ===
            plot_grad_chart(ax_grad_pos[ax_idx], d.theta_grad, pos, top_k_grad,
                          f'Pos {pos}: Combined ↓prob', True, '#922b21')
            plot_grad_chart(ax_grad_neg[ax_idx], d.theta_grad, pos, top_k_grad,
                          f'Pos {pos}: Combined ↑prob', False, '#145a32')

        # === Info panels for gradient rows ===
        # Task gradient info
        ax_task_pos_info.axis('off')
        task_norm = torch.norm(d.task_grad).item()
        ax_task_pos_info.text(0.05, 0.95, f"TASK LOSS\nGradients\n\n||grad||:\n{task_norm:.3f}\n\n↓prob = tokens\nhurting task",
                             transform=ax_task_pos_info.transAxes, fontsize=8,
                             verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='#f5b7b1', alpha=0.8))
        ax_task_neg_info.axis('off')
        ax_task_neg_info.text(0.05, 0.95, f"TASK LOSS\n\n↑prob = tokens\nhelping task\nperformance",
                             transform=ax_task_neg_info.transAxes, fontsize=8,
                             verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='#abebc6', alpha=0.8))

        # Fluency gradient info
        ax_fluency_pos_info.axis('off')
        fl_norm = torch.norm(d.fluency_grad).item()
        ax_fluency_pos_info.text(0.05, 0.95, f"FLUENCY LOSS\nGradients\n\n||grad||:\n{fl_norm:.3f}\n\n↓prob = tokens\nhurting fluency",
                                transform=ax_fluency_pos_info.transAxes, fontsize=8,
                                verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8))
        ax_fluency_neg_info.axis('off')
        ax_fluency_neg_info.text(0.05, 0.95, f"FLUENCY LOSS\n\n↑prob = tokens\nimproving text\nfluency",
                                transform=ax_fluency_neg_info.transAxes, fontsize=8,
                                verticalalignment='top', fontfamily='monospace',
                                bbox=dict(boxstyle='round', facecolor='#d5f5e3', alpha=0.8))

        # Combined gradient info
        ax_grad_info.axis('off')
        total_norm = torch.norm(d.theta_grad).item()
        ax_grad_info.text(0.05, 0.95, f"COMBINED\nGradients\n\n||grad||:\n{total_norm:.3f}\n\nTask + Fluency",
                         transform=ax_grad_info.transAxes, fontsize=8,
                         verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='#e8daef', alpha=0.8))
        ax_grad_neg_info.axis('off')
        ax_grad_neg_info.text(0.05, 0.95, f"COMBINED\n\nNet effect of\nboth losses on\ntoken probs",
                             transform=ax_grad_neg_info.transAxes, fontsize=8,
                             verticalalignment='top', fontfamily='monospace',
                             bbox=dict(boxstyle='round', facecolor='#d4efdf', alpha=0.8))

        # === Loss curves (per-model fluency scores) ===
        current_step = frame_idx + 1
        ax_loss.plot(steps[:current_step], fluency_model1[:current_step], 'b-', label='Qwen3-0.6B', linewidth=2)
        ax_loss.plot(steps[:current_step], fluency_model2[:current_step], 'r-', label='Qwen3-1.7B', linewidth=2)
        ax_loss.scatter([d.step], [d.fluency_loss_model1], color='blue', s=100, zorder=5)
        ax_loss.scatter([d.step], [d.fluency_loss_model2], color='red', s=100, zorder=5)
        ax_loss.set_xlabel('Iteration', fontsize=10)
        ax_loss.set_ylabel('Fluency Loss', fontsize=10)
        ax_loss.set_title(f'Fluency Loss\n0.6B={d.fluency_loss_model1:.2f}, 1.7B={d.fluency_loss_model2:.2f}',
                         fontsize=11, fontweight='bold')
        ax_loss.legend(loc='upper right', fontsize=9)
        ax_loss.set_xlim(-0.5, max(steps) + 0.5)
        all_fluencies = fluency_model1 + fluency_model2
        if max(all_fluencies) > 0:
            ax_loss.set_ylim(0, max(all_fluencies) * 1.1)
        ax_loss.grid(True, alpha=0.3)

        # === Info panel ===
        ax_info.axis('off')

        # Count unique samples
        unique_samples = len(set(tuple(s) for s in d.all_sampled_tokens))

        # Wrap long prompts
        def wrap_text(text, width=20):
            if len(text) > width:
                lines = [text[i:i+width] for i in range(0, len(text), width)]
                return '\n'.join(lines[:2])  # Max 2 lines
            return text

        info_text = f"Step: {d.step}\nTemp: {d.temperature:.2f}\n"
        info_text += f"\n--- Argmax(theta) ---\n\"{wrap_text(d.argmax_text)}\""
        info_text += f"\n\n--- {NUM_SAMPLES} Samples ---"
        info_text += f"\n({unique_samples} unique)"
        for i, txt in enumerate(d.all_sampled_texts[:4]):  # Show first 4
            marker = "=" if d.all_sampled_tokens[i] == d.argmax_tokens else "≠"
            info_text += f"\n{i+1}{marker} \"{txt[:18]}\""
        if len(d.all_sampled_texts) > 4:
            info_text += f"\n  ...+{len(d.all_sampled_texts)-4} more"

        ax_info.text(0.05, 0.98, info_text, transform=ax_info.transAxes,
                    fontsize=7, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # === Per-token fluency heatmap ===
        positions = list(range(prompt_length))
        fl_data = np.array(d.per_token_fluency).reshape(1, -1)
        # Compute max excluding position 0 (which is always 0 and messes up the scale)
        max_fl = max(max(dd.per_token_fluency[1:]) for dd in data if len(dd.per_token_fluency) > 1) if data else 1.0
        min_fl = min(min(dd.per_token_fluency[1:]) for dd in data if len(dd.per_token_fluency) > 1) if data else 0.0
        im = ax_fluency.imshow(fl_data, aspect='auto', cmap='Reds', vmin=min_fl, vmax=max_fl)

        # Add token labels
        for pos in positions:
            tok = d.top_tokens_per_pos[pos][0][0][:8]
            color = 'blue' if pos in focus_positions else 'black'
            weight = 'bold' if pos in focus_positions else 'normal'
            ax_fluency.text(pos, 0, tok, ha='center', va='center', fontsize=10,
                           color=color, fontweight=weight)

        ax_fluency.set_xticks(positions)
        ax_fluency.set_xticklabels([f'{i}' for i in positions], fontsize=10)
        ax_fluency.set_yticks([])
        ax_fluency.set_title(f'Per-Token Fluency Loss | Full Prompt: "{d.argmax_text}"', fontsize=11)
        ax_fluency.set_xlabel('Token Position (focused positions in blue)', fontsize=10)

        fig.suptitle(f'GBDA Optimization Visualization - Step {d.step}/{max(steps)}, Temperature={d.temperature:.2f}',
                    fontsize=16, fontweight='bold')

        return []

    # Create animation - slower frame rate
    anim = animation.FuncAnimation(fig, animate, frames=len(data), interval=1500, blit=False)

    # Save as GIF with slower fps
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=0.5)  # 0.5 fps = 2 seconds per frame
    print("Done!")

    plt.close(fig)
    return anim


def main():
    print("=" * 70)
    print("GBDA Optimization Visualization")
    print("=" * 70)

    # Simple antonyms task
    train_examples = [
        ("hot -> ", "cold"), ("big -> ", "small"), ("fast -> ", "slow"),
        ("up -> ", "down"), ("happy -> ", "sad"), ("light -> ", "dark"),
        ("old -> ", "young"), ("tall -> ", "short"),
    ]
    seed_prompt = None  # Uniform initialization

    viz = GBDAVisualizer()

    # Run once with 100 steps
    print(f"\n{'='*70}")
    print("Running GBDA optimization...")
    print("=" * 70)

    data = viz.run_with_visualization(
        seed_prompt=seed_prompt,
        train_examples=train_examples,
        num_iterations=100,
        prompt_length=16,
        save_every=10
    )

    # Get final iteration data
    final = data[-1]
    print(f"\n{'='*70}")
    print(f"FINAL RESULT:")
    print(f"  Fluency (Qwen3-0.6B): {final.fluency_loss_model1:.2f}")
    print(f"  Fluency (Qwen3-1.7B): {final.fluency_loss_model2:.2f}")
    print(f"  Fluency (avg): {final.fluency_loss:.2f}")
    print(f"  Prompt: {repr(final.argmax_text)}")
    print("=" * 70)

    # Create animation
    create_animation(data, "results/gbda_optimization.gif", prompt_length=16,
                    focus_positions=[0, 8], tokenizer=viz.tokenizer,
                    train_examples=train_examples)

    print("\nAnimation saved to results/gbda_optimization.gif")


if __name__ == "__main__":
    main()
