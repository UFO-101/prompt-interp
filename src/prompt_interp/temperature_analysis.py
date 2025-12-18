#!/usr/bin/env python3
"""Analyze fluency scores at different temperatures for fluent vs random prompts."""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

from prompt_interp.gbda import gumbel_softmax_sample


def main():
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).cuda()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.num_embeddings

    def encode(text):
        return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].cuda()

    prompt_length = 16

    # Create theta for fluent prompt
    fluent_text = "The quick brown fox jumps over the lazy dog."
    fluent_tokens = encode(fluent_text)[:prompt_length]
    if len(fluent_tokens) < prompt_length:
        pad = encode(".")[0].item()
        padding = torch.full((prompt_length - len(fluent_tokens),), pad, device="cuda", dtype=torch.long)
        fluent_tokens = torch.cat([fluent_tokens, padding])

    # Test multiple theta values to find the balance point
    # Max Gumbel noise over vocab_size samples ≈ log(vocab_size) ≈ 12
    theta_values = [13.0, 13.5, 14.0, 14.5, 15.0]
    temperatures = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    n_samples = 20

    print(f"Fluent prompt: {tokenizer.decode(fluent_tokens.tolist())}")
    print(f"log(vocab_size) ≈ {np.log(vocab_size):.1f}")

    # Store results for each theta
    all_results = {}

    for theta_val in theta_values:
        print(f"\n=== Testing theta={theta_val} ===")

        theta_fluent = torch.zeros(prompt_length, vocab_size, device="cuda")
        for i, tok_id in enumerate(fluent_tokens):
            theta_fluent[i, tok_id] = theta_val

        fluent_losses = []
        fluent_stds = []
        wrong_token_pcts = []

        for temp in temperatures:
            fluent_samples = []
            wrong_count = 0

            for _ in range(n_samples):
                soft_fluent = gumbel_softmax_sample(theta_fluent.unsqueeze(0), temp, hard=False)[0]
                selected = soft_fluent.argmax(dim=-1)
                wrong_count += (selected != fluent_tokens).sum().item()

                soft_embeds = torch.matmul(
                    soft_fluent.unsqueeze(0).to(embed_layer.weight.dtype),
                    embed_layer.weight
                )
                with torch.no_grad():
                    outputs = model(inputs_embeds=soft_embeds)
                    logits = outputs.logits[0]
                    log_probs = F.log_softmax(logits[:-1], dim=-1)
                    loss = -(soft_fluent[1:] * log_probs).sum(dim=-1).mean()
                    fluent_samples.append(loss.item())

            fluent_losses.append(np.mean(fluent_samples))
            fluent_stds.append(np.std(fluent_samples))
            wrong_pct = 100 * wrong_count / (n_samples * prompt_length)
            wrong_token_pcts.append(wrong_pct)

            print(f"  T={temp:.1f}: fluency={fluent_losses[-1]:.2f}±{fluent_stds[-1]:.2f}, wrong={wrong_pct:.1f}%")

        all_results[theta_val] = {
            'losses': fluent_losses,
            'stds': fluent_stds,
            'wrong_pcts': wrong_token_pcts
        }

    # Plot all theta values on single charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 1, len(theta_values)))

    for i, theta_val in enumerate(theta_values):
        results = all_results[theta_val]
        ax1.errorbar(temperatures, results['losses'], yerr=results['stds'],
                     marker='o', capsize=3, color=colors[i], linewidth=2,
                     markersize=6, label=f'θ={theta_val}')
        ax2.plot(temperatures, results['wrong_pcts'], marker='s',
                 color=colors[i], linewidth=2, markersize=6, label=f'θ={theta_val}')

    ax1.axhline(y=5.32, color='black', linestyle='--', alpha=0.5, label='True fluency (hard)')
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Fluency Loss (soft)', fontsize=12)
    ax1.set_title('Fluency Loss vs Temperature', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Wrong Token Selections (%)', fontsize=12)
    ax2.set_title('Token Selection Errors vs Temperature', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')

    plt.suptitle(f'"The quick brown fox..." - Multiple theta values (log(vocab)≈{np.log(vocab_size):.1f})',
                fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('results/temperature_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/temperature_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
