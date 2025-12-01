#!/usr/bin/env python3
"""Calculate LoRA configuration to match soft prompt parameter count."""
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# Target parameter count (from soft prompt)
SOFT_PROMPT_PARAMS = 20 * 896  # 17,920 parameters

print("=" * 70)
print("LoRA CONFIGURATION CALCULATOR")
print("=" * 70)

# Load model config
print("\nLoading Qwen3-0.6B-Base config...")
config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)

print(f"\nModel Architecture:")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Number of layers: {config.num_hidden_layers}")
print(f"  Number of attention heads: {config.num_attention_heads}")
print(f"  Intermediate size (MLP): {config.intermediate_size}")

hidden_size = config.hidden_size
num_layers = config.num_hidden_layers

print(f"\n{'=' * 70}")
print("TARGET: Match soft prompt parameter count")
print(f"  Soft prompt parameters: {SOFT_PROMPT_PARAMS:,}")
print(f"{'=' * 70}")

# LoRA parameters calculation:
# For a linear layer with dim (d_in, d_out), LoRA adds:
#   - Matrix A: (d_in, r)
#   - Matrix B: (r, d_out)
#   - Total: r * (d_in + d_out)

print("\n" + "=" * 70)
print("LoRA CONFIGURATION OPTIONS")
print("=" * 70)

# Common target modules for attention
attention_modules = {
    "q_proj and v_proj only (2 matrices)": ["q_proj", "v_proj"],
    "all attention (4 matrices)": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

# For attention matrices: typically (hidden_size, hidden_size)
for strategy_name, modules in attention_modules.items():
    print(f"\n{strategy_name.upper()}:")
    print(f"  Modules: {modules}")

    num_matrices_per_layer = len(modules)
    params_per_rank = num_matrices_per_layer * hidden_size * 2  # *2 for (in + out)

    # Try different ranks
    for r in [1, 2, 3, 4, 5, 8, 16]:
        params_one_layer = r * params_per_rank

        # Try different numbers of layers
        for n_layers in [1, 2, 4, num_layers]:
            total_params = params_one_layer * n_layers
            diff = total_params - SOFT_PROMPT_PARAMS
            pct_diff = 100 * diff / SOFT_PROMPT_PARAMS

            if abs(pct_diff) < 50:  # Only show configs within 50% of target
                marker = "  <<<" if abs(pct_diff) < 10 else ""
                print(f"    r={r:2d}, layers={n_layers:2d}: {total_params:7,} params ({pct_diff:+6.1f}%){marker}")

print("\n" + "=" * 70)
print("RECOMMENDED CONFIGURATION")
print("=" * 70)

# Calculate exact match options
# For q_proj, v_proj (2 matrices): 2 * hidden_size * 2 * r = 4 * hidden_size * r
params_per_r_2matrices = 4 * hidden_size
# For all attention (4 matrices): 4 * hidden_size * 2 * r = 8 * hidden_size * r
params_per_r_4matrices = 8 * hidden_size

# Find best match
print("\nOPTION 1 (Closest match with q_proj, v_proj):")
r_opt1 = round(SOFT_PROMPT_PARAMS / params_per_r_2matrices)
params_opt1 = r_opt1 * params_per_r_2matrices
print(f"  Rank: {r_opt1}")
print(f"  Target modules: ['q_proj', 'v_proj']")
print(f"  Layers to adapt: All {num_layers} layers")
print(f"  Total parameters: {params_opt1:,}")
print(f"  Difference: {params_opt1 - SOFT_PROMPT_PARAMS:+,} ({100*(params_opt1-SOFT_PROMPT_PARAMS)/SOFT_PROMPT_PARAMS:+.1f}%)")

print("\nOPTION 2 (All attention matrices, fewer layers):")
# For 1 layer, all attention
r_opt2 = round(SOFT_PROMPT_PARAMS / params_per_r_4matrices)
params_opt2 = r_opt2 * params_per_r_4matrices
print(f"  Rank: {r_opt2}")
print(f"  Target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
print(f"  Layers to adapt: All {num_layers} layers")
print(f"  Total parameters: {params_opt2:,}")
print(f"  Difference: {params_opt2 - SOFT_PROMPT_PARAMS:+,} ({100*(params_opt2-SOFT_PROMPT_PARAMS)/SOFT_PROMPT_PARAMS:+.1f}%)")

print("\nOPTION 3 (Conservative - all attention, low rank):")
r_opt3 = 2
params_opt3 = r_opt3 * params_per_r_4matrices
print(f"  Rank: {r_opt3}")
print(f"  Target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']")
print(f"  Layers to adapt: All {num_layers} layers")
print(f"  Total parameters: {params_opt3:,}")
print(f"  Difference: {params_opt3 - SOFT_PROMPT_PARAMS:+,} ({100*(params_opt3-SOFT_PROMPT_PARAMS)/SOFT_PROMPT_PARAMS:+.1f}%)")

print("\n" + "=" * 70)
print(f"SELECTED: Option 3 (r={r_opt3}, all attention, standard approach)")
print("=" * 70)
