#%%
"""
Visualize feature visualization results for a specific layer and neuron.

Creates a heatmap showing top prompts and their predictions, ordered by activation.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datetime import datetime

from prompt_interp import REPO_ROOT

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "feature_vis"
DEFAULT_PLOTS_DIR = REPO_ROOT / "results" / "feature_vis_plots"


def load_results_for_neuron(
    results_dir: str, layer_idx: int, neuron_idx: int
) -> list[dict]:
    """Load all results JSON files for a specific layer and neuron."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results

    pattern = f"layer{layer_idx}_neuron{neuron_idx}_*.json"
    for filepath in results_path.glob(pattern):
        with open(filepath) as f:
            data = json.load(f)
            data["source_file"] = str(filepath)
            results.append(data)

    return results


def extract_top_prompts_per_seed(results: list[dict], top_k_per_seed: int = 3) -> list[dict]:
    """
    Extract top prompts from each seed separately.
    Returns list of dicts with prompt, prediction, activation, activation_diff, source, grouped by seed.
    Sorted by activation_diff (neuron - layer_mean).
    """
    # Group prompts by init_text (seed)
    prompts_by_seed: dict[str, list[dict]] = {}

    for result in results:
        init_text = result["init_text"]
        if init_text not in prompts_by_seed:
            prompts_by_seed[init_text] = []

        # Extract from trajectory - all logged steps have activation_diff
        for step in result.get("trajectory", []):
            activation = step["activation"]
            layer_mean = step.get("layer_mean_activation", 0)
            activation_diff = step.get("activation_diff", activation - layer_mean)
            prompts_by_seed[init_text].append({
                "prompt": step["decoded_z"],
                "prediction": step["decoded_pred"],
                "activation": activation,
                "layer_mean_activation": layer_mean,
                "activation_diff": activation_diff,
                "source": f"step_{step['step']}",
                "init_text": init_text,
                "did_llm_rephrase": step.get("did_llm_rephrase", False),
            })

    # For each seed: deduplicate, sort by activation_diff, take top_k_per_seed
    top_prompts_by_seed: dict[str, list[dict]] = {}
    for seed, prompts in prompts_by_seed.items():
        # Deduplicate by prompt text, keeping highest activation_diff
        seen = {}
        for p in prompts:
            key = p["prompt"]
            if key not in seen or p["activation_diff"] > seen[key]["activation_diff"]:
                seen[key] = p
        # Sort by activation_diff (descending) and take top_k_per_seed
        sorted_prompts = sorted(seen.values(), key=lambda x: x["activation_diff"], reverse=True)
        top_prompts_by_seed[seed] = sorted_prompts[:top_k_per_seed]

    return top_prompts_by_seed


def wrap_text(text: str, max_chars: int = 30) -> str:
    """Wrap text to fit in cells."""
    words = text.split()
    lines = []
    current_line = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 <= max_chars:
            current_line.append(word)
            current_len += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def create_heatmap_by_seed(
    top_prompts_by_seed: dict[str, list[dict]],
    layer_idx: int,
    neuron_idx: int,
    output_path: str | None = None,
    n_cols: int = 3,
    row_height: float = 1.0,
):
    """
    Create a grid visualization of top prompts grouped by seed.
    Uses simple rectangles for a clean, compact table layout.
    """
    from matplotlib.patches import Rectangle

    n_seeds = len(top_prompts_by_seed)
    if n_seeds == 0:
        print("No prompts to visualize")
        return

    # Get global min/max activation_diff for consistent coloring
    all_diffs = [
        p["activation_diff"]
        for prompts in top_prompts_by_seed.values()
        for p in prompts
    ]
    min_diff = min(all_diffs)
    max_diff = max(all_diffs)
    diff_range = max_diff - min_diff if max_diff != min_diff else 1

    # Sort seeds by their best activation_diff (highest first)
    sorted_seeds = sorted(
        top_prompts_by_seed.keys(),
        key=lambda s: max(p["activation_diff"] for p in top_prompts_by_seed[s]),
        reverse=True,
    )

    # Grid layout
    n_rows_grid = (n_seeds + n_cols - 1) // n_cols
    max_prompts = max(len(p) for p in top_prompts_by_seed.values())

    # Create figure with subplots grid
    fig, axes = plt.subplots(n_rows_grid, n_cols, figsize=(11.5, n_rows_grid * max_prompts * row_height * 0.4))
    if n_rows_grid == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    cmap = plt.cm.YlOrRd

    for idx, seed in enumerate(sorted_seeds):
        prompts = top_prompts_by_seed[seed]
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Seed title
        truncated_seed = seed[:45] + "..." if len(seed) > 45 else seed
        ax.set_title(truncated_seed, fontsize=7, fontweight="bold", color="darkblue", pad=2)

        # Draw rows
        for i, p in enumerate(prompts):
            y = (max_prompts - i - 1) * row_height
            norm_diff = (p["activation_diff"] - min_diff) / diff_range
            color = cmap(norm_diff * 0.8 + 0.1)

            # Activation diff cell
            ax.add_patch(Rectangle((0, y), 0.08, row_height, facecolor=color, edgecolor="none"))
            ax.text(0.04, y + row_height / 2, f"{p['activation_diff']:.1f}", ha="center", va="center", fontsize=6)

            # Text cell with wrapped text (truncate to 200 chars)
            ax.add_patch(Rectangle((0.08, y), 0.92, row_height, facecolor=color, alpha=0.4, edgecolor="none"))
            prompt_text = p['prompt'][:200] + "..." if len(p['prompt']) > 200 else p['prompt']
            pred_text = p['prediction'][:200] + "..." if len(p['prediction']) > 200 else p['prediction']
            prompt_wrapped = wrap_text(prompt_text, max_chars=80)
            pred_wrapped = wrap_text(f"-> {pred_text}", max_chars=80)
            full_text = f"{prompt_wrapped}\n{pred_wrapped}"
            ax.text(0.09, y + row_height * 0.9, full_text, ha="left", va="top", fontsize=5, linespacing=0.9)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, max_prompts * row_height)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n_seeds, n_rows_grid * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    plt.suptitle(
        f"Feature Visualization: Layer {layer_idx}, Neuron {neuron_idx} ({n_seeds} seeds)",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {output_path}")

    plt.show()


#%%
# Configuration - edit these values
layer_idx = 14
neuron_idx = 105
results_dir = DEFAULT_RESULTS_DIR
plots_dir = DEFAULT_PLOTS_DIR
top_k_per_seed = 3

# Load and visualize results
results = load_results_for_neuron(results_dir, layer_idx, neuron_idx)
print(f"Found {len(results)} result files for layer {layer_idx}, neuron {neuron_idx}")

if results:
    top_prompts_by_seed = extract_top_prompts_per_seed(results, top_k_per_seed=top_k_per_seed)
    n_seeds = len(top_prompts_by_seed)
    total_prompts = sum(len(p) for p in top_prompts_by_seed.values())
    print(f"Extracted {total_prompts} prompts from {n_seeds} seeds (top {top_k_per_seed} per seed)")

    # Generate output path
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = plots_dir / f"layer{layer_idx}_neuron{neuron_idx}_{timestamp}.png"

    create_heatmap_by_seed(top_prompts_by_seed, layer_idx, neuron_idx, output_path, row_height=1)
else:
    print("No results found. Run feature visualization first.")

# %%
