#!/usr/bin/env python3
"""Create visualizations comparing all four model approaches."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from prompt_interp.evaluate_all_models import load_all_models, predict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']  # Red, Blue, Orange, Green


def evaluate_and_collect_results():
    """Run evaluation and collect detailed results."""
    print("Loading models and running evaluation...")
    models = load_all_models()

    # Test cases organized by category
    test_categories = {
        'Simple (1-10)': [
            ("1 2 3 4", 5),
            ("5 6 7 8", 9),
        ],
        'In Training Range': [
            ("10 11 12 13", 14),
            ("32 33 34 35", 36),
            ("100 101 102 103", 104),
            ("255 256 257 258", 259),
            ("0 1 2 3", 4),
            ("99 100 101 102", 103),
        ],
        'Beyond Training Range': [
            ("600 601 602 603", 604),
            ("1000 1001 1002 1003", 1004),
        ],
        'Different Patterns': [
            ("2 4 6 8", 10),
            ("5 10 15 20", 25),
        ],
    }

    # Collect results
    results = {
        'base': {'correct': 0, 'total': 0, 'by_category': {}},
        'soft_prompt': {'correct': 0, 'total': 0, 'by_category': {}},
        'lora': {'correct': 0, 'total': 0, 'by_category': {}},
        'finetuned': {'correct': 0, 'total': 0, 'by_category': {}}
    }

    model_names = ['base', 'soft_prompt', 'lora', 'finetuned']

    for category, cases in test_categories.items():
        for model_name in model_names:
            results[model_name]['by_category'][category] = {'correct': 0, 'total': 0}

        for input_seq, expected in cases:
            # Base model
            if models['base'][0] is not None:
                pred = predict(models['base'][0], models['base'][1], input_seq)
                try:
                    is_correct = int(pred) == expected
                except:
                    is_correct = False
                results['base']['correct'] += is_correct
                results['base']['total'] += 1
                results['base']['by_category'][category]['correct'] += is_correct
                results['base']['by_category'][category]['total'] += 1

            # Soft prompt
            if models['soft_prompt'][0] is not None:
                pred = predict(models['soft_prompt'][0], models['soft_prompt'][1], input_seq, use_soft_prompt=True)
                try:
                    is_correct = int(pred) == expected
                except:
                    is_correct = False
                results['soft_prompt']['correct'] += is_correct
                results['soft_prompt']['total'] += 1
                results['soft_prompt']['by_category'][category]['correct'] += is_correct
                results['soft_prompt']['by_category'][category]['total'] += 1

            # LoRA
            if models['lora'][0] is not None:
                pred = predict(models['lora'][0], models['lora'][1], input_seq)
                try:
                    is_correct = int(pred) == expected
                except:
                    is_correct = False
                results['lora']['correct'] += is_correct
                results['lora']['total'] += 1
                results['lora']['by_category'][category]['correct'] += is_correct
                results['lora']['by_category'][category]['total'] += 1

            # Finetuned
            if models['finetuned'][0] is not None:
                pred = predict(models['finetuned'][0], models['finetuned'][1], input_seq)
                try:
                    is_correct = int(pred) == expected
                except:
                    is_correct = False
                results['finetuned']['correct'] += is_correct
                results['finetuned']['total'] += 1
                results['finetuned']['by_category'][category]['correct'] += is_correct
                results['finetuned']['by_category'][category]['total'] += 1

    return results, test_categories


def create_visualizations():
    """Create comprehensive visualizations."""
    results, test_categories = evaluate_and_collect_results()

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))

    # 1. Overall Accuracy Comparison (top left)
    ax1 = plt.subplot(2, 3, 1)
    model_labels = ['Base\nModel', 'Soft\nPrompt', 'LoRA', 'Finetuned\nModel']
    accuracies = [
        100 * results['base']['correct'] / results['base']['total'],
        100 * results['soft_prompt']['correct'] / results['soft_prompt']['total'],
        100 * results['lora']['correct'] / results['lora']['total'],
        100 * results['finetuned']['correct'] / results['finetuned']['total'],
    ]
    bars = ax1.bar(model_labels, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim([0, 105])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%\n({results[["base", "soft_prompt", "lora", "finetuned"][i]]["correct"]}/{results[["base", "soft_prompt", "lora", "finetuned"][i]]["total"]})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Accuracy by Category (top middle)
    ax2 = plt.subplot(2, 3, 2)
    categories = list(test_categories.keys())
    x = np.arange(len(categories))
    width = 0.2

    base_accs = [100 * results['base']['by_category'][cat]['correct'] / results['base']['by_category'][cat]['total']
                 for cat in categories]
    soft_accs = [100 * results['soft_prompt']['by_category'][cat]['correct'] / results['soft_prompt']['by_category'][cat]['total']
                 for cat in categories]
    lora_accs = [100 * results['lora']['by_category'][cat]['correct'] / results['lora']['by_category'][cat]['total']
                 for cat in categories]
    fine_accs = [100 * results['finetuned']['by_category'][cat]['correct'] / results['finetuned']['by_category'][cat]['total']
                 for cat in categories]

    ax2.bar(x - 1.5*width, base_accs, width, label='Base', color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
    ax2.bar(x - 0.5*width, soft_accs, width, label='Soft Prompt', color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)
    ax2.bar(x + 0.5*width, lora_accs, width, label='LoRA', color=colors[2], alpha=0.8, edgecolor='black', linewidth=1)
    ax2.bar(x + 1.5*width, fine_accs, width, label='Finetuned', color=colors[3], alpha=0.8, edgecolor='black', linewidth=1)

    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Test Category', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([cat.replace(' ', '\n') for cat in categories], fontsize=8)
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.grid(axis='y', alpha=0.3)

    # 3. Parameter Efficiency (top right)
    ax3 = plt.subplot(2, 3, 3)
    base_params = 600_000_000
    soft_prompt_params = 20 * 1024  # 20 tokens × 1024 hidden dim
    lora_params = 573_440  # From training output
    finetuned_params = 600_000_000

    trainable = [0, soft_prompt_params, lora_params, finetuned_params]

    # Log scale for better visualization
    ax3.barh(model_labels, [np.log10(p) if p > 0 else 0 for p in trainable],
             color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Trainable Parameters (log₁₀ scale)', fontsize=12, fontweight='bold')
    ax3.set_title('Parameter Efficiency', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(axis='x', alpha=0.3)

    # Add labels
    labels_text = ['0\n(frozen)', f'{soft_prompt_params:,}\n(~20K)', f'{lora_params:,}\n(~573K)', f'{finetuned_params:,}\n(600M)']
    for i, (label, params) in enumerate(zip(model_labels, trainable)):
        ax3.text(np.log10(params) if params > 0 else 0.5, i,
                f'  {labels_text[i]}',
                va='center', fontsize=8, fontweight='bold')

    # 4. Model Size Comparison (bottom left)
    ax4 = plt.subplot(2, 3, 4)
    soft_prompt_size = 0.084  # 84KB
    lora_size = 2.3  # 2.3MB
    finetuned_size = 1192  # ~1.2GB

    sizes = [0, soft_prompt_size, lora_size, finetuned_size]
    bars = ax4.bar(model_labels, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax4.set_title('Saved Model Size', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    size_labels = ['0 MB\n(no training)', '0.084 MB\n(84 KB)', '2.3 MB', '1,192 MB\n(1.2 GB)']
    for bar, size_label in zip(bars, size_labels):
        height = bar.get_height()
        y_pos = height + (finetuned_size * 0.05) if height > 0 else finetuned_size * 0.05
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                size_label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 5. Accuracy vs Parameters Trade-off (bottom middle)
    ax5 = plt.subplot(2, 3, 5)

    trainable_for_plot = [1e3, soft_prompt_params, lora_params, finetuned_params]
    ax5.scatter([np.log10(trainable_for_plot[0])], [accuracies[0]],
               s=500, color=colors[0], alpha=0.8, edgecolor='black', linewidth=2, label='Base', zorder=3)
    ax5.scatter([np.log10(trainable_for_plot[1])], [accuracies[1]],
               s=500, color=colors[1], alpha=0.8, edgecolor='black', linewidth=2, label='Soft Prompt', zorder=3)
    ax5.scatter([np.log10(trainable_for_plot[2])], [accuracies[2]],
               s=500, color=colors[2], alpha=0.8, edgecolor='black', linewidth=2, label='LoRA', zorder=3)
    ax5.scatter([np.log10(trainable_for_plot[3])], [accuracies[3]],
               s=500, color=colors[3], alpha=0.8, edgecolor='black', linewidth=2, label='Finetuned', zorder=3)

    ax5.set_xlabel('Trainable Parameters (log₁₀)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Accuracy vs Parameters Trade-off', fontsize=14, fontweight='bold', pad=15)
    ax5.set_ylim([-5, 105])
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)

    # Add annotations
    ax5.annotate('No training\n0%',
                xy=(np.log10(trainable_for_plot[0]), accuracies[0]),
                xytext=(np.log10(trainable_for_plot[0])-0.5, accuracies[0]-15),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax5.annotate('~20K params\n91.7%',
                xy=(np.log10(trainable_for_plot[1]), accuracies[1]),
                xytext=(np.log10(trainable_for_plot[1]), accuracies[1]+10),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax5.annotate('~573K params\n91.7%',
                xy=(np.log10(trainable_for_plot[2]), accuracies[2]),
                xytext=(np.log10(trainable_for_plot[2]), accuracies[2]-15),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax5.annotate('600M params\n100%',
                xy=(np.log10(trainable_for_plot[3]), accuracies[3]),
                xytext=(np.log10(trainable_for_plot[3])+0.5, accuracies[3]-15),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', lw=1.5))

    # 6. Summary Statistics (bottom right)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    EXPERIMENT SUMMARY
    ══════════════════════════════════════

    Task: Integer Sequence Completion
    Dataset: 10,000 training sequences (0-500)
    Base Model: Qwen3-0.6B-Base (600M params)

    RESULTS:
    ──────────────────────────────────────

    Base Model (No Training):
      • Accuracy: {accuracies[0]:.1f}%
      • Trainable params: 0
      • Model size: 0 MB
      • Just repeats first number

    Soft Prompt Tuning:
      • Accuracy: {accuracies[1]:.1f}%
      • Trainable params: {soft_prompt_params:,} (~0.003%)
      • Model size: 84 KB
      • Failed on even number pattern

    LoRA (r=2):
      • Accuracy: {accuracies[2]:.1f}%
      • Trainable params: {lora_params:,} (~0.096%)
      • Model size: 2.3 MB
      • Failed on multiples of 5 pattern

    Full Finetuning:
      • Accuracy: {accuracies[3]:.1f}%
      • Trainable params: {finetuned_params:,} (100%)
      • Model size: 1.2 GB
      • Perfect performance

    KEY INSIGHTS:
    ──────────────────────────────────────
    Both soft prompts and LoRA achieve
    91.7% accuracy with minimal parameters!

    LoRA uses 28x more params than soft
    prompt but still 1000x fewer than full
    finetuning.
    """

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Parameter-Efficient Training Comparison: Soft Prompt vs LoRA vs Finetuning',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = Path('results_comparison_with_lora.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    create_visualizations()
