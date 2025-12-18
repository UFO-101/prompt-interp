#!/usr/bin/env python3
"""Generate comparison chart for GBDA vs EPO."""
import matplotlib.pyplot as plt
import numpy as np

# Results from the comparison run
results = {
    "antonyms": {
        "description": "Word → Opposite",
        "baseline": 80,
        "gbda": 80,
        "epo": 90,
        "gbda_prompt": "Opposites: hot -> cold, big -> small.",
        "epo_prompt": 'Opposites: migrationBuilder\n cold, big -> small",@"',
    },
    "plurals": {
        "description": "Singular → Plural",
        "baseline": 30,
        "gbda": 30,
        "epo": 10,
        "gbda_prompt": "Plurals: cat -> cats, child -> children.",
        "epo_prompt": "Plurals有毒 exp.WaitFor cats,夏 children聚会",
    }
}

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("GBDA vs EPO (Fluent Dreaming) Prompt Optimization", fontsize=14, fontweight='bold')

# Colors
colors = ['#95a5a6', '#3498db', '#e74c3c']  # gray, blue, red
labels = ['Baseline', 'GBDA', 'EPO']

# Bar chart for each task
for idx, (task, data) in enumerate(results.items()):
    ax = axes[idx]

    values = [data['baseline'], data['gbda'], data['epo']]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Styling
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f"{task.upper()}\n({data['description']})", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=data['baseline'], color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Add delta annotations
    gbda_delta = data['gbda'] - data['baseline']
    epo_delta = data['epo'] - data['baseline']

    delta_text = f"Δ GBDA: {gbda_delta:+d}%\nΔ EPO: {epo_delta:+d}%"
    ax.text(0.98, 0.98, delta_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/gbda_vs_epo_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: results/gbda_vs_epo_comparison.png")

# Create a second figure with prompts
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.axis('off')

# Create table data
table_data = [
    ["Task", "Method", "Test Acc", "Optimized Prompt"],
    ["", "", "", ""],
    ["ANTONYMS", "Baseline", "80%", "Opposites: hot -> cold, big -> small. Now:"],
    ["(Word→Opposite)", "GBDA", "80%", "Opposites: hot -> cold, big -> small."],
    ["", "EPO", "90% ⬆", 'Opposites: migrationBuilder cold, big -> small",@"'],
    ["", "", "", ""],
    ["PLURALS", "Baseline", "30%", "Plurals: cat -> cats, child -> children. Now:"],
    ["(Singular→Plural)", "GBDA", "30%", "Plurals: cat -> cats, child -> children."],
    ["", "EPO", "10% ⬇", "Plurals有毒 exp.WaitFor cats,夏 children聚会"],
]

# Title
ax2.text(0.5, 0.95, "GBDA vs EPO: Optimized Prompts Comparison",
         transform=ax2.transAxes, fontsize=14, fontweight='bold',
         ha='center', va='top')

# Draw table manually with better formatting
y_start = 0.85
line_height = 0.08

for i, row in enumerate(table_data):
    if i == 0:  # Header
        ax2.text(0.05, y_start, row[0], fontsize=11, fontweight='bold', va='top')
        ax2.text(0.22, y_start, row[1], fontsize=11, fontweight='bold', va='top')
        ax2.text(0.35, y_start, row[2], fontsize=11, fontweight='bold', va='top')
        ax2.text(0.48, y_start, row[3], fontsize=11, fontweight='bold', va='top')
        ax2.axhline(y=y_start - 0.02, xmin=0.03, xmax=0.97, color='black', linewidth=1)
    elif row[0] == "":
        pass  # Skip empty rows
    else:
        y = y_start - i * line_height
        ax2.text(0.05, y, row[0], fontsize=10, va='top', fontweight='bold' if row[1] == "Baseline" else 'normal')
        ax2.text(0.22, y, row[1], fontsize=10, va='top',
                color='#3498db' if row[1] == "GBDA" else '#e74c3c' if row[1] == "EPO" else 'black')
        ax2.text(0.35, y, row[2], fontsize=10, va='top',
                color='green' if '⬆' in row[2] else 'red' if '⬇' in row[2] else 'black')
        # Truncate long prompts
        prompt = row[3][:55] + "..." if len(row[3]) > 55 else row[3]
        ax2.text(0.48, y, f'"{prompt}"', fontsize=9, va='top', family='monospace',
                style='italic' if row[1] != "Baseline" else 'normal')

# Add observations
obs_y = 0.15
ax2.text(0.05, obs_y, "Key Observations:", fontsize=11, fontweight='bold', va='top')
observations = [
    "• GBDA: No change from baseline (conservative optimization)",
    "• EPO: +10% on antonyms, but -20% on plurals (overfitting)",
    "• Both methods insert nonsense tokens (Chinese, code snippets)",
    "• EPO explores more aggressively but less stable"
]
for i, obs in enumerate(observations):
    ax2.text(0.05, obs_y - 0.04 - i*0.04, obs, fontsize=10, va='top')

plt.savefig('results/gbda_vs_epo_prompts.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: results/gbda_vs_epo_prompts.png")

plt.show()
print("\nDone!")
