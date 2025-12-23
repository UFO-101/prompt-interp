"""Run multi-init experiments to study z optimization dynamics."""

import json
import torch
from datetime import datetime

from prompt_interp.sonar_wrapper import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator
from prompt_interp.experiments import run_multi_init_experiments


# Target sentences - plausible "second sentences" in a story
TARGET_SENTENCES = [
    "She loved to play outside with her toys.",  # Actual sentence 1 from Lily story
    "He decided to go on an adventure.",  # Alternative story continuation
    "The sun was shining brightly that day.",  # Another plausible second sentence
]

# Init sentences - random/arbitrary starting points
INIT_SENTENCES = [
    "I like cheese.",
    "The weather is nice today.",
]


def main():
    print("Loading models...")
    sonar_wrapper = SonarWrapper()

    # Freeze SONAR decoder parameters
    for p in sonar_wrapper.decoder.model.parameters():
        p.requires_grad = False

    generator = SonarLLMGenerator.from_pretrained("raxtemur/sonar-llm-900m")

    # Freeze generator parameters
    for p in generator.parameters():
        p.requires_grad = False

    print(f"Running {len(TARGET_SENTENCES)} targets x {len(INIT_SENTENCES)} inits = {len(TARGET_SENTENCES) * len(INIT_SENTENCES)} experiments")
    print()

    results = run_multi_init_experiments(
        target_sentences=TARGET_SENTENCES,
        init_sentences=INIT_SENTENCES,
        sonar_wrapper=sonar_wrapper,
        generator=generator,
        n_steps=10,
        lr=0.01,
        verbose=True,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/multi_init_experiments_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump({
            "target_sentences": TARGET_SENTENCES,
            "init_sentences": INIT_SENTENCES,
            "n_steps": 10,
            "lr": 0.01,
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Also save a markdown summary
    md_path = f"results/multi_init_experiments_{timestamp}.md"
    write_markdown_summary(results, TARGET_SENTENCES, INIT_SENTENCES, md_path)
    print(f"Markdown summary saved to {md_path}")


def write_markdown_summary(results, targets, inits, output_path):
    """Write detailed markdown summary of experiments."""
    lines = [
        "# Multi-Init Next-Sentence Prediction Experiments",
        "",
        "**Goal**: Find embeddings z that cause SONAR-LLM to predict specific target sentences.",
        "",
        "**Method**: Gradient descent with decoder cross-entropy loss. We optimize z so that",
        "SONAR-LLM(z) produces an embedding that decodes to the target sentence.",
        "",
        "**Scientific question**: Does the optimized z converge to something semantically",
        "similar to what would naturally precede the target sentence?",
        "",
        "---",
        "",
        "## Configuration",
        "",
        f"- **Steps**: 10",
        f"- **Learning rate**: 0.01",
        f"- **Optimizer**: Adam",
        "",
        "### Target Sentences",
        "",
    ]

    for i, t in enumerate(targets):
        lines.append(f"{i}. \"{t}\"")

    lines.extend([
        "",
        "### Init Sentences",
        "",
    ])

    for i, s in enumerate(inits):
        lines.append(f"{i}. \"{s}\"")

    lines.extend([
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Experiment | Target | Init | Final Loss | Success |",
        "|------------|--------|------|------------|---------|",
    ])

    for r in results:
        lines.append(
            f"| {r['experiment_name']} | {r['target_idx']} | {r['init_idx']} | "
            f"{r['final_loss']:.3f} | {'Yes' if r['success'] else 'No'} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Trajectories",
        "",
    ])

    for r in results:
        lines.extend([
            f"### {r['experiment_name']}",
            "",
            f"**Target**: \"{r['target_text']}\"",
            "",
            f"**Init**: \"{r['init_text']}\"",
            "",
            "| Step | Loss | Sim | Decoded z | Decoded pred |",
            "|------|------|-----|-----------|--------------|",
        ])

        for t in r['trajectory']:
            z_short = t['decoded_z'][:50].replace("|", "/").replace("\n", " ")
            pred_short = t['decoded_pred'][:50].replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {t['step']} | {t['loss']:.3f} | {t['similarity']:.3f} | "
                f"{z_short}... | {pred_short}... |"
            )

        lines.extend([
            "",
            f"**Final z**: \"{r['final_z']}\"",
            "",
            f"**Final pred**: \"{r['final_pred']}\"",
            "",
            f"**Success**: {r['success']}",
            "",
            "---",
            "",
        ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
