"""Study how much noise can be added to SONAR embeddings while retaining semantic content."""

import json
import torch
from datetime import datetime

from prompt_interp.sonar_wrapper import SonarWrapper


PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "The weather is nice today.",
    "I like cheese.",
    "He decided to go on an adventure.",
    "She loved to play outside with her toys.",
]

NOISE_LEVELS = [0.0, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]


def add_noise(z: torch.Tensor, noise_level: float, project: bool) -> torch.Tensor:
    """Add Gaussian noise to embedding, optionally projecting back to original norm."""
    orig_norm = z.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(z) * noise_level * orig_norm
    z_noisy = z + noise
    if project:
        z_noisy = z_noisy * (orig_norm / (z_noisy.norm(dim=-1, keepdim=True) + 1e-8))
    return z_noisy


def run_experiment(sonar: SonarWrapper):
    results = []

    for prompt in PROMPTS:
        z = sonar.encode([prompt])  # (1, 1024)
        orig_norm = z.norm().item()

        for noise_level in NOISE_LEVELS:
            for project in [False, True]:
                z_noisy = add_noise(z, noise_level, project)
                decoded = sonar.decode(z_noisy)[0]
                cos_sim = torch.nn.functional.cosine_similarity(z, z_noisy, dim=-1).item()
                noisy_norm = z_noisy.norm().item()

                results.append({
                    "prompt": prompt,
                    "noise_level": noise_level,
                    "project": project,
                    "decoded": decoded,
                    "cosine_sim": cos_sim,
                    "orig_norm": orig_norm,
                    "noisy_norm": noisy_norm,
                    "exact_match": decoded.strip() == prompt.strip(),
                })

                print(f"noise={noise_level:.2f} proj={project} sim={cos_sim:.3f}")
                print(f"  orig: {prompt[:60]}")
                print(f"  decoded: {decoded[:60]}")
                print()

    return results


def write_markdown(results, path):
    lines = [
        "# SONAR Embedding Noise Robustness",
        "",
        "How much noise can be added to a SONAR embedding while retaining semantic content?",
        "",
        "## Method",
        "- Add Gaussian noise scaled by `noise_level * original_norm`",
        "- Test with and without projecting back to original norm",
        "",
        "## Results by Prompt",
        "",
    ]

    for prompt in PROMPTS:
        lines.append(f"### \"{prompt[:50]}...\"" if len(prompt) > 50 else f"### \"{prompt}\"")
        lines.append("")
        lines.append("| Noise | Project | Cos Sim | Decoded |")
        lines.append("|-------|---------|---------|---------|")

        for r in results:
            if r["prompt"] == prompt:
                dec_short = r["decoded"][:40].replace("|", "/").replace("\n", " ")
                match = " ✓" if r["exact_match"] else ""
                lines.append(
                    f"| {r['noise_level']:.2f} | {r['project']} | {r['cosine_sim']:.3f} | {dec_short}...{match} |"
                )
        lines.append("")

    lines.extend([
        "## Summary",
        "",
        "| Noise Level | Exact Match Rate (no proj) | Exact Match Rate (proj) |",
        "|-------------|---------------------------|------------------------|",
    ])

    for noise_level in NOISE_LEVELS:
        no_proj = [r for r in results if r["noise_level"] == noise_level and not r["project"]]
        proj = [r for r in results if r["noise_level"] == noise_level and r["project"]]
        no_proj_rate = sum(r["exact_match"] for r in no_proj) / len(no_proj) if no_proj else 0
        proj_rate = sum(r["exact_match"] for r in proj) / len(proj) if proj else 0
        lines.append(f"| {noise_level:.2f} | {no_proj_rate:.0%} | {proj_rate:.0%} |")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    print("Loading SONAR...")
    sonar = SonarWrapper()

    print("Running experiments...")
    results = run_experiment(sonar)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/noise_robustness_{timestamp}.json"
    md_path = f"results/noise_robustness_{timestamp}.md"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")

    write_markdown(results, md_path)
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
