# SONAR Prompt Optimization

Experiments in learning interpretable text prompts via optimization in SONAR embedding space.

## Approach

Two-stage optimization system:
1. **Stage 1**: Optimize a z embedding vector that generates prompt tokens via SONAR decoder
2. **Stage 2**: Use the generated prompt with z=0 (unconditioned) to solve tasks

Key technique: Straight-through gradient estimation via embedding geometry.

## Key Finding

Using PPL regularization (weight=0.1) provides stability without dominating the task loss. The optimization achieves 67% accuracy on an antonym completion task, with specific examples (`hot -> cold`, `happy -> sad`) consistently failing.

## Usage

```bash
uv run python scripts/optimize_prompt.py
```

## Structure

```
scripts/
  optimize_prompt.py    # Main optimization script
src/prompt_interp/      # Package stub
papers/                 # Reference papers (SONAR, EPO, ContextBench)
```

## Requirements

- Python 3.12+
- CUDA GPU
- SONAR (`sonar-space`)
- PyTorch

Install dependencies:
```bash
uv sync
```


Ideas to try:
 - [done] Add perplexity term.
 - [done] Each iteration, update z to be the re-encoding.
 - PCA with different learning rates.


 - Test if jailbreaks transfer to normal TinyStories models