# SONAR Continuous Prompt Optimization - Experiment Summary

## Task: Antonyms
- Input: "hot ->" → Target: "cold"
- 6 examples: hot/cold, big/small, fast/slow, up/down, happy/sad, light/dark

## Baseline
- **0/6 accuracy** - Without any prefix, Qwen outputs "10000" for all inputs

## Best Result
- **2/6 accuracy (33%)** at step 29
- Seed: "Antonyms: words with opposite meanings."
- lr=0.0005, grad_clip=0.1
- Correct predictions: big→small, fast→slow
- Decoded text stayed coherent: "Antonyms: words with opposite meanings."

## Key Findings

### 1. Optimization Dynamics
- Loss decreases steadily from ~14 to ~7 over 30 steps
- Accuracy improves: 0/6 → 1/6 (step 20) → 2/6 (step 29)
- Embedding changes but decoded text stays the same for ~30 steps
- After step 30-31, text suddenly degenerates ("The words are 'Two'...")

### 2. Best Hyperparameters
- Learning rate: 0.0005 (too low = no learning, too high = instability)
- Gradient clipping: 0.1
- Optimal steps: ~25-30 (before text degenerates)

### 3. SONAR Decoder Robustness
- Small perturbations in embedding space don't change decoded text
- There's a critical threshold after which decoding jumps to different output
- This creates a "plateau" in text space while loss can still decrease

### 4. Prediction Quality
At step 29:
- ✓ big → small (correct)
- ✓ fast → slow (correct)
- ✗ hot → 低温cold (contains "cold" but with Chinese prefix)
- ✗ light → heavy (semantically correct antonym, but not "dark")
- ✗ happy → unfavorably (wrong)
- ✗ up → (empty)

## Conclusions
1. SONAR continuous optimization CAN improve task performance (0% → 33%)
2. The optimization finds valid regions where decoded text stays coherent
3. There's a trade-off: more optimization steps = lower loss but risks text degeneration
4. Early stopping around step 25-30 preserves coherent text while capturing improvements

## Next Steps
- Try more diverse seeds
- Explore regularization to stay in valid SONAR space longer
- Consider ensemble of multiple optimized embeddings
- Test on other tasks
