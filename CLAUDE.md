# CLAUDE.md - Project Context for Future Sessions

## Project: SONAR-LLM Embedding Optimization

### What is SONAR-LLM?
SONAR-LLM is a sentence-level language model that operates in SONAR embedding space (1024-dim). Given a sentence embedding z, it predicts the embedding of the next sentence. The model was trained on TinyStories.

### The Optimization Experiment

**Goal**: Find an embedding z such that SONAR-LLM(z) decodes to a specific target sentence.

**The pipeline**:
```
z (optimizable, 1024-dim)
    → SONAR-LLM (frozen)
    → pred_z (predicted next sentence embedding)
    → SONAR decoder (frozen)
    → predicted text
```

**Loss function**: Decoder cross-entropy loss between pred_z and target sentence tokens. We do NOT optimize z to decode to anything specific - we optimize so that the MODEL'S PREDICTION decodes correctly.

**What we're NOT doing**:
- NOT finding z that decodes to the target (that would be trivial - just encode the target)
- NOT using teacher forcing with multiple sentences
- NOT trying to predict whole stories autoregressively

**Scientific question**: When we optimize z to make SONAR-LLM predict a specific next sentence, does z converge to something semantically meaningful (like a plausible preceding sentence), or does it find adversarial embeddings?

**Current finding**: The optimized z decodes to garbage/adversarial text, but successfully causes the model to predict the exact target sentence. Different random initializations converge to different garbage z's but produce the same target prediction.

### Important: Editing Guidelines

**When asked to make a specific change, ONLY make that change.** Do not:
- Reorganize file structure
- Remove or move cells/code blocks
- "Clean up" or refactor unrelated code
- Change formatting or style of unchanged code

The user's file organization (e.g., `#%%` cell markers, code placement) is intentional.

**Do not make semantic changes to optimization/loss functions without explicit approval.** When diagnosing issues like OOM errors, explain the options but do not implement fixes (like adding `.detach()`) that change the optimization behavior without the user's go-ahead.
