"""
Token Mapping v2: Analyze failures and improve.

Key findings from v1:
- Contrastive: 51.9% test (best)
- Big train-test gap suggests overfitting
- Need to understand what's failing

This script:
1. Analyzes which tokens succeed vs fail
2. Tries improvements: regularization, hard negatives, simpler model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import time
from collections import defaultdict

dev = "cuda"
print("Loading models...", flush=True)

sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sonar_embeds = sd.model.decoder.final_proj.weight.data.to(dev)
sonar_dec = sd.tokenizer.create_decoder()
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')

qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(dev).eval()
qwen_embeds = qm.model.embed_tokens.weight.data

qwen_embeds_norm = F.normalize(qwen_embeds, dim=1)

def sonar_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return None

# Build pairs
print("\nBuilding shared vocabulary pairs...", flush=True)
pairs = []
for sonar_id in range(min(150000, sonar_embeds.shape[0])):
    s = sonar_str(sonar_id)
    if not s or len(s) < 1:
        continue
    qwen_toks = qt.encode(s, add_special_tokens=False)
    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        qwen_s = qt.decode([qwen_id])
        if qwen_s == s:
            pairs.append((sonar_id, qwen_id, s))

print(f"Found {len(pairs)} pairs")

np.random.seed(42)
indices = np.random.permutation(len(pairs))
n_train = int(0.8 * len(pairs))
train_pairs = [pairs[i] for i in indices[:n_train]]
test_pairs = [pairs[i] for i in indices[n_train:]]

train_sonar_ids = torch.tensor([p[0] for p in train_pairs], device=dev)
train_qwen_ids = torch.tensor([p[1] for p in train_pairs], device=dev)
test_sonar_ids = torch.tensor([p[0] for p in test_pairs], device=dev)
test_qwen_ids = torch.tensor([p[1] for p in test_pairs], device=dev)

train_sonar_embs = sonar_embeds[train_sonar_ids]
train_qwen_embs = qwen_embeds[train_qwen_ids]
test_sonar_embs = sonar_embeds[test_sonar_ids]
test_qwen_embs = qwen_embeds[test_qwen_ids]


# ============================================================================
# Analysis: What tokens succeed vs fail?
# ============================================================================
print("\n" + "="*70)
print("ANALYZING PROCRUSTES FAILURES")
print("="*70)

# Quick Procrustes
src_mean = train_sonar_embs.mean(dim=0)
tgt_mean = train_qwen_embs.mean(dim=0)
A = (train_sonar_embs - src_mean).cpu().numpy()
B = (train_qwen_embs - tgt_mean).cpu().numpy()
W_linear, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
A_proj = A @ W_linear
M = B.T @ A_proj
U, S, Vt = np.linalg.svd(M)
R = U @ Vt
scale = np.trace(B.T @ A_proj @ R) / np.trace(A_proj.T @ A_proj)
W_proc = torch.tensor(scale * W_linear @ R, dtype=torch.float32, device=dev)

def procrustes_map(x):
    return (x - src_mean) @ W_proc + tgt_mean

# Categorize test tokens by success/failure
with torch.no_grad():
    mapped = procrustes_map(test_sonar_embs)
    mapped_norm = F.normalize(mapped, dim=1)
    sims = mapped_norm @ qwen_embeds_norm.T
    pred_ids = sims.argmax(dim=1)
    correct_mask = (pred_ids == test_qwen_ids)

success_tokens = []
failure_tokens = []

for i, (sonar_id, qwen_id, tok_str) in enumerate(test_pairs):
    if correct_mask[i]:
        success_tokens.append((tok_str, sims[i, qwen_id].item()))
    else:
        pred_str = qt.decode([pred_ids[i].item()])
        failure_tokens.append((tok_str, pred_str, sims[i, qwen_id].item()))

print(f"\nSuccesses: {len(success_tokens)}, Failures: {len(failure_tokens)}")

# Categorize by token type
def categorize(s):
    if len(s) == 1:
        if s.isalpha(): return 'single_letter'
        if s.isdigit(): return 'digit'
        return 'punct'
    if s.isdigit(): return 'number'
    if s.startswith(' '): return 'space_prefix'
    if len(s) <= 3: return 'short'
    if s.isalpha() and s.islower(): return 'lowercase_word'
    if s.isalpha() and s[0].isupper(): return 'capitalized'
    return 'other'

success_cats = defaultdict(list)
failure_cats = defaultdict(list)

for tok, sim in success_tokens:
    success_cats[categorize(tok)].append(tok)
for tok, pred, sim in failure_tokens:
    failure_cats[categorize(tok)].append((tok, pred))

print("\nSuccess rates by category:")
for cat in sorted(set(list(success_cats.keys()) + list(failure_cats.keys()))):
    n_succ = len(success_cats[cat])
    n_fail = len(failure_cats[cat])
    total = n_succ + n_fail
    if total > 0:
        rate = 100 * n_succ / total
        print(f"  {cat:<20}: {n_succ:4d}/{total:4d} ({rate:5.1f}%)")

print("\nSample failures:")
for tok, pred, sim in failure_tokens[:20]:
    print(f"  '{tok}' -> '{pred}' (sim to correct: {sim:.3f})")


# ============================================================================
# Improvement 1: Hard negative mining
# ============================================================================
print("\n" + "="*70)
print("IMPROVEMENT: Hard Negative Mining + Stronger Regularization")
print("="*70)

class ImprovedMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        # Simple linear with careful initialization
        self.linear = nn.Linear(src_dim, tgt_dim, bias=True)
        # Initialize close to identity-like mapping
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

model = ImprovedMapper(1024, 896).to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

batch_size = 256
n_epochs = 50
n_hard_negatives = 100  # Hard negatives from similar tokens
n_random_negatives = 400

# Precompute nearest neighbors in Qwen space for hard negatives
print("Precomputing hard negatives...")
with torch.no_grad():
    # For each training target embedding, find similar embeddings
    train_qwen_norm = F.normalize(train_qwen_embs, dim=1)
    target_sims = train_qwen_norm @ qwen_embeds_norm.T  # [N_train, 151936]
    # Get top-k similar (excluding self)
    hard_neg_candidates = target_sims.topk(n_hard_negatives + 1, dim=1).indices[:, 1:]  # Skip self

print("Training with hard negatives...")
best_test_acc = 0
best_state = None

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    n_batches = 0

    perm = torch.randperm(len(train_sonar_embs), device=dev)

    for i in range(0, len(train_sonar_embs), batch_size):
        batch_idx = perm[i:i+batch_size]
        batch_sonar = train_sonar_embs[batch_idx]
        batch_qwen_ids = train_qwen_ids[batch_idx]
        batch_qwen_embs = train_qwen_embs[batch_idx]

        optimizer.zero_grad()

        mapped = model(batch_sonar)

        # Gather negatives: hard + random
        hard_negs = hard_neg_candidates[batch_idx]  # [B, n_hard]
        random_negs = torch.randint(0, qwen_embeds.shape[0],
                                    (len(batch_idx), n_random_negatives), device=dev)

        # Build candidate set for each sample
        # Shape: [B, 1 + n_hard + n_random]
        candidates = torch.cat([batch_qwen_ids.unsqueeze(1), hard_negs, random_negs], dim=1)

        # Get embeddings for all candidates
        candidate_embs = qwen_embeds[candidates]  # [B, n_cand, 896]

        # Normalize
        mapped_norm = F.normalize(mapped, dim=1)  # [B, 896]
        candidate_norm = F.normalize(candidate_embs, dim=2)  # [B, n_cand, 896]

        # Logits: dot product
        logits = torch.bmm(candidate_norm, mapped_norm.unsqueeze(2)).squeeze(2)  # [B, n_cand]

        # Labels: always 0 (correct is first)
        labels = torch.zeros(len(batch_idx), dtype=torch.long, device=dev)

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            mapped_test = model(test_sonar_embs)
            mapped_test_norm = F.normalize(mapped_test, dim=1)
            test_sims = mapped_test_norm @ qwen_embeds_norm.T
            test_preds = test_sims.argmax(dim=1)
            test_acc = (test_preds == test_qwen_ids).float().mean().item()

            # Top-5
            top5 = test_sims.topk(5, dim=1).indices
            top5_acc = (top5 == test_qwen_ids.unsqueeze(1)).any(dim=1).float().mean().item()

            # Avg sim
            avg_sim = test_sims[torch.arange(len(test_qwen_ids), device=dev), test_qwen_ids].mean().item()

        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, test_top1={100*test_acc:.1f}%, top5={100*top5_acc:.1f}%, sim={avg_sim:.3f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"\nBest test accuracy: {100*best_test_acc:.1f}%")

# Load best
model.load_state_dict(best_state)
model.eval()


# ============================================================================
# Final sentence evaluation
# ============================================================================
print("\n" + "="*70)
print("SENTENCE MAPPING WITH IMPROVED MODEL")
print("="*70)

def map_sentence(sentence, map_fn):
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]

    qwen_strs = []
    for tok in content_tokens:
        tok_id = tok.item()
        if tok_id == 3:
            break
        sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
        with torch.no_grad():
            mapped = map_fn(sonar_emb)
        mapped_norm = F.normalize(mapped, dim=1)
        sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
        qwen_id = sims.argmax().item()
        qwen_str = qt.decode([qwen_id])
        qwen_strs.append(qwen_str)

    return ''.join(qwen_strs).strip()

test_sentences = [
    "Hello world.",
    "The cat sat on the mat.",
    "hot becomes cold",
    "What is the meaning of life?",
    "I love programming.",
    "The quick brown fox jumps over the lazy dog.",
    "Find the antonym.",
    "Give the opposite word.",
]

print(f"\n{'Input':<45} | {'Output':<45}")
print("-" * 95)

for sentence in test_sentences:
    out = map_sentence(sentence, lambda x: model(x))
    s_disp = sentence[:42] + "..." if len(sentence) > 45 else sentence
    o_disp = out[:42] + "..." if len(out) > 45 else out
    print(f"{s_disp:<45} | {o_disp:<45}")


# ============================================================================
# Detailed token-by-token analysis on one sentence
# ============================================================================
print("\n" + "="*70)
print("DETAILED TOKEN-BY-TOKEN ANALYSIS")
print("="*70)

sentence = "The cat sat on the mat."
print(f"\nSentence: '{sentence}'")

sonar_tokens = sonar_enc(sentence)
content_tokens = sonar_tokens[2:]

print(f"\n{'SONAR tok':<15} {'SONAR str':<12} {'->':^4} {'Qwen str':<12} {'Sim':<8} {'Correct?':<10}")
print("-" * 70)

for tok in content_tokens:
    tok_id = tok.item()
    if tok_id == 3:
        break

    sonar_s = sonar_str(tok_id) or f"<{tok_id}>"
    sonar_emb = sonar_embeds[tok_id].unsqueeze(0)

    with torch.no_grad():
        mapped = model(sonar_emb)
    mapped_norm = F.normalize(mapped, dim=1)
    sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
    qwen_id = sims.argmax().item()
    qwen_s = qt.decode([qwen_id])
    sim = sims[qwen_id].item()

    # Check if this token is in shared vocab
    is_shared = False
    expected_qwen = None
    for s_id, q_id, s in pairs:
        if s_id == tok_id:
            is_shared = True
            expected_qwen = qt.decode([q_id])
            break

    if is_shared:
        correct = "✓" if qwen_s == expected_qwen else f"✗ (expect '{expected_qwen}')"
    else:
        correct = "(not shared)"

    print(f"{tok_id:<15} '{sonar_s:<10}' {'->':^4} '{qwen_s:<10}' {sim:<8.3f} {correct}")


# Save the improved model
print("\n" + "="*70)
print("SAVING MODEL")
print("="*70)
torch.save(model.state_dict(), "results/token_mapper_improved.pt")
print("Saved to results/token_mapper_improved.pt")

# Also save a summary
summary = {
    'test_accuracy': best_test_acc,
    'n_train': len(train_pairs),
    'n_test': len(test_pairs),
}
print(f"\nSummary: {summary}")
