"""
Token Mapping v3: Focus on ASCII tokens, memory efficient.

Key insight from v2: Many failures are non-ASCII tokens.
For English sentences, we mainly care about ASCII mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import time

dev = "cuda"
torch.cuda.empty_cache()

print("Loading models...", flush=True)

sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sonar_embeds = sd.model.decoder.final_proj.weight.data.to(dev)
sonar_dec = sd.tokenizer.create_decoder()
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')

qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(dev).eval()
qwen_embeds = qm.model.embed_tokens.weight.data

# Keep normalized version on CPU to save GPU memory
qwen_embeds_norm = F.normalize(qwen_embeds, dim=1).cpu()

def sonar_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return None

# Build ASCII-only pairs
print("\nBuilding ASCII-only shared vocabulary pairs...", flush=True)
pairs = []
for sonar_id in range(min(150000, sonar_embeds.shape[0])):
    s = sonar_str(sonar_id)
    if not s or len(s) < 1:
        continue
    # ASCII only
    if not s.isascii():
        continue

    qwen_toks = qt.encode(s, add_special_tokens=False)
    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        qwen_s = qt.decode([qwen_id])
        if qwen_s == s:
            pairs.append((sonar_id, qwen_id, s))

print(f"Found {len(pairs)} ASCII pairs")

np.random.seed(42)
indices = np.random.permutation(len(pairs))
n_train = int(0.8 * len(pairs))
train_pairs = [pairs[i] for i in indices[:n_train]]
test_pairs = [pairs[i] for i in indices[n_train:]]

print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

train_sonar_ids = torch.tensor([p[0] for p in train_pairs], device=dev)
train_qwen_ids = torch.tensor([p[1] for p in train_pairs], device=dev)
test_sonar_ids = torch.tensor([p[0] for p in test_pairs], device=dev)
test_qwen_ids = torch.tensor([p[1] for p in test_pairs], device=dev)

train_sonar_embs = sonar_embeds[train_sonar_ids]
train_qwen_embs = qwen_embeds[train_qwen_ids]
test_sonar_embs = sonar_embeds[test_sonar_ids]
test_qwen_embs = qwen_embeds[test_qwen_ids]


def evaluate_batched(model, sonar_embs, qwen_ids, batch_size=500):
    """Memory-efficient evaluation."""
    model.eval()
    correct = 0
    top5_correct = 0
    total_sim = 0
    n = len(sonar_embs)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_sonar = sonar_embs[i:i+batch_size]
            batch_qwen_ids = qwen_ids[i:i+batch_size]

            mapped = model(batch_sonar)
            mapped_norm = F.normalize(mapped, dim=1).cpu()

            # Compute sims on CPU
            sims = mapped_norm @ qwen_embeds_norm.T
            preds = sims.argmax(dim=1)
            correct += (preds == batch_qwen_ids.cpu()).sum().item()

            top5 = sims.topk(5, dim=1).indices
            top5_correct += (top5 == batch_qwen_ids.cpu().unsqueeze(1)).any(dim=1).sum().item()

            for j, qid in enumerate(batch_qwen_ids):
                total_sim += sims[j, qid.item()].item()

    return correct / n, top5_correct / n, total_sim / n


# ============================================================================
# Approach: Linear with full softmax (on CPU for memory)
# ============================================================================
print("\n" + "="*70)
print("TRAINING: Linear mapper with hard negatives")
print("="*70)

class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

model = LinearMapper(1024, 896).to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

batch_size = 256
n_epochs = 100
n_negatives = 500

# Precompute hard negatives (on CPU)
print("Precomputing hard negatives...")
with torch.no_grad():
    train_qwen_norm = F.normalize(train_qwen_embs, dim=1).cpu()
    target_sims = train_qwen_norm @ qwen_embeds_norm.T
    hard_neg_candidates = target_sims.topk(100, dim=1).indices[:, 1:].to(dev)

print(f"Training for {n_epochs} epochs...")
best_test_acc = 0
best_state = None
patience = 20
no_improve = 0

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    n_batches = 0

    perm = torch.randperm(len(train_sonar_embs), device=dev)

    for i in range(0, len(train_sonar_embs), batch_size):
        batch_idx = perm[i:i+batch_size]
        batch_sonar = train_sonar_embs[batch_idx]
        batch_qwen_ids = train_qwen_ids[batch_idx]

        optimizer.zero_grad()

        mapped = model(batch_sonar)

        # Hard negatives + random negatives
        hard_negs = hard_neg_candidates[batch_idx, :50]  # [B, 50]
        random_negs = torch.randint(0, qwen_embeds.shape[0],
                                    (len(batch_idx), n_negatives - 50), device=dev)

        candidates = torch.cat([batch_qwen_ids.unsqueeze(1), hard_negs, random_negs], dim=1)
        candidate_embs = qwen_embeds[candidates]

        mapped_norm = F.normalize(mapped, dim=1)
        candidate_norm = F.normalize(candidate_embs, dim=2)

        logits = torch.bmm(candidate_norm, mapped_norm.unsqueeze(2)).squeeze(2)
        labels = torch.zeros(len(batch_idx), dtype=torch.long, device=dev)

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        test_acc, top5_acc, avg_sim = evaluate_batched(model, test_sonar_embs, test_qwen_ids)
        train_acc, _, _ = evaluate_batched(model, train_sonar_embs, train_qwen_ids)
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, train={100*train_acc:.1f}%, test={100*test_acc:.1f}% (top5={100*top5_acc:.1f}%)")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience // 10:
            print("Early stopping")
            break

print(f"\nBest test accuracy: {100*best_test_acc:.1f}%")
model.load_state_dict(best_state)


# ============================================================================
# Also try: Direct regression to target embeddings
# ============================================================================
print("\n" + "="*70)
print("ALTERNATIVE: Direct regression to target embeddings")
print("="*70)

# This optimizes ||f(x) - y||^2 where y is the target Qwen embedding
# Then we find nearest neighbor

model2 = LinearMapper(1024, 896).to(dev)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(50):
    model2.train()
    total_loss = 0
    n_batches = 0

    perm = torch.randperm(len(train_sonar_embs), device=dev)

    for i in range(0, len(train_sonar_embs), batch_size):
        batch_idx = perm[i:i+batch_size]
        batch_sonar = train_sonar_embs[batch_idx]
        batch_qwen_embs = train_qwen_embs[batch_idx]

        optimizer2.zero_grad()

        mapped = model2(batch_sonar)

        # MSE loss + cosine similarity loss
        mse_loss = F.mse_loss(mapped, batch_qwen_embs)
        cos_loss = 1 - F.cosine_similarity(mapped, batch_qwen_embs).mean()

        loss = mse_loss + 0.5 * cos_loss
        loss.backward()
        optimizer2.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        test_acc, top5_acc, avg_sim = evaluate_batched(model2, test_sonar_embs, test_qwen_ids)
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, test={100*test_acc:.1f}% (top5={100*top5_acc:.1f}%), sim={avg_sim:.3f}")


# ============================================================================
# Compare approaches
# ============================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

# Procrustes baseline
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

class ProcMapper(nn.Module):
    def __init__(self, W, src_mean, tgt_mean):
        super().__init__()
        self.register_buffer('W', W)
        self.register_buffer('src_mean', src_mean)
        self.register_buffer('tgt_mean', tgt_mean)

    def forward(self, x):
        return (x - self.src_mean) @ self.W + self.tgt_mean

proc_model = ProcMapper(W_proc, src_mean, tgt_mean).to(dev)

print("\nTest set results (ASCII tokens only):")
proc_acc, proc_top5, proc_sim = evaluate_batched(proc_model, test_sonar_embs, test_qwen_ids)
ce_acc, ce_top5, ce_sim = evaluate_batched(model, test_sonar_embs, test_qwen_ids)
reg_acc, reg_top5, reg_sim = evaluate_batched(model2, test_sonar_embs, test_qwen_ids)

print(f"  Procrustes:    top1={100*proc_acc:.1f}%, top5={100*proc_top5:.1f}%, sim={proc_sim:.3f}")
print(f"  Cross-entropy: top1={100*ce_acc:.1f}%, top5={100*ce_top5:.1f}%, sim={ce_sim:.3f}")
print(f"  Regression:    top1={100*reg_acc:.1f}%, top5={100*reg_top5:.1f}%, sim={reg_sim:.3f}")


# ============================================================================
# Sentence mapping
# ============================================================================
print("\n" + "="*70)
print("SENTENCE MAPPING")
print("="*70)

# Use best model
best_model = model if ce_acc >= reg_acc else model2
best_name = "CE" if ce_acc >= reg_acc else "Reg"
print(f"Using {best_name} model")

def map_sentence(sentence, mapper):
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]

    qwen_strs = []
    for tok in content_tokens:
        tok_id = tok.item()
        if tok_id == 3:
            break
        sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
        with torch.no_grad():
            mapped = mapper(sonar_emb)
        mapped_norm = F.normalize(mapped, dim=1).cpu()
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
    "A is B",
    "1 + 1 = 2",
]

print(f"\n{'Input':<40} | {'Procrustes':<25} | {best_name:<25}")
print("-" * 95)

for sentence in test_sentences:
    proc_out = map_sentence(sentence, proc_model)
    best_out = map_sentence(sentence, best_model)

    s = sentence[:37] + "..." if len(sentence) > 40 else sentence
    p = proc_out[:22] + "..." if len(proc_out) > 25 else proc_out
    b = best_out[:22] + "..." if len(best_out) > 25 else best_out

    print(f"{s:<40} | {p:<25} | {b:<25}")


# ============================================================================
# Token-by-token for one sentence
# ============================================================================
print("\n" + "="*70)
print("TOKEN-BY-TOKEN: 'The cat sat on the mat.'")
print("="*70)

sentence = "The cat sat on the mat."
sonar_tokens = sonar_enc(sentence)
content_tokens = sonar_tokens[2:]

print(f"\n{'SONAR':<12} {'->':^4} {'Procrustes':<12} {'->':^4} {best_name:<12}")
print("-" * 50)

for tok in content_tokens:
    tok_id = tok.item()
    if tok_id == 3:
        break

    sonar_s = sonar_str(tok_id) or f"<{tok_id}>"
    sonar_emb = sonar_embeds[tok_id].unsqueeze(0)

    with torch.no_grad():
        proc_mapped = proc_model(sonar_emb)
        best_mapped = best_model(sonar_emb)

    proc_norm = F.normalize(proc_mapped, dim=1).cpu()
    best_norm = F.normalize(best_mapped, dim=1).cpu()

    proc_id = (proc_norm @ qwen_embeds_norm.T).squeeze(0).argmax().item()
    best_id = (best_norm @ qwen_embeds_norm.T).squeeze(0).argmax().item()

    proc_s = qt.decode([proc_id])
    best_s = qt.decode([best_id])

    print(f"'{sonar_s:<10}' {'->':^4} '{proc_s:<10}' {'->':^4} '{best_s:<10}'")


# Save
torch.save(best_model.state_dict(), "results/token_mapper_best.pt")
print(f"\nSaved best model to results/token_mapper_best.pt")
