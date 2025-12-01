"""
Clean token mapping: Learn a linear map between SONAR and Qwen embedding spaces.

Goals:
- Use ALL available training data (not just ASCII)
- Map tokens to BOTH space-prefixed and non-space-prefixed Qwen versions
- No hardcoding or post-processing hacks
- Evaluate the raw vector space mapping quality
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
qwen_embeds_norm = F.normalize(qwen_embeds, dim=1).cpu()

def sonar_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return None


# ============================================================================
# Build training data: include BOTH space and non-space variants
# ============================================================================
print("\nBuilding training pairs (including space variants)...", flush=True)

pairs = []  # (sonar_id, qwen_id)

for sonar_id in range(min(200000, sonar_embeds.shape[0])):
    s = sonar_str(sonar_id)
    if not s or len(s) < 1:
        continue

    # Try exact match
    qwen_toks = qt.encode(s, add_special_tokens=False)
    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        qwen_s = qt.decode([qwen_id])
        if qwen_s == s:
            pairs.append((sonar_id, qwen_id, s))

    # Also try space-prefixed version: SONAR "cat" -> Qwen " cat"
    s_with_space = " " + s
    qwen_toks_space = qt.encode(s_with_space, add_special_tokens=False)
    if len(qwen_toks_space) == 1:
        qwen_id_space = qwen_toks_space[0]
        qwen_s_space = qt.decode([qwen_id_space])
        if qwen_s_space == s_with_space:
            pairs.append((sonar_id, qwen_id_space, s_with_space))

print(f"Total pairs: {len(pairs)}")

# Count unique SONAR tokens
unique_sonar = len(set(p[0] for p in pairs))
print(f"Unique SONAR tokens: {unique_sonar}")
print(f"Average Qwen targets per SONAR token: {len(pairs)/unique_sonar:.2f}")

# Train/test split (by SONAR token to avoid leakage)
np.random.seed(42)
unique_sonar_ids = list(set(p[0] for p in pairs))
np.random.shuffle(unique_sonar_ids)
n_train = int(0.8 * len(unique_sonar_ids))
train_sonar_ids = set(unique_sonar_ids[:n_train])
test_sonar_ids = set(unique_sonar_ids[n_train:])

train_pairs = [p for p in pairs if p[0] in train_sonar_ids]
test_pairs = [p for p in pairs if p[0] in test_sonar_ids]

print(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

# Create tensors
train_sonar = torch.tensor([p[0] for p in train_pairs], device=dev)
train_qwen = torch.tensor([p[1] for p in train_pairs], device=dev)
test_sonar = torch.tensor([p[0] for p in test_pairs], device=dev)
test_qwen = torch.tensor([p[1] for p in test_pairs], device=dev)

train_sonar_embs = sonar_embeds[train_sonar]
train_qwen_embs = qwen_embeds[train_qwen]
test_sonar_embs = sonar_embeds[test_sonar]
test_qwen_embs = qwen_embeds[test_qwen]


# ============================================================================
# Training: Linear mapping with cross-entropy
# ============================================================================
print("\n" + "="*70)
print("TRAINING LINEAR MAPPING")
print("="*70)

class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearMapper(1024, 896).to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

batch_size = 512
n_epochs = 50
n_negatives = 500

# Precompute hard negatives
print("Precomputing hard negatives...")
with torch.no_grad():
    train_qwen_norm = F.normalize(train_qwen_embs, dim=1).cpu()
    target_sims = train_qwen_norm @ qwen_embeds_norm.T
    hard_neg_candidates = target_sims.topk(100, dim=1).indices[:, 1:].to(dev)

print(f"Training for {n_epochs} epochs...")

def evaluate(model, sonar_embs, qwen_ids, batch_size=1000):
    """Evaluate top-1 and top-5 accuracy."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    n = len(sonar_embs)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_sonar = sonar_embs[i:i+batch_size]
            batch_qwen = qwen_ids[i:i+batch_size]

            mapped = model(batch_sonar)
            mapped_norm = F.normalize(mapped, dim=1).cpu()

            sims = mapped_norm @ qwen_embeds_norm.T
            top1 = sims.argmax(dim=1)
            top5 = sims.topk(5, dim=1).indices

            correct_top1 += (top1 == batch_qwen.cpu()).sum().item()
            correct_top5 += (top5 == batch_qwen.cpu().unsqueeze(1)).any(dim=1).sum().item()

    return correct_top1 / n, correct_top5 / n

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
        batch_qwen_ids = train_qwen[batch_idx]

        optimizer.zero_grad()

        mapped = model(batch_sonar)

        # Hard negatives + random negatives
        hard_negs = hard_neg_candidates[batch_idx, :50]
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
        train_top1, train_top5 = evaluate(model, train_sonar_embs, train_qwen)
        test_top1, test_top5 = evaluate(model, test_sonar_embs, test_qwen)

        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")
        print(f"  Train: top1={100*train_top1:.1f}%, top5={100*train_top5:.1f}%")
        print(f"  Test:  top1={100*test_top1:.1f}%, top5={100*test_top5:.1f}%")

        if test_top1 > best_test_acc:
            best_test_acc = test_top1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"\nBest test top-1 accuracy: {100*best_test_acc:.1f}%")
model.load_state_dict(best_state)


# ============================================================================
# Evaluate on sentences (no post-processing)
# ============================================================================
print("\n" + "="*70)
print("SENTENCE MAPPING (raw, no post-processing)")
print("="*70)

def map_sentence_raw(sentence):
    """Map sentence with NO post-processing - just raw nearest neighbor."""
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]  # Skip BOS and lang tag

    output_strs = []

    with torch.no_grad():
        for tok in content_tokens:
            tok_id = tok.item()
            if tok_id == 3:  # EOS
                break

            sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
            mapped = model(sonar_emb)
            mapped_norm = F.normalize(mapped, dim=1).cpu()

            sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
            qwen_id = sims.argmax().item()
            qwen_s = qt.decode([qwen_id])

            output_strs.append(qwen_s)

    return ''.join(output_strs)

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
    "Machine learning is fascinating.",
]

print(f"\n{'Input':<45} | {'Output':<45}")
print("-" * 95)

for sentence in test_sentences:
    output = map_sentence_raw(sentence)
    s = sentence[:42] + "..." if len(sentence) > 45 else sentence
    o = output[:42] + "..." if len(output) > 45 else output
    print(f"{s:<45} | {o:<45}")


# ============================================================================
# Detailed token-by-token
# ============================================================================
print("\n" + "="*70)
print("TOKEN-BY-TOKEN ANALYSIS")
print("="*70)

for sentence in ["The cat sat on the mat.", "hot becomes cold"]:
    print(f"\n'{sentence}'")
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]

    print(f"  {'SONAR':<12} {'->':^4} {'Qwen':<15} {'Top-3 candidates':<40}")
    print(f"  {'-'*75}")

    with torch.no_grad():
        for tok in content_tokens:
            tok_id = tok.item()
            if tok_id == 3:
                break

            sonar_s = sonar_str(tok_id) or f"<{tok_id}>"
            sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
            mapped = model(sonar_emb)
            mapped_norm = F.normalize(mapped, dim=1).cpu()

            sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
            top3 = sims.topk(3)

            qwen_id = top3.indices[0].item()
            qwen_s = qt.decode([qwen_id])

            top3_strs = [f"'{qt.decode([i.item()])}'" for i in top3.indices]

            print(f"  '{sonar_s:<10}' {'->':^4} '{qwen_s:<13}' {', '.join(top3_strs)}")


# ============================================================================
# Save
# ============================================================================
print("\n" + "="*70)
print("SAVING")
print("="*70)

torch.save({
    'model_state': model.state_dict(),
    'train_pairs': len(train_pairs),
    'test_pairs': len(test_pairs),
    'best_test_acc': best_test_acc,
}, "results/token_mapper_clean.pt")

print(f"Saved to results/token_mapper_clean.pt")
print(f"Best test accuracy: {100*best_test_acc:.1f}%")
