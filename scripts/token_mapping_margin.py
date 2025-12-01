"""
Token mapping with margin-based loss to improve discrimination.

The issue: punctuation tokens (",", " ", ".") are close to many content words.
Solution: Use triplet/margin loss to push negatives further away.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

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
qwen_embeds_norm = F.normalize(qwen_embeds, dim=1)

def sonar_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return None

# Build training pairs (same as before)
print("\nBuilding training pairs...", flush=True)
pairs = []
for sonar_id in range(min(200000, sonar_embeds.shape[0])):
    s = sonar_str(sonar_id)
    if not s or len(s) < 1:
        continue

    # Exact match
    qwen_toks = qt.encode(s, add_special_tokens=False)
    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        if qt.decode([qwen_id]) == s:
            pairs.append((sonar_id, qwen_id))

    # Space-prefixed version
    s_space = " " + s
    qwen_toks_space = qt.encode(s_space, add_special_tokens=False)
    if len(qwen_toks_space) == 1:
        qwen_id_space = qwen_toks_space[0]
        if qt.decode([qwen_id_space]) == s_space:
            pairs.append((sonar_id, qwen_id_space))

print(f"Total pairs: {len(pairs)}")

# Split
np.random.seed(42)
unique_sonar_ids = list(set(p[0] for p in pairs))
np.random.shuffle(unique_sonar_ids)
n_train = int(0.8 * len(unique_sonar_ids))
train_sonar_set = set(unique_sonar_ids[:n_train])

train_pairs = [p for p in pairs if p[0] in train_sonar_set]
test_pairs = [p for p in pairs if p[0] not in train_sonar_set]

print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

train_sonar = torch.tensor([p[0] for p in train_pairs], device=dev)
train_qwen = torch.tensor([p[1] for p in train_pairs], device=dev)
test_sonar = torch.tensor([p[0] for p in test_pairs], device=dev)
test_qwen = torch.tensor([p[1] for p in test_pairs], device=dev)

train_sonar_embs = sonar_embeds[train_sonar]
train_qwen_embs = qwen_embeds[train_qwen]


# ============================================================================
# Training with combined loss: cross-entropy + triplet margin
# ============================================================================
print("\n" + "="*70)
print("TRAINING WITH MARGIN LOSS")
print("="*70)

class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearMapper(1024, 896).to(dev)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

batch_size = 256
n_epochs = 100
margin = 0.1  # Minimum margin between positive and negative

# Hard negatives (computed in batches on CPU to save memory)
print("Precomputing hard negatives...")
hard_neg_candidates = []
with torch.no_grad():
    train_qwen_norm_cpu = F.normalize(train_qwen_embs, dim=1).cpu()
    qwen_embeds_norm_cpu = qwen_embeds_norm.cpu()

    for i in range(0, len(train_qwen_norm_cpu), 1000):
        batch = train_qwen_norm_cpu[i:i+1000]
        sims = batch @ qwen_embeds_norm_cpu.T
        topk = sims.topk(50, dim=1).indices[:, 1:]
        hard_neg_candidates.append(topk)

    hard_neg_candidates = torch.cat(hard_neg_candidates, dim=0)

def evaluate(model):
    model.eval()
    correct = 0
    n = len(test_sonar)
    qwen_embeds_norm_cpu = qwen_embeds_norm.cpu()

    with torch.no_grad():
        for i in range(0, n, 1000):
            batch_sonar = sonar_embeds[test_sonar[i:i+1000]]
            batch_qwen = test_qwen[i:i+1000]

            mapped = model(batch_sonar)
            mapped_norm = F.normalize(mapped, dim=1).cpu()
            sims = mapped_norm @ qwen_embeds_norm_cpu.T
            preds = sims.argmax(dim=1)
            correct += (preds == batch_qwen.cpu()).sum().item()

    return correct / n

best_acc = 0
best_state = None

print("Training...")
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    n_batches = 0

    perm = torch.randperm(len(train_sonar_embs), device=dev)

    for i in range(0, len(train_sonar_embs), batch_size):
        batch_idx = perm[i:i+batch_size].cpu()
        batch_sonar = train_sonar_embs[batch_idx.to(dev)]
        batch_qwen_ids = train_qwen[batch_idx.to(dev)]
        batch_qwen_embs = train_qwen_embs[batch_idx.to(dev)]

        optimizer.zero_grad()

        mapped = model(batch_sonar)
        mapped_norm = F.normalize(mapped, dim=1)
        target_norm = F.normalize(batch_qwen_embs, dim=1)

        # Positive similarity
        pos_sim = (mapped_norm * target_norm).sum(dim=1)

        # Hard negatives
        hard_negs = hard_neg_candidates[batch_idx, :20].to(dev)
        random_negs = torch.randint(0, qwen_embeds.shape[0], (len(batch_idx), 80), device=dev)
        neg_ids = torch.cat([hard_negs, random_negs], dim=1)
        neg_embs = qwen_embeds[neg_ids]
        neg_norm = F.normalize(neg_embs, dim=2)

        # Negative similarities
        neg_sims = torch.bmm(neg_norm, mapped_norm.unsqueeze(2)).squeeze(2)
        max_neg_sim = neg_sims.max(dim=1).values

        # Triplet margin loss: want pos_sim > max_neg_sim + margin
        triplet_loss = F.relu(max_neg_sim - pos_sim + margin).mean()

        # Also use cross-entropy for ranking
        all_sims = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        ce_loss = F.cross_entropy(all_sims * 10, torch.zeros(len(batch_idx), dtype=torch.long, device=dev))

        loss = ce_loss + triplet_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 20 == 0:
        acc = evaluate(model)
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, test_acc={100*acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

print(f"\nBest test accuracy: {100*best_acc:.1f}%")
model.load_state_dict(best_state)


# ============================================================================
# Evaluate on sentences
# ============================================================================
print("\n" + "="*70)
print("SENTENCE MAPPING")
print("="*70)

qwen_embeds_norm_cpu = qwen_embeds_norm.cpu()

def map_sentence(sentence):
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]

    output_strs = []
    with torch.no_grad():
        for tok in content_tokens:
            tok_id = tok.item()
            if tok_id == 3:
                break

            sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
            mapped = model(sonar_emb)
            mapped_norm = F.normalize(mapped, dim=1).cpu()

            sims = (mapped_norm @ qwen_embeds_norm_cpu.T).squeeze(0)
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
    "Find the antonym.",
    "A is B",
]

print(f"\n{'Input':<40} | {'Output':<40}")
print("-" * 85)

for sentence in test_sentences:
    output = map_sentence(sentence)
    print(f"{sentence:<40} | {output:<40}")


# Token-by-token for problematic sentence
print("\n" + "="*70)
print("TOKEN-BY-TOKEN: 'The cat sat on the mat.'")
print("="*70)

sentence = "The cat sat on the mat."
sonar_tokens = sonar_enc(sentence)
content_tokens = sonar_tokens[2:]

print(f"\n{'SONAR':<12} {'->':^4} {'Qwen':<12} {'Top-3':<50}")
print("-" * 80)

with torch.no_grad():
    for tok in content_tokens:
        tok_id = tok.item()
        if tok_id == 3:
            break

        sonar_s = sonar_str(tok_id) or f"<{tok_id}>"
        sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
        mapped = model(sonar_emb)
        mapped_norm = F.normalize(mapped, dim=1).cpu()

        sims = (mapped_norm @ qwen_embeds_norm_cpu.T).squeeze(0)
        top3 = sims.topk(3)

        qwen_id = top3.indices[0].item()
        qwen_s = qt.decode([qwen_id])
        top3_strs = [f"'{qt.decode([i.item()])}' ({s:.2f})" for i, s in zip(top3.indices, top3.values)]

        print(f"'{sonar_s:<10}' {'->':^4} '{qwen_s:<10}' {', '.join(top3_strs)}")

# Save
torch.save(model.state_dict(), "results/token_mapper_margin.pt")
print(f"\nSaved to results/token_mapper_margin.pt")
