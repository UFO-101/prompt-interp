"""
Token Mapping Experiments: Find the best SONAR → Qwen token mapping.

Goal: Map SONAR token embeddings to Qwen space such that nearest-neighbor
retrieval gives semantically correct tokens.

Approaches:
1. Baseline: Procrustes (MSE objective)
2. Linear + Cross-entropy (classification objective)
3. MLP + Cross-entropy (non-linear, with regularization)
4. Contrastive learning (efficient alternative to full softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
import time

dev = "cuda"
print("Loading models...", flush=True)

# Load SONAR
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sonar_embeds = sd.model.decoder.final_proj.weight.data.to(dev)
sonar_dec = sd.tokenizer.create_decoder()
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')

# Load Qwen
qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(dev).eval()
qwen_embeds = qm.model.embed_tokens.weight.data  # [151936, 896]

print(f"SONAR vocab: {sonar_embeds.shape}")
print(f"Qwen vocab: {qwen_embeds.shape}")

# ============================================================================
# Build training data: shared vocabulary pairs
# ============================================================================
print("\nBuilding shared vocabulary pairs...", flush=True)

def sonar_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return None

# Find tokens that exist in both vocabularies
pairs = []  # (sonar_id, qwen_id, token_str)

for sonar_id in range(min(150000, sonar_embeds.shape[0])):
    s = sonar_str(sonar_id)
    if not s or len(s) < 1:
        continue

    # Try to find exact match in Qwen
    qwen_toks = qt.encode(s, add_special_tokens=False)

    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        qwen_s = qt.decode([qwen_id])

        # Check for exact match
        if qwen_s == s:
            pairs.append((sonar_id, qwen_id, s))

print(f"Found {len(pairs)} shared vocabulary pairs")

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(pairs))
n_train = int(0.8 * len(pairs))
train_idx = indices[:n_train]
test_idx = indices[n_train:]

train_pairs = [pairs[i] for i in train_idx]
test_pairs = [pairs[i] for i in test_idx]

print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

# Create tensors
train_sonar_ids = torch.tensor([p[0] for p in train_pairs], device=dev)
train_qwen_ids = torch.tensor([p[1] for p in train_pairs], device=dev)
test_sonar_ids = torch.tensor([p[0] for p in test_pairs], device=dev)
test_qwen_ids = torch.tensor([p[1] for p in test_pairs], device=dev)

train_sonar_embs = sonar_embeds[train_sonar_ids]
train_qwen_embs = qwen_embeds[train_qwen_ids]
test_sonar_embs = sonar_embeds[test_sonar_ids]
test_qwen_embs = qwen_embeds[test_qwen_ids]

print(f"Train embeddings: {train_sonar_embs.shape} -> {train_qwen_embs.shape}")

# Normalize Qwen embeddings for efficient nearest neighbor
qwen_embeds_norm = F.normalize(qwen_embeds, dim=1)


def evaluate_mapping(map_fn, sonar_embs, qwen_ids, desc=""):
    """Evaluate a mapping function on nearest-neighbor accuracy."""
    with torch.no_grad():
        mapped = map_fn(sonar_embs)
        mapped_norm = F.normalize(mapped, dim=1)

        # Nearest neighbor in Qwen vocab
        sims = mapped_norm @ qwen_embeds_norm.T  # [N, 151936]
        pred_ids = sims.argmax(dim=1)

        correct = (pred_ids == qwen_ids).float().mean().item()

        # Also compute average similarity to correct token
        correct_sims = sims[torch.arange(len(qwen_ids), device=dev), qwen_ids]
        avg_sim = correct_sims.mean().item()

        # Top-5 accuracy
        top5 = sims.topk(5, dim=1).indices
        top5_correct = (top5 == qwen_ids.unsqueeze(1)).any(dim=1).float().mean().item()

    if desc:
        print(f"  {desc}: top1={100*correct:.1f}%, top5={100*top5_correct:.1f}%, avg_sim={avg_sim:.3f}")

    return correct, top5_correct, avg_sim


# ============================================================================
# Approach 1: Baseline Procrustes
# ============================================================================
print("\n" + "="*70)
print("APPROACH 1: Procrustes (baseline)")
print("="*70)

# Center the data
src_mean = train_sonar_embs.mean(dim=0)
tgt_mean = train_qwen_embs.mean(dim=0)

A = (train_sonar_embs - src_mean).cpu().numpy()
B = (train_qwen_embs - tgt_mean).cpu().numpy()

# Procrustes with dimension reduction
W_linear, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
A_proj = A @ W_linear

M = B.T @ A_proj
U, S, Vt = np.linalg.svd(M)
R = U @ Vt
scale = np.trace(B.T @ A_proj @ R) / np.trace(A_proj.T @ A_proj)
W_procrustes = torch.tensor(scale * W_linear @ R, dtype=torch.float32, device=dev)

def procrustes_map(x):
    return (x - src_mean) @ W_procrustes + tgt_mean

evaluate_mapping(procrustes_map, train_sonar_embs, train_qwen_ids, "Train")
evaluate_mapping(procrustes_map, test_sonar_embs, test_qwen_ids, "Test")


# ============================================================================
# Approach 2: Linear + Cross-Entropy
# ============================================================================
print("\n" + "="*70)
print("APPROACH 2: Linear + Cross-Entropy")
print("="*70)

class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)

    def forward(self, x):
        return self.linear(x)

linear_model = LinearMapper(1024, 896).to(dev)
optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3)

# Use smaller batches for memory efficiency
batch_size = 512
n_epochs = 20

# We'll use sampled softmax for efficiency (full vocab is 151k)
# Sample negative examples + the correct one
n_negatives = 1000

print(f"Training with sampled softmax ({n_negatives} negatives)...")
start_time = time.time()

for epoch in range(n_epochs):
    linear_model.train()
    total_loss = 0
    n_batches = 0

    perm = torch.randperm(len(train_sonar_embs), device=dev)

    for i in range(0, len(train_sonar_embs), batch_size):
        batch_idx = perm[i:i+batch_size]
        batch_sonar = train_sonar_embs[batch_idx]
        batch_qwen_ids = train_qwen_ids[batch_idx]

        optimizer.zero_grad()

        mapped = linear_model(batch_sonar)  # [B, 896]

        # Sample negatives
        neg_ids = torch.randint(0, qwen_embeds.shape[0], (n_negatives,), device=dev)

        # Combine with correct IDs
        all_ids = torch.cat([batch_qwen_ids, neg_ids])
        all_embeds = qwen_embeds[all_ids]  # [B + n_neg, 896]

        # Compute similarities
        mapped_norm = F.normalize(mapped, dim=1)
        embeds_norm = F.normalize(all_embeds, dim=1)

        # Each sample's logits: similarity to its correct embed + all negatives
        # The correct one is at position i for sample i (first B positions)
        logits = mapped_norm @ embeds_norm.T  # [B, B + n_neg]

        # Labels: position of correct embed for each sample
        labels = torch.arange(len(batch_idx), device=dev)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 5 == 0:
        linear_model.eval()
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")
        evaluate_mapping(lambda x: linear_model(x), test_sonar_embs, test_qwen_ids, "Test")

print(f"Training time: {time.time() - start_time:.1f}s")
linear_model.eval()
evaluate_mapping(lambda x: linear_model(x), train_sonar_embs, train_qwen_ids, "Train")
evaluate_mapping(lambda x: linear_model(x), test_sonar_embs, test_qwen_ids, "Test")


# ============================================================================
# Approach 3: MLP + Cross-Entropy
# ============================================================================
print("\n" + "="*70)
print("APPROACH 3: MLP + Cross-Entropy")
print("="*70)

class MLPMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(src_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tgt_dim)
        )

    def forward(self, x):
        return self.net(x)

mlp_model = MLPMapper(1024, 896, hidden_dim=512, dropout=0.1).to(dev)
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"Training MLP with sampled softmax...")
start_time = time.time()

best_test_acc = 0
best_mlp_state = None

for epoch in range(30):
    mlp_model.train()
    total_loss = 0
    n_batches = 0

    perm = torch.randperm(len(train_sonar_embs), device=dev)

    for i in range(0, len(train_sonar_embs), batch_size):
        batch_idx = perm[i:i+batch_size]
        batch_sonar = train_sonar_embs[batch_idx]
        batch_qwen_ids = train_qwen_ids[batch_idx]

        optimizer.zero_grad()

        mapped = mlp_model(batch_sonar)

        neg_ids = torch.randint(0, qwen_embeds.shape[0], (n_negatives,), device=dev)
        all_ids = torch.cat([batch_qwen_ids, neg_ids])
        all_embeds = qwen_embeds[all_ids]

        mapped_norm = F.normalize(mapped, dim=1)
        embeds_norm = F.normalize(all_embeds, dim=1)

        logits = mapped_norm @ embeds_norm.T
        labels = torch.arange(len(batch_idx), device=dev)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 5 == 0:
        mlp_model.eval()
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")
        test_acc, _, _ = evaluate_mapping(lambda x: mlp_model(x), test_sonar_embs, test_qwen_ids, "Test")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_mlp_state = {k: v.clone() for k, v in mlp_model.state_dict().items()}

print(f"Training time: {time.time() - start_time:.1f}s")
print(f"Best test accuracy: {100*best_test_acc:.1f}%")

# Load best model
mlp_model.load_state_dict(best_mlp_state)
mlp_model.eval()
evaluate_mapping(lambda x: mlp_model(x), train_sonar_embs, train_qwen_ids, "Train (best)")
evaluate_mapping(lambda x: mlp_model(x), test_sonar_embs, test_qwen_ids, "Test (best)")


# ============================================================================
# Approach 4: InfoNCE Contrastive Learning
# ============================================================================
print("\n" + "="*70)
print("APPROACH 4: InfoNCE Contrastive")
print("="*70)

class ContrastiveMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(src_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tgt_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, x):
        return self.net(x)

contrastive_model = ContrastiveMapper(1024, 896, hidden_dim=512).to(dev)
optimizer = torch.optim.Adam(contrastive_model.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"Training with InfoNCE loss...")
start_time = time.time()

best_test_acc = 0
best_contrastive_state = None

for epoch in range(30):
    contrastive_model.train()
    total_loss = 0
    n_batches = 0

    perm = torch.randperm(len(train_sonar_embs), device=dev)

    for i in range(0, len(train_sonar_embs), batch_size):
        batch_idx = perm[i:i+batch_size]
        batch_sonar = train_sonar_embs[batch_idx]
        batch_qwen_ids = train_qwen_ids[batch_idx]
        batch_qwen_embs = train_qwen_embs[batch_idx]

        optimizer.zero_grad()

        mapped = contrastive_model(batch_sonar)

        # InfoNCE: use in-batch negatives
        mapped_norm = F.normalize(mapped, dim=1)
        target_norm = F.normalize(batch_qwen_embs, dim=1)

        # Similarity matrix
        logits = (mapped_norm @ target_norm.T) / contrastive_model.temperature.exp()

        # Diagonal should be high (positive pairs)
        labels = torch.arange(len(batch_idx), device=dev)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 5 == 0:
        contrastive_model.eval()
        print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, temp={contrastive_model.temperature.exp().item():.3f}")
        test_acc, _, _ = evaluate_mapping(lambda x: contrastive_model(x), test_sonar_embs, test_qwen_ids, "Test")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_contrastive_state = {k: v.clone() for k, v in contrastive_model.state_dict().items()}

print(f"Training time: {time.time() - start_time:.1f}s")
print(f"Best test accuracy: {100*best_test_acc:.1f}%")

contrastive_model.load_state_dict(best_contrastive_state)
contrastive_model.eval()
evaluate_mapping(lambda x: contrastive_model(x), train_sonar_embs, train_qwen_ids, "Train (best)")
evaluate_mapping(lambda x: contrastive_model(x), test_sonar_embs, test_qwen_ids, "Test (best)")


# ============================================================================
# Summary and Sentence Evaluation
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nFinal test accuracies:")
print(f"  Procrustes:  ", end="")
evaluate_mapping(procrustes_map, test_sonar_embs, test_qwen_ids, "")
print(f"  Linear CE:   ", end="")
evaluate_mapping(lambda x: linear_model(x), test_sonar_embs, test_qwen_ids, "")
print(f"  MLP CE:      ", end="")
evaluate_mapping(lambda x: mlp_model(x), test_sonar_embs, test_qwen_ids, "")
print(f"  Contrastive: ", end="")
evaluate_mapping(lambda x: contrastive_model(x), test_sonar_embs, test_qwen_ids, "")


# ============================================================================
# Sentence-level evaluation
# ============================================================================
print("\n" + "="*70)
print("SENTENCE MAPPING TEST")
print("="*70)

def map_sentence(sentence, map_fn, name=""):
    """Map a sentence through SONAR -> mapper -> Qwen."""
    # SONAR tokenization
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]  # Skip BOS and lang tag

    # Map each token
    qwen_strs = []
    for tok in content_tokens:
        tok_id = tok.item()
        if tok_id == 3:  # EOS
            break

        sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
        mapped = map_fn(sonar_emb)
        mapped_norm = F.normalize(mapped, dim=1)

        sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
        qwen_id = sims.argmax().item()
        qwen_str = qt.decode([qwen_id])
        qwen_strs.append(qwen_str)

    result = ''.join(qwen_strs).strip()
    return result

test_sentences = [
    "Hello world.",
    "The cat sat on the mat.",
    "hot becomes cold",
    "What is the meaning of life?",
    "I love programming.",
    "The quick brown fox jumps over the lazy dog.",
]

print(f"\n{'Input':<45} | {'Procrustes':<30} | {'MLP':<30}")
print("-" * 110)

for sentence in test_sentences:
    with torch.no_grad():
        proc_out = map_sentence(sentence, procrustes_map)
        mlp_out = map_sentence(sentence, lambda x: mlp_model(x))

    # Truncate for display
    s_disp = sentence[:42] + "..." if len(sentence) > 45 else sentence
    p_disp = proc_out[:27] + "..." if len(proc_out) > 30 else proc_out
    m_disp = mlp_out[:27] + "..." if len(mlp_out) > 30 else mlp_out

    print(f"{s_disp:<45} | {p_disp:<30} | {m_disp:<30}")


# Save the best model
print("\n" + "="*70)
print("SAVING BEST MODEL")
print("="*70)

# Determine best model
models = {
    'linear': (linear_model, lambda x: linear_model(x)),
    'mlp': (mlp_model, lambda x: mlp_model(x)),
    'contrastive': (contrastive_model, lambda x: contrastive_model(x)),
}

best_name = None
best_acc = 0
for name, (model, fn) in models.items():
    acc, _, _ = evaluate_mapping(fn, test_sonar_embs, test_qwen_ids, "")
    if acc > best_acc:
        best_acc = acc
        best_name = name

print(f"Best model: {best_name} with {100*best_acc:.1f}% accuracy")
torch.save(models[best_name][0].state_dict(), f"results/best_token_mapper_{best_name}.pt")
print(f"Saved to results/best_token_mapper_{best_name}.pt")
