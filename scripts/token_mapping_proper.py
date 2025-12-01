"""
Proper token mapping: distinguish word-initial vs continuation tokens.

SONAR has two types of tokens for many subwords:
- Word-initial: appears at start of a word (after space)
- Continuation: appears mid-word

We should map:
- Word-initial SONAR → Qwen space-prefixed tokens
- Continuation SONAR → Qwen non-space tokens
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


# ============================================================================
# Build proper training data: distinguish word-initial vs continuation
# ============================================================================
print("\nClassifying SONAR tokens as word-initial vs continuation...", flush=True)

# For each string, find which token ID is word-initial (appears when string is alone)
word_initial_tokens = {}  # string -> token_id that appears when alone
continuation_tokens = {}  # string -> list of other token_ids

token_to_string = {}
string_to_tokens = {}

for tok_id in range(1, min(200000, sonar_embeds.shape[0])):
    s = sonar_str(tok_id)
    if not s or len(s) < 1:
        continue
    token_to_string[tok_id] = s
    if s not in string_to_tokens:
        string_to_tokens[s] = []
    string_to_tokens[s].append(tok_id)

# For strings with multiple tokens, determine which is word-initial
for s, tok_ids in string_to_tokens.items():
    if len(s) < 1:
        continue

    # The word-initial version is the one that appears when the string is alone
    try:
        alone_toks = sonar_enc(s)
        content = alone_toks[2:-1] if alone_toks[-1] == 3 else alone_toks[2:]

        if len(content) == 1:
            # Single token for this string when alone = word-initial
            word_initial_id = content[0].item()
            word_initial_tokens[s] = word_initial_id

            # Others are continuation
            for tid in tok_ids:
                if tid != word_initial_id:
                    if s not in continuation_tokens:
                        continuation_tokens[s] = []
                    continuation_tokens[s].append(tid)
    except:
        pass

print(f"Word-initial tokens: {len(word_initial_tokens)}")
print(f"Strings with continuation variants: {len(continuation_tokens)}")


# ============================================================================
# Build training pairs with proper space handling
# ============================================================================
print("\nBuilding proper training pairs...", flush=True)

pairs = []  # (sonar_id, qwen_id, is_word_initial)

for s, sonar_id in word_initial_tokens.items():
    # Word-initial SONAR token should map to space-prefixed Qwen token (for mid-sentence)
    s_space = " " + s
    qwen_toks_space = qt.encode(s_space, add_special_tokens=False)
    if len(qwen_toks_space) == 1:
        qwen_id = qwen_toks_space[0]
        if qt.decode([qwen_id]) == s_space:
            pairs.append((sonar_id, qwen_id, True))

    # Also allow exact match for sentence-initial position
    qwen_toks = qt.encode(s, add_special_tokens=False)
    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        if qt.decode([qwen_id]) == s:
            pairs.append((sonar_id, qwen_id, True))

for s, cont_ids in continuation_tokens.items():
    for sonar_id in cont_ids:
        # Continuation SONAR token should map to non-space Qwen token ONLY
        qwen_toks = qt.encode(s, add_special_tokens=False)
        if len(qwen_toks) == 1:
            qwen_id = qwen_toks[0]
            if qt.decode([qwen_id]) == s:
                pairs.append((sonar_id, qwen_id, False))
        # Do NOT add space-prefixed version for continuation tokens!

print(f"Total pairs: {len(pairs)}")
print(f"Word-initial pairs: {sum(1 for p in pairs if p[2])}")
print(f"Continuation pairs: {sum(1 for p in pairs if not p[2])}")

# Train/test split
np.random.seed(42)
unique_sonar_ids = list(set(p[0] for p in pairs))
np.random.shuffle(unique_sonar_ids)
n_train = int(0.8 * len(unique_sonar_ids))
train_sonar_set = set(unique_sonar_ids[:n_train])

train_pairs = [(s, q) for s, q, _ in pairs if s in train_sonar_set]
test_pairs = [(s, q) for s, q, _ in pairs if s not in train_sonar_set]

print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

train_sonar = torch.tensor([p[0] for p in train_pairs], device=dev)
train_qwen = torch.tensor([p[1] for p in train_pairs], device=dev)
test_sonar = torch.tensor([p[0] for p in test_pairs], device=dev)
test_qwen = torch.tensor([p[1] for p in test_pairs], device=dev)

train_sonar_embs = sonar_embeds[train_sonar]
train_qwen_embs = qwen_embeds[train_qwen]


# ============================================================================
# Training
# ============================================================================
print("\n" + "="*70)
print("TRAINING")
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
margin = 0.1

# Hard negatives
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

        pos_sim = (mapped_norm * target_norm).sum(dim=1)

        hard_negs = hard_neg_candidates[batch_idx, :20].to(dev)
        random_negs = torch.randint(0, qwen_embeds.shape[0], (len(batch_idx), 80), device=dev)
        neg_ids = torch.cat([hard_negs, random_negs], dim=1)
        neg_embs = qwen_embeds[neg_ids]
        neg_norm = F.normalize(neg_embs, dim=2)

        neg_sims = torch.bmm(neg_norm, mapped_norm.unsqueeze(2)).squeeze(2)
        max_neg_sim = neg_sims.max(dim=1).values

        triplet_loss = F.relu(max_neg_sim - pos_sim + margin).mean()

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
# Evaluate
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
    "running quickly",
    "unbelievable",
]

print(f"\n{'Input':<40} | {'Output':<40}")
print("-" * 85)

for sentence in test_sentences:
    output = map_sentence(sentence)
    print(f"{sentence:<40} | {output:<40}")

# Token-by-token
print("\n" + "="*70)
print("TOKEN-BY-TOKEN: 'The cat sat on the mat.'")
print("="*70)

sentence = "The cat sat on the mat."
sonar_tokens = sonar_enc(sentence)
content_tokens = sonar_tokens[2:]

print(f"\n{'SONAR':<12} {'Type':<12} {'->':^4} {'Qwen':<12}")
print("-" * 50)

with torch.no_grad():
    for tok in content_tokens:
        tok_id = tok.item()
        if tok_id == 3:
            break

        sonar_s = sonar_str(tok_id) or f"<{tok_id}>"

        # Determine if word-initial or continuation
        is_word_initial = tok_id in word_initial_tokens.values()
        tok_type = "word-init" if is_word_initial else "continuation"

        sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
        mapped = model(sonar_emb)
        mapped_norm = F.normalize(mapped, dim=1).cpu()

        sims = (mapped_norm @ qwen_embeds_norm_cpu.T).squeeze(0)
        qwen_id = sims.argmax().item()
        qwen_s = qt.decode([qwen_id])

        print(f"'{sonar_s:<10}' {tok_type:<12} {'->':^4} '{qwen_s}'")

torch.save(model.state_dict(), "results/token_mapper_proper.pt")
print(f"\nSaved to results/token_mapper_proper.pt")
