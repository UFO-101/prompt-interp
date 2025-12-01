"""
Better token mapping - handle SONAR's EOS token issue.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

dev = "cuda"
print("Loading...", flush=True)
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(dev).eval()

a = np.load("results/nllb_qwen_alignment.npz")
W = torch.tensor(a['W'], dtype=torch.float32, device=dev)
src_mean = torch.tensor(a['src_mean'], dtype=torch.float32, device=dev)
tgt_mean = torch.tensor(a['tgt_mean'], dtype=torch.float32, device=dev)

sonar_embeds = sd.model.decoder.final_proj.weight.data.to(dev)
qwen_embeds = qm.model.embed_tokens.weight.data
sonar_dec = sd.tokenizer.create_decoder()
print("Ready.\n", flush=True)


def bridge(sonar_emb):
    return (sonar_emb - src_mean) @ W + tgt_mean


def find_nearest_qwen(qwen_emb, k=5):
    qwen_emb_norm = F.normalize(qwen_emb.unsqueeze(0), dim=1)
    qwen_embeds_norm = F.normalize(qwen_embeds, dim=1)
    sims = (qwen_emb_norm @ qwen_embeds_norm.T).squeeze(0)
    topk = torch.topk(sims, k)
    return topk.indices.tolist(), topk.values.tolist()


def sonar_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return f"<{tok_id}>"


def qwen_str(tok_id):
    return qt.decode([tok_id])


print("=" * 90)
print("SONAR TOKEN ID -> EMBEDDING -> BRIDGE -> NEAREST QWEN TOKENS")
print("=" * 90)

# Sample SONAR tokens directly by ID (skip the encoding issue)
# Look for tokens that decode to recognizable strings
interesting_tokens = []

print("\nScanning SONAR vocab for recognizable English tokens...")
for i in range(1000, 150000, 100):  # Sample every 100th token
    s = sonar_str(i)
    # Look for ASCII strings that look like words/subwords
    if s and 2 <= len(s) <= 10 and s.replace(' ', '').isalpha() and s.isascii():
        interesting_tokens.append((i, s))

print(f"Found {len(interesting_tokens)} interesting tokens\n")

print(f"{'SONAR ID':<10} {'SONAR':<15} -> {'Qwen Top-5 (cosine similarity)':<60}")
print("-" * 90)

for tok_id, tok_s in interesting_tokens[:50]:
    sonar_emb = sonar_embeds[tok_id]
    qwen_emb = bridge(sonar_emb)
    nearest_ids, nearest_sims = find_nearest_qwen(qwen_emb, k=5)

    nearest_info = []
    for qid, sim in zip(nearest_ids, nearest_sims):
        qs = qwen_str(qid).replace('\n', '\\n')
        nearest_info.append(f"'{qs}'({sim:.2f})")

    print(f"{tok_id:<10} '{tok_s:<13}' -> {', '.join(nearest_info)}")


# Now let's look at the SEMANTIC quality
print("\n" + "=" * 90)
print("SEMANTIC ANALYSIS: Do similar SONAR tokens map to similar Qwen tokens?")
print("=" * 90)

# Find tokens for related concepts
concept_groups = [
    ("numbers", [str(i) for i in range(10)]),
    ("colors", ["red", "blue", "green", "black", "white"]),
    ("animals", ["cat", "dog", "bird", "fish"]),
    ("actions", ["run", "walk", "eat", "sleep"]),
]

# Since individual words are multi-token, let's look at subword patterns
# Find SONAR tokens that START with certain letters
print("\nSONAR tokens starting with 'un' (negation prefix):")
un_tokens = [(i, sonar_str(i)) for i in range(1000, 100000)
             if sonar_str(i).lower().startswith('un') and len(sonar_str(i)) < 10][:10]

for tok_id, tok_s in un_tokens:
    sonar_emb = sonar_embeds[tok_id]
    qwen_emb = bridge(sonar_emb)
    nearest_ids, nearest_sims = find_nearest_qwen(qwen_emb, k=3)
    nearest = [(qwen_str(qid), sim) for qid, sim in zip(nearest_ids, nearest_sims)]
    print(f"  SONAR '{tok_s}' -> Qwen: {nearest}")


print("\nSONAR tokens starting with 'pre' (prefix):")
pre_tokens = [(i, sonar_str(i)) for i in range(1000, 100000)
              if sonar_str(i).lower().startswith('pre') and len(sonar_str(i)) < 10][:10]

for tok_id, tok_s in pre_tokens:
    sonar_emb = sonar_embeds[tok_id]
    qwen_emb = bridge(sonar_emb)
    nearest_ids, nearest_sims = find_nearest_qwen(qwen_emb, k=3)
    nearest = [(qwen_str(qid), sim) for qid, sim in zip(nearest_ids, nearest_sims)]
    print(f"  SONAR '{tok_s}' -> Qwen: {nearest}")


# Check if the alignment preserves ANY semantic structure
print("\n" + "=" * 90)
print("DOES THE BRIDGE PRESERVE RELATIVE DISTANCES?")
print("=" * 90)

# Pick some token pairs that should be similar in SONAR
# and check if they're also similar after bridging
test_pairs = []

# Find some tokens
for i in range(5000, 50000, 1000):
    s = sonar_str(i)
    if s and s.isascii() and len(s) > 2:
        test_pairs.append((i, s))
    if len(test_pairs) >= 20:
        break

print(f"\nComparing {len(test_pairs)} tokens pairwise...")
print("If bridge preserves structure, similar SONAR tokens should map to similar Qwen regions\n")

# Compute pairwise similarities in SONAR space and bridged Qwen space
sonar_embs = torch.stack([sonar_embeds[i] for i, _ in test_pairs])
qwen_embs = torch.stack([bridge(sonar_embeds[i]) for i, _ in test_pairs])

sonar_sims = F.cosine_similarity(sonar_embs.unsqueeze(1), sonar_embs.unsqueeze(0), dim=2)
qwen_sims = F.cosine_similarity(qwen_embs.unsqueeze(1), qwen_embs.unsqueeze(0), dim=2)

# Get upper triangle (excluding diagonal)
mask = torch.triu(torch.ones_like(sonar_sims), diagonal=1).bool()
sonar_flat = sonar_sims[mask]
qwen_flat = qwen_sims[mask]

correlation = torch.corrcoef(torch.stack([sonar_flat, qwen_flat]))[0, 1].item()
print(f"Correlation between SONAR pairwise sims and bridged Qwen pairwise sims: {correlation:.3f}")

print("\nSample pairs:")
print(f"{'Token A':<15} {'Token B':<15} {'SONAR sim':<12} {'Qwen sim':<12}")
print("-" * 60)
for idx, ((i, s1), (j, s2)) in enumerate([(test_pairs[a], test_pairs[b])
                                           for a in range(len(test_pairs))
                                           for b in range(a+1, len(test_pairs))][:10]):
    a_idx = [x[0] for x in test_pairs].index(i)
    b_idx = [x[0] for x in test_pairs].index(j)
    print(f"'{s1:<13}' '{s2:<13}' {sonar_sims[a_idx, b_idx]:.3f}        {qwen_sims[a_idx, b_idx]:.3f}")
