"""
Map SONAR tokens through the bridge to Qwen and see what tokens they become.

SONAR token -> SONAR embedding -> bridge -> Qwen space -> nearest Qwen token
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

# Load alignment
a = np.load("results/nllb_qwen_alignment.npz")
W = torch.tensor(a['W'], dtype=torch.float32, device=dev)
src_mean = torch.tensor(a['src_mean'], dtype=torch.float32, device=dev)
tgt_mean = torch.tensor(a['tgt_mean'], dtype=torch.float32, device=dev)

# Get embedding matrices
sonar_embeds = sd.model.decoder.final_proj.weight.data.to(dev)  # [256206, 1024]
qwen_embeds = qm.model.embed_tokens.weight.data  # [151936, 896]

sonar_dec = sd.tokenizer.create_decoder()
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')

print(f"SONAR vocab size: {sonar_embeds.shape[0]}")
print(f"Qwen vocab size: {qwen_embeds.shape[0]}")
print("Ready.\n", flush=True)


def bridge(sonar_emb):
    """Map SONAR embedding to Qwen space."""
    return (sonar_emb - src_mean) @ W + tgt_mean


def find_nearest_qwen_tokens(qwen_emb, k=5):
    """Find k nearest Qwen tokens to an embedding."""
    # Normalize for cosine similarity
    qwen_emb_norm = F.normalize(qwen_emb.unsqueeze(0), dim=1)
    qwen_embeds_norm = F.normalize(qwen_embeds, dim=1)

    # Cosine similarity
    sims = (qwen_emb_norm @ qwen_embeds_norm.T).squeeze(0)

    # Top k
    topk = torch.topk(sims, k)
    return topk.indices.tolist(), topk.values.tolist()


def sonar_token_to_str(tok_id):
    """Convert SONAR token ID to string."""
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return f"<{tok_id}>"


def qwen_token_to_str(tok_id):
    """Convert Qwen token ID to string."""
    return qt.decode([tok_id])


# Test with specific SONAR tokens
print("=" * 80)
print("SONAR TOKEN -> BRIDGE -> QWEN TOKEN MAPPING")
print("=" * 80)

# Find some interesting SONAR tokens by encoding words
test_words = [
    "hello", "world", "the", "is", "hot", "cold", "big", "small",
    "good", "bad", "happy", "sad", "fast", "slow", "light", "dark",
    "cat", "dog", "run", "walk", "red", "blue", "one", "two"
]

print("\nMapping individual words (if single token in SONAR):\n")
print(f"{'SONAR Token':<20} {'SONAR Str':<15}    {'Top 5 Qwen Tokens (cosine sim)':<50}")
print("-" * 100)

for word in test_words:
    # Encode with SONAR
    sonar_toks = sonar_enc(word)
    # Skip BOS and lang tag
    content_toks = sonar_toks[2:]

    if len(content_toks) != 1:
        # Multi-token, show the pieces
        pieces = [sonar_token_to_str(t.item()) for t in content_toks]
        print(f"'{word}' -> multi-token: {pieces}")
        continue

    tok_id = content_toks[0].item()
    tok_str = sonar_token_to_str(tok_id)

    # Get SONAR embedding
    sonar_emb = sonar_embeds[tok_id]

    # Bridge to Qwen space
    qwen_emb = bridge(sonar_emb)

    # Find nearest Qwen tokens
    nearest_ids, nearest_sims = find_nearest_qwen_tokens(qwen_emb, k=5)
    nearest_strs = [qwen_token_to_str(i) for i in nearest_ids]

    # Format output
    nearest_info = ", ".join([f"'{s}'({sim:.2f})" for s, sim in zip(nearest_strs, nearest_sims)])
    print(f"{tok_id:<20} '{tok_str:<15}' -> {nearest_info}")


# Now let's look at some random SONAR tokens
print("\n" + "=" * 80)
print("RANDOM SONAR TOKENS -> QWEN")
print("=" * 80)

# Sample some tokens from different ranges
sample_ids = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]

print(f"\n{'SONAR ID':<10} {'SONAR Str':<20} -> {'Top 5 Qwen Tokens':<60}")
print("-" * 100)

for tok_id in sample_ids:
    tok_str = sonar_token_to_str(tok_id)
    sonar_emb = sonar_embeds[tok_id]
    qwen_emb = bridge(sonar_emb)
    nearest_ids, nearest_sims = find_nearest_qwen_tokens(qwen_emb, k=5)
    nearest_strs = [qwen_token_to_str(i) for i in nearest_ids]
    nearest_info = ", ".join([f"'{s}'({sim:.2f})" for s, sim in zip(nearest_strs, nearest_sims)])
    print(f"{tok_id:<10} '{tok_str:<20}' -> {nearest_info}")


# Check the alignment quality on shared vocabulary
print("\n" + "=" * 80)
print("ALIGNMENT QUALITY: Tokens that exist in both vocabularies")
print("=" * 80)

# Find tokens that are identical strings in both vocabularies
shared_found = 0
good_alignments = []
bad_alignments = []

# Check common English words/subwords
check_tokens = []
for i in range(min(50000, sonar_embeds.shape[0])):
    try:
        s = sonar_token_to_str(i)
        if s and len(s) > 1 and s.isascii() and not s.startswith('<'):
            check_tokens.append((i, s))
    except:
        pass

print(f"\nChecking {len(check_tokens)} SONAR tokens for matches in Qwen vocab...")

for sonar_id, sonar_str in check_tokens[:500]:  # Check first 500
    # Try to find this exact string in Qwen vocab
    qwen_toks = qt.encode(sonar_str, add_special_tokens=False)

    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        qwen_str = qwen_token_to_str(qwen_id)

        if qwen_str.strip() == sonar_str.strip():
            # Found a shared token!
            shared_found += 1

            # Check if alignment is good
            sonar_emb = sonar_embeds[sonar_id]
            qwen_emb = bridge(sonar_emb)
            nearest_ids, nearest_sims = find_nearest_qwen_tokens(qwen_emb, k=1)

            if nearest_ids[0] == qwen_id:
                good_alignments.append((sonar_str, nearest_sims[0]))
            else:
                nearest_str = qwen_token_to_str(nearest_ids[0])
                bad_alignments.append((sonar_str, qwen_str, nearest_str, nearest_sims[0]))

print(f"\nFound {shared_found} shared tokens")
print(f"Good alignments (maps to correct token): {len(good_alignments)}")
print(f"Bad alignments (maps to wrong token): {len(bad_alignments)}")

if good_alignments:
    print(f"\nExamples of GOOD alignments:")
    for s, sim in good_alignments[:10]:
        print(f"  '{s}' -> '{s}' (sim={sim:.3f})")

if bad_alignments:
    print(f"\nExamples of BAD alignments:")
    for sonar_s, qwen_s, nearest_s, sim in bad_alignments[:10]:
        print(f"  '{sonar_s}' should -> '{qwen_s}', but maps to '{nearest_s}' (sim={sim:.3f})")
