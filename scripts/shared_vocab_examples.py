"""
Show examples of tokens in shared vocab vs not in shared vocab.
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


def bridge(emb):
    return (emb - src_mean) @ W + tgt_mean


def nearest_qwen(qwen_emb, k=3):
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
    s = qt.decode([tok_id])
    # Make whitespace visible
    return repr(s)[1:-1]  # Remove quotes from repr


# Build shared vocabulary lookup
print("Building shared vocabulary index...")
shared_vocab = {}  # sonar_str -> (sonar_id, qwen_id)
not_shared = []  # (sonar_id, sonar_str) - in SONAR but not Qwen

for sonar_id in range(min(100000, sonar_embeds.shape[0])):
    s = sonar_str(sonar_id)
    if not s or len(s) < 1:
        continue

    # Try to find exact match in Qwen
    qwen_toks = qt.encode(s, add_special_tokens=False)

    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        qwen_s = qt.decode([qwen_id])

        # Check if it's an exact match
        if qwen_s == s:
            shared_vocab[s] = (sonar_id, qwen_id)
        else:
            not_shared.append((sonar_id, s))
    else:
        not_shared.append((sonar_id, s))

print(f"Found {len(shared_vocab)} shared tokens")
print(f"Found {len(not_shared)} non-shared tokens")


# Show examples of SHARED vocabulary
print("\n" + "=" * 90)
print("SHARED VOCABULARY: Tokens that exist identically in both SONAR and Qwen")
print("=" * 90)

shared_items = list(shared_vocab.items())[:50]  # First 50

print(f"\n{'Token':<15} {'SONAR ID':<10} {'Qwen ID':<10} {'Maps to':<20} {'Correct?':<10} {'Sim':<8}")
print("-" * 90)

shared_correct = 0
shared_wrong = 0

for tok_str, (sonar_id, expected_qwen_id) in shared_items:
    sonar_emb = sonar_embeds[sonar_id]
    qwen_emb = bridge(sonar_emb)
    nearest_ids, nearest_sims = nearest_qwen(qwen_emb, k=1)
    actual_qwen_id = nearest_ids[0]
    actual_qwen_str = qwen_str(actual_qwen_id)
    sim = nearest_sims[0]

    correct = actual_qwen_id == expected_qwen_id
    if correct:
        shared_correct += 1
        symbol = "✓"
    else:
        shared_wrong += 1
        symbol = "✗"

    display_tok = repr(tok_str)[1:-1]
    print(f"{display_tok:<15} {sonar_id:<10} {expected_qwen_id:<10} {actual_qwen_str:<20} {symbol:<10} {sim:.3f}")

print(f"\nShared vocab accuracy: {shared_correct}/{shared_correct + shared_wrong} ({100*shared_correct/(shared_correct+shared_wrong):.1f}%)")


# Show examples of NOT SHARED vocabulary
print("\n" + "=" * 90)
print("NOT SHARED: Tokens in SONAR but not in Qwen (or different tokenization)")
print("=" * 90)

# Get diverse examples
not_shared_examples = []
for sonar_id, s in not_shared:
    if s and 2 <= len(s) <= 12 and s[0].isalpha():
        not_shared_examples.append((sonar_id, s))
    if len(not_shared_examples) >= 50:
        break

print(f"\n{'SONAR Token':<15} {'SONAR ID':<10} {'Maps to Qwen':<25} {'Sim':<8}")
print("-" * 90)

for sonar_id, tok_str in not_shared_examples:
    sonar_emb = sonar_embeds[sonar_id]
    qwen_emb = bridge(sonar_emb)
    nearest_ids, nearest_sims = nearest_qwen(qwen_emb, k=1)
    qwen_id = nearest_ids[0]
    qwen_s = qwen_str(qwen_id)
    sim = nearest_sims[0]

    display_tok = repr(tok_str)[1:-1]
    print(f"{display_tok:<15} {sonar_id:<10} {qwen_s:<25} {sim:.3f}")


# Analysis: what kinds of tokens are shared?
print("\n" + "=" * 90)
print("ANALYSIS: What kinds of tokens are in the shared vocabulary?")
print("=" * 90)

# Categorize shared tokens
categories = {
    'single_letters': [],
    'two_letter': [],
    'punctuation': [],
    'numbers': [],
    'subwords': [],
    'full_words': [],
}

for tok_str, (sonar_id, qwen_id) in shared_vocab.items():
    if len(tok_str) == 1:
        if tok_str.isalpha():
            categories['single_letters'].append(tok_str)
        elif tok_str.isdigit():
            categories['numbers'].append(tok_str)
        else:
            categories['punctuation'].append(tok_str)
    elif len(tok_str) == 2:
        categories['two_letter'].append(tok_str)
    elif tok_str.isalpha() and tok_str.islower() and len(tok_str) <= 4:
        categories['subwords'].append(tok_str)
    else:
        categories['full_words'].append(tok_str)

print(f"\nSingle letters: {len(categories['single_letters'])} - {categories['single_letters'][:20]}")
print(f"Two letter: {len(categories['two_letter'])} - {categories['two_letter'][:20]}")
print(f"Punctuation: {len(categories['punctuation'])} - {categories['punctuation'][:20]}")
print(f"Numbers: {len(categories['numbers'])} - {categories['numbers'][:10]}")
print(f"Short subwords: {len(categories['subwords'])} - {categories['subwords'][:20]}")
print(f"Full words: {len(categories['full_words'])} - {categories['full_words'][:20]}")
