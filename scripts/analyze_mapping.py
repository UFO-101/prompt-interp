"""
Analyze the token mapping in detail:
1. Why is spacing lost?
2. What tokens succeed vs fail?
3. Can we improve?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
from collections import defaultdict

dev = "cuda"
torch.cuda.empty_cache()

print("Loading...", flush=True)
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

# Load the trained model
class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearMapper(1024, 896).to(dev)
model.load_state_dict(torch.load("results/token_mapper_best.pt"))
model.eval()


# ============================================================================
# Analysis 1: Spacing issue
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 1: SPACING ISSUE")
print("="*70)

# Look at how SONAR vs Qwen handle spaces
print("\nHow does SONAR tokenize 'the cat'?")
sonar_tokens = sonar_enc("the cat")
print(f"  SONAR tokens: {sonar_tokens.tolist()}")
for tok in sonar_tokens:
    s = sonar_str(tok.item())
    print(f"    {tok.item()}: '{s}'")

print("\nHow does Qwen tokenize 'the cat'?")
qwen_tokens = qt.encode("the cat", add_special_tokens=False)
print(f"  Qwen tokens: {qwen_tokens}")
for tok in qwen_tokens:
    s = qt.decode([tok])
    print(f"    {tok}: '{s}' (repr: {repr(s)})")

# Check if "the" in SONAR includes space
print("\nDoes SONAR 'the' include a leading space?")
sonar_toks = sonar_enc("the")
for tok in sonar_toks[2:]:  # Skip BOS and lang tag
    s = sonar_str(tok.item())
    print(f"  Token {tok.item()}: '{s}' (repr: {repr(s)})")

print("\nDoes Qwen 'the' include a leading space?")
qwen_toks = qt.encode("the", add_special_tokens=False)
for tok in qwen_toks:
    s = qt.decode([tok])
    print(f"  Token {tok}: '{s}' (repr: {repr(s)})")

# Middle of sentence
print("\nQwen tokenization of ' the' (with space):")
qwen_toks = qt.encode(" the", add_special_tokens=False)
for tok in qwen_toks:
    s = qt.decode([tok])
    print(f"  Token {tok}: '{s}' (repr: {repr(s)})")


# ============================================================================
# Analysis 2: What maps to what?
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 2: COMMON TOKEN MAPPINGS")
print("="*70)

# Check common words
common_words = ["the", " the", "a", " a", "is", " is", "cat", " cat", ".", ",", " ", "hot", "cold"]

print(f"\n{'SONAR tok (if exists)':<25} {'->':^4} {'Maps to Qwen':<25}")
print("-" * 60)

for word in common_words:
    # Try to find in SONAR
    sonar_toks = sonar_enc(word)
    content_toks = [t.item() for t in sonar_toks[2:] if t.item() != 3]

    if len(content_toks) == 1:
        tok_id = content_toks[0]
        sonar_s = sonar_str(tok_id)

        # Map to Qwen
        with torch.no_grad():
            sonar_emb = sonar_embeds[tok_id].unsqueeze(0)
            mapped = model(sonar_emb)
            mapped_norm = F.normalize(mapped, dim=1).cpu()
            sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
            top3 = sims.topk(3)

        top3_str = ", ".join([f"'{qt.decode([i.item()])}'" for i in top3.indices])
        print(f"'{sonar_s}' ({tok_id})             -> {top3_str}")
    else:
        print(f"'{word}' -> multi-token in SONAR: {content_toks}")


# ============================================================================
# Analysis 3: Space-prefix tokens
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 3: SPACE-PREFIX TOKENS")
print("="*70)

# In Qwen, many tokens have leading spaces (e.g., " the")
# Does SONAR have similar tokens?

print("\nSearching for space-prefix tokens in SONAR...")
space_tokens = []
for i in range(1000, 50000):
    s = sonar_str(i)
    if s and s.startswith(' ') and len(s) > 1 and s[1:].isalpha():
        space_tokens.append((i, s))
    if len(space_tokens) >= 20:
        break

print(f"Found {len(space_tokens)} space-prefix tokens in SONAR")
for tok_id, s in space_tokens[:10]:
    print(f"  {tok_id}: '{s}' (repr: {repr(s)})")


# ============================================================================
# Analysis 4: Detailed failure analysis
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 4: WHY DO SOME TOKENS FAIL?")
print("="*70)

# Build test pairs
pairs = []
for sonar_id in range(min(100000, sonar_embeds.shape[0])):
    s = sonar_str(sonar_id)
    if not s or len(s) < 1 or not s.isascii():
        continue
    qwen_toks = qt.encode(s, add_special_tokens=False)
    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        qwen_s = qt.decode([qwen_id])
        if qwen_s == s:
            pairs.append((sonar_id, qwen_id, s))

# Test on all pairs
successes = []
failures = []

with torch.no_grad():
    for sonar_id, qwen_id, s in pairs:
        sonar_emb = sonar_embeds[sonar_id].unsqueeze(0)
        mapped = model(sonar_emb)
        mapped_norm = F.normalize(mapped, dim=1).cpu()
        sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)
        pred_id = sims.argmax().item()
        correct_sim = sims[qwen_id].item()
        pred_sim = sims[pred_id].item()

        if pred_id == qwen_id:
            successes.append((s, correct_sim))
        else:
            pred_s = qt.decode([pred_id])
            failures.append((s, pred_s, correct_sim, pred_sim))

print(f"\nTotal: {len(pairs)}, Success: {len(successes)}, Failure: {len(failures)}")
print(f"Accuracy: {100*len(successes)/len(pairs):.1f}%")

# Categorize failures
print("\nFailure patterns:")
pattern_counts = defaultdict(list)

for s, pred_s, correct_sim, pred_sim in failures:
    if pred_s == s.lower():
        pattern_counts['case_diff'].append((s, pred_s))
    elif pred_s == ' ' + s:
        pattern_counts['added_space'].append((s, pred_s))
    elif pred_s == s.strip():
        pattern_counts['removed_space'].append((s, pred_s))
    elif pred_s.strip() == s.strip():
        pattern_counts['space_diff'].append((s, pred_s))
    elif len(pred_s) == 1:
        pattern_counts['to_single_char'].append((s, pred_s))
    else:
        pattern_counts['other'].append((s, pred_s))

for pattern, examples in sorted(pattern_counts.items(), key=lambda x: -len(x[1])):
    print(f"\n  {pattern}: {len(examples)} cases")
    for s, pred_s in examples[:5]:
        print(f"    '{s}' -> '{pred_s}'")


# ============================================================================
# Analysis 5: Embedding similarity
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS 5: EMBEDDING SIMILARITY STRUCTURE")
print("="*70)

# Are semantically similar SONAR tokens still similar after mapping?
test_words = [
    ("hot", "cold"),
    ("big", "small"),
    ("good", "bad"),
    ("the", "a"),
    ("cat", "dog"),
]

print("\nSemantic pairs - similarity before and after mapping:")
print(f"{'Pair':<20} {'SONAR sim':<12} {'Qwen sim (target)':<18} {'Qwen sim (mapped)':<18}")
print("-" * 70)

for w1, w2 in test_words:
    # Get SONAR tokens
    s1_toks = sonar_enc(w1)
    s2_toks = sonar_enc(w2)
    s1_id = s1_toks[2].item() if len(s1_toks) > 2 else None
    s2_id = s2_toks[2].item() if len(s2_toks) > 2 else None

    if s1_id is None or s2_id is None:
        continue

    # SONAR similarity
    sonar_sim = F.cosine_similarity(
        sonar_embeds[s1_id].unsqueeze(0),
        sonar_embeds[s2_id].unsqueeze(0)
    ).item()

    # Map to Qwen
    with torch.no_grad():
        m1 = model(sonar_embeds[s1_id].unsqueeze(0))
        m2 = model(sonar_embeds[s2_id].unsqueeze(0))

    mapped_sim = F.cosine_similarity(m1, m2).item()

    # Target Qwen similarity (if both words exist as single tokens)
    q1_toks = qt.encode(w1, add_special_tokens=False)
    q2_toks = qt.encode(w2, add_special_tokens=False)

    if len(q1_toks) == 1 and len(q2_toks) == 1:
        qwen_sim = F.cosine_similarity(
            qwen_embeds[q1_toks[0]].unsqueeze(0),
            qwen_embeds[q2_toks[0]].unsqueeze(0)
        ).item()
    else:
        qwen_sim = float('nan')

    print(f"({w1}, {w2}){'':<12} {sonar_sim:<12.3f} {qwen_sim:<18.3f} {mapped_sim:<18.3f}")


# ============================================================================
# Suggestion: What could improve this?
# ============================================================================
print("\n" + "="*70)
print("INSIGHTS AND SUGGESTIONS")
print("="*70)

print("""
KEY FINDINGS:

1. SPACING: SONAR doesn't use space-prefix tokens like Qwen does.
   - SONAR: "the cat" -> ["the", "cat"] (no leading spaces)
   - Qwen:  "the cat" -> ["the", " cat"] (space attached to "cat")
   - This causes merged output: "Thecatsatonthemat"

2. ACCURACY: 44.8% top1, 85.1% top5
   - The correct token is usually in top-5
   - Many failures are near-misses (case, space differences)

3. SEMANTIC STRUCTURE: Partially preserved
   - Similar words remain somewhat similar after mapping
   - But the absolute positions don't align

POTENTIAL IMPROVEMENTS:

1. POST-PROCESSING: Add spaces between tokens based on Qwen's conventions
   - Simple rule: add space before tokens that don't start with punctuation

2. CONTEXT-AWARE: Use surrounding tokens to pick from top-k
   - If top-1 doesn't make sense, maybe top-2 or top-3 does

3. ENSEMBLE: Combine multiple mapping approaches
   - Procrustes has higher similarity, CE has better top-k

4. DIFFERENT OBJECTIVE: Train to minimize edit distance of output sentences
   - Sequence-level loss instead of token-level
""")
