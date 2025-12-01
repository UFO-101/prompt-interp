"""
Restricted vocabulary SONAR decoding.

Test SONAR decode with vocabulary restricted to tokens that exist in both
SONAR and Qwen vocabularies. This should produce outputs that can be
tokenized 1:1 between the two systems.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout

dev = "cuda"
torch.cuda.empty_cache()

print("Loading models...", flush=True)
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

sdm = sd.model.to(dev)
sonar_dec = sd.tokenizer.create_decoder()
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')

# Get SONAR vocab size from final projection
sonar_vocab_size = sdm.decoder.final_proj.weight.shape[0]
print(f"SONAR vocab size: {sonar_vocab_size}")
print(f"Qwen vocab size: {qt.vocab_size}")

# ============================================================================
# Build overlapping vocabulary
# ============================================================================
print("\nBuilding overlapping vocabulary...", flush=True)

def sonar_str(tok_id):
    """Decode a single SONAR token to string."""
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return None

# Find all SONAR tokens and their strings
sonar_token_to_string = {}
for tok_id in range(min(200000, sonar_vocab_size)):
    s = sonar_str(tok_id)
    if s is not None and len(s) > 0:
        sonar_token_to_string[tok_id] = s

print(f"SONAR tokens with strings: {len(sonar_token_to_string)}")

# Find overlapping tokens (exact string match in both vocabs)
# For each SONAR token string, check if Qwen has a single token for it
overlap_sonar_ids = set()
overlap_info = {}  # sonar_id -> (string, qwen_id)

for sonar_id, s in sonar_token_to_string.items():
    # Check exact match
    qwen_toks = qt.encode(s, add_special_tokens=False)
    if len(qwen_toks) == 1:
        qwen_id = qwen_toks[0]
        if qt.decode([qwen_id]) == s:
            overlap_sonar_ids.add(sonar_id)
            overlap_info[sonar_id] = (s, qwen_id)

    # Also check space-prefixed version
    s_space = " " + s
    qwen_toks_space = qt.encode(s_space, add_special_tokens=False)
    if len(qwen_toks_space) == 1:
        qwen_id = qwen_toks_space[0]
        if qt.decode([qwen_id]) == s_space:
            overlap_sonar_ids.add(sonar_id)
            if sonar_id not in overlap_info:
                overlap_info[sonar_id] = (s, qwen_id)

# Always include special tokens
special_tokens = [0, 1, 2, 3]  # PAD, UNK, BOS, EOS
for tok in special_tokens:
    overlap_sonar_ids.add(tok)

# Include language tag (256047 for eng_Latn)
overlap_sonar_ids.add(256047)

# Include sentence-boundary punctuation tokens (248xxx series)
# These are SONAR's preferred sentence-ending tokens
sentence_punct = [248075, 248079, 248130, 248203]  # . , ? !
for tok in sentence_punct:
    overlap_sonar_ids.add(tok)

print(f"Added sentence punctuation tokens: {sentence_punct}")

overlap_sonar_ids = sorted(overlap_sonar_ids)
print(f"Overlapping vocabulary size: {len(overlap_sonar_ids)}")
print(f"Overlap percentage: {100*len(overlap_sonar_ids)/sonar_vocab_size:.1f}%")

# Create a mask for allowed tokens
allowed_mask = torch.zeros(sonar_vocab_size, device=dev)
for tok_id in overlap_sonar_ids:
    if tok_id < sonar_vocab_size:
        allowed_mask[tok_id] = 1.0

# For tokens not allowed, we set logits to -inf
disallowed_mask = (allowed_mask == 0)
print(f"Tokens masked out: {disallowed_mask.sum().item()}")


# ============================================================================
# Restricted decoding function
# ============================================================================
def decode_restricted(embedding, max_len=60, use_restriction=True):
    """Decode embedding with restricted vocabulary."""
    e = embedding.detach().unsqueeze(0) if embedding.dim() == 1 else embedding.detach()
    eo = e.unsqueeze(1)

    generated = [3, 256047]  # BOS, eng_Latn

    for _ in range(max_len):
        di = torch.tensor([generated], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        if use_restriction:
            # Mask out disallowed tokens
            logits = logits.clone()
            logits[disallowed_mask] = float('-inf')

        next_token = logits.argmax().item()
        generated.append(next_token)

        if next_token == 3:  # EOS
            break

    return torch.tensor(generated, device=dev)


def tokens_to_text(tokens):
    """Convert token IDs to text."""
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


# ============================================================================
# Test on sentences
# ============================================================================
print("\n" + "=" * 80)
print("TESTING RESTRICTED VOCABULARY DECODING")
print("=" * 80)

test_sentences = [
    # Simple sentences
    "Hello world.",
    "The cat sat on the mat.",
    "I love you.",
    "How are you today?",
    "What is your name?",

    # Antonym prompts
    "hot becomes cold",
    "big becomes small",
    "fast becomes slow",
    "Find the antonym.",

    # Longer/complex sentences
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating.",
    "I think therefore I am.",
    "Can you help me with this problem?",
    "Programming in Python is fun.",

    # Numbers (known problematic)
    "The answer is 42.",
    "I have 10 apples.",
    "The year is 2024.",

    # Technical/rare words
    "Quantum physics is complex.",
    "The algorithm is efficient.",
    "Neural networks learn patterns.",

    # Edge cases
    "Yes.",
    "No!",
    "Why?",
    "Hello, how are you?",
]

print(f"\n{'Input':<45} | {'Normal Decode':<35} | {'Restricted':<35}")
print("-" * 120)

results = []
for sentence in test_sentences:
    # Encode
    with torch.no_grad():
        embedding = se.predict([sentence], source_lang='eng_Latn').to(dev)

    # Normal decode
    with torch.no_grad():
        normal_tokens = decode_restricted(embedding, use_restriction=False)
        normal_text = tokens_to_text(normal_tokens)

    # Restricted decode
    with torch.no_grad():
        restricted_tokens = decode_restricted(embedding, use_restriction=True)
        restricted_text = tokens_to_text(restricted_tokens)

    results.append((sentence, normal_text, restricted_text))

    # Truncate for display
    s = sentence[:42] + "..." if len(sentence) > 45 else sentence
    n = normal_text[:32] + "..." if len(normal_text) > 35 else normal_text
    r = restricted_text[:32] + "..." if len(restricted_text) > 35 else restricted_text

    print(f"{s:<45} | {n:<35} | {r:<35}")


# ============================================================================
# Analyze quality
# ============================================================================
print("\n" + "=" * 80)
print("QUALITY ANALYSIS")
print("=" * 80)

exact_normal = sum(1 for s, n, r in results if n.strip() == s.strip())
exact_restricted = sum(1 for s, n, r in results if r.strip() == s.strip())

print(f"\nExact match rate:")
print(f"  Normal decode: {exact_normal}/{len(results)} ({100*exact_normal/len(results):.0f}%)")
print(f"  Restricted decode: {exact_restricted}/{len(results)} ({100*exact_restricted/len(results):.0f}%)")

# Check character-level similarity
def char_similarity(a, b):
    a, b = a.lower().strip(), b.lower().strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matches = sum(1 for i, c in enumerate(a) if i < len(b) and c == b[i])
    return matches / max(len(a), len(b))

normal_sim = sum(char_similarity(s, n) for s, n, r in results) / len(results)
restricted_sim = sum(char_similarity(s, r) for s, n, r in results) / len(results)

print(f"\nCharacter-level similarity:")
print(f"  Normal decode: {100*normal_sim:.1f}%")
print(f"  Restricted decode: {100*restricted_sim:.1f}%")


# ============================================================================
# Show detailed comparison for a few examples
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED TOKEN ANALYSIS")
print("=" * 80)

for sentence in ["The cat sat on the mat.", "hot becomes cold", "Hello world."]:
    print(f"\n'{sentence}'")

    with torch.no_grad():
        embedding = se.predict([sentence], source_lang='eng_Latn').to(dev)
        normal_tokens = decode_restricted(embedding, use_restriction=False)
        restricted_tokens = decode_restricted(embedding, use_restriction=True)

    normal_content = normal_tokens[2:-1] if normal_tokens[-1] == 3 else normal_tokens[2:]
    restricted_content = restricted_tokens[2:-1] if restricted_tokens[-1] == 3 else restricted_tokens[2:]

    print(f"  Normal tokens: {normal_content.tolist()}")
    print(f"  Normal text: '{tokens_to_text(normal_tokens)}'")
    print(f"  Restricted tokens: {restricted_content.tolist()}")
    print(f"  Restricted text: '{tokens_to_text(restricted_tokens)}'")

    # Check if restricted tokens are in overlap
    in_overlap = [t.item() in overlap_sonar_ids for t in restricted_content]
    print(f"  All in overlap: {all(in_overlap)}")


# ============================================================================
# Show some examples of tokens that are NOT in the overlap
# ============================================================================
print("\n" + "=" * 80)
print("TOKENS NOT IN OVERLAP (examples)")
print("=" * 80)

not_in_overlap = []
for tok_id, s in sonar_token_to_string.items():
    if tok_id not in overlap_sonar_ids:
        not_in_overlap.append((tok_id, s))

print(f"\nTotal tokens not in overlap: {len(not_in_overlap)}")
print("\nSample of excluded tokens:")
for tok_id, s in not_in_overlap[:30]:
    print(f"  {tok_id}: '{s}'")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Vocabulary overlap: {len(overlap_sonar_ids)} tokens ({100*len(overlap_sonar_ids)/sonar_vocab_size:.1f}% of SONAR vocab)

This restricted decoding forces SONAR to use only tokens that have exact
equivalents in Qwen's vocabulary. The tradeoff is:
- Worse reconstruction quality (may need more tokens / approximations)
- But 1:1 token correspondence between SONAR and Qwen

This is useful for bridging between the two systems at the token level.
""")
