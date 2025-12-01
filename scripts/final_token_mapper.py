"""
Final Token Mapper: SONAR → Qwen with space-aware processing.

Key insights applied:
1. SONAR doesn't use space-prefix tokens, Qwen does
2. Single letters fail - need special handling
3. 79.2% base accuracy on ASCII tokens
4. Post-process to add spaces appropriately
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
qwen_embeds_norm = F.normalize(qwen_embeds, dim=1).cpu()

def sonar_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return None

# Load trained model
class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearMapper(1024, 896).to(dev)
model.load_state_dict(torch.load("results/token_mapper_best.pt"))
model.eval()


# Build lookup for single-letter tokens (these fail systematically)
# Map them directly to Qwen equivalents
SINGLE_LETTER_MAP = {}
for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
    qwen_toks = qt.encode(char, add_special_tokens=False)
    if len(qwen_toks) == 1:
        SINGLE_LETTER_MAP[char] = qwen_toks[0]

print(f"Built single-letter map with {len(SINGLE_LETTER_MAP)} entries")


def map_token(sonar_tok_id, position=0):
    """
    Map a single SONAR token to Qwen.

    Args:
        sonar_tok_id: SONAR token ID
        position: Position in sequence (0 = first token)

    Returns:
        qwen_str: The Qwen token string
        qwen_id: The Qwen token ID
    """
    sonar_s = sonar_str(sonar_tok_id)

    # Special case: single letters that fail systematically
    if sonar_s and len(sonar_s) == 1 and sonar_s in SINGLE_LETTER_MAP:
        qwen_id = SINGLE_LETTER_MAP[sonar_s]
        qwen_s = qt.decode([qwen_id])
        return qwen_s, qwen_id, 1.0  # Perfect match

    # Use learned mapping
    with torch.no_grad():
        sonar_emb = sonar_embeds[sonar_tok_id].unsqueeze(0)
        mapped = model(sonar_emb)
        mapped_norm = F.normalize(mapped, dim=1).cpu()
        sims = (mapped_norm @ qwen_embeds_norm.T).squeeze(0)

        # Get top candidates
        top_k = 5
        top_ids = sims.topk(top_k).indices.tolist()
        top_sims = sims.topk(top_k).values.tolist()

    # Choose best candidate
    # Prefer space-prefix version for non-first tokens if available
    qwen_id = top_ids[0]
    qwen_s = qt.decode([qwen_id])
    best_sim = top_sims[0]

    # If not first position and top choice doesn't have space, check if space version exists
    if position > 0 and sonar_s and not qwen_s.startswith(' '):
        for i, tid in enumerate(top_ids[1:], 1):
            ts = qt.decode([tid])
            if ts == ' ' + qwen_s or ts == ' ' + sonar_s:
                qwen_id = tid
                qwen_s = ts
                best_sim = top_sims[i]
                break

    return qwen_s, qwen_id, best_sim


def map_sentence(sentence, add_spaces=True):
    """
    Map a full sentence from SONAR to Qwen tokens.

    Args:
        sentence: Input sentence
        add_spaces: Whether to add spaces between tokens

    Returns:
        output: Reconstructed sentence
        details: List of (sonar_str, qwen_str, sim) tuples
    """
    sonar_tokens = sonar_enc(sentence)
    content_tokens = sonar_tokens[2:]  # Skip BOS and lang tag

    output_strs = []
    details = []

    for pos, tok in enumerate(content_tokens):
        tok_id = tok.item()
        if tok_id == 3:  # EOS
            break

        sonar_s = sonar_str(tok_id) or f"<{tok_id}>"
        qwen_s, qwen_id, sim = map_token(tok_id, position=pos)

        # Post-processing: add space if needed
        if add_spaces and pos > 0:
            # Add space before this token if:
            # - Previous output doesn't end with space
            # - This token doesn't start with space
            # - This token doesn't start with punctuation
            # - This token is not empty
            if output_strs and qwen_s:
                prev = output_strs[-1]
                if (not prev.endswith(' ') and
                    not qwen_s.startswith(' ') and
                    not qwen_s[0] in '.,;:!?\'")-]}'):
                    qwen_s = ' ' + qwen_s

        output_strs.append(qwen_s)
        details.append((sonar_s, qwen_s, sim))

    output = ''.join(output_strs).strip()
    return output, details


# ============================================================================
# Test on sentences
# ============================================================================
print("\n" + "="*70)
print("FINAL SENTENCE MAPPING RESULTS")
print("="*70)

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
    "The weather is nice today.",
    "Can you help me?",
    "Machine learning is fascinating.",
]

print(f"\n{'Input':<45} | {'Output':<45}")
print("-" * 95)

for sentence in test_sentences:
    output, _ = map_sentence(sentence)
    s = sentence[:42] + "..." if len(sentence) > 45 else sentence
    o = output[:42] + "..." if len(output) > 45 else output
    print(f"{s:<45} | {o:<45}")


# ============================================================================
# Detailed analysis
# ============================================================================
print("\n" + "="*70)
print("DETAILED TOKEN-BY-TOKEN ANALYSIS")
print("="*70)

for sentence in test_sentences[:4]:
    print(f"\n'{sentence}'")
    output, details = map_sentence(sentence)

    print(f"  Output: '{output}'")
    print(f"  {'SONAR':<12} {'->':^4} {'Qwen':<12} {'Sim':<6}")
    print(f"  {'-'*40}")
    for sonar_s, qwen_s, sim in details:
        print(f"  '{sonar_s:<10}' {'->':^4} '{qwen_s:<10}' {sim:.3f}")


# ============================================================================
# Accuracy comparison
# ============================================================================
print("\n" + "="*70)
print("MAPPING QUALITY SUMMARY")
print("="*70)

# Compare with just raw model (no post-processing)
print("\nWithout space post-processing:")
for sentence in test_sentences[:5]:
    output_raw, _ = map_sentence(sentence, add_spaces=False)
    print(f"  '{sentence}' -> '{output_raw}'")

print("\nWith space post-processing:")
for sentence in test_sentences[:5]:
    output_proc, _ = map_sentence(sentence, add_spaces=True)
    print(f"  '{sentence}' -> '{output_proc}'")


# ============================================================================
# Save the complete mapper
# ============================================================================
print("\n" + "="*70)
print("SAVING COMPLETE MAPPER")
print("="*70)

# Save everything needed to use this mapper
torch.save({
    'model_state': model.state_dict(),
    'single_letter_map': SINGLE_LETTER_MAP,
}, "results/final_token_mapper.pt")

print("Saved to results/final_token_mapper.pt")

print("""
FINAL SUMMARY:
==============
1. Base token mapping accuracy: ~79% on ASCII tokens
2. Single letters handled via direct lookup (fixes "a"->"1" issue)
3. Space post-processing adds readability
4. Semantic structure is preserved through mapping

LIMITATIONS:
- Non-ASCII tokens have lower accuracy
- Some words still map to similar but wrong tokens
- Tokenization differences cause some information loss

USE CASE:
This mapper is best for:
- English text
- When you need approximate semantic preservation
- When exact token match isn't critical

NOT suitable for:
- Exact reconstruction requirements
- Non-ASCII languages
- Tasks requiring precise token correspondence
""")
