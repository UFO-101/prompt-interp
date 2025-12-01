"""
Map full sentences: SONAR tokens -> embeddings -> bridge -> Qwen tokens
Show input/output sentences side by side.
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
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')
sonar_dec = sd.tokenizer.create_decoder()
print("Ready.\n", flush=True)


def bridge(emb):
    return (emb - src_mean) @ W + tgt_mean


def nearest_qwen_token(qwen_emb):
    """Find single nearest Qwen token."""
    qwen_emb_norm = F.normalize(qwen_emb.unsqueeze(0), dim=1)
    qwen_embeds_norm = F.normalize(qwen_embeds, dim=1)
    sims = (qwen_emb_norm @ qwen_embeds_norm.T).squeeze(0)
    idx = sims.argmax().item()
    return idx, sims[idx].item()


def sonar_tok_to_str(tok_id):
    try:
        return sonar_dec(torch.tensor([tok_id]))
    except:
        return f"<{tok_id}>"


def qwen_tok_to_str(tok_id):
    return qt.decode([tok_id])


def map_sentence(sentence):
    """Map a sentence through the full pipeline."""
    print(f"\n{'='*80}")
    print(f"INPUT: \"{sentence}\"")
    print('='*80)

    # Step 1: SONAR tokenization
    sonar_tokens = sonar_enc(sentence)
    # Remove BOS (token 3) and lang tag (256047)
    content_tokens = sonar_tokens[2:]  # Keep EOS if present

    print(f"\n1. SONAR TOKENIZATION ({len(content_tokens)} tokens):")
    sonar_strs = []
    for i, tok in enumerate(content_tokens):
        tok_id = tok.item()
        tok_str = sonar_tok_to_str(tok_id)
        sonar_strs.append(tok_str)
        print(f"   [{i:2d}] {tok_id:6d} -> '{tok_str}'")

    # Step 2: Get SONAR embeddings and bridge to Qwen
    print(f"\n2. EMBED -> BRIDGE -> NEAREST QWEN TOKEN:")
    qwen_tokens = []
    qwen_strs = []
    for i, tok in enumerate(content_tokens):
        tok_id = tok.item()
        sonar_emb = sonar_embeds[tok_id]
        qwen_emb = bridge(sonar_emb)
        qwen_tok, sim = nearest_qwen_token(qwen_emb)
        qwen_str = qwen_tok_to_str(qwen_tok)
        qwen_tokens.append(qwen_tok)
        qwen_strs.append(qwen_str)

        sonar_s = sonar_tok_to_str(tok_id)
        print(f"   [{i:2d}] SONAR '{sonar_s:<12}' -> Qwen '{qwen_str:<12}' (sim={sim:.3f})")

    # Step 3: Reconstruct sentences
    print(f"\n3. RECONSTRUCTED SENTENCES:")
    sonar_sentence = ''.join(sonar_strs).replace('</s>', '').strip()
    qwen_sentence = ''.join(qwen_strs).replace('<|endoftext|>', '').strip()

    print(f"   SONAR reconstruction: \"{sonar_sentence}\"")
    print(f"   Qwen reconstruction:  \"{qwen_sentence}\"")

    # Step 4: Compare to actual Qwen tokenization
    print(f"\n4. ACTUAL QWEN TOKENIZATION (for reference):")
    actual_qwen_tokens = qt.encode(sentence, add_special_tokens=False)
    print(f"   ({len(actual_qwen_tokens)} tokens):")
    for i, tok in enumerate(actual_qwen_tokens):
        tok_str = qwen_tok_to_str(tok)
        print(f"   [{i:2d}] {tok:6d} -> '{tok_str}'")

    return sonar_sentence, qwen_sentence


# Test sentences
sentences = [
    "Hello world.",
    "The cat sat on the mat.",
    "Give the opposite word.",
    "hot becomes cold",
    "Find the antonym.",
    "What is the meaning of life?",
    "The quick brown fox jumps over the lazy dog.",
    "I love programming.",
]

print("="*80)
print("SENTENCE MAPPING: SONAR -> Bridge -> Qwen")
print("="*80)

results = []
for sentence in sentences:
    sonar_out, qwen_out = map_sentence(sentence)
    results.append((sentence, sonar_out, qwen_out))

# Summary
print("\n" + "="*80)
print("SUMMARY: Input vs Mapped Output")
print("="*80)
print(f"\n{'Input':<45} {'Mapped to Qwen':<45}")
print("-"*90)
for inp, _, qwen_out in results:
    # Truncate for display
    inp_disp = inp[:42] + "..." if len(inp) > 45 else inp
    out_disp = qwen_out[:42] + "..." if len(qwen_out) > 45 else qwen_out
    print(f"{inp_disp:<45} {out_disp:<45}")
