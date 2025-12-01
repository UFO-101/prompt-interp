"""
New approach: Use SONAR token embeddings, not hidden states.

Instead of: z -> decoder hidden states -> bridge -> Qwen
Try: z -> decoded tokens -> SONAR token embeddings -> bridge -> Qwen

The Procrustes alignment was trained on token embeddings,
so it should work on token embeddings!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout

dev = "cuda"
print("Loading...", flush=True)
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(dev).eval()
a = np.load("results/nllb_qwen_alignment.npz")
W = torch.tensor(a['W'], dtype=torch.float32, device=dev)
sm = torch.tensor(a['src_mean'], dtype=torch.float32, device=dev)
tm = torch.tensor(a['tgt_mean'], dtype=torch.float32, device=dev)
sdm = sd.model.to(dev)
tdc = sd.tokenizer.create_decoder()

# Get SONAR/NLLB embedding matrix
sonar_embed_matrix = sdm.decoder.final_proj.weight.data  # [vocab_size, 1024]
print(f"SONAR embedding matrix shape: {sonar_embed_matrix.shape}")
print("Ready.\n", flush=True)

def enc(t): return se.predict([t], source_lang='eng_Latn').to(dev)

def dec_t(e, max_len=60):
    """Decode embedding to tokens."""
    e = e.detach().unsqueeze(0) if e.dim() == 1 else e.detach()
    eo = e.unsqueeze(1)
    g = [3, 256047]
    for _ in range(max_len):
        di = torch.tensor([g], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        nt = sdm.decoder.final_proj(h)[0, -1, :].argmax().item()
        g.append(nt)
        if nt == 3: break
    return torch.tensor(g, device=dev)

def dec(e):
    """Decode embedding to text."""
    t = dec_t(e)
    tt = t[2:-1] if t[-1] == 3 else t[2:]
    return tdc(tt.cpu()) if len(tt) > 0 else ""

def get_token_embeddings(tokens):
    """Get SONAR token embeddings for a sequence of tokens."""
    # Skip BOS (3) and lang tag (256047), and EOS (3)
    content_tokens = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    embeds = sonar_embed_matrix[content_tokens]  # [seq_len, 1024]
    return embeds.unsqueeze(0)  # [1, seq_len, 1024]

def bridge(sonar_embeds):
    """Map SONAR token embeddings to Qwen embedding space."""
    return (sonar_embeds - sm) @ W + tm

def generate_soft(prefix_embeds, input_text):
    """Generate from soft prefix embeddings."""
    task_text = f"{input_text} "
    task_tokens = qt(task_text, return_tensors="pt").input_ids.to(dev)
    task_embeds = qm.model.embed_tokens(task_tokens)
    full_embeds = torch.cat([prefix_embeds, task_embeds], dim=1)

    outputs = qm(inputs_embeds=full_embeds)
    generated = []
    for _ in range(5):
        nt = outputs.logits[0, -1, :].argmax().item()
        generated.append(nt)
        if nt == qt.eos_token_id: break
        ne = qm.model.embed_tokens(torch.tensor([[nt]], device=dev))
        full_embeds = torch.cat([full_embeds, ne], dim=1)
        outputs = qm(inputs_embeds=full_embeds)
    return qt.decode(generated).strip()

def generate_hard(prefix_text, input_text):
    """Generate from text prompt."""
    full_prompt = f"{prefix_text}\n{input_text} "
    input_ids = qt(full_prompt, return_tensors="pt").input_ids.to(dev)

    outputs = qm(input_ids=input_ids)
    generated = []
    for _ in range(5):
        nt = outputs.logits[0, -1, :].argmax().item()
        generated.append(nt)
        if nt == qt.eos_token_id: break
        input_ids = torch.cat([input_ids, torch.tensor([[nt]], device=dev)], dim=1)
        outputs = qm(input_ids=input_ids)
    return qt.decode(generated).strip()

def get_first_word(text):
    text = text.lower()
    words = text.split()
    if not words: return ''
    return ''.join(c for c in words[0] if c.isalpha())


examples = [
    ("hot ->", "cold"),
    ("big ->", "small"),
    ("fast ->", "slow"),
    ("up ->", "down"),
    ("happy ->", "sad"),
    ("light ->", "dark"),
]

print("=" * 70)
print("TEST: Token embeddings vs Hidden states")
print("=" * 70)

prompts = [
    "Give the opposite word for each input.",
    "Find the antonym: hot becomes cold, big becomes small.",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 70)

    with torch.no_grad():
        z = enc(prompt)
        tokens = dec_t(z)
        decoded = dec(z)
        print(f"Decoded: '{decoded}'")

        # Get token embeddings
        token_embeds = get_token_embeddings(tokens)  # [1, seq, 1024]
        print(f"Token embeddings shape: {token_embeds.shape}")
        print(f"Token embeddings stats: mean={token_embeds.mean():.4f}, std={token_embeds.std():.4f}")

        # Bridge to Qwen
        qwen_embeds = bridge(token_embeds)
        print(f"Bridged embeddings stats: mean={qwen_embeds.mean():.4f}, std={qwen_embeds.std():.4f}")

        # Compare to actual Qwen embeddings
        qwen_actual = qm.model.embed_tokens(qt(prompt, return_tensors="pt").input_ids.to(dev))
        print(f"Actual Qwen embeds stats: mean={qwen_actual.mean():.4f}, std={qwen_actual.std():.4f}")

        # Test soft eval with token embeddings
        print(f"\nSoft eval (SONAR token embeddings -> bridge -> Qwen):")
        soft_correct = 0
        for inp, tgt in examples:
            pred = generate_soft(qwen_embeds, inp)
            fw = get_first_word(pred)
            correct = fw == tgt.lower()
            soft_correct += correct
            sym = '✓' if correct else '✗'
            print(f"  {inp} -> '{fw}' ({tgt}) {sym}")

        # Test hard eval
        print(f"\nHard eval (text -> Qwen tokenizer -> Qwen):")
        hard_correct = 0
        for inp, tgt in examples:
            pred = generate_hard(prompt, inp)
            fw = get_first_word(pred)
            correct = fw == tgt.lower()
            hard_correct += correct
            sym = '✓' if correct else '✗'
            print(f"  {inp} -> '{fw}' ({tgt}) {sym}")

        print(f"\nSummary: soft={soft_correct}/6, hard={hard_correct}/6")


# Now let's check the alignment quality more carefully
print("\n" + "=" * 70)
print("ALIGNMENT QUALITY CHECK")
print("=" * 70)

# Find some shared tokens between SONAR and Qwen
test_words = ["hot", "cold", "big", "small", "word", "opposite"]

print("\nComparing embeddings for shared vocabulary:")
for word in test_words:
    # Get Qwen embedding
    qwen_tok = qt.encode(word, add_special_tokens=False)
    if len(qwen_tok) != 1:
        print(f"  '{word}': multi-token in Qwen, skipping")
        continue
    qwen_emb = qm.model.embed_tokens(torch.tensor([qwen_tok], device=dev))[0, 0]

    # Get SONAR embedding and bridge it
    # Need to find SONAR token for this word
    try:
        sonar_tok = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')(word)
        if len(sonar_tok) > 3:  # More than BOS + lang + word
            print(f"  '{word}': multi-token in SONAR, skipping")
            continue
        # Get the actual word token (skip BOS and lang tag)
        word_tok = sonar_tok[-1].item()
        sonar_emb = sonar_embed_matrix[word_tok]
        bridged_emb = bridge(sonar_emb.unsqueeze(0).unsqueeze(0))[0, 0]

        # Compare
        cosine = F.cosine_similarity(qwen_emb.unsqueeze(0), bridged_emb.unsqueeze(0)).item()
        print(f"  '{word}': cosine={cosine:.4f}, qwen_norm={qwen_emb.norm():.4f}, bridged_norm={bridged_emb.norm():.4f}")
    except Exception as e:
        print(f"  '{word}': error - {e}")
