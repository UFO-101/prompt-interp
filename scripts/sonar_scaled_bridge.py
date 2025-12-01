"""
SONAR optimization with properly scaled bridge.
The issue: SONAR hidden states have ~19x larger std than Qwen embeddings.
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
print("Ready.\n", flush=True)

def enc(t): return se.predict([t], source_lang='eng_Latn').to(dev)

def dec_t(e, max_len=60):
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
    t = dec_t(e)
    tt = t[2:-1] if t[-1] == 3 else t[2:]
    return tdc(tt.cpu()) if len(tt) > 0 else ""

def geth(e, t):
    e = e.unsqueeze(0) if e.dim() == 1 else e
    return sdm.decode(t[:-1].unsqueeze(0), BatchLayout.of(t[:-1].unsqueeze(0)), e.unsqueeze(1), BatchLayout.of(e.unsqueeze(1)))

def bridge_unscaled(h):
    """Original bridge - produces embeddings with wrong scale."""
    return (h - sm) @ W + tm

def bridge_scaled(h, target_std=0.015):
    """Scaled bridge - normalize to match Qwen embedding statistics."""
    transformed = (h - sm) @ W + tm
    # Normalize each embedding vector
    current_std = transformed.std()
    scaled = transformed * (target_std / current_std)
    return scaled

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
    if not words:
        return ''
    return ''.join(c for c in words[0] if c.isalpha())


examples = [
    ("hot ->", "cold"),
    ("big ->", "small"),
    ("fast ->", "slow"),
    ("up ->", "down"),
    ("happy ->", "sad"),
    ("light ->", "dark"),
]

print("=" * 70, flush=True)
print("TEST: Scaled vs Unscaled Bridge", flush=True)
print("=" * 70, flush=True)

prompts = [
    "Give the opposite word for each input.",
    "Find the antonym: hot becomes cold, big becomes small.",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'", flush=True)
    print("-" * 70, flush=True)

    with torch.no_grad():
        z = enc(prompt)
        tokens = dec_t(z)
        hidden = geth(z, tokens)

        # Unscaled bridge
        unscaled_embeds = bridge_unscaled(hidden)
        print(f"Unscaled bridge std: {unscaled_embeds.std().item():.4f}", flush=True)

        # Scaled bridge
        scaled_embeds = bridge_scaled(hidden)
        print(f"Scaled bridge std: {scaled_embeds.std().item():.4f}", flush=True)

        print(f"\nUnscaled soft eval:", flush=True)
        unscaled_correct = 0
        for inp, tgt in examples:
            pred = generate_soft(unscaled_embeds, inp)
            fw = get_first_word(pred)
            correct = fw == tgt.lower()
            unscaled_correct += correct
            sym = '✓' if correct else '✗'
            print(f"  {inp} -> '{fw}' ({tgt}) {sym}", flush=True)

        print(f"\nScaled soft eval:", flush=True)
        scaled_correct = 0
        for inp, tgt in examples:
            pred = generate_soft(scaled_embeds, inp)
            fw = get_first_word(pred)
            correct = fw == tgt.lower()
            scaled_correct += correct
            sym = '✓' if correct else '✗'
            print(f"  {inp} -> '{fw}' ({tgt}) {sym}", flush=True)

        print(f"\nHard eval (reference):", flush=True)
        hard_correct = 0
        for inp, tgt in examples:
            pred = generate_hard(prompt, inp)
            fw = get_first_word(pred)
            correct = fw == tgt.lower()
            hard_correct += correct
            sym = '✓' if correct else '✗'
            print(f"  {inp} -> '{fw}' ({tgt}) {sym}", flush=True)

        print(f"\nSummary: unscaled={unscaled_correct}/6, scaled={scaled_correct}/6, hard={hard_correct}/6", flush=True)


# Now test different scaling factors
print("\n" + "=" * 70, flush=True)
print("SWEEP: Different scaling factors", flush=True)
print("=" * 70, flush=True)

prompt = "Give the opposite word for each input."
with torch.no_grad():
    z = enc(prompt)
    tokens = dec_t(z)
    hidden = geth(z, tokens)

    for target_std in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        scaled = bridge_scaled(hidden, target_std=target_std)
        correct = 0
        for inp, tgt in examples:
            pred = generate_soft(scaled, inp)
            fw = get_first_word(pred)
            if fw == tgt.lower():
                correct += 1
        print(f"target_std={target_std:.3f}: {correct}/6", flush=True)
