"""
Analyze why soft eval doesn't match hard eval.
Check if the bridge preserves semantic meaning from SONAR → Qwen.
"""

import torch
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

def dec_t(e):
    e = e.detach().unsqueeze(0) if e.dim() == 1 else e.detach()
    eo = e.unsqueeze(1)
    g = [3, 256047]
    for _ in range(60):
        di = torch.tensor([g], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        nt = sdm.decoder.final_proj(h)[0, -1, :].argmax().item()
        g.append(nt)
        if nt == 3: break
    return torch.tensor(g, device=dev)

def geth(e, t):
    e = e.unsqueeze(0) if e.dim() == 1 else e
    return sdm.decode(t[:-1].unsqueeze(0), BatchLayout.of(t[:-1].unsqueeze(0)), e.unsqueeze(1), BatchLayout.of(e.unsqueeze(1)))

def bridge(h): return (h - sm) @ W + tm

def generate_from_embeds(prefix_embeds, input_text):
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

def generate_from_text(prefix_text, input_text):
    """Generate from text prompt (hard eval)."""
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


print("=" * 70, flush=True)
print("ANALYSIS: Bridge preserves semantic meaning?", flush=True)
print("=" * 70, flush=True)

prompts = [
    "Give the opposite word for each input.",
    "Antonyms: words with opposite meanings.",
    "Find the antonym: hot becomes cold, big becomes small.",
]

examples = [
    ("hot ->", "cold"),
    ("big ->", "small"),
    ("fast ->", "slow"),
]

for prompt in prompts:
    print(f"\n{'='*70}", flush=True)
    print(f"Prompt: '{prompt}'", flush=True)
    print("=" * 70, flush=True)

    # Encode and decode through SONAR
    with torch.no_grad():
        z = enc(prompt)
        tokens = dec_t(z)
        decoded = tdc(tokens[2:-1].cpu()) if tokens[-1] == 3 else tdc(tokens[2:].cpu())

        # Get SONAR hidden states
        hidden = geth(z, tokens)  # [1, seq_len, 1024]

        # Bridge to Qwen
        prefix_embeds = bridge(hidden)  # [1, seq_len, 896]

        print(f"Decoded: '{decoded}'", flush=True)
        print(f"SONAR hidden shape: {hidden.shape}", flush=True)
        print(f"Bridge output shape: {prefix_embeds.shape}", flush=True)

        # Compare: what do these embeddings look like vs actual Qwen embeddings?
        qwen_tokens = qt(prompt, return_tensors="pt").input_ids.to(dev)
        qwen_embeds = qm.model.embed_tokens(qwen_tokens)

        print(f"Qwen tokens shape: {qwen_tokens.shape}", flush=True)
        print(f"Qwen embeds shape: {qwen_embeds.shape}", flush=True)

        # Embedding statistics
        print(f"\nBridge output stats:", flush=True)
        print(f"  Mean: {prefix_embeds.mean().item():.4f}, Std: {prefix_embeds.std().item():.4f}", flush=True)
        print(f"  Min: {prefix_embeds.min().item():.4f}, Max: {prefix_embeds.max().item():.4f}", flush=True)

        print(f"Qwen embeds stats:", flush=True)
        print(f"  Mean: {qwen_embeds.mean().item():.4f}, Std: {qwen_embeds.std().item():.4f}", flush=True)
        print(f"  Min: {qwen_embeds.min().item():.4f}, Max: {qwen_embeds.max().item():.4f}", flush=True)

        # Test on examples
        print(f"\nSoft eval (SONAR -> bridge -> Qwen):", flush=True)
        for inp, tgt in examples:
            pred = generate_from_embeds(prefix_embeds, inp)
            sym = '✓' if tgt.lower() in pred.lower() else '✗'
            print(f"  {inp} -> '{pred}' ({tgt}) {sym}", flush=True)

        print(f"\nHard eval (text -> Qwen tokenizer -> Qwen):", flush=True)
        for inp, tgt in examples:
            pred = generate_from_text(prompt, inp)
            sym = '✓' if tgt.lower() in pred.lower() else '✗'
            print(f"  {inp} -> '{pred}' ({tgt}) {sym}", flush=True)


# Also test: what if we use Qwen embeds directly as prefix?
print("\n" + "=" * 70, flush=True)
print("CONTROL: Using Qwen embeds directly as prefix", flush=True)
print("=" * 70, flush=True)

prompt = "Give the opposite word for each input."
with torch.no_grad():
    qwen_tokens = qt(prompt + "\n", return_tensors="pt").input_ids.to(dev)
    qwen_embeds = qm.model.embed_tokens(qwen_tokens)

    print(f"Prompt: '{prompt}'", flush=True)
    print(f"Using actual Qwen embeddings as prefix:", flush=True)
    for inp, tgt in examples:
        pred = generate_from_embeds(qwen_embeds, inp)
        sym = '✓' if tgt.lower() in pred.lower() else '✗'
        print(f"  {inp} -> '{pred}' ({tgt}) {sym}", flush=True)
