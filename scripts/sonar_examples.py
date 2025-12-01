"""
Show concrete examples of learned prompts and how they appear to SONAR vs Qwen.
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

def bridge(h): return (h - sm) @ W + tm

def compute_loss(prefix_embeds, inp, tgt):
    task_text = f"{inp} "
    task_tokens = qt(task_text, return_tensors="pt").input_ids.to(dev)
    task_embeds = qm.model.embed_tokens(task_tokens)
    target_tokens = qt.encode(tgt, add_special_tokens=False)
    target_embeds = qm.model.embed_tokens(torch.tensor([target_tokens], device=dev))
    full_embeds = torch.cat([prefix_embeds, task_embeds, target_embeds], dim=1)
    outputs = qm(inputs_embeds=full_embeds)
    prefix_len = prefix_embeds.shape[1]
    task_len = task_tokens.shape[1]
    start_pos = prefix_len + task_len - 1
    total_loss = sum(F.cross_entropy(outputs.logits[0, start_pos + i, :].unsqueeze(0),
                    torch.tensor([t]).to(dev)) for i, t in enumerate(target_tokens))
    return total_loss / len(target_tokens)

def generate_soft(prefix_embeds, input_text):
    """What Qwen generates when given SONAR hidden states through bridge."""
    task_text = f"{input_text} "
    task_tokens = qt(task_text, return_tensors="pt").input_ids.to(dev)
    task_embeds = qm.model.embed_tokens(task_tokens)
    full_embeds = torch.cat([prefix_embeds, task_embeds], dim=1)
    outputs = qm(inputs_embeds=full_embeds)
    generated = []
    for _ in range(10):
        nt = outputs.logits[0, -1, :].argmax().item()
        generated.append(nt)
        if nt == qt.eos_token_id: break
        ne = qm.model.embed_tokens(torch.tensor([[nt]], device=dev))
        full_embeds = torch.cat([full_embeds, ne], dim=1)
        outputs = qm(inputs_embeds=full_embeds)
    return qt.decode(generated).strip()

def generate_hard(prefix_text, input_text):
    """What Qwen generates when given the decoded text as a real prompt."""
    full_prompt = f"{prefix_text}\n{input_text} "
    input_ids = qt(full_prompt, return_tensors="pt").input_ids.to(dev)
    outputs = qm(input_ids=input_ids)
    generated = []
    for _ in range(10):
        nt = outputs.logits[0, -1, :].argmax().item()
        generated.append(nt)
        if nt == qt.eos_token_id: break
        input_ids = torch.cat([input_ids, torch.tensor([[nt]], device=dev)], dim=1)
        outputs = qm(input_ids=input_ids)
    return qt.decode(generated).strip()


examples = [
    ("hot ->", "cold"),
    ("big ->", "small"),
    ("fast ->", "slow"),
    ("up ->", "down"),
    ("happy ->", "sad"),
    ("light ->", "dark"),
]

seed = "Find the antonym: hot becomes cold, big becomes small."

print("=" * 80)
print("SONAR OPTIMIZATION: Watching prompts evolve")
print("=" * 80)
print(f"\nSeed: '{seed}'")
print(f"Task: Antonyms (hot->cold, big->small, etc.)")

with torch.no_grad():
    zi = enc(seed)

z = nn.Parameter(zi.clone())
opt = torch.optim.Adam([z], lr=0.0005)

checkpoints = [0, 10, 20, 30, 40, 50, 60]

for step in range(61):
    opt.zero_grad()
    with torch.no_grad():
        tokens = dec_t(z)
    hidden = geth(z, tokens)
    prefix_embeds = bridge(hidden)

    loss = sum(compute_loss(prefix_embeds, inp, tgt) for inp, tgt in examples) / len(examples)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)
    opt.step()

    if step in checkpoints:
        with torch.no_grad():
            decoded = dec(z)
            tokens = dec_t(z)
            hidden = geth(z, tokens)
            prefix_embeds = bridge(hidden)

        print(f"\n{'='*80}")
        print(f"STEP {step} | Loss: {loss.item():.2f}")
        print(f"{'='*80}")
        print(f"\nDecoded prompt (what SONAR thinks z means):")
        print(f"  '{decoded}'")

        print(f"\nSOFT EVAL: SONAR hidden states -> bridge -> Qwen embeddings")
        print(f"  (This is what we optimize for)")
        soft_correct = 0
        for inp, tgt in examples:
            pred = generate_soft(prefix_embeds, inp)
            first_word = ''.join(c for c in pred.split()[0] if c.isalpha()).lower() if pred.split() else ''
            correct = first_word == tgt.lower()
            soft_correct += correct
            sym = '✓' if correct else '✗'
            print(f"    {inp} -> '{pred[:30]}' {sym}")
        print(f"  Soft accuracy: {soft_correct}/6")

        print(f"\nHARD EVAL: Decoded text -> Qwen tokenizer -> Qwen")
        print(f"  (This is what would happen at deployment)")
        hard_correct = 0
        for inp, tgt in examples:
            pred = generate_hard(decoded, inp)
            first_word = ''.join(c for c in pred.split()[0] if c.isalpha()).lower() if pred.split() else ''
            correct = first_word == tgt.lower()
            hard_correct += correct
            sym = '✓' if correct else '✗'
            print(f"    {inp} -> '{pred[:30]}' {sym}")
        print(f"  Hard accuracy: {hard_correct}/6")

        print(f"\n  Gap: soft={soft_correct}/6 vs hard={hard_correct}/6")


print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("""
The optimization finds embeddings that:
1. Decode to increasingly nonsensical text through SONAR
2. But somehow minimize loss when passed through the bridge to Qwen

This is because the bridge doesn't preserve semantic meaning - it's just
a linear transform that the optimizer learns to exploit.

The "learned prompts" like "Find the adonums: What turns on and on..."
work through the bridge but are meaningless as actual text prompts.
""")
