"""Debug: verify z is actually changing and being re-decoded."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
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

def dec_t_debug(e):
    """Decode with debug info about first token logits."""
    e = e.detach().unsqueeze(0) if e.dim() == 1 else e.detach()
    eo = e.unsqueeze(1)
    g = [3, 256047]  # BOS + lang tag

    # Get first real token prediction
    di = torch.tensor([g], device=dev)
    h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
    logits = sdm.decoder.final_proj(h)[0, -1, :]

    # Get top 5 tokens and their probs
    probs = torch.softmax(logits, dim=0)
    top5 = torch.topk(probs, 5)
    top5_tokens = top5.indices.tolist()
    top5_probs = top5.values.tolist()

    # Continue decoding
    for _ in range(50):
        di = torch.tensor([g], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        nt = sdm.decoder.final_proj(h)[0, -1, :].argmax().item()
        g.append(nt)
        if nt == 3: break

    tokens = torch.tensor(g, device=dev)
    tt = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    text = tdc(tt.cpu()) if len(tt) > 0 else ""

    return text, top5_tokens, top5_probs

def geth(e, t):
    e = e.unsqueeze(0) if e.dim() == 1 else e
    di = t[:-1].unsqueeze(0)
    return sdm.decode(di, BatchLayout.of(di), e.unsqueeze(1), BatchLayout.of(e.unsqueeze(1)))

def dec_t(e):
    e = e.detach().unsqueeze(0) if e.dim() == 1 else e.detach()
    eo = e.unsqueeze(1)
    g = [3, 256047]
    for _ in range(50):
        di = torch.tensor([g], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        nt = sdm.decoder.final_proj(h)[0, -1, :].argmax().item()
        g.append(nt)
        if nt == 3: break
    return torch.tensor(g, device=dev)

def bridge(h): return (h - sm) @ W + tm

def loss(pe, inp, tgt):
    ti = qt(f"{inp} ", return_tensors="pt").input_ids.to(dev)
    te = qm.model.embed_tokens(ti)
    tgt_t = qt.encode(tgt, add_special_tokens=False)
    fe = torch.cat([pe, te, qm.model.embed_tokens(torch.tensor([tgt_t], device=dev))], dim=1)
    o = qm(inputs_embeds=fe)
    sp = pe.shape[1] + ti.shape[1] - 1
    return sum(F.cross_entropy(o.logits[0, sp + i, :].unsqueeze(0), torch.tensor([t]).to(dev)) for i, t in enumerate(tgt_t)) / len(tgt_t)

ex = [("hot ->", "cold"), ("big ->", "small"), ("fast ->", "slow")]
seed = "Antonyms: words with opposite meanings."

print(f"Seed: '{seed}'", flush=True)
print("=" * 80, flush=True)

with torch.no_grad():
    zi = enc(seed)
    print(f"Initial z norm: {zi.norm().item():.4f}", flush=True)
    print(f"Initial z[0,:5]: {zi[0,:5].tolist()}", flush=True)

z = nn.Parameter(zi.clone())
opt = torch.optim.Adam([z], lr=0.0005)

for st in range(35):
    opt.zero_grad()

    # Debug: show z values before decoding
    z_sample = z.data[0, :5].tolist()

    with torch.no_grad():
        t = dec_t(z)
        decoded, top5_tok, top5_prob = dec_t_debug(z)

    pe = bridge(geth(z, t))
    l = sum(loss(pe, i, t) for i, t in ex) / len(ex)
    l.backward()

    grad_norm = z.grad.norm().item() if z.grad is not None else 0
    torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)
    opt.step()

    z_drift = (z.data - zi).norm().item()

    print(f"\nStep {st}: loss={l.item():.2f}, z_drift={z_drift:.4f}, grad_norm={grad_norm:.2f}", flush=True)
    print(f"  z[0,:5]: {z_sample}", flush=True)
    print(f"  First token top-5: {[(t, f'{p:.3f}') for t, p in zip(top5_tok, top5_prob)]}", flush=True)
    print(f"  Decoded: '{decoded[:60]}'", flush=True)
