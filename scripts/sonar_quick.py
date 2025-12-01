"""Ultra-fast SONAR test - minimal output."""
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
def dec_t(e):
    e = e.detach().unsqueeze(0) if e.dim() == 1 else e.detach()
    eo = e.unsqueeze(1)
    g = [3, 256047]
    for _ in range(40):
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
def loss(pe, inp, tgt):
    ti = qt(f"{inp} ", return_tensors="pt").input_ids.to(dev)
    te = qm.model.embed_tokens(ti)
    tgt_t = qt.encode(tgt, add_special_tokens=False)
    fe = torch.cat([pe, te, qm.model.embed_tokens(torch.tensor([tgt_t], device=dev))], dim=1)
    o = qm(inputs_embeds=fe)
    sp = pe.shape[1] + ti.shape[1] - 1
    return sum(F.cross_entropy(o.logits[0, sp + i, :].unsqueeze(0), torch.tensor([t]).to(dev)) for i, t in enumerate(tgt_t)) / len(tgt_t)
def evl(z, ex):
    with torch.no_grad():
        t = dec_t(z)
        pe = bridge(geth(z, t))
        r = []
        for inp, tgt in ex:
            ti = qt(f"{inp} ", return_tensors="pt").input_ids.to(dev)
            fe = torch.cat([pe, qm.model.embed_tokens(ti)], dim=1)
            o = qm(inputs_embeds=fe)
            g = []
            for _ in range(5):
                nt = o.logits[0, -1, :].argmax().item()
                g.append(nt)
                if nt == qt.eos_token_id: break
                fe = torch.cat([fe, qm.model.embed_tokens(torch.tensor([[nt]], device=dev))], dim=1)
                o = qm(inputs_embeds=fe)
            pt = qt.decode(g).strip().lower()
            fw = ''.join(c for c in pt.split()[0] if c.isalpha()) if pt.split() else ''
            r.append((inp, tgt, fw, fw == tgt.lower()))
        return r

ex = [("hot ->", "cold"), ("big ->", "small"), ("fast ->", "slow"), ("up ->", "down"), ("happy ->", "sad"), ("light ->", "dark")]
seeds = ["Find the opposite word.", "Opposite: hot is cold.", "Task: output antonyms."]

for s in seeds:
    print(f"Seed: '{s}'", flush=True)
    with torch.no_grad(): zi = enc(s)
    z = nn.Parameter(zi.clone())
    opt = torch.optim.Adam([z], lr=0.0005)
    ba, bs = 0, 0
    for st in range(25):
        opt.zero_grad()
        with torch.no_grad(): t = dec_t(z)
        pe = bridge(geth(z, t))
        l = sum(loss(pe, i, t) for i, t in ex) / len(ex)
        l.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)
        opt.step()
        with torch.no_grad():
            er = evl(z, ex)
            c = sum(1 for _, _, _, x in er if x)
            if c > ba: ba, bs = c, st
    er = evl(z, ex)
    print(f"  Best: {ba}/6 at step {bs}", flush=True)
    for inp, tgt, pred, c in er:
        print(f"    {inp} -> '{pred}' ({tgt}) {'✓' if c else '✗'}", flush=True)
    print(flush=True)
