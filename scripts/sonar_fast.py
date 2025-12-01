"""Fast multi-seed test - 25 steps per seed."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout

class Opt:
    def __init__(self, device="cuda"):
        self.device = device
        self.sonar_encoder = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
        self.sonar_decoder = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
        self.qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        self.qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(device)
        self.qwen.eval()
        a = np.load("results/nllb_qwen_alignment.npz")
        self.W = torch.tensor(a['W'], dtype=torch.float32, device=device)
        self.src_mean = torch.tensor(a['src_mean'], dtype=torch.float32, device=device)
        self.tgt_mean = torch.tensor(a['tgt_mean'], dtype=torch.float32, device=device)
        self.sonar_dec = self.sonar_decoder.model.to(device)
        self.tok_dec = self.sonar_decoder.tokenizer.create_decoder()

    def encode(self, t): return self.sonar_encoder.predict([t], source_lang='eng_Latn').to(self.device)

    def decode_tokens(self, e, max_len=40):
        e = e.detach()
        if e.dim() == 1: e = e.unsqueeze(0)
        eo = e.unsqueeze(1)
        el = BatchLayout.of(eo)
        g = [3, 256047]
        for _ in range(max_len):
            di = torch.tensor([g], device=self.device)
            sl = BatchLayout.of(di)
            h = self.sonar_dec.decode(di, sl, eo, el)
            lo = self.sonar_dec.decoder.final_proj(h)
            nt = lo[0, -1, :].argmax().item()
            g.append(nt)
            if nt == 3: break
        return torch.tensor(g, device=self.device)

    def decode_text(self, e):
        t = self.decode_tokens(e)
        tt = t[2:-1] if t[-1] == 3 else t[2:]
        return self.tok_dec(tt.cpu()) if len(tt) > 0 else ""

    def get_h(self, e, t):
        if e.dim() == 1: e = e.unsqueeze(0)
        di = t[:-1].unsqueeze(0)
        eo = e.unsqueeze(1)
        return self.sonar_dec.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))

    def bridge(self, h): return (h - self.src_mean) @ self.W + self.tgt_mean

    def loss(self, pe, inp, tgt):
        tt = f"{inp} "
        ti = self.qwen_tok(tt, return_tensors="pt").input_ids.to(self.device)
        te = self.qwen.model.embed_tokens(ti)
        tgt_t = self.qwen_tok.encode(tgt, add_special_tokens=False)
        tgt_e = self.qwen.model.embed_tokens(torch.tensor([tgt_t], device=self.device))
        fe = torch.cat([pe, te, tgt_e], dim=1)
        o = self.qwen(inputs_embeds=fe)
        pl, tl = pe.shape[1], ti.shape[1]
        sp = pl + tl - 1
        return sum(F.cross_entropy(o.logits[0, sp + i, :].unsqueeze(0), torch.tensor([t]).to(self.device)) for i, t in enumerate(tgt_t)) / len(tgt_t)

    def evaluate(self, z, ex):
        with torch.no_grad():
            t = self.decode_tokens(z)
            h = self.get_h(z, t)
            pe = self.bridge(h)
            r = []
            for inp, tgt in ex:
                tt = f"{inp} "
                ti = self.qwen_tok(tt, return_tensors="pt").input_ids.to(self.device)
                te = self.qwen.model.embed_tokens(ti)
                fe = torch.cat([pe, te], dim=1)
                o = self.qwen(inputs_embeds=fe)
                g = []
                for _ in range(5):
                    nt = o.logits[0, -1, :].argmax().item()
                    g.append(nt)
                    if nt == self.qwen_tok.eos_token_id: break
                    ne = self.qwen.model.embed_tokens(torch.tensor([[nt]], device=self.device))
                    fe = torch.cat([fe, ne], dim=1)
                    o = self.qwen(inputs_embeds=fe)
                pt = self.qwen_tok.decode(g).strip().lower()
                fw = ''.join(c for c in pt.split()[0] if c.isalpha()) if pt.split() else ''
                r.append((inp, tgt, fw, fw == tgt.lower()))
            return r

    def run(self, seed, ex, lr=0.0005, steps=25):
        with torch.no_grad(): zi = self.encode(seed)
        z = nn.Parameter(zi.clone())
        opt = torch.optim.Adam([z], lr=lr)
        best_acc, best_step = 0, 0
        for s in range(steps):
            opt.zero_grad()
            with torch.no_grad(): t = self.decode_tokens(z)
            h = self.get_h(z, t)
            pe = self.bridge(h)
            l = sum(self.loss(pe, i, t) for i, t in ex) / len(ex)
            l.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)
            opt.step()
            with torch.no_grad():
                er = self.evaluate(z, ex)
                c = sum(1 for _, _, _, x in er if x)
                if c > best_acc: best_acc, best_step = c, s
        return best_acc, best_step, self.evaluate(z, ex), self.decode_text(z)

def main():
    ex = [("hot ->", "cold"), ("big ->", "small"), ("fast ->", "slow"), ("up ->", "down"), ("happy ->", "sad"), ("light ->", "dark")]
    seeds = [
        "Antonyms: words with opposite meanings.",
        "Find the opposite: hot is cold, fast is slow.",
        "Output the antonym of the input word.",
    ]
    print("Loading...", flush=True)
    o = Opt()
    print("Ready.\n", flush=True)
    results = []
    for i, s in enumerate(seeds):
        ba, bs, er, fd = o.run(s, ex)
        results.append((s, ba, bs, er))
        print(f"Seed {i+1}: '{s[:40]}...'", flush=True)
        print(f"  Best: {ba}/6 at step {bs}", flush=True)
        for inp, tgt, pred, c in er:
            sym = '✓' if c else '✗'
            print(f"    {inp} -> '{pred}' ({tgt}) {sym}", flush=True)
        print(flush=True)
    print("=" * 50, flush=True)
    best = max(results, key=lambda x: x[1])
    print(f"Best: '{best[0][:40]}' -> {best[1]}/6", flush=True)

if __name__ == "__main__":
    main()
