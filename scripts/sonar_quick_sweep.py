"""
Quick SONAR optimization sweep - focused experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout


class QuickOptimizer:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading models...")

        self.sonar_encoder = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder'
        )
        self.sonar_decoder = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder'
        )
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        self.qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(device)
        self.qwen_model.eval()

        alignment = np.load("results/nllb_qwen_alignment.npz")
        self.W = torch.tensor(alignment['W'], dtype=torch.float32, device=device)
        self.src_mean = torch.tensor(alignment['src_mean'], dtype=torch.float32, device=device)
        self.tgt_mean = torch.tensor(alignment['tgt_mean'], dtype=torch.float32, device=device)

        self.sonar_decoder_model = self.sonar_decoder.model.to(device)
        self.tok_decoder = self.sonar_decoder.tokenizer.create_decoder()
        print("Models loaded.")

    def encode(self, text):
        return self.sonar_encoder.predict([text], source_lang='eng_Latn').to(self.device)

    def decode_tokens(self, emb, max_len=40):
        emb = emb.detach()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        encoder_output = emb.unsqueeze(1)
        enc_layout = BatchLayout.of(encoder_output)
        generated = [3, 256047]
        for _ in range(max_len):
            decoder_input = torch.tensor([generated], device=self.device)
            seqs_layout = BatchLayout.of(decoder_input)
            hidden = self.sonar_decoder_model.decode(decoder_input, seqs_layout, encoder_output, enc_layout)
            logits = self.sonar_decoder_model.decoder.final_proj(hidden)
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)
            if next_token == 3:
                break
        return torch.tensor(generated, device=self.device)

    def decode_text(self, emb):
        tokens = self.decode_tokens(emb)
        text_tokens = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
        return self.tok_decoder(text_tokens.cpu()) if len(text_tokens) > 0 else ""

    def get_hidden(self, emb, tokens):
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        decoder_input = tokens[:-1].unsqueeze(0)
        encoder_output = emb.unsqueeze(1)
        seqs_layout = BatchLayout.of(decoder_input)
        enc_layout = BatchLayout.of(encoder_output)
        return self.sonar_decoder_model.decode(decoder_input, seqs_layout, encoder_output, enc_layout)

    def bridge(self, h):
        return (h - self.src_mean) @ self.W + self.tgt_mean

    def compute_loss(self, prefix_embeds, input_text, target):
        task_text = f"{input_text} "
        task_tokens = self.qwen_tokenizer(task_text, return_tensors="pt").input_ids.to(self.device)
        task_embeds = self.qwen_model.model.embed_tokens(task_tokens)
        target_tokens = self.qwen_tokenizer.encode(target, add_special_tokens=False)
        target_embeds = self.qwen_model.model.embed_tokens(torch.tensor([target_tokens], device=self.device))
        full_embeds = torch.cat([prefix_embeds, task_embeds, target_embeds], dim=1)
        outputs = self.qwen_model(inputs_embeds=full_embeds)
        prefix_len = prefix_embeds.shape[1]
        task_len = task_tokens.shape[1]
        start_pos = prefix_len + task_len - 1
        total_loss = sum(
            F.cross_entropy(outputs.logits[0, start_pos + i, :].unsqueeze(0),
                          torch.tensor([t]).to(self.device))
            for i, t in enumerate(target_tokens)
        )
        return total_loss / len(target_tokens)

    def evaluate(self, z, examples):
        with torch.no_grad():
            tokens = self.decode_tokens(z)
            hidden = self.get_hidden(z, tokens)
            prefix_embeds = self.bridge(hidden)
            results = []
            for inp, tgt in examples:
                task_text = f"{inp} "
                task_tokens = self.qwen_tokenizer(task_text, return_tensors="pt").input_ids.to(self.device)
                task_embeds = self.qwen_model.model.embed_tokens(task_tokens)
                full_embeds = torch.cat([prefix_embeds, task_embeds], dim=1)
                outputs = self.qwen_model(inputs_embeds=full_embeds)
                generated = []
                for _ in range(5):
                    next_token = outputs.logits[0, -1, :].argmax().item()
                    generated.append(next_token)
                    if next_token == self.qwen_tokenizer.eos_token_id:
                        break
                    next_embed = self.qwen_model.model.embed_tokens(torch.tensor([[next_token]], device=self.device))
                    full_embeds = torch.cat([full_embeds, next_embed], dim=1)
                    outputs = self.qwen_model(inputs_embeds=full_embeds)
                pred_text = self.qwen_tokenizer.decode(generated).strip().lower()
                first_word = ''.join(c for c in pred_text.split()[0] if c.isalpha()) if pred_text.split() else ''
                results.append((inp, tgt, first_word, first_word == tgt.lower()))
            return results

    def run_one(self, seed, examples, lr, grad_clip, z_reg, num_steps):
        """Run optimization and return trajectory info."""
        with torch.no_grad():
            z_init = self.encode(seed)
        z = nn.Parameter(z_init.clone())
        optimizer = torch.optim.Adam([z], lr=lr)

        trajectory = []
        for step in range(num_steps):
            optimizer.zero_grad()
            with torch.no_grad():
                tokens = self.decode_tokens(z)
            hidden = self.get_hidden(z, tokens)
            prefix = self.bridge(hidden)

            loss = sum(self.compute_loss(prefix, inp, tgt) for inp, tgt in examples) / len(examples)
            if z_reg > 0:
                loss = loss + z_reg * (z - z_init.detach()).pow(2).sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
            optimizer.step()

            # Log every 10 steps
            if step % 10 == 0 or step == num_steps - 1:
                with torch.no_grad():
                    decoded = self.decode_text(z)
                    eval_results = self.evaluate(z, examples)
                    correct = sum(1 for _, _, _, c in eval_results if c)
                trajectory.append({
                    'step': step,
                    'loss': loss.item(),
                    'decoded': decoded,
                    'correct': correct,
                    'predictions': [(inp, tgt, pred) for inp, tgt, pred, _ in eval_results]
                })

        return trajectory


def main():
    print("=" * 70)
    print("Quick SONAR Optimization Sweep")
    print("=" * 70)

    examples = [
        ("hot ->", "cold"),
        ("big ->", "small"),
        ("fast ->", "slow"),
        ("up ->", "down"),
        ("happy ->", "sad"),
        ("light ->", "dark"),
    ]

    # Key seeds
    seeds = [
        "Antonyms: words with opposite meanings.",
        "Find the opposite: hot is cold, fast is slow.",
    ]

    # Key configs to test
    configs = [
        {"lr": 0.0001, "grad_clip": 0.1, "z_reg": 0.0, "name": "very_low_lr"},
        {"lr": 0.0005, "grad_clip": 0.1, "z_reg": 0.0, "name": "low_lr"},
        {"lr": 0.001, "grad_clip": 0.1, "z_reg": 0.0, "name": "med_lr"},
        {"lr": 0.001, "grad_clip": 0.1, "z_reg": 0.5, "name": "med_lr_reg"},
        {"lr": 0.002, "grad_clip": 0.1, "z_reg": 0.5, "name": "high_lr_reg"},
    ]

    opt = QuickOptimizer()

    all_results = []

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED: '{seed}'")
        print("=" * 70)

        for cfg in configs:
            print(f"\n  Config: {cfg['name']} (lr={cfg['lr']}, reg={cfg['z_reg']})")

            trajectory = opt.run_one(
                seed, examples,
                lr=cfg['lr'],
                grad_clip=cfg['grad_clip'],
                z_reg=cfg['z_reg'],
                num_steps=50
            )

            # Show trajectory
            for t in trajectory:
                decoded_short = t['decoded'][:50] + '...' if len(t['decoded']) > 50 else t['decoded']
                print(f"    Step {t['step']:2d}: loss={t['loss']:.2f}, acc={t['correct']}/6, decoded='{decoded_short}'")

            final = trajectory[-1]
            all_results.append({
                'seed': seed,
                'config': cfg['name'],
                'final_loss': final['loss'],
                'final_acc': final['correct'],
                'final_decoded': final['decoded'],
                'predictions': final['predictions'],
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Group by config
    for cfg in configs:
        results = [r for r in all_results if r['config'] == cfg['name']]
        accs = [r['final_acc'] for r in results]
        losses = [r['final_loss'] for r in results]
        print(f"  {cfg['name']}: acc={np.mean(accs):.1f}/6, loss={np.mean(losses):.2f}")

    # Show best
    best = max(all_results, key=lambda r: (r['final_acc'], -r['final_loss']))
    print(f"\n  Best: {best['config']} on '{best['seed'][:40]}...'")
    print(f"    Accuracy: {best['final_acc']}/6")
    print(f"    Loss: {best['final_loss']:.2f}")
    print(f"    Decoded: '{best['final_decoded']}'")
    print(f"    Predictions:")
    for inp, tgt, pred in best['predictions']:
        symbol = '✓' if pred == tgt.lower() else '✗'
        print(f"      {inp} -> '{pred}' (target: {tgt}) {symbol}")


if __name__ == "__main__":
    main()
