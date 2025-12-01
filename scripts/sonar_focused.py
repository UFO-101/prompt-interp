"""
Focused SONAR experiment - just the best configs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout


class Optimizer:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading...", flush=True)
        self.sonar_encoder = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
        self.sonar_decoder = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        self.qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(device)
        self.qwen_model.eval()
        alignment = np.load("results/nllb_qwen_alignment.npz")
        self.W = torch.tensor(alignment['W'], dtype=torch.float32, device=device)
        self.src_mean = torch.tensor(alignment['src_mean'], dtype=torch.float32, device=device)
        self.tgt_mean = torch.tensor(alignment['tgt_mean'], dtype=torch.float32, device=device)
        self.sonar_decoder_model = self.sonar_decoder.model.to(device)
        self.tok_decoder = self.sonar_decoder.tokenizer.create_decoder()
        print("Ready.", flush=True)

    def encode(self, text):
        return self.sonar_encoder.predict([text], source_lang='eng_Latn').to(self.device)

    def decode_tokens(self, emb, max_len=40):
        emb = emb.detach()
        if emb.dim() == 1: emb = emb.unsqueeze(0)
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
            if next_token == 3: break
        return torch.tensor(generated, device=self.device)

    def decode_text(self, emb):
        tokens = self.decode_tokens(emb)
        text_tokens = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
        return self.tok_decoder(text_tokens.cpu()) if len(text_tokens) > 0 else ""

    def get_hidden(self, emb, tokens):
        if emb.dim() == 1: emb = emb.unsqueeze(0)
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
        total_loss = sum(F.cross_entropy(outputs.logits[0, start_pos + i, :].unsqueeze(0),
                        torch.tensor([t]).to(self.device)) for i, t in enumerate(target_tokens))
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
                    if next_token == self.qwen_tokenizer.eos_token_id: break
                    next_embed = self.qwen_model.model.embed_tokens(torch.tensor([[next_token]], device=self.device))
                    full_embeds = torch.cat([full_embeds, next_embed], dim=1)
                    outputs = self.qwen_model(inputs_embeds=full_embeds)
                pred_text = self.qwen_tokenizer.decode(generated).strip().lower()
                first_word = ''.join(c for c in pred_text.split()[0] if c.isalpha()) if pred_text.split() else ''
                results.append((inp, tgt, first_word, pred_text, first_word == tgt.lower()))
            return results


def main():
    print("=" * 60, flush=True)
    print("Focused SONAR Experiment", flush=True)
    print("=" * 60, flush=True)

    examples = [
        ("hot ->", "cold"),
        ("big ->", "small"),
        ("fast ->", "slow"),
        ("up ->", "down"),
        ("happy ->", "sad"),
        ("light ->", "dark"),
    ]

    opt = Optimizer()

    seed = "Antonyms: words with opposite meanings."
    lr = 0.0005
    num_steps = 80

    print(f"\nSeed: '{seed}'", flush=True)
    print(f"Config: lr={lr}, steps={num_steps}", flush=True)

    with torch.no_grad():
        z_init = opt.encode(seed)
    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()
        with torch.no_grad():
            tokens = opt.decode_tokens(z)
        hidden = opt.get_hidden(z, tokens)
        prefix = opt.bridge(hidden)

        loss = sum(opt.compute_loss(prefix, inp, tgt) for inp, tgt in examples) / len(examples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)
        optimizer.step()

        # Log every 5 steps with detailed predictions
        if step % 5 == 0 or step == num_steps - 1:
            with torch.no_grad():
                decoded = opt.decode_text(z)
                eval_results = opt.evaluate(z, examples)
                correct = sum(1 for _, _, _, _, c in eval_results if c)

            print(f"\n--- Step {step} ---", flush=True)
            print(f"Loss: {loss.item():.2f}", flush=True)
            print(f"Decoded: '{decoded[:70]}{'...' if len(decoded) > 70 else ''}'", flush=True)
            print(f"Accuracy: {correct}/6", flush=True)
            for inp, tgt, first_word, full_pred, is_correct in eval_results:
                symbol = '✓' if is_correct else '✗'
                print(f"  {inp} -> '{first_word}' (full: '{full_pred[:20]}') {symbol}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULT", flush=True)
    print("=" * 60, flush=True)
    with torch.no_grad():
        decoded = opt.decode_text(z)
        eval_results = opt.evaluate(z, examples)
        correct = sum(1 for _, _, _, _, c in eval_results if c)
    print(f"Decoded prompt: '{decoded}'", flush=True)
    print(f"Accuracy: {correct}/6 ({100*correct/6:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
