"""
Full SONAR optimization experiment with soft and hard evaluation.

Soft eval: SONAR hidden states → bridge → Qwen embeddings (training mode)
Hard eval: Decode z → string → Qwen tokenizer → Qwen (deployment mode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout
from dataclasses import dataclass
from typing import List, Tuple
import json
from datetime import datetime


@dataclass
class Result:
    step: int
    loss: float
    decoded: str
    soft_correct: int
    hard_correct: int
    soft_preds: List[Tuple[str, str, str]]  # (input, target, prediction)
    hard_preds: List[Tuple[str, str, str]]


class SONAROptimizer:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading SONAR encoder...", flush=True)
        self.sonar_encoder = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder',
            tokenizer='text_sonar_basic_encoder'
        )
        print("Loading SONAR decoder...", flush=True)
        self.sonar_decoder = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder',
            tokenizer='text_sonar_basic_encoder'
        )
        print("Loading Qwen...", flush=True)
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        ).to(device)
        self.qwen_model.eval()

        print("Loading alignment...", flush=True)
        alignment = np.load("results/nllb_qwen_alignment.npz")
        self.W = torch.tensor(alignment['W'], dtype=torch.float32, device=device)
        self.src_mean = torch.tensor(alignment['src_mean'], dtype=torch.float32, device=device)
        self.tgt_mean = torch.tensor(alignment['tgt_mean'], dtype=torch.float32, device=device)

        self.sonar_decoder_model = self.sonar_decoder.model.to(device)
        self.tok_decoder = self.sonar_decoder.tokenizer.create_decoder()
        print("Ready.\n", flush=True)

    def encode(self, text: str) -> torch.Tensor:
        return self.sonar_encoder.predict([text], source_lang='eng_Latn').to(self.device)

    def decode_to_tokens(self, emb: torch.Tensor, max_len: int = 60) -> torch.Tensor:
        """Decode embedding to SONAR tokens."""
        emb = emb.detach()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        encoder_output = emb.unsqueeze(1)
        enc_layout = BatchLayout.of(encoder_output)

        generated = [3, 256047]  # BOS + lang tag
        for _ in range(max_len):
            decoder_input = torch.tensor([generated], device=self.device)
            seqs_layout = BatchLayout.of(decoder_input)
            hidden = self.sonar_decoder_model.decode(
                decoder_input, seqs_layout, encoder_output, enc_layout
            )
            logits = self.sonar_decoder_model.decoder.final_proj(hidden)
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)
            if next_token == 3:  # EOS
                break
        return torch.tensor(generated, device=self.device)

    def decode_to_text(self, emb: torch.Tensor) -> str:
        """Decode embedding to text string."""
        tokens = self.decode_to_tokens(emb)
        text_tokens = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
        return self.tok_decoder(text_tokens.cpu()) if len(text_tokens) > 0 else ""

    def get_decoder_hidden(self, emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Get SONAR decoder hidden states (differentiable)."""
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        decoder_input = tokens[:-1].unsqueeze(0)
        encoder_output = emb.unsqueeze(1)
        seqs_layout = BatchLayout.of(decoder_input)
        enc_layout = BatchLayout.of(encoder_output)
        return self.sonar_decoder_model.decode(
            decoder_input, seqs_layout, encoder_output, enc_layout
        )

    def bridge(self, sonar_hidden: torch.Tensor) -> torch.Tensor:
        """Map SONAR hidden states to Qwen embedding space."""
        return (sonar_hidden - self.src_mean) @ self.W + self.tgt_mean

    def compute_loss(self, prefix_embeds: torch.Tensor, input_text: str, target: str) -> torch.Tensor:
        """Compute NLL loss for target prediction (soft prefix)."""
        task_text = f"{input_text} "
        task_tokens = self.qwen_tokenizer(task_text, return_tensors="pt").input_ids.to(self.device)
        task_embeds = self.qwen_model.model.embed_tokens(task_tokens)

        target_tokens = self.qwen_tokenizer.encode(target, add_special_tokens=False)
        target_embeds = self.qwen_model.model.embed_tokens(
            torch.tensor([target_tokens], device=self.device)
        )

        full_embeds = torch.cat([prefix_embeds, task_embeds, target_embeds], dim=1)
        outputs = self.qwen_model(inputs_embeds=full_embeds)

        prefix_len = prefix_embeds.shape[1]
        task_len = task_tokens.shape[1]
        start_pos = prefix_len + task_len - 1

        total_loss = sum(
            F.cross_entropy(
                outputs.logits[0, start_pos + i, :].unsqueeze(0),
                torch.tensor([t]).to(self.device)
            )
            for i, t in enumerate(target_tokens)
        )
        return total_loss / len(target_tokens)

    def evaluate_soft(self, z: torch.Tensor, examples: List[Tuple[str, str]]) -> Tuple[int, List]:
        """
        Soft evaluation: SONAR hidden → bridge → Qwen embeddings.
        This is what we use during training.
        """
        with torch.no_grad():
            tokens = self.decode_to_tokens(z)
            hidden = self.get_decoder_hidden(z, tokens)
            prefix_embeds = self.bridge(hidden)

            results = []
            correct = 0
            for input_text, target in examples:
                task_text = f"{input_text} "
                task_tokens = self.qwen_tokenizer(task_text, return_tensors="pt").input_ids.to(self.device)
                task_embeds = self.qwen_model.model.embed_tokens(task_tokens)
                full_embeds = torch.cat([prefix_embeds, task_embeds], dim=1)

                # Generate
                outputs = self.qwen_model(inputs_embeds=full_embeds)
                generated = []
                for _ in range(5):
                    next_token = outputs.logits[0, -1, :].argmax().item()
                    generated.append(next_token)
                    if next_token == self.qwen_tokenizer.eos_token_id:
                        break
                    next_embed = self.qwen_model.model.embed_tokens(
                        torch.tensor([[next_token]], device=self.device)
                    )
                    full_embeds = torch.cat([full_embeds, next_embed], dim=1)
                    outputs = self.qwen_model(inputs_embeds=full_embeds)

                pred_text = self.qwen_tokenizer.decode(generated).strip().lower()
                first_word = ''.join(c for c in pred_text.split()[0] if c.isalpha()) if pred_text.split() else ''
                is_correct = first_word == target.lower()
                correct += is_correct
                results.append((input_text, target, first_word))

            return correct, results

    def evaluate_hard(self, prompt_text: str, examples: List[Tuple[str, str]]) -> Tuple[int, List]:
        """
        Hard evaluation: Use decoded text as actual string prompt.
        This is what we'd use at deployment time.
        """
        with torch.no_grad():
            results = []
            correct = 0
            for input_text, target in examples:
                # Construct full prompt as string
                full_prompt = f"{prompt_text}\n{input_text} "

                # Tokenize with Qwen tokenizer
                input_ids = self.qwen_tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)

                # Generate
                outputs = self.qwen_model(input_ids=input_ids)
                generated = []
                for _ in range(5):
                    next_token = outputs.logits[0, -1, :].argmax().item()
                    generated.append(next_token)
                    if next_token == self.qwen_tokenizer.eos_token_id:
                        break
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)
                    outputs = self.qwen_model(input_ids=input_ids)

                pred_text = self.qwen_tokenizer.decode(generated).strip().lower()
                first_word = ''.join(c for c in pred_text.split()[0] if c.isalpha()) if pred_text.split() else ''
                is_correct = first_word == target.lower()
                correct += is_correct
                results.append((input_text, target, first_word))

            return correct, results

    def optimize(
        self,
        seed_text: str,
        examples: List[Tuple[str, str]],
        num_steps: int = 100,
        lr: float = 0.0005,
        eval_every: int = 10,
        grad_clip: float = 0.1,
    ) -> List[Result]:
        """Run optimization and return trajectory."""

        print(f"Seed: '{seed_text}'", flush=True)
        print(f"Steps: {num_steps}, LR: {lr}", flush=True)
        print("-" * 70, flush=True)

        with torch.no_grad():
            z_init = self.encode(seed_text)

        z = nn.Parameter(z_init.clone())
        optimizer = torch.optim.Adam([z], lr=lr)

        trajectory = []
        best_hard_acc = 0
        best_hard_prompt = seed_text

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward pass
            with torch.no_grad():
                tokens = self.decode_to_tokens(z)
            hidden = self.get_decoder_hidden(z, tokens)
            prefix_embeds = self.bridge(hidden)

            # Compute loss
            total_loss = sum(
                self.compute_loss(prefix_embeds, inp, tgt)
                for inp, tgt in examples
            ) / len(examples)

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
            optimizer.step()

            # Evaluate periodically
            if step % eval_every == 0 or step == num_steps - 1:
                with torch.no_grad():
                    decoded = self.decode_to_text(z)
                    soft_correct, soft_preds = self.evaluate_soft(z, examples)
                    hard_correct, hard_preds = self.evaluate_hard(decoded, examples)

                result = Result(
                    step=step,
                    loss=total_loss.item(),
                    decoded=decoded,
                    soft_correct=soft_correct,
                    hard_correct=hard_correct,
                    soft_preds=soft_preds,
                    hard_preds=hard_preds,
                )
                trajectory.append(result)

                # Track best hard accuracy
                if hard_correct > best_hard_acc:
                    best_hard_acc = hard_correct
                    best_hard_prompt = decoded

                # Print progress
                soft_str = f"soft={soft_correct}/{len(examples)}"
                hard_str = f"hard={hard_correct}/{len(examples)}"
                print(f"Step {step:3d}: loss={total_loss.item():.2f}, {soft_str}, {hard_str}", flush=True)
                print(f"         Decoded: '{decoded[:60]}{'...' if len(decoded) > 60 else ''}'", flush=True)

                # Show predictions if there's a difference
                if soft_correct != hard_correct:
                    print(f"         Soft preds: {[(p[0], p[2]) for p in soft_preds]}", flush=True)
                    print(f"         Hard preds: {[(p[0], p[2]) for p in hard_preds]}", flush=True)

        print("-" * 70, flush=True)
        print(f"Best hard prompt ({best_hard_acc}/{len(examples)}): '{best_hard_prompt}'", flush=True)

        return trajectory, best_hard_prompt, best_hard_acc


def main():
    print("=" * 70, flush=True)
    print("SONAR Full Experiment - Soft vs Hard Evaluation", flush=True)
    print("=" * 70, flush=True)

    examples = [
        ("hot ->", "cold"),
        ("big ->", "small"),
        ("fast ->", "slow"),
        ("up ->", "down"),
        ("happy ->", "sad"),
        ("light ->", "dark"),
    ]

    opt = SONAROptimizer()

    # First, evaluate baseline (no prompt)
    print("\n" + "=" * 70, flush=True)
    print("BASELINE (no prompt)", flush=True)
    print("=" * 70, flush=True)
    baseline_correct, baseline_preds = opt.evaluate_hard("", examples)
    print(f"Accuracy: {baseline_correct}/{len(examples)}", flush=True)
    for inp, tgt, pred in baseline_preds:
        sym = '✓' if pred == tgt.lower() else '✗'
        print(f"  {inp} -> '{pred}' ({tgt}) {sym}", flush=True)

    # Seeds to try
    seeds = [
        "Antonyms: words with opposite meanings.",
        "Give the opposite word for each input.",
        "Find the antonym: hot becomes cold, big becomes small.",
    ]

    all_results = []

    for seed in seeds:
        print("\n" + "=" * 70, flush=True)
        trajectory, best_prompt, best_acc = opt.optimize(
            seed_text=seed,
            examples=examples,
            num_steps=80,
            lr=0.0005,
            eval_every=5,
        )
        all_results.append({
            'seed': seed,
            'best_prompt': best_prompt,
            'best_hard_acc': best_acc,
            'trajectory': [(r.step, r.loss, r.decoded, r.soft_correct, r.hard_correct)
                          for r in trajectory]
        })

    # Final summary
    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Baseline: {baseline_correct}/{len(examples)}", flush=True)

    for res in all_results:
        print(f"\nSeed: '{res['seed'][:40]}...'", flush=True)
        print(f"  Best hard acc: {res['best_hard_acc']}/{len(examples)}", flush=True)
        print(f"  Best prompt: '{res['best_prompt']}'", flush=True)

    # Find overall best
    best = max(all_results, key=lambda x: x['best_hard_acc'])
    print(f"\n** OVERALL BEST **", flush=True)
    print(f"Prompt: '{best['best_prompt']}'", flush=True)
    print(f"Hard accuracy: {best['best_hard_acc']}/{len(examples)}", flush=True)

    # Final evaluation of best prompt
    print(f"\nFinal evaluation of best prompt:", flush=True)
    final_correct, final_preds = opt.evaluate_hard(best['best_prompt'], examples)
    for inp, tgt, pred in final_preds:
        sym = '✓' if pred == tgt.lower() else '✗'
        print(f"  {inp} -> '{pred}' ({tgt}) {sym}", flush=True)


if __name__ == "__main__":
    main()
