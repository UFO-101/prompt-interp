"""
Batched SONAR optimization experiments with hyperparameter sweep.

Runs multiple seeds in parallel and tests different hyperparameter configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

# SONAR imports
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout


@dataclass
class HyperParams:
    lr: float
    grad_clip: float
    z_reg: float  # L2 regularization to keep z near original
    num_steps: int
    name: str


class BatchedSONAROptimizer:
    """Batched SONAR optimization for multiple seeds in parallel."""

    def __init__(self, device: str = "cuda"):
        self.device = device

        print("Loading SONAR encoder...")
        self.sonar_encoder = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder',
            tokenizer='text_sonar_basic_encoder'
        )

        print("Loading SONAR decoder...")
        self.sonar_decoder = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder',
            tokenizer='text_sonar_basic_encoder'
        )

        print("Loading Qwen model...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        ).to(device)
        self.qwen_model.eval()

        print("Loading alignment...")
        alignment = np.load("results/nllb_qwen_alignment.npz")
        self.W = torch.tensor(alignment['W'], dtype=torch.float32, device=device)
        self.src_mean = torch.tensor(alignment['src_mean'], dtype=torch.float32, device=device)
        self.tgt_mean = torch.tensor(alignment['tgt_mean'], dtype=torch.float32, device=device)

        self.sonar_decoder_model = self.sonar_decoder.model.to(device)
        self.tok_decoder = self.sonar_decoder.tokenizer.create_decoder()

    def encode_sentences(self, texts: list[str]) -> torch.Tensor:
        """Encode multiple texts to SONAR embeddings."""
        emb = self.sonar_encoder.predict(texts, source_lang='eng_Latn')
        return emb.to(self.device)

    def decode_single_embedding(self, embedding: torch.Tensor, max_len: int = 50) -> tuple[torch.Tensor, str]:
        """Decode one embedding to tokens and text."""
        emb = embedding.detach()
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        encoder_output = emb.unsqueeze(1)
        enc_layout = BatchLayout.of(encoder_output)

        bos_token, lang_token, eos_token = 3, 256047, 3
        generated = [bos_token, lang_token]

        for _ in range(max_len):
            decoder_input = torch.tensor([generated], device=self.device)
            seqs_layout = BatchLayout.of(decoder_input)
            hidden = self.sonar_decoder_model.decode(decoder_input, seqs_layout, encoder_output, enc_layout)
            logits = self.sonar_decoder_model.decoder.final_proj(hidden)
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)
            if next_token == eos_token:
                break

        tokens = torch.tensor(generated, device=self.device)
        text_tokens = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
        text = self.tok_decoder(text_tokens.cpu()) if len(text_tokens) > 0 else ""
        return tokens, text

    def get_decoder_hidden_single(self, embedding: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Get decoder hidden states for single embedding."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        decoder_input = tokens[:-1].unsqueeze(0)
        encoder_output = embedding.unsqueeze(1)
        seqs_layout = BatchLayout.of(decoder_input)
        enc_layout = BatchLayout.of(encoder_output)
        return self.sonar_decoder_model.decode(decoder_input, seqs_layout, encoder_output, enc_layout)

    def bridge(self, sonar_hidden: torch.Tensor) -> torch.Tensor:
        """Map SONAR hidden states to Qwen embedding space."""
        centered = sonar_hidden - self.src_mean
        transformed = centered @ self.W
        return transformed + self.tgt_mean

    def compute_task_loss(self, prefix_embeds: torch.Tensor, input_text: str, target: str) -> torch.Tensor:
        """Compute NLL loss for predicting target."""
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

        total_loss = 0
        for i, target_tok in enumerate(target_tokens):
            logits = outputs.logits[0, start_pos + i, :]
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_tok]).to(self.device))
            total_loss += loss

        return total_loss / len(target_tokens)

    def evaluate(self, z: torch.Tensor, examples: list) -> tuple[int, list]:
        """Evaluate accuracy with optimized embedding."""
        with torch.no_grad():
            tokens, decoded = self.decode_single_embedding(z)
            hidden = self.get_decoder_hidden_single(z, tokens)
            prefix_embeds = self.bridge(hidden)

            correct = 0
            results = []
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
                    next_embed = self.qwen_model.model.embed_tokens(torch.tensor([[next_token]], device=self.device))
                    full_embeds = torch.cat([full_embeds, next_embed], dim=1)
                    outputs = self.qwen_model(inputs_embeds=full_embeds)

                pred_text = self.qwen_tokenizer.decode(generated).strip().lower()
                first_word = ''.join(c for c in pred_text.split()[0] if c.isalpha()) if pred_text.split() else ''
                is_correct = first_word == target.lower()
                correct += is_correct
                results.append((input_text, target, first_word, pred_text, is_correct))

            return correct, results, decoded

    def optimize_single(
        self,
        seed_text: str,
        examples: list,
        hp: HyperParams,
        seed_idx: int = 0
    ) -> dict:
        """Optimize a single seed with given hyperparameters."""
        with torch.no_grad():
            z_init = self.encode_sentences([seed_text])
            z_init_norm = z_init.norm().item()

        z = nn.Parameter(z_init.clone())
        optimizer = torch.optim.Adam([z], lr=hp.lr)

        history = []
        best_loss = float('inf')
        best_z = z.data.clone()
        best_decoded = ""

        for step in range(hp.num_steps):
            optimizer.zero_grad()

            with torch.no_grad():
                tokens, decoded = self.decode_single_embedding(z)

            hidden = self.get_decoder_hidden_single(z, tokens)
            prefix_embeds = self.bridge(hidden)

            # Task loss
            total_loss = 0
            for input_seq, target in examples:
                loss = self.compute_task_loss(prefix_embeds, input_seq, target)
                total_loss += loss
            avg_loss = total_loss / len(examples)

            # Z regularization - penalize drift from init
            if hp.z_reg > 0:
                z_drift = (z - z_init.detach()).pow(2).sum()
                avg_loss = avg_loss + hp.z_reg * z_drift

            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=hp.grad_clip)
            optimizer.step()

            loss_val = avg_loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_z = z.data.clone()
                best_decoded = decoded

            # Log periodically
            if step % 10 == 0 or step == hp.num_steps - 1:
                history.append({
                    'step': step,
                    'loss': loss_val,
                    'z_norm': z.data.norm().item(),
                    'decoded': decoded[:80]
                })

        # Final evaluation
        correct, eval_results, final_decoded = self.evaluate(best_z, examples)

        return {
            'seed_idx': seed_idx,
            'seed_text': seed_text,
            'hp': hp.name,
            'lr': hp.lr,
            'grad_clip': hp.grad_clip,
            'z_reg': hp.z_reg,
            'best_loss': best_loss,
            'best_decoded': best_decoded,
            'final_decoded': final_decoded,
            'accuracy': correct / len(examples),
            'correct': correct,
            'total': len(examples),
            'eval_results': eval_results,
            'history': history,
            'z_init_norm': z_init_norm
        }


def run_experiments():
    print("=" * 70)
    print("Batched SONAR Optimization Experiments")
    print("=" * 70)

    # Task: Antonyms
    examples = [
        ("hot ->", "cold"),
        ("big ->", "small"),
        ("fast ->", "slow"),
        ("up ->", "down"),
        ("happy ->", "sad"),
        ("light ->", "dark"),
    ]

    # Fewer seeds for faster experiments
    seeds = [
        "Give the opposite word. For example, hot becomes cold and big becomes small.",
        "Antonyms: words with opposite meanings.",
        "Find the opposite: hot is cold, fast is slow.",
        "Output the antonym of the input word.",
    ]

    # Focused hyperparameter sweep - 50 steps is enough to see trends
    hps = [
        # Varying learning rate
        HyperParams(lr=0.0001, grad_clip=0.1, z_reg=0.0, num_steps=50, name="lr_0.0001"),
        HyperParams(lr=0.0005, grad_clip=0.1, z_reg=0.0, num_steps=50, name="lr_0.0005"),
        HyperParams(lr=0.001, grad_clip=0.1, z_reg=0.0, num_steps=50, name="lr_0.001"),
        HyperParams(lr=0.002, grad_clip=0.1, z_reg=0.0, num_steps=50, name="lr_0.002"),
        # With regularization to keep z coherent
        HyperParams(lr=0.001, grad_clip=0.1, z_reg=0.1, num_steps=50, name="lr_0.001_reg_0.1"),
        HyperParams(lr=0.001, grad_clip=0.1, z_reg=1.0, num_steps=50, name="lr_0.001_reg_1.0"),
        HyperParams(lr=0.002, grad_clip=0.1, z_reg=0.1, num_steps=50, name="lr_0.002_reg_0.1"),
        # Very conservative
        HyperParams(lr=0.0001, grad_clip=0.05, z_reg=0.1, num_steps=50, name="conservative"),
    ]

    print(f"\nRunning {len(seeds)} seeds × {len(hps)} hyperparameter configs = {len(seeds) * len(hps)} experiments\n")

    opt = BatchedSONAROptimizer()

    # First evaluate baseline (no prefix)
    print("\n" + "=" * 70)
    print("BASELINE (no prefix)")
    print("=" * 70)

    baseline_correct = 0
    for input_text, target in examples:
        task_text = f"{input_text} "
        input_ids = opt.qwen_tokenizer(task_text, return_tensors="pt").input_ids.to(opt.device)
        outputs = opt.qwen_model.generate(input_ids, max_new_tokens=5, do_sample=False)
        pred = opt.qwen_tokenizer.decode(outputs[0][input_ids.shape[1]:]).strip().lower()
        first_word = ''.join(c for c in pred.split()[0] if c.isalpha()) if pred.split() else ''
        is_correct = first_word == target.lower()
        baseline_correct += is_correct
        symbol = '✓' if is_correct else '✗'
        print(f"  {input_text} -> '{first_word}' (target: {target}) {symbol}")
    print(f"  Baseline accuracy: {baseline_correct}/{len(examples)} ({100*baseline_correct/len(examples):.1f}%)")

    all_results = []

    for hp in hps:
        print(f"\n{'=' * 70}")
        print(f"Hyperparams: {hp.name} (lr={hp.lr}, clip={hp.grad_clip}, z_reg={hp.z_reg})")
        print("=" * 70)

        hp_results = []
        for i, seed in enumerate(seeds):
            print(f"\n  Seed {i+1}/{len(seeds)}: '{seed[:50]}...'")
            result = opt.optimize_single(seed, examples, hp, seed_idx=i)
            hp_results.append(result)

            print(f"    Best loss: {result['best_loss']:.4f}")
            print(f"    Accuracy: {result['correct']}/{result['total']} ({100*result['accuracy']:.1f}%)")
            print(f"    Best decoded: '{result['best_decoded'][:60]}...'")

            # Show predictions
            for inp, tgt, pred_word, pred_full, is_correct in result['eval_results']:
                symbol = '✓' if is_correct else '✗'
                print(f"      {inp} -> '{pred_word}' (target: {tgt}) {symbol}")

        all_results.extend(hp_results)

        # Summary for this HP config
        accuracies = [r['accuracy'] for r in hp_results]
        losses = [r['best_loss'] for r in hp_results]
        print(f"\n  {hp.name} summary:")
        print(f"    Mean accuracy: {np.mean(accuracies)*100:.1f}% ± {np.std(accuracies)*100:.1f}%")
        print(f"    Mean best loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        print(f"    Best accuracy: {max(accuracies)*100:.1f}%")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    # Group by HP
    for hp in hps:
        hp_results = [r for r in all_results if r['hp'] == hp.name]
        accuracies = [r['accuracy'] for r in hp_results]
        print(f"  {hp.name}: {np.mean(accuracies)*100:.1f}% ± {np.std(accuracies)*100:.1f}% (best: {max(accuracies)*100:.1f}%)")

    # Best overall
    best_result = max(all_results, key=lambda r: (r['accuracy'], -r['best_loss']))
    print(f"\n  Best overall: {best_result['hp']} on seed {best_result['seed_idx']}")
    print(f"    Accuracy: {best_result['correct']}/{best_result['total']} ({100*best_result['accuracy']:.1f}%)")
    print(f"    Best loss: {best_result['best_loss']:.4f}")
    print(f"    Decoded: '{best_result['best_decoded']}'")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"results/sonar_experiments_{timestamp}.json")

    # Convert results for JSON serialization
    json_results = []
    for r in all_results:
        jr = {k: v for k, v in r.items() if k != 'eval_results'}
        jr['eval_results'] = [(inp, tgt, pred, full, int(correct)) for inp, tgt, pred, full, correct in r['eval_results']]
        json_results.append(jr)

    with open(results_path, 'w') as f:
        json.dump({
            'baseline_accuracy': baseline_correct / len(examples),
            'examples': examples,
            'seeds': seeds,
            'hyperparams': [{'name': hp.name, 'lr': hp.lr, 'grad_clip': hp.grad_clip, 'z_reg': hp.z_reg} for hp in hps],
            'results': json_results
        }, f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    run_experiments()
