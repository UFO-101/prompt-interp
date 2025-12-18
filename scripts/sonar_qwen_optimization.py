#!/usr/bin/env python3
"""
SONAR-Qwen Prompt Optimization v2

Approach:
1. Start with z in SONAR embedding space
2. Decode autoregressively from SONAR decoder (no grads) to get discrete tokens
3. For gradient flow: use soft logits from SONAR decoder to create soft embeddings
4. Pass soft embeddings through Qwen with task appended
5. Backprop loss to z via the soft decoder logits

Key trick: Use Gumbel-softmax or straight-through to make decoding differentiable.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)
from fairseq2.nn.batch_layout import BatchLayout


# =============================================================================
# Task Definition
# =============================================================================

ANTONYMS_TRAIN = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
    ("good -> ", "bad"),
    ("old -> ", "young"),
    ("high -> ", "low"),
    ("long -> ", "short"),
]

ANTONYMS_TEST = [
    ("wet -> ", "dry"),
    ("loud -> ", "quiet"),
    ("rich -> ", "poor"),
    ("full -> ", "empty"),
    ("strong -> ", "weak"),
]


# =============================================================================
# Vocabulary Mapping
# =============================================================================

def build_vocab_mapping(sonar_decoder_tokenizer, qwen_tokenizer, max_sonar_id=50000):
    """Build mapping of overlapping tokens between SONAR and Qwen."""
    print("Building vocabulary mapping...")

    sonar_decode = sonar_decoder_tokenizer.create_decoder()

    # Map SONAR token IDs to their text
    sonar_to_text = {}
    for tok_id in range(4, max_sonar_id):  # Skip special tokens 0-3
        try:
            text = sonar_decode(torch.tensor([tok_id]))
            if text:
                sonar_to_text[tok_id] = text
        except:
            pass

    # Create SONAR -> Qwen token mapping
    sonar_to_qwen = {}
    for sonar_id, text in sonar_to_text.items():
        # Encode the text with Qwen
        qwen_tokens = qwen_tokenizer.encode(text, add_special_tokens=False)
        if len(qwen_tokens) == 1:
            # Perfect 1:1 mapping
            sonar_to_qwen[sonar_id] = qwen_tokens[0]

    print(f"  SONAR tokens decoded: {len(sonar_to_text)}")
    print(f"  1:1 mapping with Qwen: {len(sonar_to_qwen)}")

    return sonar_to_qwen, sonar_to_text


# =============================================================================
# SONAR-Qwen Optimizer
# =============================================================================

class SonarQwenOptimizer:
    """Optimizes prompts using SONAR embeddings and Qwen evaluation."""

    def __init__(self, device="cuda"):
        self.device = device

        print("Loading SONAR models...")
        self.encoder_pipeline = TextToEmbeddingModelPipeline(
            encoder='text_sonar_basic_encoder',
            tokenizer='text_sonar_basic_encoder'
        )
        self.decoder_pipeline = EmbeddingToTextModelPipeline(
            decoder='text_sonar_basic_decoder',
            tokenizer='text_sonar_basic_encoder'
        )

        # Get SONAR decoder model for manual decoding
        self.sonar_decoder = self.decoder_pipeline.model.to(device)
        self.sonar_tokenizer = self.decoder_pipeline.tokenizer
        self.sonar_text_decoder = self.sonar_tokenizer.create_decoder()

        # SONAR embedding layer for vocab constraint
        self.sonar_embed = self.sonar_decoder.decoder.decoder_frontend.embed
        self.sonar_vocab_size = self.sonar_embed.num_embeddings

        print("Loading Qwen model...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
        )
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)
        self.qwen_embed = self.qwen_model.get_input_embeddings()

        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        # Build vocab mapping
        self.sonar_to_qwen, self.sonar_to_text = build_vocab_mapping(
            self.sonar_tokenizer, self.qwen_tokenizer
        )

        # Create mapping tensors
        self.sonar_ids = torch.tensor(list(self.sonar_to_qwen.keys()), device=device)
        self.qwen_ids = torch.tensor(list(self.sonar_to_qwen.values()), device=device)

        # Create mask for allowed SONAR tokens
        self.allowed_mask = torch.zeros(self.sonar_vocab_size, device=device)
        self.allowed_mask[self.sonar_ids] = 1.0
        self.allowed_mask[3] = 1.0  # Also allow EOS

        # Get target norm from real sentences
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, how are you today?",
            "Opposites: hot is to cold as big is to small.",
        ]
        with torch.no_grad():
            sample_embeds = self.encoder_pipeline.predict(sample_texts, source_lang='eng_Latn')
            self.target_norm = torch.norm(sample_embeds, dim=1).mean().item()
        print(f"Target embedding norm: {self.target_norm:.4f}")

        print("Models loaded.\n")

    def decode_with_gradients(
        self,
        z: torch.Tensor,
        max_len: int = 15,
        temperature: float = 0.5,
    ) -> tuple[list[int], torch.Tensor]:
        """
        Decode from z with gradient-friendly soft outputs.

        Returns:
            tokens: List of hard token IDs (for display)
            soft_qwen_embeds: Soft Qwen embeddings (for gradients)
        """
        # z shape: (1, 1024)
        tokens = [3, 256047]  # BOS=3, eng_Latn lang token
        e = z.unsqueeze(1)  # (1, 1, 1024)

        all_soft_logits = []

        for step in range(max_len):
            di = torch.tensor([tokens], device=self.device)
            h = self.sonar_decoder.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)

            # Get logits for last position
            logits = self.sonar_decoder.decoder.final_proj(h)[0, -1, :]  # (vocab_size,)

            # Mask to allowed tokens (hard constraint)
            masked_logits = logits.clone()
            masked_logits[self.allowed_mask == 0] = float('-inf')

            # Get hard token for next step
            next_tok = masked_logits.argmax().item()
            tokens.append(next_tok)

            if next_tok == 3:  # EOS
                break

            # Store soft logits (with temperature) for gradient flow
            soft = F.softmax(masked_logits / temperature, dim=-1)
            all_soft_logits.append(soft)

        # Convert soft logits over SONAR vocab to soft Qwen embeddings
        # For each position, weighted sum of Qwen embeddings
        if len(all_soft_logits) == 0:
            return tokens[2:], torch.zeros(0, self.qwen_embed.embedding_dim, device=self.device)

        soft_logits = torch.stack(all_soft_logits)  # (seq_len, sonar_vocab)

        # Map to Qwen vocab: for each SONAR token that has a Qwen mapping,
        # contribute its probability * Qwen embedding
        # This is sparse: only ~10k SONAR tokens map to Qwen

        # Get Qwen embeddings for mapped tokens
        qwen_embeds = self.qwen_embed(self.qwen_ids)  # (n_mapped, qwen_dim)

        # Extract probabilities for mapped SONAR tokens
        mapped_probs = soft_logits[:, self.sonar_ids]  # (seq_len, n_mapped)

        # Normalize (the unmapped probability mass is lost)
        mapped_probs = mapped_probs / (mapped_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute soft embeddings
        soft_qwen_embeds = torch.matmul(
            mapped_probs.to(qwen_embeds.dtype),
            qwen_embeds
        )  # (seq_len, qwen_dim)

        # Remove special tokens from hard tokens for display
        content_tokens = tokens[2:]
        if content_tokens and content_tokens[-1] == 3:
            content_tokens = content_tokens[:-1]

        return content_tokens, soft_qwen_embeds

    def get_prompt_text(self, tokens: list[int]) -> str:
        """Convert SONAR tokens to text."""
        if not tokens:
            return ""
        return self.sonar_text_decoder(torch.tensor(tokens))

    def compute_task_loss(
        self,
        soft_prompt_embeds: torch.Tensor,
        examples: list[tuple[str, str]],
    ) -> tuple[torch.Tensor, float, list[tuple[str, str, str]]]:
        """
        Compute task loss with soft prompt embeddings.

        Returns:
            loss: Differentiable loss tensor
            accuracy: Fraction correct (for logging)
            predictions: List of (input, expected, predicted)
        """
        total_loss = torch.tensor(0.0, device=self.device)
        correct = 0
        predictions = []

        for input_text, expected in examples:
            # Tokenize the suffix (input text)
            suffix_ids = self.qwen_tokenizer.encode(
                " " + input_text, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            suffix_embeds = self.qwen_embed(suffix_ids[0])

            # Combine: soft_prompt + suffix
            combined_embeds = torch.cat([
                soft_prompt_embeds.to(suffix_embeds.dtype),
                suffix_embeds
            ], dim=0).unsqueeze(0)  # (1, total_len, dim)

            # Forward through Qwen
            outputs = self.qwen_model(inputs_embeds=combined_embeds)
            logits = outputs.logits[0, -1, :]  # Last position

            # Target token
            target_ids = self.qwen_tokenizer.encode(
                expected, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            target_token = target_ids[0, 0]

            # Cross-entropy loss
            loss = F.cross_entropy(logits.unsqueeze(0).float(), target_token.unsqueeze(0))
            total_loss = total_loss + loss

            # Check accuracy (non-differentiable)
            with torch.no_grad():
                predicted_token = logits.argmax().item()
                predicted_text = self.qwen_tokenizer.decode([predicted_token]).strip()
                if predicted_token == target_token.item():
                    correct += 1
                predictions.append((input_text, expected, predicted_text))

        avg_loss = total_loss / len(examples)
        accuracy = correct / len(examples)

        return avg_loss, accuracy, predictions

    def optimize(
        self,
        train_examples: list[tuple[str, str]],
        test_examples: list[tuple[str, str]],
        n_steps: int = 100,
        lr: float = 0.01,
        seed_text: str = None,
        temperature: float = 0.3,
        anchor_weight: float = 0.0,
    ):
        """Optimize z to find a good prompt."""

        # Initialize z
        if seed_text:
            print(f"Initializing from seed: '{seed_text}'")
            with torch.no_grad():
                z = self.encoder_pipeline.predict([seed_text], source_lang='eng_Latn')
                z = z.to(self.device).float()
                z_anchor = z.clone()  # Keep anchor for regularization
        else:
            print("Initializing with random z")
            z = torch.randn(1, 1024, device=self.device)
            z = z / torch.norm(z) * self.target_norm
            z_anchor = z.clone()

        z = z.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)

        best_z = z.detach().clone()
        best_acc = 0.0
        best_prompt = ""

        print(f"\nStarting optimization (lr={lr}, steps={n_steps}, temp={temperature}, anchor={anchor_weight})")
        print("=" * 70)

        for step in range(n_steps):
            optimizer.zero_grad()

            # Decode with soft outputs
            tokens, soft_embeds = self.decode_with_gradients(z, temperature=temperature)
            prompt_text = self.get_prompt_text(tokens)

            if soft_embeds.shape[0] == 0:
                print(f"Step {step:3d} | Empty prompt, skipping...")
                continue

            # Compute loss on train examples
            train_loss, train_acc, _ = self.compute_task_loss(soft_embeds, train_examples)

            # Add anchor regularization to stay near seed embedding
            if anchor_weight > 0:
                anchor_loss = torch.nn.functional.mse_loss(z, z_anchor)
                train_loss = train_loss + anchor_weight * anchor_loss

            # Backward pass
            train_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)

            # Update z
            optimizer.step()

            # Project z to maintain norm (important for SONAR stability)
            with torch.no_grad():
                z.data = z.data / torch.norm(z.data) * self.target_norm

            # Evaluate on test (no grad)
            with torch.no_grad():
                test_tokens, test_soft = self.decode_with_gradients(z, temperature=temperature)
                test_prompt = self.get_prompt_text(test_tokens)
                if test_soft.shape[0] > 0:
                    _, test_acc, test_preds = self.compute_task_loss(test_soft, test_examples)
                else:
                    test_acc = 0.0
                    test_preds = []

            # Track best
            if test_acc > best_acc:
                best_acc = test_acc
                best_z = z.detach().clone()
                best_prompt = test_prompt

            # Log
            if step % 5 == 0 or test_acc > 0:
                grad_norm = z.grad.norm().item() if z.grad is not None else 0
                print(f"Step {step:3d} | Train: loss={train_loss.item():.3f} acc={train_acc:.0%} | "
                      f"Test: acc={test_acc:.0%} | grad={grad_norm:.4f}")
                print(f"         Prompt: '{prompt_text[:70]}'" + ("..." if len(prompt_text) > 70 else ""))

        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print(f"Best test accuracy: {best_acc:.0%}")
        print(f"Best prompt: '{best_prompt}'")

        # Final evaluation with best z
        with torch.no_grad():
            final_tokens, final_soft = self.decode_with_gradients(best_z, temperature=temperature)
            final_prompt = self.get_prompt_text(final_tokens)
            if final_soft.shape[0] > 0:
                _, final_acc, final_preds = self.compute_task_loss(final_soft, test_examples)
                print("\nFinal test predictions:")
                for inp, exp, pred in final_preds:
                    status = "✓" if pred.strip() == exp.strip() else "✗"
                    print(f"  {status} {inp} expected='{exp}' got='{pred}'")

        return best_z, best_prompt, best_acc


def main():
    print("SONAR-Qwen Prompt Optimization v2")
    print("=" * 70)

    optimizer = SonarQwenOptimizer()

    # Run optimization with anchor weight to stay near seed
    best_z, best_prompt, best_acc = optimizer.optimize(
        train_examples=ANTONYMS_TRAIN,
        test_examples=ANTONYMS_TEST,
        n_steps=100,
        lr=0.005,
        seed_text="Opposites: hot -> cold, big -> small. Now:",
        temperature=0.5,
        anchor_weight=100.0,  # Strong regularization to stay near seed
    )

    print(f"\nFinal result:")
    print(f"  Best prompt: '{best_prompt}'")
    print(f"  Test accuracy: {best_acc:.0%}")


if __name__ == "__main__":
    main()
