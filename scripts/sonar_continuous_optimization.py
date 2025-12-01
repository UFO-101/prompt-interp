"""
Continuous Prompt Optimization using SONAR embeddings.

Pipeline:
1. Start with SONAR sentence embedding z (1024D)
2. Decode z through SONAR decoder with teacher forcing → hidden states
3. Map hidden states to Qwen embedding space via Procrustes
4. Run Qwen forward pass with soft prefix embeddings
5. Compute NLL loss for the task
6. Backprop to optimize z

Since SONAR has tied embeddings, the Procrustes alignment trained on token
embeddings should work directly on decoder hidden states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# SONAR imports
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout


class SONARQwenBridge(nn.Module):
    """Bridge between SONAR decoder output and Qwen input embeddings."""

    def __init__(self, alignment_path: str = "results/nllb_qwen_alignment.npz"):
        super().__init__()

        # Load Procrustes alignment
        alignment = np.load(alignment_path)
        self.register_buffer('W', torch.tensor(alignment['W'], dtype=torch.float32))
        self.register_buffer('src_mean', torch.tensor(alignment['src_mean'], dtype=torch.float32))
        self.register_buffer('tgt_mean', torch.tensor(alignment['tgt_mean'], dtype=torch.float32))

        print(f"Loaded alignment: W={self.W.shape}, maps 1024D → 896D")

    def forward(self, sonar_hidden: torch.Tensor) -> torch.Tensor:
        """
        Map SONAR decoder hidden states to Qwen embedding space.

        Args:
            sonar_hidden: [batch, seq_len, 1024] - SONAR decoder hidden states

        Returns:
            qwen_embeds: [batch, seq_len, 896] - Qwen-compatible embeddings
        """
        # Center, transform, uncenter
        centered = sonar_hidden - self.src_mean
        transformed = centered @ self.W
        return transformed + self.tgt_mean


class ContinuousPromptOptimizer:
    """Optimize SONAR embeddings for downstream Qwen task performance."""

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

        print("Loading SONAR-Qwen bridge...")
        self.bridge = SONARQwenBridge().to(device)

        # Get SONAR's internal model for differentiable forward pass
        # Note: We move to device here; the pipeline will stay on CPU for text decoding
        self.sonar_decoder_model = self.sonar_decoder.model.to(device)
        self.sonar_tokenizer = self.sonar_decoder.tokenizer
        self.tok_encoder = self.sonar_tokenizer.create_encoder(mode='target', lang='eng_Latn')
        self.tok_decoder = self.sonar_tokenizer.create_decoder()

    def encode_sentence(self, text: str) -> torch.Tensor:
        """Encode text to SONAR embedding."""
        emb = self.sonar_encoder.predict([text], source_lang='eng_Latn')
        return emb.to(self.device)

    def decode_embedding_to_tokens(self, embedding: torch.Tensor, max_len: int = 50) -> torch.Tensor:
        """Decode SONAR embedding to token sequence using greedy decoding (non-differentiable)."""
        model = self.sonar_decoder_model
        device = self.device

        # Ensure embedding is on device and has right shape
        emb = embedding.detach().to(device)  # Detach - this is non-differentiable
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)  # [1, 1024]

        encoder_output = emb.unsqueeze(1)  # [1, 1, 1024]
        enc_layout = BatchLayout.of(encoder_output)

        # Start with BOS token (3) and language tag (256047 for eng_Latn)
        bos_token = 3
        lang_token = 256047
        eos_token = 3
        generated = [bos_token, lang_token]

        for _ in range(max_len):
            # Prepare decoder input
            decoder_input = torch.tensor([generated], device=device)
            seqs_layout = BatchLayout.of(decoder_input)

            # Get hidden states and project to logits
            hidden = model.decode(decoder_input, seqs_layout, encoder_output, enc_layout)
            logits = model.decoder.final_proj(hidden)

            # Greedy: take argmax of last position
            next_token = logits[0, -1, :].argmax().item()
            generated.append(next_token)

            # Stop on EOS (but not the BOS at position 0)
            if next_token == eos_token:
                break

        return torch.tensor(generated, device=device)

    def decode_embedding(self, embedding: torch.Tensor, max_len: int = 50) -> str:
        """Decode SONAR embedding to text string."""
        tokens = self.decode_embedding_to_tokens(embedding, max_len)
        # Remove BOS, lang tag, and EOS for text decoding
        text_tokens = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
        if len(text_tokens) == 0:
            return ""
        return self.tok_decoder(text_tokens.cpu())

    def get_decoder_hidden_states(
        self,
        embedding: torch.Tensor,
        tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Get SONAR decoder hidden states with teacher forcing.

        This is the differentiable forward pass through the decoder.

        Args:
            embedding: [1, 1024] - SONAR sentence embedding (with gradients)
            tokens: Token sequence from greedy decoding (including BOS, lang tag, EOS)

        Returns:
            hidden_states: [1, seq_len, 1024] - decoder hidden states
        """
        # Prepare decoder input (all tokens except last - teacher forcing)
        decoder_input = tokens[:-1].unsqueeze(0)  # [1, seq-1]

        # Encoder output is the sentence embedding (single vector)
        # This is where gradients flow through!
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        encoder_output = embedding.unsqueeze(1)  # [1, 1, 1024]

        # Create batch layouts
        seqs_layout = BatchLayout.of(decoder_input)
        enc_layout = BatchLayout.of(encoder_output)

        # Forward pass through decoder - returns hidden states [1, seq, 1024]
        # Gradients flow: loss → hidden_states → cross-attention → encoder_output → embedding
        hidden_states = self.sonar_decoder_model.decode(
            decoder_input, seqs_layout, encoder_output, enc_layout
        )

        return hidden_states

    def compute_task_loss(
        self,
        prefix_embeds: torch.Tensor,
        input_text: str,
        target: str
    ) -> torch.Tensor:
        """
        Compute task loss: NLL of predicting the target string.

        No few-shot - the soft prefix must do all the work.

        Args:
            prefix_embeds: [1, prefix_len, 896] - soft prefix embeddings
            input_text: e.g., "hot ->"
            target: e.g., "cold"
        """
        # Just the input - prefix must provide context
        task_text = f"{input_text} "

        task_tokens = self.qwen_tokenizer(task_text, return_tensors="pt").input_ids.to(self.device)
        task_embeds = self.qwen_model.model.embed_tokens(task_tokens)

        # Target tokens
        target_tokens = self.qwen_tokenizer.encode(target, add_special_tokens=False)

        # Concatenate: prefix + task + target (for teacher forcing)
        target_embeds = self.qwen_model.model.embed_tokens(
            torch.tensor([target_tokens], device=self.device)
        )
        full_embeds = torch.cat([prefix_embeds, task_embeds, target_embeds], dim=1)

        # Forward pass
        outputs = self.qwen_model(inputs_embeds=full_embeds)

        # Compute loss on target positions
        prefix_len = prefix_embeds.shape[1]
        task_len = task_tokens.shape[1]
        start_pos = prefix_len + task_len - 1  # -1 because we predict next token

        total_loss = 0
        for i, target_tok in enumerate(target_tokens):
            logits = outputs.logits[0, start_pos + i, :]
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_tok]).to(self.device))
            total_loss += loss

        return total_loss / len(target_tokens)

    def optimize(
        self,
        seed_text: str,
        examples: list[tuple[str, int]],
        num_steps: int = 100,
        lr: float = 0.01
    ):
        """
        Optimize the SONAR embedding for task performance.

        Args:
            seed_text: Initial text to encode
            examples: List of (input_sequence, target_number) pairs
            num_steps: Optimization steps
            lr: Learning rate
        """
        print(f"\nStarting optimization from seed: '{seed_text}'")

        # Encode seed text
        with torch.no_grad():
            z = self.encode_sentence(seed_text)
            print(f"  Encoded z shape: {z.shape}, device: {z.device}")

            # Verify decode works before optimization
            test_decode = self.decode_embedding(z)
            print(f"  Initial decode: '{test_decode}'")

        # Make z a parameter we can optimize
        z = nn.Parameter(z.clone())
        optimizer = torch.optim.Adam([z], lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            # Step 1: Decode z to tokens (non-differentiable, detached)
            with torch.no_grad():
                tokens = self.decode_embedding_to_tokens(z)

            # Step 2: Differentiable forward pass with teacher forcing
            # The gradients flow through z via cross-attention in the decoder
            hidden = self.get_decoder_hidden_states(z, tokens)  # [1, seq, 1024]

            # Map to Qwen space
            prefix_embeds = self.bridge(hidden)  # [1, seq, 896]

            # Compute loss over all examples
            total_loss = 0
            for input_seq, target in examples:
                loss = self.compute_task_loss(prefix_embeds, input_seq, target)
                total_loss += loss

            avg_loss = total_loss / len(examples)

            # Backward - gradients flow: loss → Qwen → bridge → hidden → decoder → z
            avg_loss.backward()

            # Debug: check gradient
            if step < 3:
                if z.grad is not None:
                    print(f"  Step {step}: z.grad norm: {z.grad.norm().item():.4f}, z norm: {z.data.norm().item():.4f}")
                else:
                    print(f"  Step {step}: WARNING: z.grad is None!")

            # Clip gradients relative to z norm to prevent explosion
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.1)

            optimizer.step()

            if step == 0:
                print(f"  z norm after step: {z.data.norm().item():.4f}")

            # Logging - every step
            with torch.no_grad():
                decoded = self.decode_embedding(z)
            print(f"Step {step}: loss={avg_loss.item():.4f}, decoded='{decoded[:60]}...' " if len(decoded) > 60 else f"Step {step}: loss={avg_loss.item():.4f}, decoded='{decoded}'")

        return z


def main():
    import sys
    def log(msg):
        print(msg)
        sys.stdout.flush()

    log("=" * 70)
    log("SONAR Continuous Prompt Optimization")
    log("=" * 70)

    # Check if alignment exists
    alignment_path = Path("results/nllb_qwen_alignment.npz")
    if not alignment_path.exists():
        log("ERROR: Run scripts/embedding_alignment.py first to create the alignment!")
        return

    # Antonym task - train examples
    examples = [
        ("hot ->", "cold"),
        ("big ->", "small"),
        ("fast ->", "slow"),
        ("up ->", "down"),
        ("happy ->", "sad"),
        ("light ->", "dark"),
    ]

    log("Creating optimizer...")
    opt = ContinuousPromptOptimizer()

    # Try a seed relevant to the antonym task
    seeds = [
        "Give the opposite word. For example, hot becomes cold and big becomes small.",
    ]

    for seed in seeds:
        log(f"\n{'='*70}")
        log(f"Trying seed: '{seed}'")
        log("=" * 70)

        z_opt = opt.optimize(
            seed_text=seed,
            examples=examples,
            num_steps=100,
            lr=0.001
        )

        # Evaluate
        log("\nEvaluation:")
        with torch.no_grad():
            decoded = opt.decode_embedding(z_opt)
            log(f"  Optimized text: '{decoded}'")

            # Get hidden states using the current decoded tokens
            tokens = opt.decode_embedding_to_tokens(z_opt)
            hidden = opt.get_decoder_hidden_states(z_opt, tokens)
            prefix_embeds = opt.bridge(hidden)

            correct = 0
            for input_text, target in examples:
                # No few-shot - just input + space
                task_text = f"{input_text} "
                task_tokens = opt.qwen_tokenizer(task_text, return_tensors="pt").input_ids.to(opt.device)
                task_embeds = opt.qwen_model.model.embed_tokens(task_tokens)

                full_embeds = torch.cat([prefix_embeds, task_embeds], dim=1)

                # Generate tokens
                outputs = opt.qwen_model(inputs_embeds=full_embeds)
                generated = []
                for _ in range(5):
                    next_token = outputs.logits[0, -1, :].argmax().item()
                    generated.append(next_token)
                    if next_token == opt.qwen_tokenizer.eos_token_id:
                        break
                    next_embed = opt.qwen_model.model.embed_tokens(torch.tensor([[next_token]], device=opt.device))
                    full_embeds = torch.cat([full_embeds, next_embed], dim=1)
                    outputs = opt.qwen_model(inputs_embeds=full_embeds)

                pred_text = opt.qwen_tokenizer.decode(generated).strip().lower()
                # Get first word only
                first_word = ''.join(c for c in pred_text.split()[0] if c.isalpha()) if pred_text.split() else ''

                is_correct = first_word == target.lower()
                # Partial credit if target appears but not as first word
                partial = target.lower() in pred_text and not is_correct
                correct += is_correct
                symbol = '✓' if is_correct else ('~' if partial else '✗')
                log(f"  {input_text} -> pred='{first_word}', target={target} {symbol}")

            log(f"  Accuracy: {correct}/{len(examples)} ({100*correct/len(examples):.1f}%)")


if __name__ == "__main__":
    main()
