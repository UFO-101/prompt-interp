"""Core optimization utilities for SONAR embedding space."""

import torch
import torch.nn.functional as F
from fairseq2.nn.batch_layout import BatchLayout

from prompt_interp.sonar_wrapper import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator


EMBEDDING_NORM_MEAN = 0.207

LILY_STORY = """Once upon a time, there was a little girl named Lily. She loved to play outside with her toys. One day, she saw a big tree in the sky. She wanted to climb it, but it was too high. Lily asked her mom to help her climb the tree. Her mom said, "No, you can't climb the tree. It's too high." Lily was sad because she wanted to climb the tree. Later that day, Lily's mom told her that she could climb the tree."""


def project_to_norm(z: torch.Tensor, target_norm: float = EMBEDDING_NORM_MEAN) -> torch.Tensor:
    """Project embeddings to target L2 norm. Shape: (batch, seq, embed_dim)."""
    assert z.dim() == 3, f"Expected (batch, seq, embed_dim), got {z.shape}"
    norm = z.norm(dim=-1, keepdim=True)
    return z * (target_norm / (norm + 1e-8))


def predict_next_embedding(z: torch.Tensor, generator: SonarLLMGenerator) -> torch.Tensor:
    """Pass z through SONAR-LLM. Shape: (batch, seq, embed_dim) -> (batch, seq, embed_dim)."""
    assert z.dim() == 3, f"Expected (batch, seq, embed_dim), got {z.shape}"
    hidden = generator.forward_proj(z)
    out = generator.llama_model(inputs_embeds=hidden, output_hidden_states=True)
    return generator.reverse_proj(out.hidden_states[-1])


def decoder_ce_loss(
    embeddings: torch.Tensor,
    target_tokens: torch.Tensor,
    sonar_wrapper: SonarWrapper,
) -> torch.Tensor:
    """Compute CE loss. embeddings: (batch, seq, embed_dim), target_tokens: (batch, seq_len)."""
    assert embeddings.dim() == 3, f"Expected (batch, seq, embed_dim), got {embeddings.shape}"
    assert target_tokens.dim() == 2, f"Expected (batch, seq_len), got {target_tokens.shape}"

    batch, seq, _ = embeddings.shape
    device = embeddings.device
    decoder = sonar_wrapper.decoder.model.decoder

    source_layout = BatchLayout(shape=(batch, seq), seq_lens=[seq] * batch, device=device)
    target_layout = BatchLayout(shape=target_tokens.shape, seq_lens=[target_tokens.size(1)] * batch, device=device)

    logits = decoder(embeddings, source_layout, target_tokens, target_layout)
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        target_tokens[:, 1:].reshape(-1),
    )


def tokenize_for_decoder(text: str, sonar_wrapper: SonarWrapper) -> torch.Tensor:
    """Tokenize text for SONAR decoder. Returns (seq_len,)."""
    device = torch.device(sonar_wrapper.device) if isinstance(sonar_wrapper.device, str) else sonar_wrapper.device
    encoder_fn = sonar_wrapper.decoder.tokenizer.create_encoder(mode="target")
    return encoder_fn(text).to(device)
