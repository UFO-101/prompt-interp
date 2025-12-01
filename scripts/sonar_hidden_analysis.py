"""
Analyze: Are SONAR decoder hidden states in the same space as token embeddings?

The Procrustes alignment was trained on token embeddings.
We're applying it to decoder hidden states.
Even with tied embeddings, these might not be equivalent.
"""

import torch
import numpy as np
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout

dev = "cuda"
print("Loading...", flush=True)
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sdm = sd.model.to(dev)
tdc = sd.tokenizer.create_decoder()
tok_encoder = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')
print("Ready.\n", flush=True)

# Get the token embedding matrix
embed_matrix = sdm.decoder.final_proj.weight.data  # This is the tied embedding/projection matrix
print(f"Embedding matrix shape: {embed_matrix.shape}")  # Should be [vocab_size, 1024]
print(f"Embedding matrix stats: mean={embed_matrix.mean():.4f}, std={embed_matrix.std():.4f}")

def enc(t): return se.predict([t], source_lang='eng_Latn').to(dev)

def decode_with_hidden(z, max_len=20):
    """Decode and return both tokens and hidden states at each position."""
    z = z.detach().unsqueeze(0) if z.dim() == 1 else z.detach()
    eo = z.unsqueeze(1)

    g = [3, 256047]  # BOS + lang tag
    all_hidden = []

    for _ in range(max_len):
        di = torch.tensor([g], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        last_hidden = h[0, -1, :]  # Hidden state at last position
        all_hidden.append(last_hidden)

        logits = sdm.decoder.final_proj(h)
        nt = logits[0, -1, :].argmax().item()
        g.append(nt)
        if nt == 3:
            break

    return g, torch.stack(all_hidden)

# Test with a sentence
text = "Give the opposite word for each input."
print(f"\nTest sentence: '{text}'")
print("=" * 70)

with torch.no_grad():
    z = enc(text)
    tokens, hiddens = decode_with_hidden(z)

    print(f"\nDecoded tokens: {tokens}")
    text_tokens = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    decoded = tdc(torch.tensor(text_tokens))
    print(f"Decoded text: '{decoded}'")

    print(f"\nHidden states shape: {hiddens.shape}")  # [seq_len, 1024]
    print(f"Hidden stats: mean={hiddens.mean():.4f}, std={hiddens.std():.4f}")

    # For each position, compare hidden state to the embedding of the next token
    print(f"\nComparing hidden states to token embeddings:")
    print(f"{'Pos':<4} {'Token':<8} {'Hidden norm':<12} {'Embed norm':<12} {'Cosine sim':<12} {'L2 dist':<12}")
    print("-" * 70)

    for i, (h, next_tok) in enumerate(zip(hiddens[:-1], tokens[2:])):  # Skip BOS+lang, compare to predicted tokens
        token_embed = embed_matrix[next_tok]  # Embedding of the token that was predicted

        h_norm = h.norm().item()
        e_norm = token_embed.norm().item()
        cosine = torch.nn.functional.cosine_similarity(h.unsqueeze(0), token_embed.unsqueeze(0)).item()
        l2 = (h - token_embed).norm().item()

        try:
            tok_str = tdc(torch.tensor([next_tok]))[:8]
        except:
            tok_str = str(next_tok)

        print(f"{i:<4} {tok_str:<8} {h_norm:<12.4f} {e_norm:<12.4f} {cosine:<12.4f} {l2:<12.4f}")

    # What about the logits? The hidden states go through final_proj to produce logits
    # final_proj is essentially h @ embed_matrix.T (since tied embeddings)
    # So hidden states are NOT meant to be in the same space as embeddings!

    print(f"\n" + "=" * 70)
    print("KEY INSIGHT: Hidden states are NOT in embedding space!")
    print("=" * 70)
    print("""
The decoder works as:
  hidden_state @ embed_matrix.T = logits

This means hidden states are meant to produce LOGITS when multiplied
with the embedding matrix, NOT to be directly compared to embeddings.

The Procrustes alignment trained on embeddings will NOT work on hidden states!
""")

    # Let's verify: hidden @ embed_matrix.T should give high score for next token
    print("\nVerifying: hidden @ embed_matrix.T gives high logit for correct token?")
    for i, (h, next_tok) in enumerate(zip(hiddens[:-1], tokens[2:])):
        logits = h @ embed_matrix.T
        top_pred = logits.argmax().item()
        next_tok_logit = logits[next_tok].item()
        max_logit = logits.max().item()

        try:
            pred_str = tdc(torch.tensor([top_pred]))[:8]
            next_str = tdc(torch.tensor([next_tok]))[:8]
        except:
            pred_str = str(top_pred)
            next_str = str(next_tok)

        match = "✓" if top_pred == next_tok else "✗"
        print(f"  Pos {i}: pred='{pred_str}', actual='{next_str}' {match} (logit={next_tok_logit:.2f}, max={max_logit:.2f})")
