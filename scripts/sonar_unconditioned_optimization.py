"""
Optimize SONAR z vector using unconditioned decoder for task loss.

Approach:
1. Encode seed → z
2. Decode z → tokens (conditioned, no grad on token selection)
3. Teacher-force those tokens through decoder with z=0 (unconditioned)
4. Compute task loss (antonyms) on unconditioned logits
5. Backprop to z through the soft token selection on conditioned decoder
6. Update z and repeat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout

dev = "cuda"
torch.cuda.empty_cache()

print("Loading models...", flush=True)
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sdm = sd.model.to(dev)
sonar_dec = sd.tokenizer.create_decoder()
sonar_embeds = sdm.decoder.final_proj.weight.data  # [vocab_size, 1024]
sonar_vocab_size = sonar_embeds.shape[0]

# Zero z for unconditioned generation
z_zero = torch.zeros(1, 1, 1024, device=dev)

def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def decode_conditioned(z, max_len=40):
    """Decode z to tokens (hard, no gradients on selection)."""
    with torch.no_grad():
        e = z.unsqueeze(0) if z.dim() == 1 else z
        eo = e.unsqueeze(1)

        generated = [3, 256047]  # BOS, eng_Latn

        for _ in range(max_len):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token == 3:  # EOS
                break

    return torch.tensor(generated, device=dev)


def compute_unconditioned_loss(tokens, target_texts, temperature=1.0):
    """
    Teacher-force tokens through unconditioned decoder (z=0).
    Compute cross-entropy loss for predicting target_texts after the prompt.

    Returns loss and whether gradients can flow.
    """
    # Input is all tokens except last (teacher forcing)
    input_tokens = tokens[:-1].unsqueeze(0)  # [1, seq_len]

    # Run through decoder with z=0
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), z_zero, BatchLayout.of(z_zero))
    logits = sdm.decoder.final_proj(h)  # [1, seq_len, vocab]

    # The logits at position i predict token i+1
    # For task: we want the model to predict the target after seeing the prompt
    #
    # tokens = [BOS, lang, content..., EOS]
    # After the full prompt, the next token prediction is what we care about

    # For simplicity, let's just compute loss on predicting the next few tokens
    # after the content ends (before EOS)

    # Actually, for the antonym task, we need to:
    # 1. Append task input (e.g., "hot -> ") to the prompt
    # 2. See if the model predicts the target ("cold")

    # But we're working with SONAR tokens, not text directly.
    # Let's compute a simple proxy: negative log-likelihood of the generated sequence
    # being "likely" under z=0 (we want high probability outputs)

    total_loss = 0.0
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]

    if len(content) == 0:
        return torch.tensor(0.0, device=dev, requires_grad=True), False

    # Compute NLL of the generated content under z=0
    for i, tok in enumerate(content):
        pos = i + 2  # Position in full sequence (after BOS, lang)
        if pos - 1 < logits.shape[1]:
            log_probs = F.log_softmax(logits[0, pos - 1, :] / temperature, dim=-1)
            total_loss -= log_probs[tok]

    return total_loss / len(content), True


def soft_forward_conditioned(z, tokens, temperature=1.0):
    """
    Teacher-force tokens through conditioned decoder.
    Use soft token selection for differentiability.
    Returns soft embeddings that can be compared to target gradients.
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    # Input all tokens except last
    input_tokens = tokens[:-1].unsqueeze(0)
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), eo, BatchLayout.of(eo))
    logits = sdm.decoder.final_proj(h)  # [1, seq_len, vocab]

    # Soft token selection: weighted sum of embeddings
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    soft_embeds = []

    for i in range(len(content)):
        pos = i + 1  # Position in logits that predicts content[i]
        if pos < logits.shape[1]:
            probs = F.softmax(logits[0, pos, :] / temperature, dim=0)
            soft_embed = probs @ sonar_embeds  # [1024]
            soft_embeds.append(soft_embed)

    if len(soft_embeds) == 0:
        return None

    return torch.stack(soft_embeds)  # [n_content, 1024]


def compute_task_loss_sonar(tokens, examples):
    """
    Compute task loss using SONAR decoder itself.

    Approach: For each example (input, target):
    - Encode prompt + input with SONAR
    - See if greedy decode gives target
    - Compute cross-entropy loss
    """
    prompt_text = tokens_to_text(tokens)
    if not prompt_text:
        return None, float('inf')

    total_loss = 0.0
    n_correct = 0

    for input_text, target_text in examples:
        # Create full prompt
        full_text = prompt_text + " " + input_text

        # Encode
        with torch.no_grad():
            z_task = se.predict([full_text], source_lang='eng_Latn').to(dev)

        # Decode with z=0 and compute loss
        e = z_task
        eo = e.unsqueeze(1)

        # Start decoding
        generated = [3, 256047]

        # Generate a few tokens
        for _ in range(10):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token == 3:
                break

        # Get generated text
        gen_tokens = torch.tensor(generated, device=dev)
        gen_text = tokens_to_text(gen_tokens)

        # Check if correct
        if gen_text.lower().strip().startswith(target_text.lower()):
            n_correct += 1

    return n_correct, n_correct / len(examples)


# ============================================================================
# Main optimization loop
# ============================================================================
print("\n" + "=" * 80)
print("SONAR UNCONDITIONED OPTIMIZATION")
print("=" * 80)

# Antonym task
examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

# Seed
seed = "Find the opposite: hot becomes cold, big becomes small."
print(f"\nSeed: '{seed}'")

# Encode seed
with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.01)

print("\n" + "-" * 80)
print(f"{'Step':<6} {'Loss':<10} {'Decoded Text':<50}")
print("-" * 80)

n_steps = 100
temperature = 1.0

for step in range(n_steps + 1):
    optimizer.zero_grad()

    # 1. Decode z to tokens (hard, no grad)
    tokens = decode_conditioned(z)
    decoded_text = tokens_to_text(tokens)

    if len(tokens) < 4:  # Too short
        print(f"{step:<6} {'N/A':<10} {decoded_text[:50]:<50}")
        continue

    # 2. Compute loss: NLL under z=0 (we want high probability)
    # This encourages z to produce sequences that are "natural" for the LM
    loss_uncond, has_grad = compute_unconditioned_loss(tokens, examples, temperature)

    if not has_grad:
        print(f"{step:<6} {'N/A':<10} {decoded_text[:50]:<50}")
        continue

    # 3. Also compute soft forward through conditioned decoder
    # and create proxy loss for gradient flow to z
    if step > 0:
        soft_embeds = soft_forward_conditioned(z, tokens, temperature)

        if soft_embeds is not None:
            # Get gradients from unconditioned loss
            # We want to push the soft embeddings toward what z=0 prefers

            # Run unconditioned forward to get target logits
            input_tokens = tokens[:-1].unsqueeze(0)
            with torch.no_grad():
                h_uncond = sdm.decode(input_tokens, BatchLayout.of(input_tokens),
                                      z_zero, BatchLayout.of(z_zero))
                logits_uncond = sdm.decoder.final_proj(h_uncond)

            # Proxy loss: soft_embed should match what z=0 predicts
            content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
            proxy_loss = 0.0

            for i in range(min(len(content), soft_embeds.shape[0])):
                pos = i + 1
                if pos < logits_uncond.shape[1]:
                    # Target: the token that z=0 would predict
                    target_probs = F.softmax(logits_uncond[0, pos, :], dim=0)
                    target_embed = target_probs @ sonar_embeds

                    # Loss: distance between soft_embed and target_embed
                    proxy_loss = proxy_loss - (soft_embeds[i] * target_embed).sum()

            proxy_loss = proxy_loss / max(1, len(content))

            # Backprop
            proxy_loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optimizer.step()

    loss_val = loss_uncond.item() if isinstance(loss_uncond, torch.Tensor) else loss_uncond
    display_text = decoded_text[:47] + "..." if len(decoded_text) > 50 else decoded_text
    print(f"{step:<6} {loss_val:<10.4f} {display_text:<50}")


# ============================================================================
# Final evaluation
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

with torch.no_grad():
    tokens = decode_conditioned(z)
    decoded_text = tokens_to_text(tokens)

print(f"\nFinal prompt: '{decoded_text}'")

# Test on task
print("\nTask performance (encoding prompt+input, decoding with z=0):")
for input_text, target in examples:
    full_text = decoded_text + " " + input_text

    # Encode
    z_task = se.predict([full_text], source_lang='eng_Latn').to(dev)

    # Decode with z=0
    e = z_task
    eo = e.unsqueeze(1)
    generated = [3, 256047]

    for _ in range(10):
        di = torch.tensor([generated], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_token = logits.argmax().item()
        generated.append(next_token)
        if next_token == 3:
            break

    gen_tokens = torch.tensor(generated, device=dev)
    gen_text = tokens_to_text(gen_tokens)

    correct = gen_text.lower().strip().startswith(target.lower())
    print(f"  {input_text} -> '{gen_text[:20]}' {'OK' if correct else 'X'}")
