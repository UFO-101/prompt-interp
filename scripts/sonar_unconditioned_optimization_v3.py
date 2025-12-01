"""
Optimize SONAR z vector using unconditioned decoder for task loss - v3.

Key insight from v2: The gradient path was broken because:
1. text_to_tokens() breaks the computational graph
2. The z=0 decoder path doesn't connect to z

New approach:
1. Decode z with CONDITIONED decoder (soft, with gradients to z)
2. Use the resulting soft embeddings to form the prompt
3. Concatenate task input tokens
4. Run through z=0 decoder to get task predictions
5. Compute CE loss on targets
6. Backprop through soft embeddings to z

The key: soft token selection on conditioned output, then feed those
soft embeddings (not hard tokens) through z=0 decoder.
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
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')
sonar_embeds = sdm.decoder.decoder_frontend.embed.weight.data  # Input embeddings [vocab_size, 1024]
sonar_vocab_size = sonar_embeds.shape[0]

# Zero z for unconditioned generation
z_zero = torch.zeros(1, 1, 1024, device=dev)


def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def text_to_tokens(text):
    """Encode text to SONAR tokens."""
    tokens = sonar_enc(text)
    return tokens.to(dev)


def decode_conditioned_hard(z, max_len=30):
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


def soft_decode_conditioned(z, max_len=20, temperature=0.5):
    """
    Decode z to SOFT embeddings (with gradients to z).

    Returns:
    - hard_tokens: the actual token sequence (for display)
    - soft_embeds: differentiable embedding sequence
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    # Start with BOS and lang tag embeddings (these are fixed)
    bos_embed = sonar_embeds[3].unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]
    lang_embed = sonar_embeds[256047].unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]

    soft_embeds = [bos_embed, lang_embed]
    hard_tokens = [3, 256047]

    for _ in range(max_len):
        # Concatenate all embeddings so far
        input_embeds = torch.cat(soft_embeds, dim=1)  # [1, seq_len, 1024]

        # Get hidden states from decoder
        # We need to use embed directly, bypassing the normal token input
        # Actually, SONAR decoder expects token IDs, not embeddings directly
        # So we need a different approach...

        # Use teacher forcing with the hard tokens we've collected
        di = torch.tensor([hard_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        logits = sdm.decoder.final_proj(h)[0, -1, :]  # [vocab]

        # Soft token selection
        probs = F.softmax(logits / temperature, dim=0)
        soft_embed = probs @ sonar_embeds  # [1024]
        soft_embeds.append(soft_embed.unsqueeze(0).unsqueeze(0))

        # Hard token for next step
        hard_token = logits.argmax().item()
        hard_tokens.append(hard_token)

        if hard_token == 3:  # EOS
            break

    # Stack soft embeddings: [1, seq_len, 1024]
    soft_embeds_tensor = torch.cat(soft_embeds, dim=1)
    hard_tokens_tensor = torch.tensor(hard_tokens, device=dev)

    return hard_tokens_tensor, soft_embeds_tensor


def compute_task_loss_soft(z, soft_prompt_embeds, hard_prompt_tokens, examples, temperature=1.0):
    """
    Compute task loss using soft prompt embeddings.

    The idea:
    1. Soft prompt embeddings from conditioned decoder (has gradients to z)
    2. Concatenate with task input tokens (embedded as hard tokens)
    3. Run through model with z=0
    4. Compute CE loss on target

    Problem: SONAR decoder takes token IDs, not embeddings directly.

    Alternative approach:
    - Use the conditioned decoder's logits at each position
    - The logits ARE differentiable w.r.t. z
    - Compute a "matching" loss between conditioned logits and what z=0 would want
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    prompt_text = tokens_to_text(hard_prompt_tokens)
    if not prompt_text:
        return None, float('inf'), 0.0

    total_loss = torch.tensor(0.0, device=dev)
    n_correct = 0

    for input_text, target_text in examples:
        # Create full text and tokenize
        full_text = prompt_text + " " + input_text
        full_tokens = text_to_tokens(full_text)

        # Get target tokens
        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]

        if len(target_content) == 0:
            continue

        # === Forward through z=0 decoder (for task loss, no grad needed) ===
        with torch.no_grad():
            full_input = full_tokens[:-1] if full_tokens[-1] == 3 else full_tokens
            full_input = full_input.unsqueeze(0)

            h_z0 = sdm.decode(full_input, BatchLayout.of(full_input),
                             z_zero, BatchLayout.of(z_zero))
            logits_z0 = sdm.decoder.final_proj(h_z0)

            # Check prediction
            last_pos = logits_z0.shape[1] - 1
            pred_token = logits_z0[0, last_pos, :].argmax().item()
            if pred_token == target_content[0].item():
                n_correct += 1

            # What token does z=0 want to predict?
            target_probs = F.softmax(logits_z0[0, last_pos, :], dim=0)

        # === Forward through conditioned decoder (for gradients to z) ===
        # Teacher-force with hard tokens from prompt
        prompt_input = hard_prompt_tokens[:-1] if hard_prompt_tokens[-1] == 3 else hard_prompt_tokens
        prompt_input = prompt_input.unsqueeze(0)

        h_cond = sdm.decode(prompt_input, BatchLayout.of(prompt_input), eo, BatchLayout.of(eo))
        logits_cond = sdm.decoder.final_proj(h_cond)  # [1, prompt_len, vocab]

        # The conditioned output probabilities should help predict the task
        # We want: conditioned decoder produces prompt that, when fed to z=0, gives right answer

        # Proxy loss: push conditioned logits toward producing tokens that z=0 "likes"
        # For each position in the prompt, encourage high prob on tokens that
        # will lead to correct z=0 predictions

        # Simple approach: just use the conditioned logits at the last position
        # and try to make them predict something useful for the task
        last_pos_cond = logits_cond.shape[1] - 1

        # KL divergence: push conditioned output toward what z=0 prefers
        # This is indirect but provides gradient signal
        log_probs_cond = F.log_softmax(logits_cond[0, last_pos_cond, :] / temperature, dim=0)
        kl = (target_probs * (target_probs.log() - log_probs_cond)).sum()

        # Also: cross-entropy on target token through conditioned path
        # This is more direct but may not help task performance
        ce_target = F.cross_entropy(logits_cond[0, last_pos_cond, :].unsqueeze(0),
                                   target_content[0:1])

        total_loss = total_loss + kl * 0.1 + ce_target

    accuracy = n_correct / len(examples)
    avg_loss = total_loss / len(examples)

    return avg_loss, avg_loss.item(), accuracy


# ============================================================================
# Main optimization loop
# ============================================================================
print("\n" + "=" * 80)
print("SONAR UNCONDITIONED OPTIMIZATION v3")
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

# Baseline
print("\nBaseline (no prompt, just z=0 on task inputs):")
for input_text, target in examples:
    tokens = text_to_tokens(input_text)
    with torch.no_grad():
        input_seq = tokens[:-1] if tokens[-1] == 3 else tokens
        input_seq = input_seq.unsqueeze(0)
        h = sdm.decode(input_seq, BatchLayout.of(input_seq), z_zero, BatchLayout.of(z_zero))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        pred = logits.argmax().item()
        pred_text = sonar_dec(torch.tensor([pred]))
    print(f"  '{input_text}' -> '{pred_text}' (want: {target})")

# Encode seed
with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.01)

print("\n" + "-" * 80)
print(f"{'Step':<6} {'Loss':<10} {'Acc':<6} {'Decoded Text':<45}")
print("-" * 80)

n_steps = 100
temperature = 1.0

best_acc = 0.0
best_z = z_init.clone()

for step in range(n_steps + 1):
    optimizer.zero_grad()

    # Soft decode with gradients
    hard_tokens, soft_embeds = soft_decode_conditioned(z, max_len=15, temperature=0.5)
    decoded_text = tokens_to_text(hard_tokens)

    if not decoded_text or len(hard_tokens) < 4:
        print(f"{step:<6} {'N/A':<10} {0.0:<6.2f} {decoded_text[:45]:<45}")
        continue

    # Compute task loss
    loss, loss_val, accuracy = compute_task_loss_soft(z, soft_embeds, hard_tokens, examples, temperature)

    if loss is None:
        print(f"{step:<6} {'N/A':<10} {0.0:<6.2f} {decoded_text[:45]:<45}")
        continue

    # Track best
    if accuracy > best_acc:
        best_acc = accuracy
        best_z = z.detach().clone()

    # Backprop
    if step > 0 and loss.requires_grad:
        loss.backward()

        # Check if gradients flowed
        if z.grad is not None and z.grad.abs().sum() > 0:
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optimizer.step()

    display_text = decoded_text[:42] + "..." if len(decoded_text) > 45 else decoded_text
    print(f"{step:<6} {loss_val:<10.4f} {accuracy:<6.2f} {display_text:<45}")


# ============================================================================
# Final evaluation
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

z_final = best_z

with torch.no_grad():
    tokens = decode_conditioned_hard(z_final)
    decoded_text = tokens_to_text(tokens)

print(f"\nBest prompt (acc={best_acc:.2f}): '{decoded_text}'")

print("\nTask performance:")
for input_text, target in examples:
    full_text = decoded_text + " " + input_text if decoded_text else input_text
    full_tokens = text_to_tokens(full_text)

    with torch.no_grad():
        full_input = full_tokens[:-1] if full_tokens[-1] == 3 else full_tokens
        full_input = full_input.unsqueeze(0)

        h = sdm.decode(full_input, BatchLayout.of(full_input), z_zero, BatchLayout.of(z_zero))
        logits = sdm.decoder.final_proj(h)

        # Generate a few tokens
        generated = full_tokens.tolist()
        for _ in range(5):
            h = sdm.decode(torch.tensor([generated], device=dev),
                          BatchLayout.of(torch.tensor([generated], device=dev)),
                          z_zero, BatchLayout.of(z_zero))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            generated.append(next_tok)
            if next_tok == 3:
                break

        gen_tokens = torch.tensor(generated, device=dev)
        gen_text = tokens_to_text(gen_tokens)

        # Extract answer
        prompt_input_text = tokens_to_text(full_tokens)
        if len(gen_text) > len(prompt_input_text):
            answer = gen_text[len(prompt_input_text):].strip()
        else:
            answer = gen_text

        correct = answer.lower().startswith(target.lower())
        mark = "OK" if correct else "X"
        print(f"  {input_text} -> '{answer[:30]}' (want: {target}) {mark}")
