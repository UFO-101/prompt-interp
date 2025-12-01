"""
Optimize SONAR z vector using unconditioned decoder for task loss - v2.

Key insight: The task loss should be computed by:
1. Take the decoded prompt text
2. Concatenate with task input (e.g., "hot -> ")
3. Encode the full thing back with SONAR encoder
4. Decode with z=0 (unconditioned)
5. Check if output matches target ("cold")

For gradient flow:
- Use soft token selection on conditioned decoder
- Compute proxy loss that pushes toward task-relevant outputs
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
sonar_embeds = sdm.decoder.final_proj.weight.data  # [vocab_size, 1024]

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


def decode_conditioned(z, max_len=30):
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


def decode_z0(tokens, max_new=10):
    """Continue decoding from tokens using z=0."""
    with torch.no_grad():
        # Remove EOS if present
        if tokens[-1] == 3:
            tokens = tokens[:-1]

        generated = tokens.tolist()

        for _ in range(max_new):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token == 3:
                break

    return torch.tensor(generated, device=dev)


def compute_task_loss(prompt_text, examples):
    """
    Compute task loss: for each (input, target) example,
    encode prompt+input, decode with z=0, compute CE loss on target.
    """
    total_loss = 0.0
    n_correct = 0

    for input_text, target_text in examples:
        # Create full prompt + input
        full_text = prompt_text + " " + input_text if prompt_text else input_text

        # Encode to tokens
        full_tokens = text_to_tokens(full_text)

        # Decode continuation with z=0
        cont_tokens = decode_z0(full_tokens, max_new=5)
        cont_text = tokens_to_text(cont_tokens)

        # Check if target is in continuation
        # The continuation after prompt+input should be the answer
        prompt_input_text = tokens_to_text(full_tokens)
        if len(cont_text) > len(prompt_input_text):
            answer = cont_text[len(prompt_input_text):].strip()
        else:
            answer = ""

        # Check correctness
        if answer.lower().startswith(target_text.lower()):
            n_correct += 1

        # Compute CE loss for target tokens
        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]

        if len(target_content) > 0:
            # Get logits for predicting after the prompt+input
            input_for_loss = full_tokens[:-1] if full_tokens[-1] == 3 else full_tokens
            input_for_loss = input_for_loss.unsqueeze(0)

            h = sdm.decode(input_for_loss, BatchLayout.of(input_for_loss),
                          z_zero, BatchLayout.of(z_zero))
            logits = sdm.decoder.final_proj(h)  # [1, seq_len, vocab]

            # Loss on first target token
            last_pos = logits.shape[1] - 1
            loss = F.cross_entropy(logits[0, last_pos, :].unsqueeze(0),
                                  target_content[0:1])
            total_loss += loss.item()

    return total_loss / len(examples), n_correct / len(examples)


def soft_decode_and_proxy_loss(z, examples, temperature=1.0):
    """
    Soft decode with z, then compute proxy loss based on task.

    The idea: we want the soft embeddings from conditioned decoder
    to produce tokens that, when fed to z=0 decoder, predict the targets.
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    # First, do hard decode to get tokens
    tokens = decode_conditioned(z)
    prompt_text = tokens_to_text(tokens)

    if not prompt_text or len(tokens) < 4:
        return None, float('inf'), 0.0

    # Now do soft forward through conditioned decoder
    input_tokens = tokens[:-1].unsqueeze(0)
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), eo, BatchLayout.of(eo))
    logits_cond = sdm.decoder.final_proj(h)  # [1, seq_len, vocab]

    # Compute task loss (for monitoring)
    task_loss, accuracy = compute_task_loss(prompt_text, examples)

    # Proxy loss: we want the conditioned output to be "good" for the task
    # One approach: minimize distance between soft embeddings and embeddings
    # that would produce good task performance

    # Simple proxy: cross-entropy between conditioned logits and target distribution
    # that encourages task-relevant tokens

    proxy_loss = torch.tensor(0.0, device=dev, requires_grad=True)

    # For each position, encourage the soft output to match what a good prompt would have
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]

    if len(content) > 0:
        # Encourage high probability on the actual generated tokens
        # (stability) plus push toward task-relevant patterns
        for i in range(min(len(content), logits_cond.shape[1] - 2)):
            pos = i + 1  # Position predicting content[i]
            if pos < logits_cond.shape[1]:
                log_probs = F.log_softmax(logits_cond[0, pos, :] / temperature, dim=0)
                # Negative log prob of actual token (encourages coherence)
                proxy_loss = proxy_loss - log_probs[content[i]] * 0.1

        # Also add task-based signal
        # Compute gradient direction from task loss
        for input_text, target_text in examples[:2]:  # Just use first 2 for speed
            full_text = prompt_text + " " + input_text
            target_tokens = text_to_tokens(target_text)
            target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]

            if len(target_content) > 0:
                # Encode full text
                full_tokens = text_to_tokens(full_text)
                full_input = full_tokens[:-1] if full_tokens[-1] == 3 else full_tokens
                full_input = full_input.unsqueeze(0)

                # Get z=0 prediction
                h_z0 = sdm.decode(full_input, BatchLayout.of(full_input),
                                 z_zero, BatchLayout.of(z_zero))
                logits_z0 = sdm.decoder.final_proj(h_z0)

                # CE loss on target
                last_pos = logits_z0.shape[1] - 1
                ce_loss = F.cross_entropy(logits_z0[0, last_pos, :].unsqueeze(0),
                                         target_content[0:1])
                proxy_loss = proxy_loss + ce_loss

    return proxy_loss, task_loss, accuracy


# ============================================================================
# Main optimization loop
# ============================================================================
print("\n" + "=" * 80)
print("SONAR UNCONDITIONED OPTIMIZATION v2")
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

# First, let's see what z=0 does on its own for the task
print("\nBaseline (no prompt, just z=0 on task inputs):")
for input_text, target in examples:
    tokens = text_to_tokens(input_text)
    cont = decode_z0(tokens, max_new=5)
    cont_text = tokens_to_text(cont)
    print(f"  '{input_text}' -> '{cont_text}'")

# Encode seed
with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.001)

print("\n" + "-" * 80)
print(f"{'Step':<6} {'TaskLoss':<10} {'Acc':<6} {'Decoded Text':<45}")
print("-" * 80)

n_steps = 100
temperature = 1.0

best_acc = 0.0
best_z = z_init.clone()

for step in range(n_steps + 1):
    optimizer.zero_grad()

    # Compute proxy loss with gradient
    proxy_loss, task_loss, accuracy = soft_decode_and_proxy_loss(z, examples, temperature)

    # Track best
    if accuracy > best_acc:
        best_acc = accuracy
        best_z = z.detach().clone()

    # Get decoded text for display
    with torch.no_grad():
        tokens = decode_conditioned(z)
        decoded_text = tokens_to_text(tokens)

    if proxy_loss is None:
        print(f"{step:<6} {'N/A':<10} {accuracy:<6.2f} {decoded_text[:45]:<45}")
        continue

    # Backprop and update
    if step > 0 and proxy_loss.requires_grad:
        proxy_loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()

    display_text = decoded_text[:42] + "..." if len(decoded_text) > 45 else decoded_text
    print(f"{step:<6} {task_loss:<10.4f} {accuracy:<6.2f} {display_text:<45}")


# ============================================================================
# Final evaluation
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

# Use best z
z_final = best_z

with torch.no_grad():
    tokens = decode_conditioned(z_final)
    decoded_text = tokens_to_text(tokens)

print(f"\nBest prompt (acc={best_acc:.2f}): '{decoded_text}'")

print("\nTask performance:")
for input_text, target in examples:
    full_text = decoded_text + " " + input_text if decoded_text else input_text
    full_tokens = text_to_tokens(full_text)
    cont = decode_z0(full_tokens, max_new=8)
    cont_text = tokens_to_text(cont)

    # Extract answer
    prompt_input_text = tokens_to_text(full_tokens)
    if len(cont_text) > len(prompt_input_text):
        answer = cont_text[len(prompt_input_text):].strip()
    else:
        answer = cont_text

    correct = answer.lower().startswith(target.lower())
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:30]}' (want: {target}) {mark}")
