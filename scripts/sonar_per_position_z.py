"""
Test per-position z for mixed conditioned/unconditioned generation.
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


def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def text_to_tokens(text):
    return sonar_enc(text).to(dev)


def decode_with_per_position_z(z, prompt_len, tokens, max_new=10):
    """
    Decode with z for first prompt_len positions, z=0 for the rest.

    Args:
        z: [1, 1024] the conditioning vector
        prompt_len: number of positions that should use z
        tokens: initial token sequence
        max_new: max new tokens to generate
    """
    seq = tokens.copy()

    for _ in range(max_new):
        seq_len = len(seq)

        # Create per-position encoder output
        # z for positions 0 to prompt_len-1, zeros for positions prompt_len onwards
        encoder_out = torch.zeros(1, seq_len, 1024, device=dev)
        encoder_out[:, :min(prompt_len, seq_len), :] = z

        di = torch.tensor([seq], device=dev)

        with torch.no_grad():
            h = sdm.decode(di, BatchLayout.of(di), encoder_out, BatchLayout.of(encoder_out))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            seq.append(next_tok)
            if next_tok == 3:
                break

    return seq


# ============================================================================
# Test 1: Compare generation modes
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Compare generation modes")
print("=" * 80)

test_text = "The opposite of hot is cold."
with torch.no_grad():
    z_test = se.predict([test_text], source_lang='eng_Latn').to(dev)

z_zero = torch.zeros(1, 1, 1024, device=dev)

# Start tokens
tokens = [3, 256047]  # BOS, lang

print("\n1. Full z generation (conditioned):")
seq = tokens.copy()
e = z_test.unsqueeze(1)
for _ in range(15):
    di = torch.tensor([seq], device=dev)
    with torch.no_grad():
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        seq.append(next_tok)
        if next_tok == 3:
            break
print(f"   Output: '{tokens_to_text(torch.tensor(seq))}'")

print("\n2. Full z=0 generation (unconditioned):")
seq = tokens.copy()
for _ in range(15):
    di = torch.tensor([seq], device=dev)
    with torch.no_grad():
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        seq.append(next_tok)
        if next_tok == 3:
            break
print(f"   Output: '{tokens_to_text(torch.tensor(seq))}'")

print("\n3. Mixed: z for first 5 positions, then z=0:")
seq = decode_with_per_position_z(z_test, prompt_len=5, tokens=tokens.copy(), max_new=15)
print(f"   Output: '{tokens_to_text(torch.tensor(seq))}'")

print("\n4. Mixed: z for first 10 positions, then z=0:")
seq = decode_with_per_position_z(z_test, prompt_len=10, tokens=tokens.copy(), max_new=15)
print(f"   Output: '{tokens_to_text(torch.tensor(seq))}'")


# ============================================================================
# Test 2: Generate prompt with z, continue with z=0
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Generate prompt with z, continue task with z=0")
print("=" * 80)

# Generate prompt with z
print("\nStep 1: Generate prompt with full z")
prompt_tokens = [3, 256047]
e = z_test.unsqueeze(1)
for _ in range(10):
    di = torch.tensor([prompt_tokens], device=dev)
    with torch.no_grad():
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        prompt_tokens.append(next_tok)
        if next_tok == 3:
            break
if prompt_tokens[-1] == 3:
    prompt_tokens = prompt_tokens[:-1]
prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
print(f"   Prompt ({len(prompt_tokens)} tokens): '{prompt_text}'")

# Add task input
task_input = "hot -> "
task_tokens = text_to_tokens(task_input)
task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
full_tokens = prompt_tokens + task_content.tolist()
print(f"\nStep 2: Full sequence: '{tokens_to_text(torch.tensor(full_tokens))}'")

# Continue with z=0 for new positions only
print("\nStep 3a: Continue with full z=0 (baseline):")
seq = full_tokens.copy()
for _ in range(8):
    di = torch.tensor([seq], device=dev)
    with torch.no_grad():
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        seq.append(next_tok)
        if next_tok == 3:
            break
answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()
print(f"   Answer: '{answer}'")

print("\nStep 3b: Continue with per-position z (z for prompt, z=0 for task+answer):")
prompt_len = len(prompt_tokens)
seq = decode_with_per_position_z(z_test, prompt_len=prompt_len, tokens=full_tokens.copy(), max_new=8)
answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()
print(f"   Answer: '{answer}'")

print("\nStep 3c: Continue with full z (for comparison):")
seq = full_tokens.copy()
for _ in range(8):
    di = torch.tensor([seq], device=dev)
    with torch.no_grad():
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        seq.append(next_tok)
        if next_tok == 3:
            break
answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()
print(f"   Answer: '{answer}'")


# ============================================================================
# Test 3: Verify gradient flow with per-position z
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Verify gradient flow with per-position z")
print("=" * 80)

z = nn.Parameter(z_test.clone())

# Build sequence
seq_len = len(full_tokens)
prompt_len = len(prompt_tokens)

# Create per-position encoder output
encoder_out = torch.zeros(1, seq_len, 1024, device=dev)
encoder_out[:, :prompt_len, :] = z  # z for prompt positions

di = torch.tensor([full_tokens], device=dev)
h = sdm.decode(di, BatchLayout.of(di), encoder_out, BatchLayout.of(encoder_out))
if h.dim() == 4:
    h = h.squeeze(1)
logits = sdm.decoder.final_proj(h)[0, -1, :]

# Target: "cold"
target_tokens = text_to_tokens("cold")
target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
target_id = target_content[0].item()

target = torch.tensor([target_id], device=dev, dtype=torch.long)
loss = F.cross_entropy(logits.unsqueeze(0), target)

print(f"Loss: {loss.item():.4f}")
print(f"Predicted: '{sonar_dec(torch.tensor([logits.argmax().item()]))}'")
print(f"Target: '{sonar_dec(torch.tensor([target_id]))}'")

loss.backward()

print(f"\nGradient check:")
print(f"  z.grad is None: {z.grad is None}")
if z.grad is not None:
    print(f"  z.grad.norm(): {z.grad.norm().item():.4f}")
    print(f"  z.grad.abs().mean(): {z.grad.abs().mean().item():.6f}")


# ============================================================================
# Test 4: Full optimization with per-position z
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Optimization with per-position z")
print("=" * 80)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
]

seed = "The opposite of hot is cold."
with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.01)

print("\nOptimizing z with per-position conditioning...")
print("(z for prompt positions, z=0 for task+answer positions)")

for step in range(21):
    optimizer.zero_grad()

    # Generate prompt with z
    with torch.no_grad():
        prompt_tokens = [3, 256047]
        e = z.unsqueeze(1)
        for _ in range(12):
            di = torch.tensor([prompt_tokens], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            prompt_tokens.append(next_tok)
            if next_tok == 3:
                break
        if prompt_tokens[-1] == 3:
            prompt_tokens = prompt_tokens[:-1]

    prompt_len = len(prompt_tokens)
    total_loss = 0.0
    n_correct = 0

    for input_text, target_text in examples:
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]

        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
        target_id = target_content[0].item()

        # Full sequence for teacher forcing
        full_tokens = prompt_tokens + task_content.tolist() + [target_id]
        seq_len = len(full_tokens) - 1  # Input is all but last

        # Per-position encoder output: z for prompt, 0 for task+answer
        encoder_out = torch.zeros(1, seq_len, 1024, device=dev)
        encoder_out[:, :prompt_len, :] = z

        di = torch.tensor([full_tokens[:-1]], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), encoder_out, BatchLayout.of(encoder_out))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        target = torch.tensor([target_id], device=dev, dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target)
        total_loss = total_loss + loss

        if logits.argmax().item() == target_id:
            n_correct += 1

    avg_loss = total_loss / len(examples)
    accuracy = n_correct / len(examples)

    if step > 0:
        avg_loss.backward()
        optimizer.step()

    if step % 5 == 0:
        prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
        print(f"Step {step}: loss={avg_loss.item():.4f}, acc={accuracy:.0%}, prompt='{prompt_text[:40]}...'")

# Final evaluation
print("\n" + "-" * 60)
print("Final evaluation with per-position z:")
print("-" * 60)

with torch.no_grad():
    prompt_tokens = [3, 256047]
    e = z.unsqueeze(1)
    for _ in range(12):
        di = torch.tensor([prompt_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        prompt_tokens.append(next_tok)
        if next_tok == 3:
            break
    if prompt_tokens[-1] == 3:
        prompt_tokens = prompt_tokens[:-1]

prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
prompt_len = len(prompt_tokens)
print(f"Prompt: '{prompt_text}'")

for input_text, target in examples:
    task_tokens = text_to_tokens(input_text)
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full_tokens = prompt_tokens + task_content.tolist()

    # Generate with per-position z
    seq = decode_with_per_position_z(z.detach(), prompt_len=prompt_len, tokens=full_tokens.copy(), max_new=5)
    answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()

    correct = answer.lower().startswith(target.lower())
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:20]}' (want: {target}) {mark}")
