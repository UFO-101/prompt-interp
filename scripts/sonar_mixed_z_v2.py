"""
Mixed z optimization v2: Use actual z=0 for task positions, not masking.

Approach: Hook the encoder output to swap between z and z_zero for different positions.
Since encoder output is used for K,V in cross-attention, we can tile it per-position.
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

z_zero = torch.zeros(1, 1, 1024, device=dev)

def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())

def text_to_tokens(text):
    tokens = sonar_enc(text)
    return tokens.to(dev)


# ============================================================================
# Key insight: We can't easily do per-position z in a single forward pass.
# The encoder output (z) is shared across all positions in cross-attention.
#
# Alternative approach:
# - Accept that answer generation uses z=0 for all positions
# - But for gradient computation, we do teacher-forcing where:
#   - Prompt tokens "see" z during their computation
#   - Answer tokens are computed (we don't need their cross-attn to z)
# - The loss is on the answer, but gradients flow through prompt hidden states
#
# Actually, the decoder is autoregressive, so each position only attends to
# previous positions via self-attention, and to z via cross-attention.
#
# The gradient flow issue: even if answer positions use z=0, the answer
# depends on the prompt TOKENS (via self-attention), not directly on z.
#
# So the real question: can we make gradients flow through the prompt
# hidden states to z, even though the answer is generated with z=0?
# ============================================================================

print("\n" + "=" * 80)
print("Simpler approach: Two-phase forward")
print("=" * 80)

# The key insight: we do TWO forward passes
# Phase 1: Forward prompt with z, get hidden states (with gradients)
# Phase 2: Forward full sequence with z=0, but inject prompt hidden states
#
# Actually this is complex. Let me try something simpler:
# Just use the gradient from the LOSS on answer to flow back through
# the decoder, which WILL flow to z if z is used in the forward pass.
#
# The issue in our previous attempts was that we MASKED the cross-attention,
# which broke the model. Instead, let's just use full z for everything
# during the teacher-forced forward, but train to predict what z=0 would predict.

# Actually, let me just verify the core optimization works:
# - Forward with full z (teacher forcing)
# - Loss on predicting a target token
# - Gradients should flow to z
# - z update should change the predictions

test_text = "The opposite of hot is cold. The opposite of big is small."
with torch.no_grad():
    z_init = se.predict([test_text], source_lang='eng_Latn').to(dev)

print(f"Seed: '{test_text}'")

# Generate prompt
with torch.no_grad():
    e = z_init.unsqueeze(1)
    prompt_tokens = [3, 256047]
    for _ in range(15):
        di = torch.tensor([prompt_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        prompt_tokens.append(next_tok)
        if next_tok == 3:
            break
    if prompt_tokens[-1] == 3:
        prompt_tokens = prompt_tokens[:-1]
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
    print(f"Prompt: '{prompt_text}'")

# What does z=0 produce after this prompt + task input?
task_input = "hot -> "
task_tokens = text_to_tokens(task_input)
task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
full_tokens = prompt_tokens + task_content.tolist()

print(f"\nFull input: '{tokens_to_text(torch.tensor(full_tokens))}'")

# Generate with z=0
print("\nGenerate with z=0:")
with torch.no_grad():
    seq = full_tokens.copy()
    for _ in range(5):
        di = torch.tensor([seq], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        seq.append(next_tok)
        if next_tok == 3:
            break
    full_text = tokens_to_text(torch.tensor(seq))
    base_text = tokens_to_text(torch.tensor(full_tokens))
    answer = full_text[len(base_text):]
    print(f"  Answer: '{answer}'")

# ============================================================================
# Now let's try optimization
# Goal: Make z=0 produce "cold" after "prompt + hot ->"
#
# Problem: z=0 is fixed (zeros), so we can't optimize it.
# What we CAN optimize is z (the prompt z), but z doesn't directly
# affect the z=0 generation.
#
# The ONLY way z affects z=0 generation is through the TOKENS
# that z produces (the prompt text).
#
# So we need: z -> prompt_tokens -> z=0 continuation
# But prompt_tokens are discrete!
#
# This is why we tried masking, but masking broke things.
# ============================================================================

print("\n" + "=" * 80)
print("Alternative: Use z for the FULL forward, train on task loss")
print("=" * 80)

# If we use z for everything (not z=0), then:
# - z affects all positions
# - Loss on answer flows to z
# - But this is not "unconditioned" generation anymore
#
# However, if z encodes "how to do the task", then the model should
# learn to produce correct answers.

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.01)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
]

print("\nOptimizing z to make CONDITIONED output predict targets...")
print("(Not z=0, but z-conditioned)")

for step in range(31):
    optimizer.zero_grad()

    total_loss = 0.0
    n_correct = 0

    for input_text, target_text in examples:
        # Generate prompt with z
        with torch.no_grad():
            e_gen = z.unsqueeze(0).unsqueeze(1)
            prompt_tokens = [3, 256047]
            for _ in range(12):
                di = torch.tensor([prompt_tokens], device=dev)
                h = sdm.decode(di, BatchLayout.of(di), e_gen, BatchLayout.of(e_gen))
                logits = sdm.decoder.final_proj(h)[0, -1, :]
                next_tok = logits.argmax().item()
                prompt_tokens.append(next_tok)
                if next_tok == 3:
                    break
            if prompt_tokens[-1] == 3:
                prompt_tokens = prompt_tokens[:-1]

        # Add task
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full_tokens = prompt_tokens + task_content.tolist()

        # Get target token
        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
        target_token = target_content[0]

        # Add target for teacher forcing
        full_with_target = full_tokens + [target_token.item()]

        # Forward with z (WITH gradients)
        # z shape: [1, 1024] -> need [1, 1, 1024] for encoder output
        e_param = z.unsqueeze(1) if z.dim() == 2 else z.unsqueeze(0).unsqueeze(1)
        input_tokens = torch.tensor(full_with_target[:-1], device=dev).unsqueeze(0)
        h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e_param, BatchLayout.of(e_param))
        # h might have extra dim, squeeze it
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        # Loss
        target_idx = torch.tensor([target_token.item()], device=dev, dtype=torch.long)
        loss = F.cross_entropy(logits.float().unsqueeze(0), target_idx)
        total_loss = total_loss + loss

        # Check prediction
        pred = logits.argmax().item()
        if pred == target_token.item():
            n_correct += 1

    avg_loss = total_loss / len(examples)
    accuracy = n_correct / len(examples)

    if step > 0:
        avg_loss.backward()
        optimizer.step()

    if step % 5 == 0:
        with torch.no_grad():
            e_gen = z.unsqueeze(0).unsqueeze(1)
            prompt_tokens = [3, 256047]
            for _ in range(12):
                di = torch.tensor([prompt_tokens], device=dev)
                h = sdm.decode(di, BatchLayout.of(di), e_gen, BatchLayout.of(e_gen))
                logits = sdm.decoder.final_proj(h)[0, -1, :]
                next_tok = logits.argmax().item()
                prompt_tokens.append(next_tok)
                if next_tok == 3:
                    break
            prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

        print(f"\nStep {step}: loss={avg_loss.item():.4f}, acc={accuracy:.2f}")
        print(f"  Prompt: '{prompt_text[:50]}'")

# Final eval
print("\n" + "=" * 80)
print("FINAL EVALUATION (with optimized z)")
print("=" * 80)

with torch.no_grad():
    e_gen = z.unsqueeze(0).unsqueeze(1)
    prompt_tokens = [3, 256047]
    for _ in range(15):
        di = torch.tensor([prompt_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e_gen, BatchLayout.of(e_gen))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        prompt_tokens.append(next_tok)
        if next_tok == 3:
            break
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
    print(f"Prompt: '{prompt_text}'")

print("\nConditioned (z) generation:")
for input_text, target in examples:
    with torch.no_grad():
        if prompt_tokens[-1] == 3:
            pt = prompt_tokens[:-1]
        else:
            pt = prompt_tokens

        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full = pt + task_content.tolist()

        for _ in range(5):
            di = torch.tensor([full], device=dev)
            e_gen = z.unsqueeze(0).unsqueeze(1)
            h = sdm.decode(di, BatchLayout.of(di), e_gen, BatchLayout.of(e_gen))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            full.append(next_tok)
            if next_tok == 3:
                break

        full_text = tokens_to_text(torch.tensor(full))
        base_text = tokens_to_text(torch.tensor(pt + task_content.tolist()))
        answer = full_text[len(base_text):].strip()

    correct = answer.lower().startswith(target.lower())
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:20]}' (want: {target}) {mark}")

print("\nUnconditioned (z=0) generation:")
for input_text, target in examples:
    with torch.no_grad():
        if prompt_tokens[-1] == 3:
            pt = prompt_tokens[:-1]
        else:
            pt = prompt_tokens

        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full = pt + task_content.tolist()

        for _ in range(5):
            di = torch.tensor([full], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            full.append(next_tok)
            if next_tok == 3:
                break

        full_text = tokens_to_text(torch.tensor(full))
        base_text = tokens_to_text(torch.tensor(pt + task_content.tolist()))
        answer = full_text[len(base_text):].strip()

    correct = answer.lower().startswith(target.lower())
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:20]}' (want: {target}) {mark}")
