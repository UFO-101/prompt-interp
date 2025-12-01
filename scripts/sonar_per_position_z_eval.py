"""
Evaluate: Does the learned prompt work with z=0 at ALL positions?
This tests if the TEXT itself carries the task information.
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
    return sonar_enc(text).to(dev)


def generate_with_z0(tokens, max_new=5):
    """Generate with z=0 for all positions."""
    seq = tokens.copy() if isinstance(tokens, list) else tokens.tolist()
    for _ in range(max_new):
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
    return seq


def decode_with_per_position_z(z, prompt_len, tokens, max_new=5):
    """Decode with z for first prompt_len positions, z=0 for the rest."""
    seq = tokens.copy() if isinstance(tokens, list) else tokens.tolist()
    for _ in range(max_new):
        seq_len = len(seq)
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


def optimize_with_per_position_z(seed_text, examples, n_steps=20):
    """Optimize z using per-position conditioning."""
    with torch.no_grad():
        z_init = se.predict([seed_text], source_lang='eng_Latn').to(dev)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=0.01)

    for step in range(n_steps + 1):
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

        for input_text, target_text in examples:
            task_tokens = text_to_tokens(input_text)
            task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]

            target_tokens = text_to_tokens(target_text)
            target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
            target_id = target_content[0].item()

            full_tokens = prompt_tokens + task_content.tolist() + [target_id]
            seq_len = len(full_tokens) - 1

            # Per-position encoder output
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

        if step > 0:
            (total_loss / len(examples)).backward()
            optimizer.step()

    # Get final prompt
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

    return z.detach(), prompt_tokens


# ============================================================================
# Test: Antonyms
# ============================================================================
print("\n" + "=" * 80)
print("ANTONYMS TASK")
print("=" * 80)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

seed = "The opposite of hot is cold."
print(f"Seed: '{seed}'")

z_opt, prompt_tokens = optimize_with_per_position_z(seed, examples, n_steps=20)
prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
prompt_len = len(prompt_tokens)

print(f"\nLearned prompt: '{prompt_text}'")

# Evaluate with per-position z (what we trained with)
print("\n1. Eval with per-position z (z for prompt, z=0 for task):")
n_correct = 0
for input_text, target in examples:
    task_tokens = text_to_tokens(input_text)
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full_tokens = prompt_tokens + task_content.tolist()

    seq = decode_with_per_position_z(z_opt, prompt_len, full_tokens, max_new=5)
    answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()

    correct = answer.lower().startswith(target.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"   {input_text} -> '{answer[:20]}' (want: {target}) {mark}")
print(f"   Accuracy: {n_correct}/{len(examples)}")

# Evaluate with z=0 everywhere (just prompt text, no z)
print("\n2. Eval with z=0 everywhere (just the prompt TEXT):")
n_correct = 0
for input_text, target in examples:
    task_tokens = text_to_tokens(input_text)
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full_tokens = prompt_tokens + task_content.tolist()

    seq = generate_with_z0(full_tokens, max_new=5)
    answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()

    correct = answer.lower().startswith(target.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"   {input_text} -> '{answer[:20]}' (want: {target}) {mark}")
print(f"   Accuracy: {n_correct}/{len(examples)}")


# ============================================================================
# Test: Does the TEXT work even from a different z?
# ============================================================================
print("\n" + "=" * 80)
print("TEST: Does the prompt TEXT work from random z?")
print("=" * 80)

# Get the prompt text and re-tokenize it (as if we only have the text, not z)
prompt_text_only = tokens_to_text(torch.tensor(prompt_tokens))
print(f"Prompt text: '{prompt_text_only}'")

# Re-encode this text
retokenized = text_to_tokens(prompt_text_only)
retokenized_full = [3, 256047] + retokenized[2:].tolist()
if retokenized_full[-1] == 3:
    retokenized_full = retokenized_full[:-1]

print(f"Re-tokenized: {len(retokenized_full)} tokens")

# Try with z=0
print("\nWith z=0 and re-tokenized prompt:")
n_correct = 0
for input_text, target in examples:
    task_tokens = text_to_tokens(input_text)
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full_tokens = retokenized_full + task_content.tolist()

    seq = generate_with_z0(full_tokens, max_new=5)
    answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()

    correct = answer.lower().startswith(target.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"   {input_text} -> '{answer[:20]}' (want: {target}) {mark}")
print(f"   Accuracy: {n_correct}/{len(examples)}")


# ============================================================================
# Control: Test handcrafted prompts with z=0
# ============================================================================
print("\n" + "=" * 80)
print("CONTROL: Handcrafted prompts with z=0")
print("=" * 80)

test_prompts = [
    "cold cold cold cold cold cold cold cold cold cold",
    "small small small small small small small small",
    "The opposite of hot is cold. The opposite of big is small.",
    "hot -> cold, big -> small, fast -> slow",
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt[:50]}...'")
    prompt_tok = text_to_tokens(prompt)
    prompt_full = [3, 256047] + prompt_tok[2:].tolist()
    if prompt_full[-1] == 3:
        prompt_full = prompt_full[:-1]

    n_correct = 0
    for input_text, target in examples[:3]:  # Just first 3
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full_tokens = prompt_full + task_content.tolist()

        seq = generate_with_z0(full_tokens, max_new=5)
        answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):].strip()

        correct = answer.lower().startswith(target.lower())
        if correct:
            n_correct += 1
    print(f"   Accuracy: {n_correct}/3")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key question: Is task performance from:
  A) The z vector at prompt positions (soft prompt)
  B) The prompt TEXT itself (hard tokens)

If "z=0 everywhere" works as well as "per-position z", then it's the text.
If "z=0 everywhere" fails, then z at prompt positions is essential.
""")
