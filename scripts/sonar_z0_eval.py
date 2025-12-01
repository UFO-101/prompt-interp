"""
Evaluate: Does the learned prompt work with z=0?
Or does it only work because z is used at inference time?
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


def generate_with_z(z, prompt_tokens, task_tokens, max_new=5):
    """Generate continuation using z for cross-attention."""
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full = prompt_tokens + task_content.tolist()

    e = z.unsqueeze(1) if z.dim() == 2 else z.unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        for _ in range(max_new):
            di = torch.tensor([full], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            full.append(next_tok)
            if next_tok == 3:
                break

    full_text = tokens_to_text(torch.tensor(full))
    base_text = tokens_to_text(torch.tensor(prompt_tokens + task_content.tolist()))
    answer = full_text[len(base_text):].strip()
    return answer


def generate_with_z0(prompt_tokens, task_tokens, max_new=5):
    """Generate continuation using z=0 for cross-attention."""
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full = prompt_tokens + task_content.tolist()

    with torch.no_grad():
        for _ in range(max_new):
            di = torch.tensor([full], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            full.append(next_tok)
            if next_tok == 3:
                break

    full_text = tokens_to_text(torch.tensor(full))
    base_text = tokens_to_text(torch.tensor(prompt_tokens + task_content.tolist()))
    answer = full_text[len(base_text):].strip()
    return answer


def optimize_z(seed_text, examples, n_steps=30):
    """Optimize z for task."""
    with torch.no_grad():
        z_init = se.predict([seed_text], source_lang='eng_Latn').to(dev)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=0.01)

    for step in range(n_steps + 1):
        optimizer.zero_grad()

        # Generate prompt
        with torch.no_grad():
            e = z.unsqueeze(1)
            prompt_tokens = [3, 256047]
            for _ in range(15):
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

        total_loss = 0.0
        for input_text, target_text in examples:
            target_tokens = text_to_tokens(target_text)
            target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
            target_token_id = target_content[0].item()

            task_tokens = text_to_tokens(input_text)
            task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
            full_tokens = prompt_tokens + task_content.tolist() + [target_token_id]

            e = z.unsqueeze(1)
            input_toks = torch.tensor([full_tokens[:-1]], device=dev)
            h = sdm.decode(input_toks, BatchLayout.of(input_toks), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]

            target = torch.tensor([target_token_id], device=dev, dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            total_loss = total_loss + loss

        if step > 0:
            (total_loss / len(examples)).backward()
            optimizer.step()

    # Return final z and prompt tokens
    with torch.no_grad():
        e = z.unsqueeze(1)
        final_prompt_tokens = [3, 256047]
        for _ in range(15):
            di = torch.tensor([final_prompt_tokens], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            final_prompt_tokens.append(next_tok)
            if next_tok == 3:
                break
        if final_prompt_tokens[-1] == 3:
            final_prompt_tokens = final_prompt_tokens[:-1]

    return z.detach(), final_prompt_tokens


# ============================================================================
# Test 1: Antonyms
# ============================================================================
print("\n" + "=" * 80)
print("TEST: ANTONYMS")
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
print(f"\nOptimizing from seed: '{seed}'")

z_opt, prompt_tokens = optimize_z(seed, examples, n_steps=30)
prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
print(f"\nLearned prompt: '{prompt_text}'")

print("\n" + "-" * 60)
print("Evaluation with OPTIMIZED z (what we trained with):")
print("-" * 60)
n_correct = 0
for input_text, target in examples:
    task_tokens = text_to_tokens(input_text)
    answer = generate_with_z(z_opt, prompt_tokens, task_tokens)
    correct = answer.lower().startswith(target.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:25]}' (want: {target}) {mark}")
print(f"Accuracy: {n_correct}/{len(examples)} = {n_correct/len(examples):.0%}")

print("\n" + "-" * 60)
print("Evaluation with z=0 (just using the prompt TEXT):")
print("-" * 60)
n_correct = 0
for input_text, target in examples:
    task_tokens = text_to_tokens(input_text)
    answer = generate_with_z0(prompt_tokens, task_tokens)
    correct = answer.lower().startswith(target.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:25]}' (want: {target}) {mark}")
print(f"Accuracy: {n_correct}/{len(examples)} = {n_correct/len(examples):.0%}")


# ============================================================================
# Test 2: What if we just use the prompt text with z=0, no optimization?
# ============================================================================
print("\n" + "=" * 80)
print("CONTROL: Just prefix with prompt text, z=0, NO optimization")
print("=" * 80)

# Manually create some prompt texts
test_prompts = [
    "slow slow slow slow slow slow slow slow slow slow",
    "cold cold cold cold cold cold cold cold cold cold",
    "The opposite of hot is cold.",
    "hot -> cold, big -> small, fast -> slow",
    "100 100 100 100 100 100 100 100 100 100",
]

for prompt_text in test_prompts:
    print(f"\nPrompt: '{prompt_text[:50]}...'")

    # Tokenize the prompt
    prompt_tokens_raw = text_to_tokens(prompt_text)
    # Add BOS and lang tag
    prompt_tokens = [3, 256047] + prompt_tokens_raw[2:].tolist()
    if prompt_tokens[-1] == 3:
        prompt_tokens = prompt_tokens[:-1]

    n_correct = 0
    for input_text, target in examples:
        task_tokens = text_to_tokens(input_text)
        answer = generate_with_z0(prompt_tokens, task_tokens)
        correct = answer.lower().startswith(target.lower())
        if correct:
            n_correct += 1
    print(f"  Accuracy with z=0: {n_correct}/{len(examples)} = {n_correct/len(examples):.0%}")


# ============================================================================
# Test 3: Translation
# ============================================================================
print("\n" + "=" * 80)
print("TEST: TRANSLATION (English to Spanish)")
print("=" * 80)

translate_examples = [
    ("cat -> ", "gato"),
    ("dog -> ", "perro"),
    ("house -> ", "casa"),
    ("water -> ", "agua"),
    ("sun -> ", "sol"),
    ("moon -> ", "luna"),
]

seed = "English to Spanish: cat is gato."
print(f"\nOptimizing from seed: '{seed}'")

z_opt, prompt_tokens = optimize_z(seed, translate_examples, n_steps=30)
prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
print(f"\nLearned prompt: '{prompt_text}'")

print("\n" + "-" * 60)
print("Evaluation with OPTIMIZED z:")
print("-" * 60)
n_correct = 0
for input_text, target in translate_examples:
    task_tokens = text_to_tokens(input_text)
    answer = generate_with_z(z_opt, prompt_tokens, task_tokens)
    correct = answer.lower().startswith(target.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:25]}' (want: {target}) {mark}")
print(f"Accuracy: {n_correct}/{len(translate_examples)} = {n_correct/len(translate_examples):.0%}")

print("\n" + "-" * 60)
print("Evaluation with z=0:")
print("-" * 60)
n_correct = 0
for input_text, target in translate_examples:
    task_tokens = text_to_tokens(input_text)
    answer = generate_with_z0(prompt_tokens, task_tokens)
    correct = answer.lower().startswith(target.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:25]}' (want: {target}) {mark}")
print(f"Accuracy: {n_correct}/{len(translate_examples)} = {n_correct/len(translate_examples):.0%}")


# ============================================================================
# Test 4: Numbers
# ============================================================================
print("\n" + "=" * 80)
print("TEST: NEXT NUMBER")
print("=" * 80)

number_examples = [
    ("1 -> ", "2"),
    ("5 -> ", "6"),
    ("10 -> ", "11"),
    ("99 -> ", "100"),
    ("7 -> ", "8"),
    ("3 -> ", "4"),
]

seed = "Next number: 3 -> 4, 7 -> 8"
print(f"\nOptimizing from seed: '{seed}'")

z_opt, prompt_tokens = optimize_z(seed, number_examples, n_steps=30)
prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
print(f"\nLearned prompt: '{prompt_text}'")

print("\n" + "-" * 60)
print("Evaluation with OPTIMIZED z:")
print("-" * 60)
n_correct = 0
for input_text, target in number_examples:
    task_tokens = text_to_tokens(input_text)
    answer = generate_with_z(z_opt, prompt_tokens, task_tokens)
    correct = answer.startswith(target)
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:25]}' (want: {target}) {mark}")
print(f"Accuracy: {n_correct}/{len(number_examples)} = {n_correct/len(number_examples):.0%}")

print("\n" + "-" * 60)
print("Evaluation with z=0:")
print("-" * 60)
n_correct = 0
for input_text, target in number_examples:
    task_tokens = text_to_tokens(input_text)
    answer = generate_with_z0(prompt_tokens, task_tokens)
    correct = answer.startswith(target)
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:25]}' (want: {target}) {mark}")
print(f"Accuracy: {n_correct}/{len(number_examples)} = {n_correct/len(number_examples):.0%}")


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key question: Is the task performance coming from:
  A) The optimized z vector (soft prompt / cross-attention)
  B) The prompt text itself (hard tokens)

If z=0 works just as well, then it's the text.
If z=0 fails, then the z vector is doing the work.
""")
