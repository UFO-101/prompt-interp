"""
Comprehensive evaluation of conditioned SONAR optimization.
- Multiple seeds
- Multiple tasks
- Track when we first achieve good performance
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


def generate_prompt(z, max_len=15):
    with torch.no_grad():
        e = z.unsqueeze(1)
        tokens = [3, 256047]
        for _ in range(max_len):
            di = torch.tensor([tokens], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            tokens.append(next_tok)
            if next_tok == 3:
                break
        if tokens[-1] == 3:
            tokens = tokens[:-1]
    return tokens


def forward_and_loss(z, full_tokens, target_token_id):
    e = z.unsqueeze(1)
    input_tokens = torch.tensor([full_tokens[:-1]], device=dev)
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e, BatchLayout.of(e))
    if h.dim() == 4:
        h = h.squeeze(1)
    logits = sdm.decoder.final_proj(h)[0, -1, :]
    target = torch.tensor([target_token_id], device=dev, dtype=torch.long)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    pred = logits.argmax().item()
    return loss, pred


def evaluate(z, examples):
    """Evaluate accuracy on examples."""
    prompt_tokens = generate_prompt(z)
    n_correct = 0
    results = []

    for input_text, target in examples:
        with torch.no_grad():
            task_tokens = text_to_tokens(input_text)
            task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
            full = prompt_tokens + task_content.tolist()

            e = z.unsqueeze(1)
            for _ in range(5):
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

        correct = answer.lower().startswith(target.lower())
        if correct:
            n_correct += 1
        results.append((input_text, answer[:20], target, correct))

    return n_correct / len(examples), results, tokens_to_text(torch.tensor(prompt_tokens))


def optimize(seed_text, examples, n_steps=50, lr=0.01):
    """Run optimization and track progress."""
    with torch.no_grad():
        z_init = se.predict([seed_text], source_lang='eng_Latn').to(dev)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=lr)

    history = []
    first_perfect = None
    first_90 = None

    for step in range(n_steps + 1):
        optimizer.zero_grad()

        prompt_tokens = generate_prompt(z)
        total_loss = 0.0
        n_correct = 0

        for input_text, target_text in examples:
            target_tokens = text_to_tokens(target_text)
            target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
            target_token_id = target_content[0].item()

            task_tokens = text_to_tokens(input_text)
            task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
            full_tokens = prompt_tokens + task_content.tolist() + [target_token_id]

            loss, pred = forward_and_loss(z, full_tokens, target_token_id)
            total_loss = total_loss + loss
            if pred == target_token_id:
                n_correct += 1

        avg_loss = total_loss / len(examples)
        accuracy = n_correct / len(examples)
        prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

        history.append({
            'step': step,
            'loss': avg_loss.item(),
            'accuracy': accuracy,
            'prompt': prompt_text
        })

        if accuracy >= 0.9 and first_90 is None:
            first_90 = step
        if accuracy >= 1.0 and first_perfect is None:
            first_perfect = step

        if step > 0:
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optimizer.step()

    return z, history, first_90, first_perfect


# ============================================================================
# TASK 1: Antonyms
# ============================================================================
print("\n" + "=" * 80)
print("TASK 1: ANTONYMS")
print("=" * 80)

antonym_examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

antonym_seeds = [
    "The opposite of hot is cold.",
    "Find the antonym: hot becomes cold.",
    "Reverse: up goes to down.",
    "Hello world, how are you today?",  # Unrelated seed
    "The quick brown fox jumps over the lazy dog.",  # Unrelated
]

print("\nTesting different seeds:")
for seed in antonym_seeds:
    print(f"\n--- Seed: '{seed[:50]}...' ---")
    z, history, first_90, first_perfect = optimize(seed, antonym_examples, n_steps=30)

    print(f"  First 90% acc at step: {first_90}")
    print(f"  First 100% acc at step: {first_perfect}")

    # Show prompts at key points
    if first_90 is not None:
        h = history[first_90]
        print(f"  Prompt at 90%: '{h['prompt'][:60]}...' (loss={h['loss']:.4f})")

    if first_perfect is not None and first_perfect != first_90:
        h = history[first_perfect]
        print(f"  Prompt at 100%: '{h['prompt'][:60]}...' (loss={h['loss']:.4f})")

    # Final
    h = history[-1]
    print(f"  Final prompt: '{h['prompt'][:60]}...' (loss={h['loss']:.4f}, acc={h['accuracy']:.0%})")

    # Evaluate
    acc, results, _ = evaluate(z, antonym_examples)
    print(f"  Task results:")
    for inp, ans, tgt, cor in results:
        mark = "OK" if cor else "X"
        print(f"    {inp} -> '{ans}' (want: {tgt}) {mark}")


# ============================================================================
# TASK 2: Capitalization
# ============================================================================
print("\n" + "=" * 80)
print("TASK 2: CAPITALIZATION")
print("=" * 80)

capital_examples = [
    ("hello -> ", "HELLO"),
    ("world -> ", "WORLD"),
    ("python -> ", "PYTHON"),
    ("code -> ", "CODE"),
    ("test -> ", "TEST"),
    ("data -> ", "DATA"),
]

capital_seeds = [
    "Convert to uppercase: hello becomes HELLO.",
    "UPPERCASE: word -> WORD",
    "Make it big: small -> SMALL",
    "The cat sat on the mat.",  # Unrelated
]

print("\nTesting different seeds:")
for seed in capital_seeds:
    print(f"\n--- Seed: '{seed[:50]}...' ---")
    z, history, first_90, first_perfect = optimize(seed, capital_examples, n_steps=30)

    print(f"  First 90% acc at step: {first_90}")
    print(f"  First 100% acc at step: {first_perfect}")

    h = history[-1]
    print(f"  Final prompt: '{h['prompt'][:60]}...' (loss={h['loss']:.4f}, acc={h['accuracy']:.0%})")

    acc, results, _ = evaluate(z, capital_examples)
    print(f"  Task results:")
    for inp, ans, tgt, cor in results[:3]:  # Show first 3
        mark = "OK" if cor else "X"
        print(f"    {inp} -> '{ans}' (want: {tgt}) {mark}")


# ============================================================================
# TASK 3: Simple arithmetic (next number)
# ============================================================================
print("\n" + "=" * 80)
print("TASK 3: NEXT NUMBER")
print("=" * 80)

number_examples = [
    ("1 -> ", "2"),
    ("5 -> ", "6"),
    ("10 -> ", "11"),
    ("99 -> ", "100"),
    ("7 -> ", "8"),
    ("3 -> ", "4"),
]

number_seeds = [
    "Add one: 1 becomes 2, 5 becomes 6.",
    "Next number: 3 -> 4",
    "Count up: 7 goes to 8",
    "I like pizza and pasta.",  # Unrelated
]

print("\nTesting different seeds:")
for seed in number_seeds:
    print(f"\n--- Seed: '{seed[:50]}...' ---")
    z, history, first_90, first_perfect = optimize(seed, number_examples, n_steps=30)

    print(f"  First 90% acc at step: {first_90}")
    print(f"  First 100% acc at step: {first_perfect}")

    h = history[-1]
    print(f"  Final prompt: '{h['prompt'][:60]}...' (loss={h['loss']:.4f}, acc={h['accuracy']:.0%})")

    acc, results, _ = evaluate(z, number_examples)
    print(f"  Task results:")
    for inp, ans, tgt, cor in results[:3]:
        mark = "OK" if cor else "X"
        print(f"    {inp} -> '{ans}' (want: {tgt}) {mark}")


# ============================================================================
# TASK 4: Translation (simple words)
# ============================================================================
print("\n" + "=" * 80)
print("TASK 4: ENGLISH TO SPANISH")
print("=" * 80)

translate_examples = [
    ("cat -> ", "gato"),
    ("dog -> ", "perro"),
    ("house -> ", "casa"),
    ("water -> ", "agua"),
    ("sun -> ", "sol"),
    ("moon -> ", "luna"),
]

translate_seeds = [
    "English to Spanish: cat is gato, dog is perro.",
    "Translate: house -> casa",
    "En español: water = agua",
    "The weather is nice today.",  # Unrelated
]

print("\nTesting different seeds:")
for seed in translate_seeds:
    print(f"\n--- Seed: '{seed[:50]}...' ---")
    z, history, first_90, first_perfect = optimize(seed, translate_examples, n_steps=40)

    print(f"  First 90% acc at step: {first_90}")
    print(f"  First 100% acc at step: {first_perfect}")

    h = history[-1]
    print(f"  Final prompt: '{h['prompt'][:60]}...' (loss={h['loss']:.4f}, acc={h['accuracy']:.0%})")

    acc, results, _ = evaluate(z, translate_examples)
    print(f"  Task results:")
    for inp, ans, tgt, cor in results[:3]:
        mark = "OK" if cor else "X"
        print(f"    {inp} -> '{ans}' (want: {tgt}) {mark}")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key observations:
1. How fast does optimization converge for different tasks?
2. Do related seeds help vs unrelated seeds?
3. What do the prompts look like at early vs late stopping?
4. Which tasks are easier/harder to optimize for?
""")
