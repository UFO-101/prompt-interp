"""
Investigate how well z=0 decoder can do tasks with different prompts.
Is this task even learnable with z=0?
"""

import torch
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


def generate_with_z0(prompt_tokens, task_tokens, max_new=8):
    """Generate continuation with z=0."""
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
    return answer, full


def evaluate_prompt(prompt_text, examples, verbose=True):
    """Evaluate a prompt with z=0."""
    prompt_tokens = text_to_tokens(prompt_text)
    prompt_full = [3, 256047] + prompt_tokens[2:].tolist()
    if prompt_full[-1] == 3:
        prompt_full = prompt_full[:-1]

    n_correct = 0
    results = []

    for input_text, target_text in examples:
        task_tokens = text_to_tokens(input_text)
        answer, _ = generate_with_z0(prompt_full, task_tokens)

        correct = answer.lower().startswith(target_text.lower())
        if correct:
            n_correct += 1
        results.append((input_text, answer[:25], target_text, correct))

    if verbose:
        print(f"\nPrompt: '{prompt_text[:60]}...'")
        for inp, ans, tgt, cor in results:
            mark = "OK" if cor else "X"
            print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")
        print(f"  Accuracy: {n_correct}/{len(examples)}")

    return n_correct / len(examples), results


# ============================================================================
# Test various prompt styles
# ============================================================================
print("\n" + "=" * 80)
print("ANTONYM TASK: Testing prompts with z=0 decoder")
print("=" * 80)

antonym_examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

prompts = [
    # Explicit instruction prompts
    "Find the antonym:",
    "Give the opposite:",
    "Opposite:",
    "The opposite is:",

    # Few-shot style
    "hot -> cold, big -> small. Now: ",
    "Antonyms: hot -> cold, big -> small. ",
    "hot is to cold as big is to small. Now ",

    # More examples
    "hot -> cold, big -> small, up -> down. ",
    "The opposite of hot is cold. The opposite of big is small. The opposite of ",

    # Different formats
    "hot:cold, big:small, ",
    "hot=cold, big=small, fast=slow, ",

    # Very explicit
    "Task: output the opposite word. Examples: hot -> cold, big -> small. Now: ",
    "I give you a word and you give its opposite. hot -> cold, big -> small. ",
]

results_summary = []
for prompt in prompts:
    acc, _ = evaluate_prompt(prompt, antonym_examples)
    results_summary.append((prompt[:50], acc))

print("\n" + "=" * 80)
print("SUMMARY (sorted by accuracy)")
print("=" * 80)
for prompt, acc in sorted(results_summary, key=lambda x: -x[1]):
    print(f"{acc:.0%} : '{prompt}...'")


# ============================================================================
# Test: What does z=0 decoder predict without any prompt?
# ============================================================================
print("\n" + "=" * 80)
print("NO PROMPT: What does z=0 decoder predict?")
print("=" * 80)

for input_text, target in antonym_examples[:3]:
    # Just task input, no prompt
    task_tokens = text_to_tokens(input_text)
    empty_prompt = [3, 256047]  # Just BOS and lang
    answer, _ = generate_with_z0(empty_prompt, task_tokens)
    print(f"  {input_text} -> '{answer[:30]}' (want: {target})")


# ============================================================================
# Test: Does z=0 decoder work better on simpler tasks?
# ============================================================================
print("\n" + "=" * 80)
print("SIMPLER TASK: Repetition")
print("=" * 80)

repeat_examples = [
    ("cat -> ", "cat"),
    ("dog -> ", "dog"),
    ("house -> ", "house"),
]

prompts_repeat = [
    "Repeat the word: cat -> cat, dog -> dog. ",
    "cat -> cat, dog -> dog, house -> house. ",
    "Echo: ",
]

for prompt in prompts_repeat:
    evaluate_prompt(prompt, repeat_examples)


print("\n" + "=" * 80)
print("SIMPLER TASK: First letter")
print("=" * 80)

letter_examples = [
    ("cat -> ", "c"),
    ("dog -> ", "d"),
    ("house -> ", "h"),
    ("apple -> ", "a"),
]

prompts_letter = [
    "First letter: cat -> c, dog -> d. ",
    "cat -> c, dog -> d, house -> h. ",
]

for prompt in prompts_letter:
    evaluate_prompt(prompt, letter_examples)


print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("""
The z=0 decoder is essentially a next-token predictor.
With a good prompt, it might do simple pattern completion.
But antonyms require semantic knowledge that may not be accessible
when z=0 (since z carries the "meaning" of the input).

Options:
1. Task might just be too hard for z=0 - need simpler task
2. Need much longer/better prompts with more examples
3. The optimization is working but the task ceiling is low
""")
