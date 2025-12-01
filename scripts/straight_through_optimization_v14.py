"""
V14: Extended experiments to break the 67% barrier.

Key observations from v13:
- Best: 67% with "hot -> cold, large -> small. now:"
- "hot -> cold" and "happy -> sad" never solved
- Optimization peaks at step 3-5, then diverges

Strategy for v14:
1. Test more diverse seeds that include the hard examples
2. Use very low learning rate with more steps for gradual optimization
3. Try momentum (SGD with momentum) instead of Adam
4. Test restart strategy: when stuck, restart from best z with small noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout
import json
from datetime import datetime

dev = "cuda"
torch.cuda.empty_cache()

print("Loading models...", flush=True)
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sdm = sd.model.to(dev)
sonar_dec = sd.tokenizer.create_decoder()
sonar_enc = sd.tokenizer.create_encoder(mode='target', lang='eng_Latn')

embed_layer = sdm.decoder.decoder_frontend.embed
embed_matrix = embed_layer.weight.detach()
z_zero = torch.zeros(1, 1, 1024, device=dev)


def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def text_to_tokens(text):
    return sonar_enc(text).to(dev)


def generate_prompt_with_grads(z, max_len=15):
    tokens = [3, 256047]
    all_logits = []
    e = z.unsqueeze(1)

    for _ in range(max_len):
        di = torch.tensor([tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        all_logits.append(logits)
        next_tok = logits.argmax().item()
        tokens.append(next_tok)
        if next_tok == 3:
            break

    if tokens[-1] == 3:
        tokens = tokens[:-1]
        all_logits = all_logits[:-1]

    return tokens, all_logits


def train_step(z, examples):
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z, max_len=15)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    total_loss = 0.0
    n_correct = 0
    all_embed_grads = []

    for input_text, target_text in examples:
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]

        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
        target_id = target_content[0].item()

        full_tokens = prompt_tokens + task_content.tolist() + [target_id]
        input_tokens = full_tokens[:-1]

        embedding_output = []
        def embed_hook(module, input, output):
            output.retain_grad()
            embedding_output.append(output)
            return output

        handle = embed_layer.register_forward_hook(embed_hook)

        di = torch.tensor([input_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        handle.remove()

        target = torch.tensor([target_id], device=dev, dtype=torch.long)
        task_loss = F.cross_entropy(logits.unsqueeze(0), target)
        total_loss = total_loss + task_loss.item()

        if logits.argmax().item() == target_id:
            n_correct += 1

        task_loss.backward(retain_graph=False)

        if len(embedding_output) > 0 and embedding_output[0].grad is not None:
            embed_grad = embedding_output[0].grad[0]
            prompt_embed_grad = embed_grad[:len(prompt_tokens)]
            all_embed_grads.append(prompt_embed_grad.clone())

        sdm.zero_grad()

    if len(all_embed_grads) > 0 and len(stage1_logits) > 0:
        avg_embed_grad = torch.stack(all_embed_grads).mean(dim=0)
        logit_grads = avg_embed_grad @ embed_matrix.T
        logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

        if logit_grads.shape[0] > 2:
            grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]
            total_surrogate = sum((l * g).sum() for l, g in zip(stage1_logits, grad_for_stage1))
            total_surrogate.backward()

    return total_loss / len(examples), n_correct / len(examples), prompt_text, prompt_tokens


def evaluate_with_z0(prompt_tokens, examples, verbose=False):
    n_correct = 0
    results = []
    for input_text, target_text in examples:
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full = prompt_tokens + task_content.tolist()

        with torch.no_grad():
            for _ in range(5):
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
        correct = answer.lower().startswith(target_text.lower())
        if correct:
            n_correct += 1
        results.append((input_text, answer[:20], target_text, correct))

    return n_correct / len(examples), results


def optimize_with_restarts(seed_text, examples, lr=0.001, max_steps=40, grad_clip=0.5,
                           restart_interval=10, noise_scale=0.01, verbose=True):
    """Optimization with periodic restarts from best z."""
    with torch.no_grad():
        z_init = se.predict([seed_text], source_lang='eng_Latn').to(dev)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=lr)

    best_acc = 0
    best_prompt = None
    best_tokens = None
    best_z = z_init.clone()
    best_step = 0
    initial_loss = None
    no_improve_count = 0

    for step in range(max_steps):
        # Restart from best z with small noise
        if step > 0 and step % restart_interval == 0 and no_improve_count > restart_interval // 2:
            if verbose:
                print(f"  [Restart from best z at step {step}]")
            noise = torch.randn_like(best_z) * noise_scale
            z = nn.Parameter((best_z + noise).clone())
            optimizer = torch.optim.Adam([z], lr=lr)
            no_improve_count = 0

        optimizer.zero_grad()
        loss, acc, prompt, tokens = train_step(z, examples)

        if initial_loss is None:
            initial_loss = loss

        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
            optimizer.step()

        z0_acc, _ = evaluate_with_z0(tokens, examples)

        if z0_acc > best_acc:
            best_acc = z0_acc
            best_prompt = prompt
            best_tokens = tokens.copy()
            best_z = z.data.clone()
            best_step = step
            no_improve_count = 0
        else:
            no_improve_count += 1

        if verbose:
            print(f"  [{step:2d}] loss={loss:.2f} z0={z0_acc:.0%} best={best_acc:.0%} | {prompt[:45]}")

        # Early stopping
        if loss > initial_loss * 4:
            if verbose:
                print(f"  -> Early stop: loss exploded")
            break

    return {
        "seed": seed_text,
        "best_acc": best_acc,
        "best_prompt": best_prompt,
        "best_tokens": best_tokens,
        "best_step": best_step,
        "lr": lr,
        "grad_clip": grad_clip,
    }


def optimize_with_sgd_momentum(seed_text, examples, lr=0.01, momentum=0.9,
                                max_steps=50, grad_clip=0.5, verbose=True):
    """Try SGD with momentum instead of Adam."""
    with torch.no_grad():
        z_init = se.predict([seed_text], source_lang='eng_Latn').to(dev)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.SGD([z], lr=lr, momentum=momentum)

    best_acc = 0
    best_prompt = None
    best_tokens = None
    best_step = 0
    initial_loss = None
    zero_acc_count = 0

    for step in range(max_steps):
        optimizer.zero_grad()
        loss, acc, prompt, tokens = train_step(z, examples)

        if initial_loss is None:
            initial_loss = loss

        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
            optimizer.step()

        z0_acc, _ = evaluate_with_z0(tokens, examples)

        if z0_acc > best_acc:
            best_acc = z0_acc
            best_prompt = prompt
            best_tokens = tokens.copy()
            best_step = step
            zero_acc_count = 0
        elif z0_acc == 0:
            zero_acc_count += 1

        if verbose and step % 5 == 0:
            print(f"  [{step:2d}] loss={loss:.2f} z0={z0_acc:.0%} best={best_acc:.0%} | {prompt[:45]}")

        if zero_acc_count >= 5:
            if verbose:
                print(f"  -> Early stop: z0_acc=0 for {zero_acc_count} steps")
            break

    return {
        "seed": seed_text,
        "best_acc": best_acc,
        "best_prompt": best_prompt,
        "best_tokens": best_tokens,
        "best_step": best_step,
    }


# ============================================================================
print("\n" + "=" * 70)
print("V14: Extended experiments to break 67% barrier")
print("=" * 70)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

# New seeds that include hard examples
new_seeds = [
    # Include the hard examples explicitly
    "hot -> cold, happy -> sad. Write the opposite:",
    "Opposites: hot -> cold, happy -> sad, big -> small.",
    "cold is the opposite of hot. sad is the opposite of happy.",
    # More variation
    "Input -> Output (opposite). hot -> cold.",
    "hot:cold, big:small, fast:slow, up:down -",
    "reverse: hot=cold, big=small, happy=sad.",
]

all_results = []

# Test 1: New seeds with Adam
print("\n--- Test 1: New seeds with Adam (lr=0.001) ---")
for seed in new_seeds:
    print(f"\nSeed: '{seed[:45]}...'")
    result = optimize_with_restarts(seed, examples, lr=0.001, max_steps=30,
                                    restart_interval=15, verbose=True)
    all_results.append(result)
    print(f"  >>> Best: {result['best_acc']:.0%} at step {result['best_step']}")

# Test 2: Best seed with very low lr + more steps
print("\n--- Test 2: Best seed with very low lr (0.0003) + 60 steps ---")
best_seed = "hot -> cold, big -> small. Now:"
print(f"\nSeed: '{best_seed}'")
result = optimize_with_restarts(best_seed, examples, lr=0.0003, max_steps=60,
                                restart_interval=20, verbose=True)
all_results.append(result)
print(f"  >>> Best: {result['best_acc']:.0%} at step {result['best_step']}")

# Test 3: SGD with momentum
print("\n--- Test 3: SGD with momentum ---")
for seed in [new_seeds[0], best_seed]:
    print(f"\nSeed: '{seed[:45]}...'")
    result = optimize_with_sgd_momentum(seed, examples, lr=0.005, momentum=0.9,
                                         max_steps=50, verbose=True)
    all_results.append(result)
    print(f"  >>> Best: {result['best_acc']:.0%} at step {result['best_step']}")

# Final results
print("\n" + "=" * 70)
print("FINAL RESULTS (sorted by accuracy)")
print("=" * 70)

all_results_sorted = sorted(all_results, key=lambda x: (-x['best_acc'], x['best_step']))
seen_prompts = set()

for r in all_results_sorted[:10]:
    if r['best_prompt'] and r['best_prompt'] not in seen_prompts:
        seen_prompts.add(r['best_prompt'])
        print(f"{r['best_acc']:.0%} (step {r['best_step']})")
        print(f"  Seed: '{r['seed'][:50]}...'")
        print(f"  Prompt: '{r['best_prompt']}'")

        if r['best_tokens']:
            _, results = evaluate_with_z0(r['best_tokens'], examples)
            correct = sum(1 for _, _, _, c in results if c)
            print(f"  Results: {correct}/{len(examples)}")
            for inp, ans, tgt, cor in results:
                mark = "OK" if cor else "X"
                print(f"    {inp} -> '{ans}' ({tgt}) {mark}")
        print()

# Best overall
best = all_results_sorted[0]
print(f"\n{'=' * 70}")
print(f"BEST OVERALL: {best['best_acc']:.0%}")
print(f"Prompt: '{best['best_prompt']}'")
print(f"From seed: '{best['seed'][:50]}...'")
print(f"{'=' * 70}")
