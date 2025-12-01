"""
V15: Systematic exploration of embedding space.

New approach: Instead of gradient-based optimization only, systematically
explore interpolations and combinations of good seeds.

Key ideas:
1. Interpolate between seeds that partially work
2. Try random perturbations in promising regions
3. Grid search around best found z vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout
import itertools

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


def decode_z(z, max_len=15):
    """Decode z to tokens (no gradients)."""
    tokens = [3, 256047]
    e = z.unsqueeze(1) if z.dim() == 2 else z.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
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


def evaluate_z(z, examples, verbose=False):
    """Evaluate a z vector on examples using z=0 for task."""
    prompt_tokens = decode_z(z)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

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

    acc = n_correct / len(examples)
    return acc, prompt_text, prompt_tokens, results


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


def quick_optimize(z_init, examples, lr=0.001, steps=10, grad_clip=0.5):
    """Quick optimization from a starting point."""
    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=lr)

    best_acc = 0
    best_z = z_init.clone()
    best_prompt = None

    for step in range(steps):
        optimizer.zero_grad()
        loss, acc, prompt, tokens = train_step(z, examples)

        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=grad_clip)
            optimizer.step()

        z0_acc, _, _, _ = evaluate_z(z.data, examples)

        if z0_acc > best_acc:
            best_acc = z0_acc
            best_z = z.data.clone()
            best_prompt = prompt

    return best_acc, best_z, best_prompt


# ============================================================================
print("\n" + "=" * 70)
print("V15: Systematic exploration of embedding space")
print("=" * 70)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

# Encode various seeds
seed_texts = [
    "hot -> cold, big -> small. Now:",
    "The opposite is:",
    "hot -> cold, happy -> sad. Write:",
    "Opposites: hot -> cold.",
    "cold is opposite of hot. sad is opposite of happy.",
    "Input -> Output: hot -> cold, big -> small.",
]

print("\n--- Encoding seeds ---")
seed_embeddings = {}
for text in seed_texts:
    with torch.no_grad():
        z = se.predict([text], source_lang='eng_Latn').to(dev)
    acc, prompt, tokens, results = evaluate_z(z, examples)
    seed_embeddings[text] = {"z": z, "acc": acc, "prompt": prompt}
    print(f"{acc:.0%}: '{text[:40]}...' -> '{prompt[:30]}...'")

# Sort by accuracy
sorted_seeds = sorted(seed_embeddings.items(), key=lambda x: -x[1]["acc"])
best_seeds = sorted_seeds[:3]
print(f"\nTop 3 seeds by initial accuracy:")
for name, data in best_seeds:
    print(f"  {data['acc']:.0%}: {name[:50]}")

# ============================================================================
print("\n--- Test 1: Interpolate between top 2 seeds ---")
z1 = best_seeds[0][1]["z"]
z2 = best_seeds[1][1]["z"]

best_interp = {"acc": 0, "alpha": None, "prompt": None, "z": None}

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    z_interp = alpha * z1 + (1 - alpha) * z2
    acc, prompt, tokens, results = evaluate_z(z_interp, examples)
    print(f"  alpha={alpha:.1f}: {acc:.0%} -> '{prompt[:40]}...'")

    if acc > best_interp["acc"]:
        best_interp = {"acc": acc, "alpha": alpha, "prompt": prompt, "z": z_interp.clone()}

print(f"\nBest interpolation: {best_interp['acc']:.0%} at alpha={best_interp['alpha']}")

# ============================================================================
print("\n--- Test 2: Random perturbations around best seed ---")
z_best = best_seeds[0][1]["z"]

best_perturb = {"acc": best_seeds[0][1]["acc"], "prompt": best_seeds[0][1]["prompt"], "z": z_best}

for scale in [0.01, 0.02, 0.05, 0.1]:
    print(f"\n  Scale: {scale}")
    for i in range(10):
        noise = torch.randn_like(z_best) * scale
        z_perturbed = z_best + noise
        acc, prompt, tokens, results = evaluate_z(z_perturbed, examples)

        if acc > best_perturb["acc"]:
            best_perturb = {"acc": acc, "prompt": prompt, "z": z_perturbed.clone()}
            print(f"    [NEW BEST] {acc:.0%}: '{prompt[:40]}...'")
        elif acc == best_perturb["acc"] and i < 3:
            print(f"    {acc:.0%}: '{prompt[:40]}...'")

print(f"\nBest perturbation: {best_perturb['acc']:.0%}")

# ============================================================================
print("\n--- Test 3: Quick optimize from promising starting points ---")
starting_points = [
    ("best_seed", best_seeds[0][1]["z"]),
    ("best_interp", best_interp["z"]),
    ("best_perturb", best_perturb["z"]),
]

best_overall = {"acc": 0, "prompt": None, "z": None, "source": None}

for name, z_start in starting_points:
    print(f"\n  Optimizing from {name}...")
    acc, z_opt, prompt = quick_optimize(z_start, examples, lr=0.001, steps=15)
    print(f"    Result: {acc:.0%} -> '{prompt[:40]}...'")

    if acc > best_overall["acc"]:
        best_overall = {"acc": acc, "prompt": prompt, "z": z_opt, "source": name}

    # Also try lower lr
    acc, z_opt, prompt = quick_optimize(z_start, examples, lr=0.0005, steps=20)
    print(f"    (lr=0.0005): {acc:.0%} -> '{prompt[:40]}...'")

    if acc > best_overall["acc"]:
        best_overall = {"acc": acc, "prompt": prompt, "z": z_opt, "source": f"{name}_low_lr"}

# ============================================================================
print("\n--- Test 4: Combining 3 seeds ---")
z1 = best_seeds[0][1]["z"]
z2 = best_seeds[1][1]["z"]
z3 = best_seeds[2][1]["z"] if len(best_seeds) > 2 else z1

best_combo = {"acc": 0, "weights": None, "prompt": None}

# Try different combinations
weights_list = [
    (0.5, 0.3, 0.2),
    (0.6, 0.2, 0.2),
    (0.4, 0.4, 0.2),
    (0.7, 0.15, 0.15),
    (0.33, 0.33, 0.34),
]

for w1, w2, w3 in weights_list:
    z_combo = w1 * z1 + w2 * z2 + w3 * z3
    acc, prompt, tokens, results = evaluate_z(z_combo, examples)
    print(f"  weights=({w1:.2f},{w2:.2f},{w3:.2f}): {acc:.0%} -> '{prompt[:35]}...'")

    if acc > best_combo["acc"]:
        best_combo = {"acc": acc, "weights": (w1, w2, w3), "prompt": prompt, "z": z_combo.clone()}

    # Quick optimize from combination
    opt_acc, opt_z, opt_prompt = quick_optimize(z_combo, examples, lr=0.001, steps=10)
    if opt_acc > best_overall["acc"]:
        best_overall = {"acc": opt_acc, "prompt": opt_prompt, "z": opt_z, "source": f"combo_{w1}_{w2}_{w3}"}

# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"\nBest overall: {best_overall['acc']:.0%}")
print(f"Source: {best_overall['source']}")
print(f"Prompt: '{best_overall['prompt']}'")

if best_overall["z"] is not None:
    _, _, tokens, results = evaluate_z(best_overall["z"], examples)
    print(f"\nDetailed results:")
    for inp, ans, tgt, cor in results:
        mark = "OK" if cor else "X"
        print(f"  {inp} -> '{ans}' ({tgt}) {mark}")

print(f"\n{'=' * 70}")
print(f"Best from interpolation: {best_interp['acc']:.0%} (alpha={best_interp['alpha']})")
print(f"Best from perturbation: {best_perturb['acc']:.0%}")
print(f"Best from combination: {best_combo['acc']:.0%} (weights={best_combo['weights']})")
print(f"{'=' * 70}")
