"""
V17: Hyperparameter exploration with PPL regularization (ppl_weight=0.1).

Key insight from v16: PPL weight 0.1 provides stability without dominating.
Now: Systematic exploration of seeds (relevant and irrelevant) with this setting.
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


def compute_ppl_with_embed_grads(prompt_tokens):
    if len(prompt_tokens) <= 2:
        return 0.0, []

    ppl_embed_grads = []
    total_nll = 0.0
    n_tokens = 0

    for i in range(2, len(prompt_tokens)):
        input_seq = prompt_tokens[:i]
        target_token = prompt_tokens[i]

        embedding_output = []
        def embed_hook(module, input, output):
            output.retain_grad()
            embedding_output.append(output)
            return output

        handle = embed_layer.register_forward_hook(embed_hook)

        di = torch.tensor([input_seq], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        handle.remove()

        target = torch.tensor([target_token], device=dev, dtype=torch.long)
        nll = F.cross_entropy(logits.unsqueeze(0), target)
        total_nll += nll.item()
        n_tokens += 1

        nll.backward(retain_graph=False)

        if len(embedding_output) > 0 and embedding_output[0].grad is not None:
            embed_grad = embedding_output[0].grad[0]
            full_grad = torch.zeros(len(prompt_tokens), embed_grad.shape[-1], device=dev)
            full_grad[:len(embed_grad)] = embed_grad
            ppl_embed_grads.append(full_grad)

        sdm.zero_grad()

    ppl_value = torch.exp(torch.tensor(total_nll / n_tokens)).item() if n_tokens > 0 else 0.0
    return ppl_value, ppl_embed_grads


def train_step(z, examples, ppl_weight=0.1):
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z, max_len=15)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    total_loss = 0.0
    n_correct = 0
    task_embed_grads = []

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
            task_embed_grads.append(prompt_embed_grad.clone())

        sdm.zero_grad()

    ppl_value, ppl_embed_grads = compute_ppl_with_embed_grads(prompt_tokens)

    if len(task_embed_grads) > 0 and len(stage1_logits) > 0:
        avg_task_grad = torch.stack(task_embed_grads).mean(dim=0)

        if len(ppl_embed_grads) > 0 and ppl_weight > 0:
            avg_ppl_grad = torch.stack(ppl_embed_grads).mean(dim=0)[:len(prompt_tokens)]
            combined_grad = avg_task_grad + ppl_weight * avg_ppl_grad
        else:
            combined_grad = avg_task_grad

        logit_grads = combined_grad @ embed_matrix.T
        logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

        if logit_grads.shape[0] > 2:
            grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]
            total_surrogate = sum((l * g).sum() for l, g in zip(stage1_logits, grad_for_stage1))
            total_surrogate.backward()

    return total_loss / len(examples), n_correct / len(examples), prompt_text, prompt_tokens, ppl_value


def evaluate_with_z0(prompt_tokens, examples):
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


def optimize_seed(seed_text, examples, lr=0.001, ppl_weight=0.1, max_steps=50, verbose=True):
    """Optimize from a seed with PPL regularization."""
    with torch.no_grad():
        z_init = se.predict([seed_text], source_lang='eng_Latn').to(dev)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=lr)

    best_acc = 0
    best_prompt = None
    best_tokens = None
    best_step = 0
    initial_loss = None
    zero_acc_count = 0

    for step in range(max_steps):
        optimizer.zero_grad()
        loss, acc, prompt, tokens, ppl = train_step(z, examples, ppl_weight=ppl_weight)

        if initial_loss is None:
            initial_loss = loss

        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.5)
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
            print(f"  [{step:2d}] loss={loss:.2f} ppl={ppl:5.0f} z0={z0_acc:.0%} best={best_acc:.0%} | {prompt[:40]}")

        # Early stopping
        if loss > initial_loss * 3:
            if verbose:
                print(f"  -> Early stop: loss exploded at step {step}")
            break
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
        "lr": lr,
        "ppl_weight": ppl_weight,
    }


# ============================================================================
print("\n" + "=" * 70)
print("V17: Hyperparameter exploration with PPL regularization")
print("=" * 70)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

# Diverse seeds: relevant, semi-relevant, and completely irrelevant
seeds = [
    # Task-relevant
    ("hot -> cold, big -> small. Now:", "relevant"),
    ("Opposites: hot -> cold.", "relevant"),
    ("The opposite is:", "semi-relevant"),
    ("Input -> Output:", "semi-relevant"),

    # Completely irrelevant
    ("The cat sat on the mat.", "irrelevant"),
    ("I went to the store yesterday.", "irrelevant"),
    ("Hello, how are you today?", "irrelevant"),
    ("The quick brown fox jumps.", "irrelevant"),
    ("1 + 1 = 2, 2 + 2 = 4.", "irrelevant"),
    ("Paris is the capital of France.", "irrelevant"),

    # Random/nonsense
    ("xyz abc 123", "random"),
    ("...", "random"),
]

# Learning rates to try
lrs = [0.0005, 0.001, 0.002]

all_results = []

print("\n--- Testing diverse seeds with ppl_weight=0.1 ---")
for seed_text, seed_type in seeds:
    print(f"\n{'='*60}")
    print(f"Seed ({seed_type}): '{seed_text[:45]}'")
    print("="*60)

    for lr in lrs:
        print(f"\n  lr={lr}:")
        result = optimize_seed(seed_text, examples, lr=lr, ppl_weight=0.1,
                              max_steps=40, verbose=True)
        result["seed_type"] = seed_type
        all_results.append(result)
        prompt_str = result['best_prompt'][:35] if result['best_prompt'] else "(no improvement)"
        print(f"  >>> Best: {result['best_acc']:.0%} at step {result['best_step']} -> '{prompt_str}...'")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Results by seed type")
print("=" * 70)

for seed_type in ["relevant", "semi-relevant", "irrelevant", "random"]:
    type_results = [r for r in all_results if r.get("seed_type") == seed_type]
    if type_results:
        best = max(type_results, key=lambda x: x["best_acc"])
        avg_acc = sum(r["best_acc"] for r in type_results) / len(type_results)
        print(f"\n{seed_type.upper()}:")
        bp = best['best_prompt'][:40] if best['best_prompt'] else "(no improvement)"
        print(f"  Best: {best['best_acc']:.0%} (lr={best['lr']}) -> '{bp}...'")
        print(f"  Average: {avg_acc:.0%}")

# Top 5 overall
print("\n" + "=" * 70)
print("TOP 5 RESULTS OVERALL")
print("=" * 70)

sorted_results = sorted(all_results, key=lambda x: (-x["best_acc"], x["best_step"]))
seen = set()

for i, r in enumerate(sorted_results):
    if r["best_prompt"] is None or r["best_prompt"] in seen:
        continue
    seen.add(r["best_prompt"])

    print(f"\n{i+1}. {r['best_acc']:.0%} at step {r['best_step']}")
    print(f"   Seed ({r['seed_type']}): '{r['seed'][:40]}...'")
    print(f"   Prompt: '{r['best_prompt']}'")
    print(f"   Config: lr={r['lr']}, ppl_weight={r['ppl_weight']}")

    if r["best_tokens"]:
        _, results = evaluate_with_z0(r["best_tokens"], examples)
        for inp, ans, tgt, cor in results:
            mark = "✓" if cor else "✗"
            print(f"     {mark} {inp} -> '{ans}' (target: {tgt})")

    if len(seen) >= 5:
        break
