"""
Straight-through estimator for learning text prompts - V8.

Insights from V7:
- Same-token filtering makes gradients too similar (no smoothing)
- We need diversity but also some consistency

New approach:
- Generate multiple prompts from noisy z's
- Compute gradient for EACH prompt (not just base)
- Average all gradients weighted by how well each prompt does
- This explores the neighborhood while preferring good directions
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
vocab_size, embed_dim = embed_matrix.shape
print(f"Embedding matrix: {embed_matrix.shape}")

z_zero = torch.zeros(1, 1, 1024, device=dev)


def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def text_to_tokens(text):
    return sonar_enc(text).to(dev)


def generate_prompt_with_grads(z, max_len=15):
    """Generate prompt tokens using z, keeping logits for gradient bridge."""
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


def compute_full_gradient(z_param, prompt_tokens, stage1_logits, examples):
    """
    Compute gradient for z given prompt tokens and logits.
    Returns (z_grad, loss, accuracy).
    """
    if len(stage1_logits) == 0:
        return None, float('inf'), 0.0

    all_embed_grads = []
    total_loss = 0.0
    n_correct = 0

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
        loss = F.cross_entropy(logits.unsqueeze(0), target)
        total_loss = total_loss + loss.item()

        if logits.argmax().item() == target_id:
            n_correct += 1

        loss.backward(retain_graph=False)

        if len(embedding_output) > 0 and embedding_output[0].grad is not None:
            embed_grad = embedding_output[0].grad[0]
            prompt_embed_grad = embed_grad[:len(prompt_tokens)]
            all_embed_grads.append(prompt_embed_grad.clone())

        sdm.zero_grad()

    if len(all_embed_grads) == 0:
        return None, total_loss / len(examples), n_correct / len(examples)

    avg_embed_grad = torch.stack(all_embed_grads).mean(dim=0)
    logit_grads = avg_embed_grad @ embed_matrix.T
    logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

    if logit_grads.shape[0] <= 2:
        return None, total_loss / len(examples), n_correct / len(examples)

    grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]

    # Backprop through stage 1 logits
    total_surrogate_loss = 0.0
    for logits_i, grad_i in zip(stage1_logits, grad_for_stage1):
        surrogate = (logits_i * grad_i).sum()
        total_surrogate_loss = total_surrogate_loss + surrogate

    total_surrogate_loss.backward()

    z_grad = z_param.grad.clone() if z_param.grad is not None else None

    return z_grad, total_loss / len(examples), n_correct / len(examples)


def evaluate_with_z0(prompt_tokens, examples):
    """Evaluate using only the prompt text with z=0."""
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


def ensemble_train_step(z, examples, n_variants=5, noise_scale=0.03):
    """
    Training step with ensemble gradient averaging.

    - Generate multiple prompts from noisy z variants
    - Compute gradient for each
    - Weight gradients by inverse loss (better prompts get more weight)
    - Return weighted average gradient
    """
    all_grads = []
    all_weights = []
    best_loss = float('inf')
    best_prompt = None
    best_tokens = None
    best_acc = 0

    for i in range(n_variants):
        # Add noise (first one is base, no noise)
        if i == 0:
            z_variant = z.clone()
        else:
            noise = torch.randn_like(z) * noise_scale
            z_variant = z + noise

        z_param = nn.Parameter(z_variant)
        tokens, logits = generate_prompt_with_grads(z_param, max_len=15)

        # Skip if empty prompt
        if len(logits) == 0:
            continue

        z_grad, loss, acc = compute_full_gradient(z_param, tokens, logits, examples)

        if z_grad is not None:
            # Weight by inverse loss (lower loss = higher weight)
            # Use softmax-style weighting
            weight = 1.0 / (loss + 0.1)  # Add small constant to avoid div by 0
            all_grads.append(z_grad)
            all_weights.append(weight)

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_prompt = tokens_to_text(torch.tensor(tokens))
            best_tokens = tokens
            best_acc = acc

    if len(all_grads) == 0:
        return None, best_loss, best_acc, best_prompt, best_tokens

    # Weighted average
    total_weight = sum(all_weights)
    weighted_grad = sum(g * (w / total_weight) for g, w in zip(all_grads, all_weights))

    return weighted_grad, best_loss, best_acc, best_prompt, best_tokens


# ============================================================================
# Main
# ============================================================================
print("\n" + "=" * 80)
print("STRAIGHT-THROUGH OPTIMIZATION V8")
print("(weighted ensemble gradient averaging)")
print("=" * 80)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

seed = "hot -> cold, big -> small. Now:"
print(f"Seed: '{seed}'")

with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

# Try different configurations
configs = [
    {"n_variants": 5, "noise_scale": 0.03, "lr": 0.001},
    {"n_variants": 8, "noise_scale": 0.05, "lr": 0.0005},
    {"n_variants": 3, "noise_scale": 0.02, "lr": 0.002},
]

overall_best = {"acc": 0, "prompt": None, "tokens": None}

for cfg in configs:
    print(f"\n{'=' * 60}")
    print(f"Config: {cfg}")
    print("=" * 60)

    z = z_init.clone()
    best_acc = 0
    best_prompt = None
    best_tokens = None

    for step in range(40):
        avg_grad, loss, acc, prompt, tokens = ensemble_train_step(
            z, examples,
            n_variants=cfg["n_variants"],
            noise_scale=cfg["noise_scale"]
        )

        if avg_grad is not None:
            grad_norm = avg_grad.norm().item()
            if grad_norm > 0.5:
                avg_grad = avg_grad * (0.5 / grad_norm)
            z = z - cfg["lr"] * avg_grad

        # Evaluate with z=0
        z0_acc, _ = evaluate_with_z0(tokens, examples)

        if z0_acc > best_acc:
            best_acc = z0_acc
            best_prompt = prompt
            best_tokens = tokens.copy() if tokens else None

        if step % 10 == 0:
            print(f"  Step {step}: loss={loss:.3f}, z0_acc={z0_acc:.0%}, best={best_acc:.0%}")
            if prompt:
                print(f"    Prompt: '{prompt[:45]}...'")

    print(f"\n  Best: {best_acc:.0%} - '{best_prompt}'")

    if best_acc > overall_best["acc"]:
        overall_best = {"acc": best_acc, "prompt": best_prompt, "tokens": best_tokens}


# Final summary
print("\n" + "=" * 80)
print("OVERALL BEST")
print("=" * 80)
print(f"Accuracy: {overall_best['acc']:.0%}")
print(f"Prompt: '{overall_best['prompt']}'")

if overall_best["tokens"]:
    print("\nFinal evaluation:")
    _, results = evaluate_with_z0(overall_best["tokens"], examples)
    for inp, ans, tgt, cor in results:
        mark = "OK" if cor else "X"
        print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")
