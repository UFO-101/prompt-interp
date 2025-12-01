"""
Straight-through estimator for learning text prompts - V7.

Key insight from V6: Averaging gradients from different z's that produce
different token sequences doesn't work - the gradients aren't comparable.

New approach:
1. Generate prompt ONCE from base z (fix the discrete tokens)
2. Compute task loss multiple times with noisy z variants
3. Average only the embedding->logit gradients (which are comparable since tokens are fixed)
4. Use that averaged gradient to update z

This should smooth the gradient while keeping discrete choices consistent.
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


def generate_prompt(z, max_len=15):
    """Generate prompt tokens using z (no gradients needed here)."""
    tokens = [3, 256047]
    e = z.unsqueeze(1)

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


def compute_embed_grad_for_prompt(prompt_tokens, examples):
    """
    Given fixed prompt tokens, compute gradient w.r.t. prompt embeddings.
    Returns embedding gradients averaged over examples.
    """
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

    if len(all_embed_grads) > 0:
        avg_embed_grad = torch.stack(all_embed_grads).mean(dim=0)
        return avg_embed_grad, total_loss / len(examples), n_correct / len(examples)

    return None, total_loss / len(examples), n_correct / len(examples)


def train_step_with_ensemble(z, examples, n_variants=5, noise_scale=0.02):
    """
    Training step with ensemble gradient averaging.

    1. Generate prompt once from base z
    2. For each noisy variant of z, compute the surrogate loss gradient
    3. Average the z gradients
    """
    # Step 1: Generate prompt from base z (with gradients for later)
    z_param = nn.Parameter(z.clone())
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z_param, max_len=15)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    # Step 2: Compute embedding gradient (this is the same regardless of z noise)
    embed_grad, loss, acc = compute_embed_grad_for_prompt(prompt_tokens, examples)

    if embed_grad is None or len(stage1_logits) == 0:
        return None, loss, acc, prompt_text, prompt_tokens

    # Convert embedding grad to logit grad
    logit_grads = embed_grad @ embed_matrix.T
    logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

    if logit_grads.shape[0] <= 2:
        return None, loss, acc, prompt_text, prompt_tokens

    grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]

    # Step 3: Compute z gradient for base z
    total_surrogate_loss = 0.0
    for logits_i, grad_i in zip(stage1_logits, grad_for_stage1):
        surrogate = (logits_i * grad_i).sum()
        total_surrogate_loss = total_surrogate_loss + surrogate

    total_surrogate_loss.backward()
    base_z_grad = z_param.grad.clone() if z_param.grad is not None else None

    # Step 4: Compute z gradient for noisy variants
    all_z_grads = []
    if base_z_grad is not None:
        all_z_grads.append(base_z_grad)

    for _ in range(n_variants - 1):
        noise = torch.randn_like(z) * noise_scale
        z_noisy = nn.Parameter(z.clone() + noise)

        # Generate with noisy z
        noisy_tokens, noisy_logits = generate_prompt_with_grads(z_noisy, max_len=15)

        # If it produces the same tokens, use the same grad (more comparable)
        # If different tokens, compute new grad
        if noisy_tokens == prompt_tokens and len(noisy_logits) == len(stage1_logits):
            # Same tokens - use same logit grads, just different z
            total_surrogate = 0.0
            for logits_i, grad_i in zip(noisy_logits, grad_for_stage1):
                surrogate = (logits_i * grad_i).sum()
                total_surrogate = total_surrogate + surrogate
            total_surrogate.backward()
            if z_noisy.grad is not None:
                all_z_grads.append(z_noisy.grad.clone())

    # Average z gradients
    if len(all_z_grads) > 0:
        avg_z_grad = torch.stack(all_z_grads).mean(dim=0)
        return avg_z_grad, loss, acc, prompt_text, prompt_tokens

    return None, loss, acc, prompt_text, prompt_tokens


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


# ============================================================================
# Main
# ============================================================================
print("\n" + "=" * 80)
print("STRAIGHT-THROUGH OPTIMIZATION V7")
print("(ensemble over z variants that produce SAME tokens)")
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

# Configuration
n_variants = 10
noise_scale = 0.02
lr = 0.001
max_steps = 50

print(f"\nConfig: n_variants={n_variants}, noise_scale={noise_scale}, lr={lr}")

z = z_init.clone()
best_acc = 0
best_prompt = None
best_tokens = None

for step in range(max_steps):
    avg_grad, loss, acc, prompt, tokens = train_step_with_ensemble(
        z, examples, n_variants=n_variants, noise_scale=noise_scale
    )

    if avg_grad is not None:
        # Manual gradient descent with clipping
        grad_norm = avg_grad.norm().item()
        if grad_norm > 0.5:
            avg_grad = avg_grad * (0.5 / grad_norm)
        z = z - lr * avg_grad

    # Evaluate with z=0
    z0_acc, _ = evaluate_with_z0(tokens, examples)

    if z0_acc > best_acc:
        best_acc = z0_acc
        best_prompt = prompt
        best_tokens = tokens.copy()

    if step % 5 == 0:
        print(f"Step {step}: loss={loss:.3f}, z0_acc={z0_acc:.0%}, best={best_acc:.0%}")
        print(f"  Prompt: '{prompt[:50]}...'")

print(f"\n{'=' * 60}")
print(f"Best: {best_acc:.0%}")
print(f"Prompt: '{best_prompt}'")

if best_tokens:
    print("\nFinal evaluation:")
    _, results = evaluate_with_z0(best_tokens, examples)
    for inp, ans, tgt, cor in results:
        mark = "OK" if cor else "X"
        print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")


# Try a second run with different seed
print("\n" + "=" * 80)
print("Second run with 'The opposite is:' seed")
print("=" * 80)

seed2 = "The opposite is:"
with torch.no_grad():
    z_init2 = se.predict([seed2], source_lang='eng_Latn').to(dev)

z = z_init2.clone()
best_acc2 = 0
best_prompt2 = None
best_tokens2 = None

for step in range(max_steps):
    avg_grad, loss, acc, prompt, tokens = train_step_with_ensemble(
        z, examples, n_variants=n_variants, noise_scale=noise_scale
    )

    if avg_grad is not None:
        grad_norm = avg_grad.norm().item()
        if grad_norm > 0.5:
            avg_grad = avg_grad * (0.5 / grad_norm)
        z = z - lr * avg_grad

    z0_acc, _ = evaluate_with_z0(tokens, examples)

    if z0_acc > best_acc2:
        best_acc2 = z0_acc
        best_prompt2 = prompt
        best_tokens2 = tokens.copy()

    if step % 5 == 0:
        print(f"Step {step}: loss={loss:.3f}, z0_acc={z0_acc:.0%}, best={best_acc2:.0%}")

print(f"\nBest: {best_acc2:.0%}")
print(f"Prompt: '{best_prompt2}'")
