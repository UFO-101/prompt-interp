"""
Straight-through estimator for learning text prompts - V6.

Key idea: Ensemble gradient averaging.
- At each step, generate K noisy variants of z
- Run each through the pipeline, get gradients
- Average the gradients to update the base z
- This smooths the gradient landscape and prevents overfitting to one path
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


def compute_grad_for_z(z, examples):
    """
    Compute gradient for a single z vector.
    Returns the gradient w.r.t. z (via surrogate loss).
    """
    # Need z to be a parameter to accumulate gradients
    z_param = nn.Parameter(z.clone())

    prompt_tokens, stage1_logits = generate_prompt_with_grads(z_param, max_len=15)
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

    # Compute gradient for z via surrogate loss
    z_grad = None
    if len(all_embed_grads) > 0 and len(stage1_logits) > 0:
        avg_embed_grad = torch.stack(all_embed_grads).mean(dim=0)
        logit_grads = avg_embed_grad @ embed_matrix.T
        logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

        if logit_grads.shape[0] > 2:
            grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]

            total_surrogate_loss = 0.0
            for logits_i, grad_i in zip(stage1_logits, grad_for_stage1):
                surrogate = (logits_i * grad_i).sum()
                total_surrogate_loss = total_surrogate_loss + surrogate

            total_surrogate_loss.backward()

            if z_param.grad is not None:
                z_grad = z_param.grad.clone()

    return z_grad, total_loss / len(examples), n_correct / len(examples), prompt_text, prompt_tokens


def ensemble_train_step(z, examples, n_variants=5, noise_scale=0.05):
    """
    Training step with ensemble gradient averaging.

    1. Create n_variants noisy versions of z
    2. Compute gradient for each
    3. Average gradients
    4. Return averaged gradient
    """
    all_grads = []
    all_losses = []
    all_accs = []
    all_prompts = []
    all_tokens = []

    # Always include the base z (no noise)
    grad, loss, acc, prompt, tokens = compute_grad_for_z(z, examples)
    if grad is not None:
        all_grads.append(grad)
    all_losses.append(loss)
    all_accs.append(acc)
    all_prompts.append(prompt)
    all_tokens.append(tokens)

    # Add noisy variants
    for i in range(n_variants - 1):
        noise = torch.randn_like(z) * noise_scale
        z_noisy = z + noise

        grad, loss, acc, prompt, tokens = compute_grad_for_z(z_noisy, examples)
        if grad is not None:
            all_grads.append(grad)
        all_losses.append(loss)
        all_accs.append(acc)
        all_prompts.append(prompt)
        all_tokens.append(tokens)

    # Average gradients
    avg_grad = None
    if len(all_grads) > 0:
        avg_grad = torch.stack(all_grads).mean(dim=0)

    # Return stats from base z (first one)
    return avg_grad, all_losses[0], all_accs[0], all_prompts[0], all_tokens[0]


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
print("STRAIGHT-THROUGH OPTIMIZATION V6 (ensemble gradient averaging)")
print("=" * 80)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

# Best seed from previous experiments
seed = "hot -> cold, big -> small. Now:"
print(f"Seed: '{seed}'")

with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

# Try different noise scales
for noise_scale in [0.01, 0.02, 0.05]:
    print(f"\n{'=' * 60}")
    print(f"Noise scale: {noise_scale}")
    print("=" * 60)

    z = z_init.clone()
    lr = 0.001

    best_acc = 0
    best_prompt = None
    best_tokens = None

    for step in range(30):
        avg_grad, loss, acc, prompt, tokens = ensemble_train_step(
            z, examples, n_variants=5, noise_scale=noise_scale
        )

        if avg_grad is not None:
            # Manual gradient descent
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
            print(f"  Step {step}: loss={loss:.3f}, z0_acc={z0_acc:.0%}, best={best_acc:.0%}")
            print(f"    Prompt: '{prompt[:50]}...'")

    print(f"\nBest: {best_acc:.0%}")
    print(f"Prompt: '{best_prompt}'")

    if best_tokens:
        print("\nFinal evaluation:")
        _, results = evaluate_with_z0(best_tokens, examples)
        for inp, ans, tgt, cor in results:
            mark = "OK" if cor else "X"
            print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")


# Also try with more variants
print("\n" + "=" * 80)
print("Trying with more variants (n=10)")
print("=" * 80)

z = z_init.clone()
best_acc = 0
best_prompt = None
best_tokens = None

for step in range(30):
    avg_grad, loss, acc, prompt, tokens = ensemble_train_step(
        z, examples, n_variants=10, noise_scale=0.02
    )

    if avg_grad is not None:
        grad_norm = avg_grad.norm().item()
        if grad_norm > 0.5:
            avg_grad = avg_grad * (0.5 / grad_norm)
        z = z - 0.001 * avg_grad

    z0_acc, _ = evaluate_with_z0(tokens, examples)

    if z0_acc > best_acc:
        best_acc = z0_acc
        best_prompt = prompt
        best_tokens = tokens.copy()

    if step % 5 == 0:
        print(f"  Step {step}: loss={loss:.3f}, z0_acc={z0_acc:.0%}, best={best_acc:.0%}")

print(f"\nBest: {best_acc:.0%}")
print(f"Prompt: '{best_prompt}'")
