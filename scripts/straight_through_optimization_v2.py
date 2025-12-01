"""
Straight-through estimator for learning text prompts - V2.

Improvements:
- Better gradient clipping
- Lower learning rate
- Entropy regularization to prevent degeneracy
- Track multiple metrics
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

# Get the embedding matrix
embed_layer = sdm.decoder.decoder_frontend.embed
embed_matrix = embed_layer.weight.detach()  # [vocab_size, dim]
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


def generate_prompt_with_grads(z, max_len=10):
    """
    Stage 1: Generate prompt tokens using z, keeping track of logits for gradient bridge.
    """
    tokens = [3, 256047]  # BOS, lang
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


def train_step(z, examples, use_entropy_reg=True, entropy_weight=0.01):
    """
    One training step with straight-through gradient estimation.
    """
    # Stage 1: Generate prompt
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z, max_len=10)
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

        # Stage 2: Forward with z=0
        full_tokens = prompt_tokens + task_content.tolist() + [target_id]
        input_tokens = full_tokens[:-1]

        # Hook to capture embedding gradient
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

        # Loss
        target = torch.tensor([target_id], device=dev, dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target)
        total_loss = total_loss + loss.item()

        if logits.argmax().item() == target_id:
            n_correct += 1

        # Backward to get embedding gradients
        loss.backward(retain_graph=False)

        # Get embedding gradients for prompt positions
        if len(embedding_output) > 0 and embedding_output[0].grad is not None:
            embed_grad = embedding_output[0].grad[0]  # [seq_len, dim]
            prompt_embed_grad = embed_grad[:len(prompt_tokens)]  # [prompt_len, dim]
            all_embed_grads.append(prompt_embed_grad.clone())

        sdm.zero_grad()

    # Average embedding gradients and convert to logit gradients
    if len(all_embed_grads) > 0:
        avg_embed_grad = torch.stack(all_embed_grads).mean(dim=0)  # [prompt_len, dim]

        # Convert to logit gradients: ∂L/∂logits = E @ ∂L/∂embed
        logit_grads = avg_embed_grad @ embed_matrix.T  # [prompt_len, vocab_size]

        # Normalize the gradients to prevent exploding
        logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

        # Stage 1 logits correspond to positions 2, 3, 4, ... (after BOS, lang)
        if len(stage1_logits) > 0 and logit_grads.shape[0] > 2:
            grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]

            total_surrogate_loss = 0.0
            for i, (logits_i, grad_i) in enumerate(zip(stage1_logits, grad_for_stage1)):
                # Surrogate: logits @ grad (dot product)
                surrogate = (logits_i * grad_i).sum()
                total_surrogate_loss = total_surrogate_loss + surrogate

                # Entropy regularization to prevent degenerate distributions
                if use_entropy_reg:
                    probs = F.softmax(logits_i, dim=-1)
                    entropy = -(probs * (probs + 1e-10).log()).sum()
                    total_surrogate_loss = total_surrogate_loss - entropy_weight * entropy

            total_surrogate_loss.backward()

    return total_loss / len(examples), n_correct / len(examples), prompt_text, prompt_tokens


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
# Main optimization loop
# ============================================================================
print("\n" + "=" * 80)
print("STRAIGHT-THROUGH OPTIMIZATION V2")
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

with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.001)  # Lower learning rate

print("\nOptimizing...")
best_acc = 0
best_prompt = None
best_tokens = None

for step in range(51):
    optimizer.zero_grad()

    loss, acc, prompt, tokens = train_step(z, examples, use_entropy_reg=True, entropy_weight=0.001)

    if z.grad is not None:
        # Gradient clipping
        grad_norm_before = z.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        grad_norm_after = z.grad.norm().item()
        optimizer.step()
    else:
        grad_norm_before = 0.0
        grad_norm_after = 0.0

    # Track best
    if acc > best_acc:
        best_acc = acc
        best_prompt = prompt
        best_tokens = tokens.copy()

    if step % 5 == 0:
        # Quick z=0 evaluation
        z0_acc, _ = evaluate_with_z0(tokens, examples)
        print(f"Step {step}: loss={loss:.4f}, train_acc={acc:.0%}, z0_acc={z0_acc:.0%}, grad_norm={grad_norm_after:.4f}")
        print(f"  Prompt: '{prompt[:50]}...'")

# Final evaluation
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

# Generate final prompt
with torch.no_grad():
    final_tokens = [3, 256047]
    e = z.unsqueeze(1)
    for _ in range(10):
        di = torch.tensor([final_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        final_tokens.append(next_tok)
        if next_tok == 3:
            break
    if final_tokens[-1] == 3:
        final_tokens = final_tokens[:-1]

final_prompt = tokens_to_text(torch.tensor(final_tokens))
print(f"\nFinal prompt: '{final_prompt}'")

print("\nEvaluation with z=0 (prompt TEXT only):")
acc, results = evaluate_with_z0(final_tokens, examples)
for inp, ans, tgt, cor in results:
    mark = "OK" if cor else "X"
    print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")
print(f"Accuracy: {acc:.0%}")

if best_tokens is not None:
    print(f"\nBest prompt seen during training: '{best_prompt}'")
    print("Evaluation with z=0:")
    acc, results = evaluate_with_z0(best_tokens, examples)
    for inp, ans, tgt, cor in results:
        mark = "OK" if cor else "X"
        print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")
    print(f"Accuracy: {acc:.0%}")

# Baseline: handcrafted prompts
print("\n" + "=" * 80)
print("BASELINES: Handcrafted prompts with z=0")
print("=" * 80)

baseline_prompts = [
    "The opposite of hot is cold.",
    "hot -> cold, big -> small, fast -> slow",
    "Antonyms: hot -> cold, up -> down",
    "Find the opposite:",
]

for prompt_text in baseline_prompts:
    prompt_tokens = text_to_tokens(prompt_text)
    prompt_full = [3, 256047] + prompt_tokens[2:].tolist()
    if prompt_full[-1] == 3:
        prompt_full = prompt_full[:-1]

    acc, results = evaluate_with_z0(prompt_full, examples)
    print(f"\nPrompt: '{prompt_text}'")
    print(f"  Accuracy: {acc:.0%}")
