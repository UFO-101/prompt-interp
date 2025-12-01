"""
Straight-through estimator for learning text prompts - V5.

Strategy: Many short runs with random restarts.
When we find a good z, perturb it and restart to explore the neighborhood.
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


def train_step(z, examples):
    """One training step with straight-through gradient estimation."""
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
        logit_grads = avg_embed_grad @ embed_matrix.T
        logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

        if len(stage1_logits) > 0 and logit_grads.shape[0] > 2:
            grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]

            total_surrogate_loss = 0.0
            for logits_i, grad_i in zip(stage1_logits, grad_for_stage1):
                surrogate = (logits_i * grad_i).sum()
                total_surrogate_loss = total_surrogate_loss + surrogate

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


def short_optimization_run(z_init, examples, steps=15, lr=0.0005):
    """Short optimization run with early stopping."""
    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=lr)

    best_acc = 0
    best_prompt = None
    best_tokens = None
    best_z = None

    for step in range(steps):
        optimizer.zero_grad()
        loss, train_acc, prompt, tokens = train_step(z, examples)

        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.5)
            optimizer.step()

        z0_acc, _ = evaluate_with_z0(tokens, examples)

        if z0_acc > best_acc:
            best_acc = z0_acc
            best_prompt = prompt
            best_tokens = tokens.copy()
            best_z = z.detach().clone()

    return best_acc, best_prompt, best_tokens, best_z


# ============================================================================
# Main
# ============================================================================
print("\n" + "=" * 80)
print("STRAIGHT-THROUGH OPTIMIZATION V5 (random restarts)")
print("=" * 80)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

# Seeds to try
seeds = [
    "The opposite is:",
    "hot -> cold, big -> small. Now:",
    "hot -> cold, big -> small, fast -> slow.",
    "Antonyms: hot -> cold, up -> down.",
    "The opposite of hot is cold.",
    "hot is to cold as big is to small.",
    "cold cold cold",  # Weird but let's see
    "opposite opposite opposite",
]

overall_best_acc = 0
overall_best_prompt = None
overall_best_tokens = None
overall_best_z = None

# Phase 1: Try all seeds
print("\n--- Phase 1: Testing seeds ---")
for seed in seeds:
    with torch.no_grad():
        z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

    acc, prompt, tokens, z = short_optimization_run(z_init, examples, steps=15, lr=0.0005)
    print(f"  {acc:.0%} : '{seed[:40]}...' -> '{prompt[:40] if prompt else 'None'}...'")

    if acc > overall_best_acc:
        overall_best_acc = acc
        overall_best_prompt = prompt
        overall_best_tokens = tokens
        overall_best_z = z

print(f"\nPhase 1 best: {overall_best_acc:.0%}")

# Phase 2: Perturb best z and try more runs
if overall_best_z is not None and overall_best_acc > 0:
    print("\n--- Phase 2: Perturbing best z ---")
    for i in range(10):
        # Add noise to best z
        noise = torch.randn_like(overall_best_z) * 0.1
        z_perturbed = overall_best_z + noise

        acc, prompt, tokens, z = short_optimization_run(z_perturbed, examples, steps=15, lr=0.0005)
        print(f"  Perturb {i}: {acc:.0%} -> '{prompt[:40] if prompt else 'None'}...'")

        if acc > overall_best_acc:
            overall_best_acc = acc
            overall_best_prompt = prompt
            overall_best_tokens = tokens
            overall_best_z = z

# Phase 3: Try with different learning rates
print("\n--- Phase 3: Trying different learning rates ---")
for lr in [0.001, 0.0002, 0.0001]:
    for seed in seeds[:3]:  # Just top 3 seeds
        with torch.no_grad():
            z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

        acc, prompt, tokens, z = short_optimization_run(z_init, examples, steps=20, lr=lr)
        if acc > overall_best_acc:
            overall_best_acc = acc
            overall_best_prompt = prompt
            overall_best_tokens = tokens
            overall_best_z = z
            print(f"  NEW BEST: {acc:.0%} with lr={lr} on '{seed[:30]}...'")

# Final summary
print("\n" + "=" * 80)
print("OVERALL BEST RESULT")
print("=" * 80)
print(f"Accuracy: {overall_best_acc:.0%}")
print(f"Prompt: '{overall_best_prompt}'")

if overall_best_tokens:
    print("\nFinal evaluation:")
    _, results = evaluate_with_z0(overall_best_tokens, examples)
    for inp, ans, tgt, cor in results:
        mark = "OK" if cor else "X"
        print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")

# Also verify the best prompt manually
if overall_best_prompt:
    print("\n--- Manual verification ---")
    prompt_tokens = text_to_tokens(overall_best_prompt)
    prompt_full = [3, 256047] + prompt_tokens[2:].tolist()
    if prompt_full[-1] == 3:
        prompt_full = prompt_full[:-1]
    acc, results = evaluate_with_z0(prompt_full, examples)
    print(f"Re-tokenized accuracy: {acc:.0%}")
