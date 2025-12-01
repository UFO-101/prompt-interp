"""
V16: Properly implement PPL loss through the embedding gradient pathway.

The key insight: PPL loss should flow through the same straight-through mechanism
as the task loss. We capture embedding gradients from PPL forward passes and
combine them with task embedding gradients.

PPL loss encourages the prompt tokens to be "natural" continuations under z=0,
which should help prevent the optimization from drifting to degenerate prompts.
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
    """
    Compute perplexity of prompt tokens under z=0, capturing embedding gradients.
    Returns (ppl_value, list of embedding gradients for each position).
    """
    if len(prompt_tokens) <= 2:
        return 0.0, []

    ppl_embed_grads = []
    total_nll = 0.0
    n_tokens = 0

    # For each position, predict next token
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

        # Backprop to get embedding gradients
        nll.backward(retain_graph=False)

        if len(embedding_output) > 0 and embedding_output[0].grad is not None:
            # Pad to full prompt length for easier combination later
            embed_grad = embedding_output[0].grad[0]  # shape: [seq_len, embed_dim]
            # We want gradients for positions 0 to i-1 (the input positions)
            # Pad to full prompt_tokens length
            full_grad = torch.zeros(len(prompt_tokens), embed_grad.shape[-1], device=dev)
            full_grad[:len(embed_grad)] = embed_grad
            ppl_embed_grads.append(full_grad)

        sdm.zero_grad()

    ppl_value = torch.exp(torch.tensor(total_nll / n_tokens)).item() if n_tokens > 0 else 0.0
    return ppl_value, ppl_embed_grads


def train_step(z, examples, ppl_weight=0.1):
    """
    Training step with PPL loss properly flowing through embedding gradients.
    """
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z, max_len=15)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    # Collect task embedding gradients
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

    # Collect PPL embedding gradients
    ppl_value, ppl_embed_grads = compute_ppl_with_embed_grads(prompt_tokens)

    # Combine task and PPL embedding gradients
    if len(task_embed_grads) > 0 and len(stage1_logits) > 0:
        # Average task gradients
        avg_task_grad = torch.stack(task_embed_grads).mean(dim=0)  # [prompt_len, embed_dim]

        # Average PPL gradients (if any)
        if len(ppl_embed_grads) > 0 and ppl_weight > 0:
            avg_ppl_grad = torch.stack(ppl_embed_grads).mean(dim=0)[:len(prompt_tokens)]
            # Combine: task gradient (minimize task loss) + ppl gradient (minimize PPL)
            # Note: task_grad points in direction to DECREASE task loss
            #       ppl_grad points in direction to DECREASE PPL (make more natural)
            combined_grad = avg_task_grad + ppl_weight * avg_ppl_grad
        else:
            combined_grad = avg_task_grad

        # Convert to logit gradients via embedding matrix
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


# ============================================================================
print("\n" + "=" * 70)
print("V16: PPL loss through embedding gradient pathway")
print("=" * 70)

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

# Test different PPL weights
ppl_weights = [0.0, 0.05, 0.1, 0.2, 0.5]

all_results = []

for ppl_weight in ppl_weights:
    print(f"\n{'=' * 60}")
    print(f"PPL weight: {ppl_weight}")
    print("=" * 60)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=0.001)

    best_acc = 0
    best_prompt = None
    best_tokens = None
    best_step = 0
    initial_loss = None
    zero_acc_count = 0

    for step in range(40):
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

        print(f"[{step:2d}] loss={loss:.2f} ppl={ppl:6.0f} z0={z0_acc:.0%} best={best_acc:.0%} | {prompt[:45]}")

        # Early stopping
        if loss > initial_loss * 3:
            print(f"  -> Early stop: loss exploded ({loss:.1f} > {initial_loss*3:.1f})")
            break
        if zero_acc_count >= 4:
            print(f"  -> Early stop: z0_acc=0 for {zero_acc_count} steps")
            break

    result = {
        "ppl_weight": ppl_weight,
        "best_acc": best_acc,
        "best_prompt": best_prompt,
        "best_tokens": best_tokens,
        "best_step": best_step,
    }
    all_results.append(result)

    print(f"\n>>> Best: {best_acc:.0%} at step {best_step}")
    print(f"    Prompt: '{best_prompt}'")

    if best_tokens:
        _, results = evaluate_with_z0(best_tokens, examples)
        for inp, ans, tgt, cor in results:
            mark = "OK" if cor else "X"
            print(f"    {inp} -> '{ans}' ({tgt}) {mark}")

# Final comparison
print("\n" + "=" * 70)
print("COMPARISON: Effect of PPL weight on optimization stability")
print("=" * 70)

for r in all_results:
    print(f"PPL weight {r['ppl_weight']:.2f}: best_acc={r['best_acc']:.0%} at step {r['best_step']}")
    print(f"  Prompt: '{r['best_prompt']}'")
