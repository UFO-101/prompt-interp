"""
Straight-through estimator for learning text prompts - V9.

Key addition: Perplexity regularization.
- In stage 2 (z=0 forward pass), compute perplexity of prompt tokens
- Add this as a regularization term to prevent degenerate prompts
- Use small weight since task performance is the priority

Goal: Find ANY prompt that gets high task accuracy with z=0,
even if it's somewhat incoherent.
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


def compute_prompt_perplexity(prompt_tokens):
    """
    Compute perplexity of prompt tokens using z=0 decoder.
    This measures how "natural" the prompt is as a continuation.
    Returns (perplexity_loss, perplexity_value).
    """
    if len(prompt_tokens) <= 2:
        return torch.tensor(0.0, device=dev), 0.0

    # Get logits for each position
    total_nll = 0.0
    n_tokens = 0

    for i in range(2, len(prompt_tokens)):
        # Input is tokens up to position i
        input_seq = prompt_tokens[:i]
        target_token = prompt_tokens[i]

        di = torch.tensor([input_seq], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        # Cross entropy for this token
        target = torch.tensor([target_token], device=dev, dtype=torch.long)
        nll = F.cross_entropy(logits.unsqueeze(0), target)
        total_nll = total_nll + nll
        n_tokens += 1

    if n_tokens == 0:
        return torch.tensor(0.0, device=dev), 0.0

    avg_nll = total_nll / n_tokens
    perplexity = torch.exp(avg_nll).item()

    return avg_nll, perplexity


def train_step(z, examples, ppl_weight=0.01):
    """
    Training step with perplexity regularization.

    Loss = task_loss + ppl_weight * perplexity_loss
    """
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z, max_len=15)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    # Compute perplexity of prompt (with gradients through embedding)
    ppl_loss, ppl_value = compute_prompt_perplexity(prompt_tokens)

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

        # Add perplexity loss (scaled)
        combined_loss = loss + ppl_weight * ppl_loss
        combined_loss.backward(retain_graph=True)

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

    return total_loss / len(examples), n_correct / len(examples), prompt_text, prompt_tokens, ppl_value


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
print("STRAIGHT-THROUGH OPTIMIZATION V9")
print("(with perplexity regularization)")
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

# Try different perplexity weights
# Low weight = prioritize task, allow weird prompts
# High weight = keep prompts natural, maybe sacrifice task perf
ppl_weights = [0.0, 0.001, 0.01, 0.1]

overall_best = {"acc": 0, "prompt": None, "tokens": None, "ppl": None}

for ppl_weight in ppl_weights:
    print(f"\n{'=' * 60}")
    print(f"Perplexity weight: {ppl_weight}")
    print("=" * 60)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=0.001)

    best_acc = 0
    best_prompt = None
    best_tokens = None
    best_ppl = None

    for step in range(50):
        optimizer.zero_grad()

        loss, train_acc, prompt, tokens, ppl = train_step(z, examples, ppl_weight=ppl_weight)

        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.5)
            optimizer.step()

        # Evaluate with z=0
        z0_acc, _ = evaluate_with_z0(tokens, examples)

        if z0_acc > best_acc:
            best_acc = z0_acc
            best_prompt = prompt
            best_tokens = tokens.copy()
            best_ppl = ppl

        if step % 10 == 0:
            print(f"  Step {step}: loss={loss:.3f}, z0_acc={z0_acc:.0%}, ppl={ppl:.1f}, best={best_acc:.0%}")
            print(f"    Prompt: '{prompt[:45]}...'")

    print(f"\n  Best: {best_acc:.0%} (ppl={best_ppl:.1f if best_ppl else 0})")
    print(f"  Prompt: '{best_prompt}'")

    if best_acc > overall_best["acc"]:
        overall_best = {"acc": best_acc, "prompt": best_prompt, "tokens": best_tokens, "ppl": best_ppl}

    if best_tokens:
        print("\n  Evaluation:")
        _, results = evaluate_with_z0(best_tokens, examples)
        for inp, ans, tgt, cor in results:
            mark = "OK" if cor else "X"
            print(f"    {inp} -> '{ans}' (want: {tgt}) {mark}")


# Final summary
print("\n" + "=" * 80)
print("OVERALL BEST")
print("=" * 80)
print(f"Accuracy: {overall_best['acc']:.0%}")
print(f"Perplexity: {overall_best['ppl']:.1f if overall_best['ppl'] else 'N/A'}")
print(f"Prompt: '{overall_best['prompt']}'")

if overall_best["tokens"]:
    print("\nFinal evaluation:")
    _, results = evaluate_with_z0(overall_best["tokens"], examples)
    for inp, ans, tgt, cor in results:
        mark = "OK" if cor else "X"
        print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")
