"""
Straight-through estimator V10 - simplified with perplexity regularization.
No ensemble, just basic optimization with PPL loss to stabilize.
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


def compute_prompt_perplexity_loss(prompt_tokens):
    """Compute NLL of prompt under z=0 model (differentiable)."""
    if len(prompt_tokens) <= 2:
        return torch.tensor(0.0, device=dev), 0.0

    total_nll = torch.tensor(0.0, device=dev)
    n_tokens = 0

    for i in range(2, len(prompt_tokens)):
        input_seq = prompt_tokens[:i]
        target_token = prompt_tokens[i]

        di = torch.tensor([input_seq], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        target = torch.tensor([target_token], device=dev, dtype=torch.long)
        nll = F.cross_entropy(logits.unsqueeze(0), target)
        total_nll = total_nll + nll
        n_tokens += 1

    avg_nll = total_nll / n_tokens if n_tokens > 0 else torch.tensor(0.0, device=dev)
    ppl = torch.exp(avg_nll).item()
    return avg_nll, ppl


def train_step(z, examples, ppl_weight=0.01):
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z, max_len=15)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    # Perplexity loss
    ppl_loss, ppl_value = compute_prompt_perplexity_loss(prompt_tokens)

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

        # Combined loss
        combined = task_loss + ppl_weight * ppl_loss
        combined.backward(retain_graph=True)

        if len(embedding_output) > 0 and embedding_output[0].grad is not None:
            embed_grad = embedding_output[0].grad[0]
            prompt_embed_grad = embed_grad[:len(prompt_tokens)]
            all_embed_grads.append(prompt_embed_grad.clone())

        sdm.zero_grad()

    # Straight-through gradient
    if len(all_embed_grads) > 0 and len(stage1_logits) > 0:
        avg_embed_grad = torch.stack(all_embed_grads).mean(dim=0)
        logit_grads = avg_embed_grad @ embed_matrix.T
        logit_grads = logit_grads / (logit_grads.norm(dim=-1, keepdim=True) + 1e-8)

        if logit_grads.shape[0] > 2:
            grad_for_stage1 = logit_grads[2:2+len(stage1_logits)]
            total_surrogate = sum((l * g).sum() for l, g in zip(stage1_logits, grad_for_stage1))
            total_surrogate.backward()

    return total_loss / len(examples), n_correct / len(examples), prompt_text, prompt_tokens, ppl_value


def evaluate_with_z0(prompt_tokens, examples):
    n_correct = 0
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
        if answer.lower().startswith(target_text.lower()):
            n_correct += 1

    return n_correct / len(examples)


# ============================================================================
print("\n" + "=" * 70)
print("V10: Simple optimization with perplexity regularization")
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

# Test different ppl weights
for ppl_weight in [0.0, 0.01, 0.1]:
    print(f"\n{'=' * 50}")
    print(f"PPL weight: {ppl_weight}")
    print("=" * 50)

    z = nn.Parameter(z_init.clone())
    optimizer = torch.optim.Adam([z], lr=0.001)

    best_acc = 0
    best_prompt = None
    best_tokens = None

    for step in range(30):
        optimizer.zero_grad()
        loss, acc, prompt, tokens, ppl = train_step(z, examples, ppl_weight=ppl_weight)

        if z.grad is not None:
            torch.nn.utils.clip_grad_norm_([z], max_norm=0.5)
            optimizer.step()

        z0_acc = evaluate_with_z0(tokens, examples)

        if z0_acc > best_acc:
            best_acc = z0_acc
            best_prompt = prompt
            best_tokens = tokens.copy()

        print(f"[{step:2d}] loss={loss:.2f} ppl={ppl:6.0f} z0={z0_acc:.0%} best={best_acc:.0%} | {prompt[:50]}")

    print(f"\nBest: {best_acc:.0%} -> '{best_prompt}'")
