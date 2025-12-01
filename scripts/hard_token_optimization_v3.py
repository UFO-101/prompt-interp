"""
Hard token optimization v3.

Key insight from debugging: hidden states are NOT in embedding space.
h @ embed.T gives logits, so h and embed are related by a dot product, not identity.

New approach:
1. Generate tokens with restricted vocab (no grad)
2. Map tokens to Qwen, run Qwen, get loss and ∂L/∂qwen_embed
3. Map gradients back: ∂L/∂sonar_embed = ∂L/∂qwen_embed @ W.T
4. Teacher-force SONAR, use SOFT token selection (softmax over logits)
5. Compute soft_embed = softmax(logits) @ sonar_embeds (differentiable)
6. Proxy loss = soft_embed · ∂L/∂sonar_embed
7. Backprop to z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout

dev = "cuda"
torch.cuda.empty_cache()

print("Loading models...", flush=True)

# SONAR
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sdm = sd.model.to(dev)
sonar_dec = sd.tokenizer.create_decoder()
sonar_embeds = sdm.decoder.final_proj.weight.data  # [vocab_size, 1024]
sonar_vocab_size = sonar_embeds.shape[0]

# Qwen
qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True).to(dev)
qm.eval()
for param in qm.parameters():
    param.requires_grad = False

# Load mapper
print("Loading token mapper...", flush=True)
checkpoint = torch.load("results/sonar_to_qwen_token_mapper.pt")
class LinearMapper(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.linear = nn.Linear(src_dim, tgt_dim)
    def forward(self, x):
        return self.linear(x)
W = LinearMapper(1024, 896).to(dev)
W.load_state_dict(checkpoint['model_state'])
W.eval()

# ============================================================================
# Build restricted vocabulary
# ============================================================================
print("Building restricted vocabulary...", flush=True)

overlap_sonar_ids = set()
sonar_to_qwen_map = {}

for tok_id in range(min(200000, sonar_vocab_size)):
    try:
        s = sonar_dec(torch.tensor([tok_id]))
        if s is None or len(s) == 0:
            continue
        qwen_toks = qt.encode(s, add_special_tokens=False)
        if len(qwen_toks) == 1 and qt.decode(qwen_toks) == s:
            overlap_sonar_ids.add(tok_id)
            sonar_to_qwen_map[tok_id] = qwen_toks[0]
        s_space = " " + s
        qwen_toks_space = qt.encode(s_space, add_special_tokens=False)
        if len(qwen_toks_space) == 1 and qt.decode(qwen_toks_space) == s_space:
            overlap_sonar_ids.add(tok_id)
            if tok_id not in sonar_to_qwen_map:
                sonar_to_qwen_map[tok_id] = qwen_toks_space[0]
    except:
        pass

for tok in [0, 1, 2, 3, 256047, 248075, 248079, 248130, 248203]:
    overlap_sonar_ids.add(tok)
for tok in [248075, 248079, 248130, 248203]:
    try:
        s = sonar_dec(torch.tensor([tok]))
        qwen_toks = qt.encode(s, add_special_tokens=False)
        if len(qwen_toks) == 1:
            sonar_to_qwen_map[tok] = qwen_toks[0]
    except:
        pass

print(f"Overlap: {len(overlap_sonar_ids)}, Mapped: {len(sonar_to_qwen_map)}")

allowed_mask = torch.zeros(sonar_vocab_size, device=dev)
for tok_id in overlap_sonar_ids:
    if tok_id < sonar_vocab_size:
        allowed_mask[tok_id] = 1.0
disallowed_mask = (allowed_mask == 0)


# ============================================================================
# Helpers
# ============================================================================

def decode_restricted(z, max_len=60):
    """Decode with restricted vocab, no gradients."""
    with torch.no_grad():
        e = z.detach().unsqueeze(0) if z.dim() == 1 else z.detach()
        eo = e.unsqueeze(1)
        generated = [3, 256047]
        for _ in range(max_len):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            logits[disallowed_mask] = float('-inf')
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token == 3:
                break
    return torch.tensor(generated, device=dev)


def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def map_sonar_to_qwen(sonar_tokens):
    content = sonar_tokens[2:-1] if sonar_tokens[-1] == 3 else sonar_tokens[2:]
    qwen_tokens = []
    for tok in content:
        tok_id = tok.item()
        if tok_id in sonar_to_qwen_map:
            qwen_tokens.append(sonar_to_qwen_map[tok_id])
        else:
            try:
                s = sonar_dec(torch.tensor([tok_id]))
                qwen_toks = qt.encode(s, add_special_tokens=False)
                if len(qwen_toks) >= 1:
                    qwen_tokens.extend(qwen_toks)
            except:
                pass
    return torch.tensor(qwen_tokens, device=dev) if qwen_tokens else None


def compute_qwen_loss_and_embed_grad(qwen_tokens, examples):
    """Get gradients w.r.t. Qwen prefix embeddings."""
    prefix_embeds = qm.model.embed_tokens(qwen_tokens.unsqueeze(0)).detach().clone()
    prefix_embeds.requires_grad_(True)

    total_loss = 0
    for input_text, target_text in examples:
        task_tokens = qt(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        task_embeds = qm.model.embed_tokens(task_tokens)
        target_ids = qt.encode(target_text, add_special_tokens=False)
        target_embeds = qm.model.embed_tokens(torch.tensor([target_ids], device=dev))

        full_embeds = torch.cat([prefix_embeds, task_embeds, target_embeds], dim=1)
        outputs = qm(inputs_embeds=full_embeds)

        prefix_len = prefix_embeds.shape[1]
        task_len = task_embeds.shape[1]
        start_pos = prefix_len + task_len - 1

        for i, t in enumerate(target_ids):
            loss = F.cross_entropy(outputs.logits[0, start_pos + i, :].unsqueeze(0),
                                  torch.tensor([t], device=dev))
            total_loss += loss

    total_loss = total_loss / len(examples)
    total_loss.backward()
    return prefix_embeds.grad.squeeze(0), total_loss.item()


def soft_decode_backward(z, sonar_tokens, sonar_embed_grads, temperature=1.0):
    """
    Teacher-force SONAR and compute proxy loss using soft token selection.

    Key insight: h[pos] predicts token at position pos+1.
    So the gradient for content token i should flow through h at position i+1
    (which is position 1 + i in the full sequence, since content starts at position 2).

    Actually simpler:
    - input_tokens = [BOS, lang, content[0], content[1], ..., content[n-1]]
    - h = decode(input_tokens) gives hidden states
    - h[0, 0] (after BOS) predicts lang
    - h[0, 1] (after lang) predicts content[0]
    - h[0, 2] (after content[0]) predicts content[1]
    - etc.

    So h[0, i+1] predicts content[i], and sonar_embed_grads[i] should flow through h[0, i+1].

    soft_embed[i] = softmax(logits[i+1] / temp) @ sonar_embeds
    proxy_loss = sum(soft_embed[i] · grad[i])
    """
    content = sonar_tokens[2:-1] if sonar_tokens[-1] == 3 else sonar_tokens[2:]
    if len(content) == 0:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    # Teacher-forced decode: input all tokens except last
    # sonar_tokens = [BOS, lang, content[0], ..., content[n-1], EOS]
    # We input [BOS, lang, content[0], ..., content[n-1]] to predict [lang, content[0], ..., EOS]
    input_tokens = sonar_tokens[:-1].unsqueeze(0)
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), eo, BatchLayout.of(eo))
    logits = sdm.decoder.final_proj(h)  # [1, seq_len, vocab]
    # logits[0, i] predicts token at position i+1

    # For each content token i:
    # - It's at position i+2 in sonar_tokens (after BOS=0, lang=1)
    # - It's predicted by logits[0, i+1] (after lang)
    # - The gradient sonar_embed_grads[i] should flow through this

    proxy_loss = torch.tensor(0.0, device=dev)
    for i in range(len(content)):
        logit_pos = i + 1  # Position in logits that predicts content[i]

        if logit_pos >= logits.shape[1] or i >= sonar_embed_grads.shape[0]:
            break

        # Soft selection with temperature
        probs = F.softmax(logits[0, logit_pos, :] / temperature, dim=0)

        # Soft embedding (weighted sum of all token embeddings)
        soft_embed = probs @ sonar_embeds  # [1024]

        # Gradient for this position (what direction should the embedding move?)
        grad = sonar_embed_grads[i]  # [1024]

        # proxy_loss = soft_embed · grad
        proxy_loss = proxy_loss + (soft_embed * grad).sum()

    return proxy_loss


# ============================================================================
# Optimization
# ============================================================================
print("\n" + "=" * 80)
print("HARD TOKEN OPTIMIZATION v3")
print("=" * 80)

# Antonym task
examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

seed = "Find the opposite: hot becomes cold, big becomes small."
print(f"\nSeed: '{seed}'")

with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.001)

print("\n" + "-" * 80)
print(f"{'Step':<6} {'Loss':<10} {'Decoded Text':<60}")
print("-" * 80)

n_steps = 100
temperature = 1.0
n_samples = 16  # Number of noisy samples (including original)
noise_scale = 0.15  # Scale of noise to add

for step in range(n_steps + 1):
    optimizer.zero_grad()

    # Create noisy versions of z
    z_samples = [z]  # First one is the original
    for _ in range(n_samples - 1):
        noise = torch.randn_like(z) * noise_scale
        z_samples.append(z + noise)

    # Accumulate gradients across samples
    total_loss = 0
    valid_samples = 0
    accumulated_proxy = 0

    # For display, use the original z
    sonar_tokens_display = decode_restricted(z)
    decoded_text = tokens_to_text(sonar_tokens_display)

    for z_sample in z_samples:
        # 1. Decode (hard, no grad)
        sonar_tokens = decode_restricted(z_sample)

        # 2. Map to Qwen
        qwen_tokens = map_sonar_to_qwen(sonar_tokens)
        if qwen_tokens is None or len(qwen_tokens) == 0:
            continue

        # 3. Qwen loss and gradients
        qwen_embed_grad, loss = compute_qwen_loss_and_embed_grad(qwen_tokens, examples)
        total_loss += loss
        valid_samples += 1

        # 4. Map gradients back to SONAR embedding space
        W_matrix = W.linear.weight.detach()
        sonar_embed_grads = qwen_embed_grad @ W_matrix

        # Normalize
        grad_norm = sonar_embed_grads.norm()
        if grad_norm > 1.0:
            sonar_embed_grads = sonar_embed_grads / grad_norm

        # 5. Soft decode backward - accumulate proxy loss
        # Note: we compute proxy w.r.t. the ORIGINAL z, not z_sample
        # This way gradients accumulate on z
        proxy = soft_decode_backward(z, sonar_tokens, sonar_embed_grads, temperature)
        accumulated_proxy = accumulated_proxy + proxy

    if valid_samples == 0:
        print(f"{step:<6} {'N/A':<10} {decoded_text[:60]:<60}")
        continue

    avg_loss = total_loss / valid_samples

    if step > 0:
        # Average the proxy loss and do gradient ascent
        avg_proxy = accumulated_proxy / valid_samples

        # Add regularization to keep z close to z_init
        reg_weight = 0.1
        reg_loss = reg_weight * ((z - z_init.detach()) ** 2).sum()

        # Total loss: minimize (-avg_proxy + reg_loss)
        total_opt_loss = -avg_proxy + reg_loss
        total_opt_loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()

    display_text = decoded_text[:57] + "..." if len(decoded_text) > 60 else decoded_text
    print(f"{step:<6} {avg_loss:<10.4f} {display_text:<60}")


# ============================================================================
# Final evaluation
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

with torch.no_grad():
    sonar_tokens = decode_restricted(z)
    decoded_text = tokens_to_text(sonar_tokens)
    qwen_tokens = map_sonar_to_qwen(sonar_tokens)

print(f"\nFinal: '{decoded_text}'")

if qwen_tokens is not None:
    print(f"\nQwen text: '{qt.decode(qwen_tokens)}'")
    prefix_embeds = qm.model.embed_tokens(qwen_tokens.unsqueeze(0))

    print("\nTask performance:")
    for input_text, target in examples:
        task_tokens = qt(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        task_embeds = qm.model.embed_tokens(task_tokens)
        full_embeds = torch.cat([prefix_embeds, task_embeds], dim=1)

        generated = []
        for _ in range(5):
            outputs = qm(inputs_embeds=full_embeds)
            next_token = outputs.logits[0, -1, :].argmax().item()
            generated.append(next_token)
            if next_token == qt.eos_token_id:
                break
            next_embed = qm.model.embed_tokens(torch.tensor([[next_token]], device=dev))
            full_embeds = torch.cat([full_embeds, next_embed], dim=1)

        pred = qt.decode(generated).strip()
        correct = pred.lower().startswith(target.lower())
        print(f"  {input_text} -> '{pred}' {'✓' if correct else '✗'}")
