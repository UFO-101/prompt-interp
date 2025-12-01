"""
Hard token optimization with gradient estimation.

Approach:
1. Generate tokens from SONAR z (no gradients, restricted vocab)
2. Map SONAR tokens to Qwen tokens (1:1 lookup)
3. Run Qwen forward, compute task loss
4. Backprop to Qwen embeddings
5. Use linear mapper transpose to estimate SONAR embedding gradients
6. Teacher-force SONAR decoder, backprop to z
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
qwen_embeds = qm.model.embed_tokens.weight.data  # [vocab_size, 896]

# Load the trained token mapper
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
for param in W.parameters():
    param.requires_grad = False

# ============================================================================
# Build restricted vocabulary and token mapping
# ============================================================================
print("Building restricted vocabulary...", flush=True)

overlap_sonar_ids = set()
sonar_to_qwen_map = {}  # sonar_token_id -> qwen_token_id

for tok_id in range(min(200000, sonar_vocab_size)):
    try:
        s = sonar_dec(torch.tensor([tok_id]))
        if s is None or len(s) == 0:
            continue

        # Check exact match
        qwen_toks = qt.encode(s, add_special_tokens=False)
        if len(qwen_toks) == 1 and qt.decode(qwen_toks) == s:
            overlap_sonar_ids.add(tok_id)
            sonar_to_qwen_map[tok_id] = qwen_toks[0]

        # Check space-prefixed (for word-initial tokens)
        s_space = " " + s
        qwen_toks_space = qt.encode(s_space, add_special_tokens=False)
        if len(qwen_toks_space) == 1 and qt.decode(qwen_toks_space) == s_space:
            overlap_sonar_ids.add(tok_id)
            # Prefer space-prefixed mapping for word-initial tokens
            if tok_id not in sonar_to_qwen_map:
                sonar_to_qwen_map[tok_id] = qwen_toks_space[0]
    except:
        pass

# Add special tokens
for tok in [0, 1, 2, 3, 256047]:
    overlap_sonar_ids.add(tok)

# Add sentence punctuation
for tok in [248075, 248079, 248130, 248203]:
    overlap_sonar_ids.add(tok)
    # Map punctuation to Qwen equivalents
    try:
        s = sonar_dec(torch.tensor([tok]))
        qwen_toks = qt.encode(s, add_special_tokens=False)
        if len(qwen_toks) == 1:
            sonar_to_qwen_map[tok] = qwen_toks[0]
    except:
        pass

print(f"Overlap vocabulary size: {len(overlap_sonar_ids)}")
print(f"Mapped tokens: {len(sonar_to_qwen_map)}")

# Create mask for restricted decoding
allowed_mask = torch.zeros(sonar_vocab_size, device=dev)
for tok_id in overlap_sonar_ids:
    if tok_id < sonar_vocab_size:
        allowed_mask[tok_id] = 1.0
disallowed_mask = (allowed_mask == 0)


# ============================================================================
# Helper functions
# ============================================================================

def decode_restricted(z, max_len=60):
    """Decode z with restricted vocabulary (no gradients)."""
    with torch.no_grad():
        e = z.detach().unsqueeze(0) if z.dim() == 1 else z.detach()
        eo = e.unsqueeze(1)

        generated = [3, 256047]  # BOS, eng_Latn

        for _ in range(max_len):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]

            # Mask disallowed tokens
            logits = logits.clone()
            logits[disallowed_mask] = float('-inf')

            next_token = logits.argmax().item()
            generated.append(next_token)

            if next_token == 3:  # EOS
                break

        return torch.tensor(generated, device=dev)


def tokens_to_text(tokens):
    """Convert SONAR tokens to text."""
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def map_sonar_to_qwen(sonar_tokens):
    """Map SONAR tokens to Qwen tokens."""
    # Skip BOS (3) and lang tag (256047), and EOS (3)
    content = sonar_tokens[2:-1] if sonar_tokens[-1] == 3 else sonar_tokens[2:]

    qwen_tokens = []
    for tok in content:
        tok_id = tok.item()
        if tok_id in sonar_to_qwen_map:
            qwen_tokens.append(sonar_to_qwen_map[tok_id])
        else:
            # Fallback: try to find any mapping
            try:
                s = sonar_dec(torch.tensor([tok_id]))
                qwen_toks = qt.encode(s, add_special_tokens=False)
                if len(qwen_toks) >= 1:
                    qwen_tokens.extend(qwen_toks)
            except:
                pass

    return torch.tensor(qwen_tokens, device=dev) if qwen_tokens else None


def compute_task_loss(qwen_tokens, input_text, target_text):
    """Compute loss for generating target_text after input_text, given prefix qwen_tokens."""
    # Get embeddings for prefix (decoded SONAR output)
    prefix_embeds = qm.model.embed_tokens(qwen_tokens.unsqueeze(0))

    # Get embeddings for task input
    task_tokens = qt(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    task_embeds = qm.model.embed_tokens(task_tokens)

    # Get embeddings for target
    target_token_ids = qt.encode(target_text, add_special_tokens=False)
    target_tokens = torch.tensor([target_token_ids], device=dev)
    target_embeds = qm.model.embed_tokens(target_tokens)

    # Concatenate: prefix + task + target
    full_embeds = torch.cat([prefix_embeds, task_embeds, target_embeds], dim=1)

    # Forward through Qwen
    outputs = qm(inputs_embeds=full_embeds)

    # Compute loss on target tokens
    prefix_len = prefix_embeds.shape[1]
    task_len = task_embeds.shape[1]
    start_pos = prefix_len + task_len - 1

    total_loss = 0
    for i, t in enumerate(target_token_ids):
        logits = outputs.logits[0, start_pos + i, :]
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([t], device=dev))
        total_loss += loss

    return total_loss / len(target_token_ids)


def get_qwen_embedding_grads(qwen_tokens, examples):
    """Get gradients w.r.t. Qwen embeddings for the task."""
    qwen_tokens = qwen_tokens.detach().requires_grad_(False)

    # Create embeddings that require gradients
    qwen_token_embeds = qm.model.embed_tokens(qwen_tokens.unsqueeze(0)).detach().clone()
    qwen_token_embeds.requires_grad_(True)

    total_loss = 0
    for input_text, target_text in examples:
        # Task embeddings
        task_tokens = qt(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        task_embeds = qm.model.embed_tokens(task_tokens)

        # Target embeddings
        target_token_ids = qt.encode(target_text, add_special_tokens=False)
        target_tokens = torch.tensor([target_token_ids], device=dev)
        target_embeds = qm.model.embed_tokens(target_tokens)

        # Forward
        full_embeds = torch.cat([qwen_token_embeds, task_embeds, target_embeds], dim=1)
        outputs = qm(inputs_embeds=full_embeds)

        # Loss
        prefix_len = qwen_token_embeds.shape[1]
        task_len = task_embeds.shape[1]
        start_pos = prefix_len + task_len - 1

        for i, t in enumerate(target_token_ids):
            logits = outputs.logits[0, start_pos + i, :]
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([t], device=dev))
            total_loss += loss

    total_loss = total_loss / len(examples)
    total_loss.backward()

    return qwen_token_embeds.grad, total_loss.item()


def teacher_force_sonar_backward(z, sonar_tokens, sonar_embed_grads):
    """
    Teacher-force SONAR decoder and backprop gradient signal from output embeddings to z.

    Uses straight-through estimator: backprop through the logit of the selected token.

    sonar_embed_grads: [seq_len, 1024] - gradients w.r.t. SONAR output embeddings
    """
    # Content tokens (skip BOS, lang tag, and EOS)
    content = sonar_tokens[2:-1] if sonar_tokens[-1] == 3 else sonar_tokens[2:]

    if len(content) == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    # Forward pass with gradients
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    # Teacher-forced input: BOS + lang_tag + all but last content token
    input_tokens = sonar_tokens[:-1]  # Everything except final token
    di = input_tokens.unsqueeze(0)

    # Get hidden states
    h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))

    # Get logits at each position
    logits = sdm.decoder.final_proj(h)  # [1, seq_len, vocab_size]

    # Straight-through estimator approach:
    # For each position, we want to maximize the logit of the token we selected
    # weighted by how much the Qwen gradient wants that embedding to change
    loss_proxy = 0
    for i, tok in enumerate(content):
        tok_id = tok.item()
        pos = i + 2  # Position in logits (after BOS and lang_tag)

        if pos >= logits.shape[1]:
            break

        if i >= sonar_embed_grads.shape[0]:
            break

        # Get the embedding of the selected token
        selected_embed = sonar_embeds[tok_id]  # [1024]

        # The gradient we want to apply to this embedding
        grad = sonar_embed_grads[i]  # [1024]

        # The logit for the selected token
        selected_logit = logits[0, pos, tok_id]

        # Weight: how much does this token's embedding align with the desired gradient?
        # Positive weight = we want to increase this token's probability
        # Negative weight = we want to decrease it
        weight = (selected_embed * grad).sum().detach()

        # Proxy loss: if weight is positive, we want higher logit (lower loss)
        # if weight is negative, we want lower logit (higher loss)
        loss_proxy = loss_proxy - weight * selected_logit

    return loss_proxy


# ============================================================================
# Optimization
# ============================================================================
print("\n" + "=" * 80)
print("HARD TOKEN OPTIMIZATION")
print("=" * 80)

# Task: Antonyms
examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

# Seed
seed = "Find the opposite: hot becomes cold, big becomes small."
print(f"\nSeed: '{seed}'")
print(f"Task: Generate antonyms")

# Encode seed
with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.001)  # Lower learning rate

# Regularization: keep z close to initial embedding
reg_weight = 0.1

print("\n" + "-" * 80)
print(f"{'Step':<6} {'Loss':<10} {'Decoded Text':<60}")
print("-" * 80)

n_steps = 50

for step in range(n_steps + 1):
    optimizer.zero_grad()

    # 1. Decode z with restricted vocabulary (no gradients)
    sonar_tokens = decode_restricted(z)
    decoded_text = tokens_to_text(sonar_tokens)

    # 2. Map to Qwen tokens
    qwen_tokens = map_sonar_to_qwen(sonar_tokens)

    if qwen_tokens is None or len(qwen_tokens) == 0:
        print(f"{step:<6} {'N/A':<10} {decoded_text[:60]:<60}")
        continue

    # 3. Get Qwen embedding gradients
    qwen_embed_grads, loss = get_qwen_embedding_grads(qwen_tokens, examples)
    # qwen_embed_grads: [1, seq_len, 896]
    qwen_embed_grads = qwen_embed_grads.squeeze(0)  # [seq_len, 896]

    # 4. Map gradients to SONAR space using W.T
    # W.linear.weight is [896, 1024], so W.T maps [896] -> [1024]
    with torch.no_grad():
        W_matrix = W.linear.weight  # [896, 1024]
        sonar_embed_grads = qwen_embed_grads @ W_matrix  # [seq_len, 1024]

    # 5. Teacher-force SONAR and backprop to z
    loss_proxy = teacher_force_sonar_backward(z, sonar_tokens, sonar_embed_grads)

    # Add regularization to keep z close to initial
    reg_loss = reg_weight * ((z - z_init.detach()) ** 2).sum()

    if step > 0:  # Don't update on step 0
        total_proxy = loss_proxy + reg_loss
        total_proxy.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=0.5)
        optimizer.step()

    # Print progress
    display_text = decoded_text[:57] + "..." if len(decoded_text) > 60 else decoded_text
    print(f"{step:<6} {loss:<10.4f} {display_text:<60}")


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

print(f"\nFinal decoded prompt: '{decoded_text}'")
print(f"\nQwen tokens: {qwen_tokens.tolist() if qwen_tokens is not None else 'None'}")
print(f"Qwen text: '{qt.decode(qwen_tokens) if qwen_tokens is not None else 'None'}'")

print("\nTask performance:")
if qwen_tokens is not None:
    prefix_embeds = qm.model.embed_tokens(qwen_tokens.unsqueeze(0))

    for input_text, target in examples:
        task_tokens = qt(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        task_embeds = qm.model.embed_tokens(task_tokens)
        full_embeds = torch.cat([prefix_embeds, task_embeds], dim=1)

        # Generate
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
        sym = "✓" if correct else "✗"
        print(f"  {input_text} -> '{pred}' (expected: {target}) {sym}")
