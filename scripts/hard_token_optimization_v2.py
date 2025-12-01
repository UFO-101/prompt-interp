"""
Hard token optimization with gradient estimation v2.

Key insight: SONAR's final hidden states are in embedding space.
final_proj is just: logits = h @ sonar_embeds.T

So we can:
1. Decode z → hidden states h (with gradients)
2. Get tokens via argmax (no gradients needed here)
3. Map to Qwen tokens, run Qwen, get loss
4. Backprop to Qwen embeddings
5. Map gradients back: ∂L/∂h = ∂L/∂qwen_embed @ W.T
6. Backprop through SONAR decoder to z
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

        # Check space-prefixed
        s_space = " " + s
        qwen_toks_space = qt.encode(s_space, add_special_tokens=False)
        if len(qwen_toks_space) == 1 and qt.decode(qwen_toks_space) == s_space:
            overlap_sonar_ids.add(tok_id)
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

def decode_restricted_with_hidden(z, max_len=60):
    """
    Decode z with restricted vocabulary.
    Returns tokens AND the final hidden states (for gradient computation).
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    generated = [3, 256047]  # BOS, eng_Latn

    # First pass: generate tokens (no grad needed for argmax)
    with torch.no_grad():
        for _ in range(max_len):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            logits[disallowed_mask] = float('-inf')
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token == 3:  # EOS
                break

    tokens = torch.tensor(generated, device=dev)

    # Second pass: get hidden states with gradients
    # Teacher-force with the generated tokens
    di = tokens[:-1].unsqueeze(0)  # Input: all but last token
    h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
    # h shape: [1, seq_len, 1024]

    return tokens, h


def tokens_to_text(tokens):
    """Convert SONAR tokens to text."""
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def map_sonar_to_qwen(sonar_tokens):
    """Map SONAR tokens to Qwen tokens."""
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
    """
    Run Qwen forward, compute loss, backprop to embeddings.
    Returns gradients w.r.t. the prefix embeddings.
    """
    # Create embeddings that require gradients
    prefix_embeds = qm.model.embed_tokens(qwen_tokens.unsqueeze(0)).detach().clone()
    prefix_embeds.requires_grad_(True)

    total_loss = 0
    for input_text, target_text in examples:
        task_tokens = qt(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        task_embeds = qm.model.embed_tokens(task_tokens)

        target_token_ids = qt.encode(target_text, add_special_tokens=False)
        target_tokens = torch.tensor([target_token_ids], device=dev)
        target_embeds = qm.model.embed_tokens(target_tokens)

        full_embeds = torch.cat([prefix_embeds, task_embeds, target_embeds], dim=1)
        outputs = qm(inputs_embeds=full_embeds)

        prefix_len = prefix_embeds.shape[1]
        task_len = task_embeds.shape[1]
        start_pos = prefix_len + task_len - 1

        for i, t in enumerate(target_token_ids):
            logits = outputs.logits[0, start_pos + i, :]
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([t], device=dev))
            total_loss += loss

    total_loss = total_loss / len(examples)
    total_loss.backward()

    return prefix_embeds.grad.squeeze(0), total_loss.item()  # [seq_len, 896]


# ============================================================================
# Optimization
# ============================================================================
print("\n" + "=" * 80)
print("HARD TOKEN OPTIMIZATION v2")
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
optimizer = torch.optim.Adam([z], lr=0.0005)

print("\n" + "-" * 80)
print(f"{'Step':<6} {'Loss':<10} {'Decoded Text':<60}")
print("-" * 80)

n_steps = 50

for step in range(n_steps + 1):
    optimizer.zero_grad()

    # 1. Decode z → tokens + hidden states (h has gradients w.r.t. z)
    sonar_tokens, h = decode_restricted_with_hidden(z)
    # h: [1, seq_len, 1024] - hidden states at each position

    decoded_text = tokens_to_text(sonar_tokens)

    # 2. Map to Qwen tokens
    qwen_tokens = map_sonar_to_qwen(sonar_tokens)

    if qwen_tokens is None or len(qwen_tokens) == 0:
        print(f"{step:<6} {'N/A':<10} {decoded_text[:60]:<60}")
        continue

    # 3. Run Qwen, get gradients w.r.t. Qwen embeddings
    qwen_embed_grad, loss = compute_qwen_loss_and_embed_grad(qwen_tokens, examples)
    # qwen_embed_grad: [num_qwen_tokens, 896]

    # 4. Map Qwen embedding gradients back to SONAR hidden state space
    # W.linear.weight is [896, 1024], so grad @ W = grad in SONAR space
    W_matrix = W.linear.weight.detach()  # [896, 1024]
    sonar_h_grad = qwen_embed_grad @ W_matrix  # [num_qwen_tokens, 1024]

    # 5. Match hidden states to gradients
    # h[0, i, :] is the hidden state AFTER processing token i, which PREDICTS token i+1
    # So h[0, 1, :] (after lang_tag) predicts the first content token
    # The gradient for the first content token's embedding should flow through h[0, 1, :]
    #
    # Indexing:
    # - sonar_tokens: [BOS=0, lang=1, content[0]=2, content[1]=3, ..., EOS]
    # - h: [0, 1, 2, 3, ...] where h[i] predicts token i+1
    # - qwen_tokens: [content[0], content[1], ...]
    # - So qwen_tokens[i] corresponds to h[0, i+1, :]

    num_content = len(qwen_tokens)

    if step > 0:
        # Create a loss proxy
        # qwen_embed_grad[i] is gradient for qwen_tokens[i]
        # This should flow through h[0, i+1, :] (the hidden that predicted that token)

        loss_proxy = 0
        for i in range(min(num_content, h.shape[1] - 1)):
            h_pos = i + 1  # h position that predicted content token i
            if h_pos < h.shape[1]:
                loss_proxy = loss_proxy + (h[0, h_pos, :] * sonar_h_grad[i]).sum()

        # Add regularization to keep z close to valid embeddings
        reg_loss = 0.01 * ((z - z_init.detach()) ** 2).sum()
        total_loss = loss_proxy + reg_loss

        total_loss.backward()
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
    sonar_tokens, _ = decode_restricted_with_hidden(z)
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
