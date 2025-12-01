"""
Debug: understand the difference between z=0 and masked cross-attention
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

def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())

def text_to_tokens(text):
    tokens = sonar_enc(text)
    return tokens.to(dev)

z_zero = torch.zeros(1, 1, 1024, device=dev)

# ============================================================================
# Check: What does cross-attention output when z=0?
# ============================================================================
print("\n" + "=" * 80)
print("Understanding cross-attention with z=0")
print("=" * 80)

# Hook to capture cross-attention outputs
cross_attn_outputs = []

def capture_hook(module, input, output):
    cross_attn_outputs.append(output.clone().detach())
    return output

# Register hooks on first layer's cross-attention
layer0 = sdm.decoder.decoder.layers[0]
h = layer0.encoder_decoder_attn.output_proj.register_forward_hook(capture_hook)

# Run with z=0
tokens = [3, 256047]  # BOS, lang
di = torch.tensor([tokens], device=dev)
with torch.no_grad():
    _ = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))

h.remove()

print(f"\nCross-attention output with z=0:")
print(f"  Shape: {cross_attn_outputs[0].shape}")
print(f"  Mean: {cross_attn_outputs[0].mean().item():.6f}")
print(f"  Std: {cross_attn_outputs[0].std().item():.6f}")
print(f"  Max: {cross_attn_outputs[0].max().item():.6f}")
print(f"  Min: {cross_attn_outputs[0].min().item():.6f}")

# Check if it's all zeros
is_zero = (cross_attn_outputs[0].abs() < 1e-6).all()
print(f"  Is all zeros: {is_zero}")

# ============================================================================
# So masking to zero IS different from z=0
# But for our optimization, maybe it doesn't matter as long as gradients flow?
# ============================================================================
print("\n" + "=" * 80)
print("Test: Does it matter for optimization?")
print("=" * 80)

# The key question: when we do teacher forcing with mixed mask,
# do the prompt positions correctly influence the answer prediction?

# Let's test with a simpler setup:
# 1. Generate a fixed prompt
# 2. Append task input
# 3. Do ONE forward pass with:
#    - Full z for all positions (baseline)
#    - z=0 for all positions (unconditioned)
#    - Mixed: z for prompt, mask=0 for task

test_text = "hot is the opposite of cold, big is the opposite of small"
with torch.no_grad():
    z_test = se.predict([test_text], source_lang='eng_Latn').to(dev)

# Prompt tokens (from z)
with torch.no_grad():
    prompt_tokens = [3, 256047]
    e = z_test.unsqueeze(1)
    for _ in range(10):
        di = torch.tensor([prompt_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        prompt_tokens.append(next_tok)
        if next_tok == 3:
            break
    if prompt_tokens[-1] == 3:
        prompt_tokens = prompt_tokens[:-1]

prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
print(f"\nPrompt: '{prompt_text}'")
print(f"Prompt length: {len(prompt_tokens)}")

# Task input
task_input = "hot -> "
task_tokens = text_to_tokens(task_input)
task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]

# Full sequence (will generate answer)
full_tokens = prompt_tokens + task_content.tolist()
print(f"Full before answer: '{tokens_to_text(torch.tensor(full_tokens))}'")

# Generate with full z
print("\n1. Generate answer with full z (conditioned on prompt z):")
with torch.no_grad():
    seq = full_tokens.copy()
    for _ in range(8):
        di = torch.tensor([seq], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        seq.append(next_tok)
        if next_tok == 3:
            break
    answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):]
    print(f"   Answer: '{answer}'")

# Generate with z=0 for everything
print("\n2. Generate answer with z=0 (fully unconditioned):")
with torch.no_grad():
    seq = full_tokens.copy()
    for _ in range(8):
        di = torch.tensor([seq], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        seq.append(next_tok)
        if next_tok == 3:
            break
    answer = tokens_to_text(torch.tensor(seq))[len(tokens_to_text(torch.tensor(full_tokens))):]
    print(f"   Answer: '{answer}'")

# ============================================================================
# The real test: teacher-forced forward with mixed mask
# ============================================================================
print("\n" + "=" * 80)
print("Teacher-forced forward pass comparison")
print("=" * 80)

# Add an answer token for teacher forcing
answer_tokens = text_to_tokens("cold")
answer_content = answer_tokens[2:-1] if answer_tokens[-1] == 3 else answer_tokens[2:]
first_answer_tok = answer_content[0].item()

full_with_answer = full_tokens + [first_answer_tok]
input_tokens = torch.tensor(full_with_answer[:-1], device=dev).unsqueeze(0)

print(f"Input (teacher-forced): '{tokens_to_text(torch.tensor(full_with_answer[:-1]))}'")
print(f"Target: '{sonar_dec(torch.tensor([first_answer_tok]))}' (token {first_answer_tok})")

# Forward with full z
with torch.no_grad():
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e, BatchLayout.of(e))
    logits_full_z = sdm.decoder.final_proj(h)[0, -1, :]
    pred_full_z = logits_full_z.argmax().item()
    prob_full_z = F.softmax(logits_full_z, dim=0)[first_answer_tok].item()

print(f"\n1. Full z:")
print(f"   Predicted: '{sonar_dec(torch.tensor([pred_full_z]))}' (token {pred_full_z})")
print(f"   P(cold): {prob_full_z:.4f}")

# Forward with z=0
with torch.no_grad():
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), z_zero, BatchLayout.of(z_zero))
    logits_z0 = sdm.decoder.final_proj(h)[0, -1, :]
    pred_z0 = logits_z0.argmax().item()
    prob_z0 = F.softmax(logits_z0, dim=0)[first_answer_tok].item()

print(f"\n2. z=0:")
print(f"   Predicted: '{sonar_dec(torch.tensor([pred_z0]))}' (token {pred_z0})")
print(f"   P(cold): {prob_z0:.4f}")

# Forward with mixed mask (prompt=1, task=0)
class MixedMaskHook:
    def __init__(self, mask):
        self.mask = mask
        self.hooks = []

    def hook_fn(self, module, input, output):
        m = self.mask.view(1, -1, 1).to(output.device).to(output.dtype)
        if m.shape[1] != output.shape[1]:
            new_m = torch.zeros(1, output.shape[1], 1, device=output.device, dtype=output.dtype)
            copy_len = min(m.shape[1], output.shape[1])
            new_m[:, :copy_len, :] = m[:, :copy_len, :]
            m = new_m
        return output * m

    def register(self, model):
        for layer in model.decoder.decoder.layers:
            h = layer.encoder_decoder_attn.output_proj.register_forward_hook(self.hook_fn)
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

prompt_len = len(prompt_tokens)
mask = torch.zeros(len(full_with_answer) - 1, device=dev)
mask[:prompt_len] = 1.0

print(f"\n3. Mixed mask (prompt positions use z, task positions masked to 0):")
print(f"   Mask: 1 for positions 0-{prompt_len-1}, 0 for positions {prompt_len}-{len(mask)-1}")

masker = MixedMaskHook(mask)
masker.register(sdm)

with torch.no_grad():
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e, BatchLayout.of(e))
    logits_mixed = sdm.decoder.final_proj(h)[0, -1, :]
    pred_mixed = logits_mixed.argmax().item()
    prob_mixed = F.softmax(logits_mixed, dim=0)[first_answer_tok].item()

masker.remove()

print(f"   Predicted: '{sonar_dec(torch.tensor([pred_mixed]))}' (token {pred_mixed})")
print(f"   P(cold): {prob_mixed:.4f}")

# ============================================================================
# Key insight: the mask approach zeros cross-attention contribution,
# which is NOT the same as z=0. But for optimization, what matters is:
# 1. Gradients flow from answer loss back to prompt positions
# 2. Prompt positions can influence the answer prediction
# ============================================================================
print("\n" + "=" * 80)
print("GRADIENT FLOW TEST")
print("=" * 80)

z = nn.Parameter(z_test.clone())
e_param = z.unsqueeze(0).unsqueeze(1)

# Create mask
mask = torch.zeros(len(full_with_answer) - 1, device=dev)
mask[:prompt_len] = 1.0

masker = MixedMaskHook(mask)
masker.register(sdm)

# Forward
h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e_param, BatchLayout.of(e_param))
logits = sdm.decoder.final_proj(h)[0, -1, :]

masker.remove()

# Loss
target = torch.tensor([first_answer_tok], device=dev)
loss = F.cross_entropy(logits.unsqueeze(0), target)

print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

print(f"z.grad norm: {z.grad.norm().item():.4f}")
print(f"z.grad mean: {z.grad.mean().item():.6f}")

# Try an optimization step
print("\nTrying one optimization step...")
z_before = z.data.clone()
with torch.no_grad():
    z.data = z.data - 0.1 * z.grad
z_after = z.data.clone()

print(f"z changed by: {(z_after - z_before).norm().item():.6f}")

# Check if prediction changed
z.grad = None
e_param = z.unsqueeze(0).unsqueeze(1)

masker = MixedMaskHook(mask)
masker.register(sdm)

with torch.no_grad():
    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e_param, BatchLayout.of(e_param))
    logits_new = sdm.decoder.final_proj(h)[0, -1, :]
    pred_new = logits_new.argmax().item()
    prob_new = F.softmax(logits_new, dim=0)[first_answer_tok].item()

masker.remove()

print(f"\nAfter update:")
print(f"  Predicted: '{sonar_dec(torch.tensor([pred_new]))}' (token {pred_new})")
print(f"  P(cold): {prob_new:.4f} (was {prob_mixed:.4f})")
