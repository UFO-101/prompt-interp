"""
Understand cross-attention behavior with z=0 vs masked.

Goal: Figure out how to make per-position z work.
"""

import torch
import torch.nn as nn
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout

dev = "cuda"
torch.cuda.empty_cache()

print("Loading models...", flush=True)
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sdm = sd.model.to(dev)
sonar_dec = sd.tokenizer.create_decoder()

def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())

# ============================================================================
# Understand the cross-attention mechanism
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-ATTENTION ANALYSIS")
print("=" * 80)

# Get a sample z
test_text = "The cat sat on the mat."
with torch.no_grad():
    z_sample = se.predict([test_text], source_lang='eng_Latn').to(dev)

z_zero = torch.zeros(1, 1, 1024, device=dev)
e_sample = z_sample.unsqueeze(1)  # [1, 1, 1024]

print(f"\nz_sample shape: {z_sample.shape}")
print(f"z_sample norm: {z_sample.norm().item():.2f}")
print(f"e_sample shape: {e_sample.shape}")

# Look at cross-attention structure
layer = sdm.decoder.decoder.layers[0]
cross_attn = layer.encoder_decoder_attn

print(f"\nCross-attention module: {type(cross_attn)}")
print(f"Cross-attention attributes: {[a for a in dir(cross_attn) if not a.startswith('_')]}")

# ============================================================================
# Test: What happens in cross-attention with z=0?
# ============================================================================
print("\n" + "=" * 80)
print("TEST: Cross-attention behavior")
print("=" * 80)

# Input tokens
tokens = torch.tensor([[3, 256047, 100]], device=dev)  # BOS, lang, one content token

# Full forward with z_sample
print("\n1. Forward with z_sample:")
with torch.no_grad():
    h1 = sdm.decode(tokens, BatchLayout.of(tokens), e_sample, BatchLayout.of(e_sample))
    if h1.dim() == 4:
        h1 = h1.squeeze(1)
    logits1 = sdm.decoder.final_proj(h1)[0, -1, :]
    pred1 = logits1.argmax().item()
    print(f"   Predicted: '{sonar_dec(torch.tensor([pred1]))}'")

# Full forward with z=0
print("\n2. Forward with z=0:")
with torch.no_grad():
    h2 = sdm.decode(tokens, BatchLayout.of(tokens), z_zero, BatchLayout.of(z_zero))
    if h2.dim() == 4:
        h2 = h2.squeeze(1)
    logits2 = sdm.decoder.final_proj(h2)[0, -1, :]
    pred2 = logits2.argmax().item()
    print(f"   Predicted: '{sonar_dec(torch.tensor([pred2]))}'")

# ============================================================================
# Test: Hook to capture what cross-attention outputs
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-ATTENTION OUTPUTS")
print("=" * 80)

cross_attn_outputs_z = []
cross_attn_outputs_z0 = []

def capture_hook_z(module, input, output):
    cross_attn_outputs_z.append(output.clone().detach())
    return output

def capture_hook_z0(module, input, output):
    cross_attn_outputs_z0.append(output.clone().detach())
    return output

# Capture with z_sample
hooks = []
for layer in sdm.decoder.decoder.layers:
    h = layer.encoder_decoder_attn.output_proj.register_forward_hook(capture_hook_z)
    hooks.append(h)

with torch.no_grad():
    _ = sdm.decode(tokens, BatchLayout.of(tokens), e_sample, BatchLayout.of(e_sample))

for h in hooks:
    h.remove()

# Capture with z=0
hooks = []
for layer in sdm.decoder.decoder.layers:
    h = layer.encoder_decoder_attn.output_proj.register_forward_hook(capture_hook_z0)
    hooks.append(h)

with torch.no_grad():
    _ = sdm.decode(tokens, BatchLayout.of(tokens), z_zero, BatchLayout.of(z_zero))

for h in hooks:
    h.remove()

print(f"\nNumber of layers: {len(cross_attn_outputs_z)}")
print(f"\nLayer 0 cross-attn output shape: {cross_attn_outputs_z[0].shape}")

print("\nComparing cross-attn outputs (z_sample vs z=0):")
for i in range(min(5, len(cross_attn_outputs_z))):
    out_z = cross_attn_outputs_z[i]
    out_z0 = cross_attn_outputs_z0[i]
    diff = (out_z - out_z0).abs()
    print(f"  Layer {i}: z_norm={out_z.norm():.2f}, z0_norm={out_z0.norm():.2f}, diff={diff.mean():.4f}")

# ============================================================================
# Key insight: z=0 still produces non-zero cross-attention output!
# This is because the attention mechanism computes softmax(Q @ K^T) @ V
# When K and V come from z=0, the output depends on the learned projections.
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print("""
z=0 produces NON-ZERO cross-attention output because:
1. encoder_output (z) is projected to K and V via linear layers
2. K = W_k @ z, V = W_v @ z
3. When z=0, K=W_k@0 + b_k = b_k (bias term!)
4. Similarly V = b_v
5. So cross-attn output = softmax(Q @ b_k^T) @ b_v

The BIAS terms in the K,V projections give non-zero output even with z=0!
""")

# ============================================================================
# Check if K,V projections have biases
# ============================================================================
print("\n" + "=" * 80)
print("CHECKING K,V PROJECTION BIASES")
print("=" * 80)

layer = sdm.decoder.decoder.layers[0]
cross_attn = layer.encoder_decoder_attn

# Check structure
print(f"Cross-attn type: {type(cross_attn)}")

# Try to find K, V projections
for name, module in cross_attn.named_modules():
    if 'proj' in name.lower() or 'linear' in name.lower():
        if hasattr(module, 'bias'):
            has_bias = module.bias is not None
            print(f"  {name}: has_bias={has_bias}")
            if has_bias:
                print(f"    bias norm: {module.bias.norm().item():.4f}")

# ============================================================================
# Solution 1: Instead of masking OUTPUT, we could mask the ATTENTION WEIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("SOLUTION EXPLORATION")
print("=" * 80)

print("""
Options to get true z=0 behavior at specific positions:

1. MASK ATTENTION WEIGHTS: Set attention weights to 0 for positions that
   shouldn't attend to z. But this is tricky because we want them to
   attend to z (with z=0 value), not ignore z entirely.

2. REPLACE ENCODER OUTPUT: For specific positions, replace the encoder
   output (z) with zeros. This requires modifying how encoder_output
   is used in cross-attention.

3. TWO FORWARD PASSES:
   - Pass 1: Forward prompt positions with z
   - Pass 2: Forward task positions with z=0
   - Combine hidden states
   But this breaks autoregressive dependencies.

4. MODIFY ENCODER OUTPUT PER-POSITION:
   - Encoder output is [batch, 1, dim] (single z repeated for all positions)
   - We could expand it to [batch, seq_len, dim] with z for some positions
     and zeros for others.
   - BUT: cross-attention typically broadcasts the encoder output...

Let's try option 4 - expand encoder output.
""")

# ============================================================================
# Test: Can we pass per-position encoder output?
# ============================================================================
print("\n" + "=" * 80)
print("TEST: Per-position encoder output")
print("=" * 80)

# Try passing expanded encoder output
seq_len = tokens.shape[1]

# All z_sample
encoder_out_all_z = z_sample.unsqueeze(1).expand(-1, seq_len, -1)  # [1, seq_len, 1024]
print(f"encoder_out_all_z shape: {encoder_out_all_z.shape}")

# All zeros
encoder_out_all_zero = torch.zeros(1, seq_len, 1024, device=dev)
print(f"encoder_out_all_zero shape: {encoder_out_all_zero.shape}")

# Mixed: z for first 2 positions, 0 for rest
encoder_out_mixed = torch.zeros(1, seq_len, 1024, device=dev)
encoder_out_mixed[:, :2, :] = z_sample

print("\nTrying forward with expanded encoder output...")
try:
    with torch.no_grad():
        h_expanded = sdm.decode(tokens, BatchLayout.of(tokens),
                               encoder_out_all_z, BatchLayout.of(encoder_out_all_z))
        print(f"Success! h_expanded shape: {h_expanded.shape}")
        if h_expanded.dim() == 4:
            h_expanded = h_expanded.squeeze(1)
        logits_exp = sdm.decoder.final_proj(h_expanded)[0, -1, :]
        pred_exp = logits_exp.argmax().item()
        print(f"Predicted with expanded z: '{sonar_dec(torch.tensor([pred_exp]))}'")
except Exception as e:
    print(f"Error: {e}")

print("\nTrying forward with all-zero expanded encoder output...")
try:
    with torch.no_grad():
        h_zero_exp = sdm.decode(tokens, BatchLayout.of(tokens),
                               encoder_out_all_zero, BatchLayout.of(encoder_out_all_zero))
        print(f"Success! h_zero_exp shape: {h_zero_exp.shape}")
        if h_zero_exp.dim() == 4:
            h_zero_exp = h_zero_exp.squeeze(1)
        logits_zero_exp = sdm.decoder.final_proj(h_zero_exp)[0, -1, :]
        pred_zero_exp = logits_zero_exp.argmax().item()
        print(f"Predicted with expanded z=0: '{sonar_dec(torch.tensor([pred_zero_exp]))}'")
except Exception as e:
    print(f"Error: {e}")

# Compare to standard z=0
print(f"\nCompare to standard z=0: '{sonar_dec(torch.tensor([pred2]))}'")

print("\nTrying forward with MIXED encoder output (z for pos 0-1, 0 for pos 2)...")
try:
    with torch.no_grad():
        h_mixed = sdm.decode(tokens, BatchLayout.of(tokens),
                            encoder_out_mixed, BatchLayout.of(encoder_out_mixed))
        print(f"Success! h_mixed shape: {h_mixed.shape}")
        if h_mixed.dim() == 4:
            h_mixed = h_mixed.squeeze(1)
        logits_mixed = sdm.decoder.final_proj(h_mixed)[0, -1, :]
        pred_mixed = logits_mixed.argmax().item()
        print(f"Predicted with mixed z: '{sonar_dec(torch.tensor([pred_mixed]))}'")
except Exception as e:
    print(f"Error: {e}")
