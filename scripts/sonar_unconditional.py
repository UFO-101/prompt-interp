"""
Experiments to make SONAR decoder work as unconditional text generator.

Ideas to neutralize cross-attention without training:
1. Zero out cross-attention output_proj weights/bias
2. Use a "mean z" computed from many sentences
3. Use zero z vector
4. Hook and zero out cross-attention output
5. Scale down cross-attention contribution gradually

"""

import torch
import torch.nn as nn
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from fairseq2.nn.batch_layout import BatchLayout
import copy

dev = "cuda"
torch.cuda.empty_cache()

print("Loading models...", flush=True)
se = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
sd = EmbeddingToTextModelPipeline(decoder='text_sonar_basic_decoder', tokenizer='text_sonar_basic_encoder')
sdm = sd.model.to(dev)
sonar_dec = sd.tokenizer.create_decoder()

# Helper to decode tokens to text
def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())

def generate(z, max_len=50, temperature=1.0, top_k=50):
    """Generate text from z vector using sampling."""
    with torch.no_grad():
        e = z.unsqueeze(0) if z.dim() == 1 else z
        eo = e.unsqueeze(1)

        generated = [3, 256047]  # BOS, eng_Latn

        for _ in range(max_len):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

            if next_token == 3:  # EOS
                break

    return torch.tensor(generated, device=dev)

# ============================================================================
# Experiment 1: Zero z vector
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: Zero z vector")
print("=" * 80)

z_zero = torch.zeros(1024, device=dev)
print("\nGenerating with z=0:")
for i in range(5):
    tokens = generate(z_zero, temperature=0.8)
    text = tokens_to_text(tokens)
    print(f"  {i+1}. {text[:80]}")

# ============================================================================
# Experiment 2: Mean z from many sentences
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: Mean z from diverse sentences")
print("=" * 80)

diverse_sentences = [
    "The cat sat on the mat.",
    "I love programming in Python.",
    "What is the meaning of life?",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating.",
    "Hello, how are you today?",
    "The weather is nice outside.",
    "Can you help me with this problem?",
    "I think therefore I am.",
    "The answer is 42.",
    "Music makes everything better.",
    "Time flies when you're having fun.",
    "Knowledge is power.",
    "Actions speak louder than words.",
    "Every cloud has a silver lining.",
    "A picture is worth a thousand words.",
    "Rome wasn't built in a day.",
    "All that glitters is not gold.",
    "When in Rome, do as the Romans do.",
    "The early bird catches the worm.",
]

with torch.no_grad():
    z_list = []
    for sent in diverse_sentences:
        z = se.predict([sent], source_lang='eng_Latn').to(dev)
        z_list.append(z)
    z_mean = torch.stack(z_list).mean(dim=0).squeeze(0)

print(f"\nMean z norm: {z_mean.norm().item():.2f}")
print("\nGenerating with mean z:")
for i in range(5):
    tokens = generate(z_mean, temperature=0.8)
    text = tokens_to_text(tokens)
    print(f"  {i+1}. {text[:80]}")

# ============================================================================
# Experiment 3: Random z from unit sphere
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 3: Random z from unit sphere (scaled to mean norm)")
print("=" * 80)

# Get typical z norm
with torch.no_grad():
    z_sample = se.predict(["Hello world."], source_lang='eng_Latn').to(dev)
    typical_norm = z_sample.norm().item()

print(f"Typical z norm: {typical_norm:.2f}")

print("\nGenerating with random z vectors:")
for i in range(5):
    z_rand = torch.randn(1024, device=dev)
    z_rand = z_rand / z_rand.norm() * typical_norm
    tokens = generate(z_rand, temperature=0.8)
    text = tokens_to_text(tokens)
    print(f"  {i+1}. {text[:80]}")

# ============================================================================
# Experiment 4: Zero out cross-attention output_proj weights
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 4: Zero out cross-attention output_proj weights")
print("=" * 80)

# Make a copy for this experiment
sdm_zeroed = copy.deepcopy(sdm)

# Zero out all cross-attention output projections
for layer in sdm_zeroed.decoder.decoder.layers:
    layer.encoder_decoder_attn.output_proj.weight.data.zero_()
    layer.encoder_decoder_attn.output_proj.bias.data.zero_()

def generate_with_model(model, z, max_len=50, temperature=1.0, top_k=50):
    """Generate with a specific model."""
    with torch.no_grad():
        e = z.unsqueeze(0) if z.dim() == 1 else z
        eo = e.unsqueeze(1)

        generated = [3, 256047]

        for _ in range(max_len):
            di = torch.tensor([generated], device=dev)
            h = model.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = model.decoder.final_proj(h)[0, -1, :]

            logits = logits / temperature
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
                logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

            if next_token == 3:
                break

    return torch.tensor(generated, device=dev)

print("\nGenerating with zeroed cross-attention (using random z):")
for i in range(5):
    z_rand = torch.randn(1024, device=dev)
    z_rand = z_rand / z_rand.norm() * typical_norm
    tokens = generate_with_model(sdm_zeroed, z_rand, temperature=0.8)
    text = tokens_to_text(tokens)
    print(f"  {i+1}. {text[:80]}")

del sdm_zeroed  # Free memory

# ============================================================================
# Experiment 5: Hook to scale down cross-attention
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 5: Hook to scale down cross-attention output")
print("=" * 80)

hooks = []
scale_factor = 0.0  # Try zeroing out

def make_scale_hook(scale):
    def hook(module, input, output):
        return output * scale
    return hook

# Register hooks on all cross-attention output projections
for layer in sdm.decoder.decoder.layers:
    h = layer.encoder_decoder_attn.output_proj.register_forward_hook(make_scale_hook(scale_factor))
    hooks.append(h)

print(f"\nGenerating with cross-attention scaled by {scale_factor}:")
for i in range(5):
    z_rand = torch.randn(1024, device=dev)
    z_rand = z_rand / z_rand.norm() * typical_norm
    tokens = generate(z_rand, temperature=0.8)
    text = tokens_to_text(tokens)
    print(f"  {i+1}. {text[:80]}")

# Remove hooks
for h in hooks:
    h.remove()
hooks = []

# ============================================================================
# Experiment 6: Use greedy decoding with mean z
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 6: Greedy decoding with mean z")
print("=" * 80)

def generate_greedy(z, max_len=50):
    """Generate text using greedy decoding."""
    with torch.no_grad():
        e = z.unsqueeze(0) if z.dim() == 1 else z
        eo = e.unsqueeze(1)

        generated = [3, 256047]

        for _ in range(max_len):
            di = torch.tensor([generated], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_token = logits.argmax().item()
            generated.append(next_token)

            if next_token == 3:
                break

    return torch.tensor(generated, device=dev)

print("\nGreedy decode with mean z:")
tokens = generate_greedy(z_mean)
text = tokens_to_text(tokens)
print(f"  Result: {text}")

print("\nGreedy decode with zero z:")
tokens = generate_greedy(z_zero)
text = tokens_to_text(tokens)
print(f"  Result: {text}")

# ============================================================================
# Experiment 7: Compute "null z" that gives uniform output distribution
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 7: Find z that maximizes entropy (more uniform)")
print("=" * 80)

# Start from mean and try to increase entropy
z_null = z_mean.clone().requires_grad_(True)
optimizer = torch.optim.Adam([z_null], lr=0.1)

print("Optimizing z for high entropy output...")
for step in range(50):
    optimizer.zero_grad()

    e = z_null.unsqueeze(0)
    eo = e.unsqueeze(1)

    # Just look at first token prediction
    di = torch.tensor([[3, 256047]], device=dev)
    h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
    logits = sdm.decoder.final_proj(h)[0, -1, :]

    # Maximize entropy
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum()

    # Minimize negative entropy
    loss = -entropy
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"  Step {step}: entropy = {entropy.item():.2f}")

print(f"\nFinal z_null norm: {z_null.norm().item():.2f}")
print("\nGenerating with entropy-optimized z:")
with torch.no_grad():
    for i in range(5):
        tokens = generate(z_null.detach(), temperature=0.8)
        text = tokens_to_text(tokens)
        print(f"  {i+1}. {text[:80]}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key findings:
- Zero z: [check output above]
- Mean z: [check output above]
- Random z: [check output above]
- Zeroed cross-attn: [check output above]
- Hooked scale=0: [check output above]
- Max entropy z: [check output above]

The question is: which approach produces coherent, diverse text
that doesn't depend on a specific input meaning?
""")
