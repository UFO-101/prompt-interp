"""
Debug version: verify the mixed-z approach is working as expected.
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


class PositionMaskedCrossAttention:
    """
    Hook to zero out cross-attention for specific positions.

    We hook into the cross-attention module itself and zero the encoder output
    for positions where we want z=0 behavior.
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.mask = None  # [seq_len] - 1 means use z, 0 means use z=0

    def set_mask(self, mask):
        self.mask = mask

    def _make_hook(self):
        def hook(module, input, output):
            if self.mask is None:
                return output
            # output from cross-attention: [batch, seq_len, dim]
            # We want positions with mask=0 to behave as if z=0
            # The simplest way: just zero out the cross-attention output
            # for those positions
            mask = self.mask.view(1, -1, 1).to(output.device).to(output.dtype)
            # Expand mask to match sequence length if needed
            if mask.shape[1] != output.shape[1]:
                # Pad or truncate
                new_mask = torch.zeros(1, output.shape[1], 1, device=output.device, dtype=output.dtype)
                copy_len = min(mask.shape[1], output.shape[1])
                new_mask[:, :copy_len, :] = mask[:, :copy_len, :]
                mask = new_mask
            return output * mask
        return hook

    def register_hooks(self):
        for layer in self.model.decoder.decoder.layers:
            h = layer.encoder_decoder_attn.output_proj.register_forward_hook(self._make_hook())
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ============================================================================
# Test 1: Verify z=0 masking works
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Verify position masking works")
print("=" * 80)

# Create a test z
test_text = "The opposite of hot is cold."
with torch.no_grad():
    z_test = se.predict([test_text], source_lang='eng_Latn').to(dev)

e = z_test.unsqueeze(1)  # [1, 1, 1024]

# Decode normally with z
print("\nDecoding with full z:")
with torch.no_grad():
    tokens = [3, 256047]
    for _ in range(15):
        di = torch.tensor([tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        tokens.append(next_tok)
        if next_tok == 3:
            break
    print(f"  Result: '{tokens_to_text(torch.tensor(tokens))}'")

# Decode with z=0 (all zeros)
print("\nDecoding with z=0:")
z_zero = torch.zeros(1, 1, 1024, device=dev)
with torch.no_grad():
    tokens = [3, 256047]
    for _ in range(15):
        di = torch.tensor([tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        tokens.append(next_tok)
        if next_tok == 3:
            break
    print(f"  Result: '{tokens_to_text(torch.tensor(tokens))}'")

# Decode with mask=0 (should be same as z=0)
print("\nDecoding with z but mask=0 for all positions:")
masker = PositionMaskedCrossAttention(sdm)
with torch.no_grad():
    tokens = [3, 256047]
    for _ in range(15):
        # Set mask to zeros for current length
        masker.set_mask(torch.zeros(len(tokens), device=dev))
        masker.register_hooks()

        di = torch.tensor([tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        tokens.append(next_tok)

        masker.remove_hooks()

        if next_tok == 3:
            break
    print(f"  Result: '{tokens_to_text(torch.tensor(tokens))}'")

# ============================================================================
# Test 2: Mixed z decoding
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Mixed z decoding (prompt with z, continuation with z=0)")
print("=" * 80)

# Generate prompt with z
print("\nStep 1: Generate prompt with z")
with torch.no_grad():
    prompt_tokens = [3, 256047]
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
    print(f"  Prompt ({len(prompt_tokens)} tokens): '{prompt_text}'")

prompt_len = len(prompt_tokens)

# Add task input
task_input = "hot -> "
task_tokens = text_to_tokens(task_input)
task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
full_tokens = prompt_tokens + task_content.tolist()
print(f"\nStep 2: Append task input")
print(f"  Full sequence ({len(full_tokens)} tokens): '{tokens_to_text(torch.tensor(full_tokens))}'")

# Continue with z=0 for answer
print("\nStep 3: Generate answer with z=0")
with torch.no_grad():
    for _ in range(8):
        di = torch.tensor([full_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        full_tokens.append(next_tok)
        if next_tok == 3:
            break
    full_text = tokens_to_text(torch.tensor(full_tokens))
    answer_text = full_text[len(tokens_to_text(torch.tensor(prompt_tokens + task_content.tolist()))):]
    print(f"  Full output: '{full_text}'")
    print(f"  Answer only: '{answer_text}'")

# ============================================================================
# Test 3: Verify gradients flow through mixed-z forward pass
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Verify gradients flow to z")
print("=" * 80)

z = nn.Parameter(z_test.clone())

# Full tokens from above
full_tokens_tensor = torch.tensor(full_tokens, device=dev)

# Target: first token of "cold"
target_tokens = text_to_tokens("cold")
target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
target_token = target_content[0]
print(f"\nTarget token: {target_token.item()} ('{sonar_dec(torch.tensor([target_token.item()]))}')")

# Forward with mixed z
e_param = z.unsqueeze(1)
input_len = len(full_tokens_tensor) - 1

# Create mask: 1 for prompt positions, 0 for task+answer
mask = torch.zeros(input_len, device=dev)
mask[:prompt_len] = 1.0
print(f"\nMask: first {prompt_len} positions = 1, rest = 0")
print(f"  mask[:5] = {mask[:5].tolist()}")
print(f"  mask[-5:] = {mask[-5:].tolist()}")

masker = PositionMaskedCrossAttention(sdm)
masker.set_mask(mask)
masker.register_hooks()

input_tokens = full_tokens_tensor[:-1].unsqueeze(0)
h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e_param, BatchLayout.of(e_param))
logits = sdm.decoder.final_proj(h)

masker.remove_hooks()

# Loss at last position
last_logits = logits[0, -1, :]
loss = F.cross_entropy(last_logits.unsqueeze(0), target_token.unsqueeze(0))
print(f"\nLoss: {loss.item():.4f}")

# Predicted token
pred_token = last_logits.argmax().item()
print(f"Predicted token: {pred_token} ('{sonar_dec(torch.tensor([pred_token]))}')")

# Backprop
loss.backward()

print(f"\nGradients:")
print(f"  z.grad is None: {z.grad is None}")
if z.grad is not None:
    print(f"  z.grad.norm(): {z.grad.norm().item():.6f}")
    print(f"  z.grad.abs().max(): {z.grad.abs().max().item():.6f}")
    print(f"  z.grad.abs().mean(): {z.grad.abs().mean().item():.6f}")

# ============================================================================
# Test 4: Quick optimization to verify it works
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Quick optimization (20 steps)")
print("=" * 80)

# Fresh z
z = nn.Parameter(z_test.clone())
optimizer = torch.optim.Adam([z], lr=0.01)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
]

for step in range(21):
    optimizer.zero_grad()

    total_loss = 0.0

    for input_text, target_text in examples:
        # Generate prompt with current z
        with torch.no_grad():
            e_gen = z.unsqueeze(0).unsqueeze(1)
            prompt_tokens = [3, 256047]
            for _ in range(12):
                di = torch.tensor([prompt_tokens], device=dev)
                h = sdm.decode(di, BatchLayout.of(di), e_gen, BatchLayout.of(e_gen))
                logits = sdm.decoder.final_proj(h)[0, -1, :]
                next_tok = logits.argmax().item()
                prompt_tokens.append(next_tok)
                if next_tok == 3:
                    break
            if prompt_tokens[-1] == 3:
                prompt_tokens = prompt_tokens[:-1]

        prompt_len = len(prompt_tokens)

        # Add task
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full_tokens = prompt_tokens + task_content.tolist()

        # Generate answer with z=0
        with torch.no_grad():
            for _ in range(5):
                di = torch.tensor([full_tokens], device=dev)
                h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
                logits = sdm.decoder.final_proj(h)[0, -1, :]
                next_tok = logits.argmax().item()
                full_tokens.append(next_tok)
                if next_tok == 3:
                    break

        full_tokens_tensor = torch.tensor(full_tokens, device=dev)

        # Get target
        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
        target_token = target_content[0]

        # Forward with gradients
        e_param = z.unsqueeze(0).unsqueeze(1)
        input_len = len(full_tokens_tensor) - 1
        mask = torch.zeros(input_len, device=dev)
        mask[:prompt_len] = 1.0

        masker = PositionMaskedCrossAttention(sdm)
        masker.set_mask(mask)
        masker.register_hooks()

        input_tokens = full_tokens_tensor[:-1].unsqueeze(0)
        h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e_param, BatchLayout.of(e_param))
        logits = sdm.decoder.final_proj(h)

        masker.remove_hooks()

        last_logits = logits[0, -1, :]
        loss = F.cross_entropy(last_logits.unsqueeze(0), target_token.unsqueeze(0))
        total_loss = total_loss + loss

    avg_loss = total_loss / len(examples)

    if step > 0:
        avg_loss.backward()
        optimizer.step()

    # Every 5 steps, show detail
    if step % 5 == 0:
        with torch.no_grad():
            e_gen = z.unsqueeze(0).unsqueeze(1)
            prompt_tokens = [3, 256047]
            for _ in range(12):
                di = torch.tensor([prompt_tokens], device=dev)
                h = sdm.decode(di, BatchLayout.of(di), e_gen, BatchLayout.of(e_gen))
                logits_p = sdm.decoder.final_proj(h)[0, -1, :]
                next_tok = logits_p.argmax().item()
                prompt_tokens.append(next_tok)
                if next_tok == 3:
                    break
            prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

        print(f"\nStep {step}: loss={avg_loss.item():.4f}")
        print(f"  Prompt: '{prompt_text}'")

        # Test on examples
        for input_text, target_text in examples:
            with torch.no_grad():
                # Use current prompt
                if prompt_tokens[-1] == 3:
                    pt = prompt_tokens[:-1]
                else:
                    pt = prompt_tokens

                task_tokens = text_to_tokens(input_text)
                task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
                full = pt + task_content.tolist()

                for _ in range(5):
                    di = torch.tensor([full], device=dev)
                    h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
                    logits_a = sdm.decoder.final_proj(h)[0, -1, :]
                    next_tok = logits_a.argmax().item()
                    full.append(next_tok)
                    if next_tok == 3:
                        break

                full_text = tokens_to_text(torch.tensor(full))
                prompt_task = tokens_to_text(torch.tensor(pt + task_content.tolist()))
                answer = full_text[len(prompt_task):].strip()

            correct = answer.lower().startswith(target_text.lower())
            mark = "OK" if correct else "X"
            print(f"  {input_text} -> '{answer[:20]}' (want: {target_text}) {mark}")
