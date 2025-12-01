"""
Mixed z optimization: z for prompt positions, z=0 for task/answer positions.

Key idea:
- Generate prompt with z (conditioned)
- Continue with task input + answer using z=0 at those positions
- Backprop loss through the whole sequence
- z receives gradients from prompt positions

Implementation:
- Hook into cross-attention to swap z for different positions
- Or: modify the encoder_output per-position
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


# ============================================================================
# Approach: Use hooks to mask cross-attention output for specific positions
# ============================================================================

class PositionMaskedCrossAttention:
    """
    Hooks into cross-attention layers to zero out the contribution
    for specific token positions (making them "unconditioned").
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.mask = None  # [seq_len] tensor, 1 = use z, 0 = use zero

    def set_mask(self, mask):
        """Set which positions use z (1) vs zero (0)."""
        self.mask = mask

    def _make_hook(self):
        def hook(module, input, output):
            if self.mask is None:
                return output
            # output shape: [batch, seq_len, dim]
            # mask shape: [seq_len]
            # Expand mask to match output
            mask = self.mask.view(1, -1, 1).to(output.device)
            # Zero out cross-attention contribution where mask is 0
            return output * mask
        return hook

    def register_hooks(self):
        """Register hooks on all cross-attention output projections."""
        for layer in self.model.decoder.decoder.layers:
            h = layer.encoder_decoder_attn.output_proj.register_forward_hook(self._make_hook())
            self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ============================================================================
# Main decoding with mixed z
# ============================================================================

def decode_with_mixed_z(z, task_tokens, max_prompt_len=20, max_answer_len=5):
    """
    Generate:
    1. Prompt tokens (using z, conditioned)
    2. Task tokens (appended, using z=0)
    3. Answer tokens (generated, using z=0)

    Returns full token sequence and the position where answer starts.
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)  # [1, 1, 1024]

    # Phase 1: Generate prompt with z (no grad needed for generation)
    with torch.no_grad():
        prompt_tokens = [3, 256047]  # BOS, eng_Latn

        for _ in range(max_prompt_len):
            di = torch.tensor([prompt_tokens], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_token = logits.argmax().item()
            prompt_tokens.append(next_token)
            if next_token == 3:  # EOS
                break

        # Remove EOS if present (we want to continue)
        if prompt_tokens[-1] == 3:
            prompt_tokens = prompt_tokens[:-1]

    prompt_len = len(prompt_tokens)

    # Phase 2: Append task tokens
    # task_tokens should NOT include BOS/lang, just content
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full_tokens = prompt_tokens + task_content.tolist()

    task_end = len(full_tokens)

    # Phase 3: Generate answer tokens with z=0 (no grad for generation)
    with torch.no_grad():
        # Create mask: 1 for prompt positions, 0 for task+answer positions
        # But for generation we just use z=0 directly
        z_zero = torch.zeros(1, 1, 1024, device=dev)

        for _ in range(max_answer_len):
            di = torch.tensor([full_tokens], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_token = logits.argmax().item()
            full_tokens.append(next_token)
            if next_token == 3:
                break

    return torch.tensor(full_tokens, device=dev), prompt_len, task_end


def forward_with_mixed_z_and_loss(z, full_tokens, prompt_len, target_token):
    """
    Forward pass through decoder with mixed z:
    - Positions 0 to prompt_len-1: use z (conditioned)
    - Positions prompt_len onwards: use z=0 (unconditioned)

    Compute loss at the last position predicting target_token.
    Gradients flow back through prompt positions to z.
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)  # [1, 1, 1024]

    # Input is all tokens except last (teacher forcing)
    input_len = len(full_tokens) - 1

    # Create position mask: 1 for prompt, 0 for rest
    # Mask length matches input_len (not full sequence)
    mask = torch.zeros(input_len, device=dev)
    mask[:min(prompt_len, input_len)] = 1.0

    # Set up hooks
    masker = PositionMaskedCrossAttention(sdm)
    masker.set_mask(mask)
    masker.register_hooks()

    try:
        # Teacher-forced forward pass (input is all tokens except last)
        input_tokens = full_tokens[:-1].unsqueeze(0)
        h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), eo, BatchLayout.of(eo))
        logits = sdm.decoder.final_proj(h)  # [1, seq_len-1, vocab]

        # Loss at last position
        last_logits = logits[0, -1, :]
        loss = F.cross_entropy(last_logits.unsqueeze(0), target_token.unsqueeze(0))

        return loss, logits
    finally:
        masker.remove_hooks()


# ============================================================================
# Main optimization
# ============================================================================
print("\n" + "=" * 80)
print("MIXED Z OPTIMIZATION")
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

# Seed
seed = "Find the opposite: hot becomes cold, big becomes small."
print(f"\nSeed: '{seed}'")

# Encode seed
with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.005)

# Regularization weight - keep z close to init
reg_weight = 0.5

print("\n" + "-" * 80)
print(f"{'Step':<6} {'Loss':<10} {'Acc':<6} {'Prompt':<35} {'Answer':<15}")
print("-" * 80)

n_steps = 100
best_acc = 0.0
best_z = z_init.clone()

for step in range(n_steps + 1):
    optimizer.zero_grad()

    total_loss = 0.0
    n_correct = 0

    # Get current prompt (for display)
    with torch.no_grad():
        e = z.unsqueeze(0) if z.dim() == 1 else z
        eo = e.unsqueeze(1)
        display_tokens = [3, 256047]
        for _ in range(20):
            di = torch.tensor([display_tokens], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_token = logits.argmax().item()
            display_tokens.append(next_token)
            if next_token == 3:
                break
        prompt_text = tokens_to_text(torch.tensor(display_tokens, device=dev))

    answers = []

    for input_text, target_text in examples:
        # Get task and target tokens
        task_tokens = text_to_tokens(input_text)
        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]

        if len(target_content) == 0:
            continue

        # Generate full sequence with mixed z
        full_tokens, prompt_len, task_end = decode_with_mixed_z(
            z, task_tokens, max_prompt_len=15, max_answer_len=5
        )

        # Extract answer for display
        answer_tokens = full_tokens[task_end:]
        answer_text = tokens_to_text(torch.cat([
            torch.tensor([3, 256047], device=dev),
            answer_tokens
        ]))
        answers.append(answer_text)

        # Check correctness
        if answer_text.lower().strip().startswith(target_text.lower()):
            n_correct += 1

        # Compute loss with gradients
        target_token = target_content[0]  # First token of target
        loss, _ = forward_with_mixed_z_and_loss(z, full_tokens, prompt_len, target_token)
        total_loss = total_loss + loss

    avg_loss = total_loss / len(examples)
    accuracy = n_correct / len(examples)

    # Track best
    if accuracy > best_acc:
        best_acc = accuracy
        best_z = z.detach().clone()

    # Add regularization loss
    reg_loss = reg_weight * ((z - z_init.detach()) ** 2).mean()
    total_opt_loss = avg_loss + reg_loss

    # Backprop
    if step > 0:
        total_opt_loss.backward()

        # Check gradients
        if z.grad is not None:
            grad_norm = z.grad.norm().item()
            if grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
                optimizer.step()

    display_prompt = prompt_text[:32] + "..." if len(prompt_text) > 35 else prompt_text
    display_answer = answers[0][:12] + "..." if len(answers) > 0 and len(answers[0]) > 15 else (answers[0] if answers else "N/A")
    print(f"{step:<6} {avg_loss.item():<10.4f} {accuracy:<6.2f} {display_prompt:<35} {display_answer:<15}")


# ============================================================================
# Final evaluation
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

# Use best z
z_eval = best_z

with torch.no_grad():
    e = z_eval.unsqueeze(0) if z_eval.dim() == 1 else z_eval
    eo = e.unsqueeze(1)
    final_tokens = [3, 256047]
    for _ in range(25):
        di = torch.tensor([final_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_token = logits.argmax().item()
        final_tokens.append(next_token)
        if next_token == 3:
            break
    prompt_text = tokens_to_text(torch.tensor(final_tokens, device=dev))

print(f"\nFinal prompt: '{prompt_text}'")

print("\nTask performance:")
for input_text, target in examples:
    task_tokens = text_to_tokens(input_text)
    full_tokens, prompt_len, task_end = decode_with_mixed_z(
        z, task_tokens, max_prompt_len=15, max_answer_len=8
    )

    answer_tokens = full_tokens[task_end:]
    answer_text = tokens_to_text(torch.cat([
        torch.tensor([3, 256047], device=dev),
        answer_tokens
    ]))

    correct = answer_text.lower().strip().startswith(target.lower())
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer_text[:25]}' (want: {target}) {mark}")
