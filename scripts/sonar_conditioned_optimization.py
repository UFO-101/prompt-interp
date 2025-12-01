"""
Simple conditioned SONAR optimization.
Use z for everything (no z=0), optimize z to solve antonym task.
This proves gradients flow and optimization works.
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
    return sonar_enc(text).to(dev)


def generate_prompt(z, max_len=15):
    """Generate prompt tokens from z."""
    with torch.no_grad():
        e = z.unsqueeze(1)  # [1, 1, 1024]
        tokens = [3, 256047]
        for _ in range(max_len):
            di = torch.tensor([tokens], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
            if h.dim() == 4:
                h = h.squeeze(1)
            logits = sdm.decoder.final_proj(h)[0, -1, :]
            next_tok = logits.argmax().item()
            tokens.append(next_tok)
            if next_tok == 3:
                break
        if tokens[-1] == 3:
            tokens = tokens[:-1]
    return tokens


def forward_and_loss(z, full_tokens, target_token_id):
    """Forward pass with gradients, return loss."""
    e = z.unsqueeze(1)  # [1, 1, 1024]
    input_tokens = torch.tensor([full_tokens[:-1]], device=dev)

    h = sdm.decode(input_tokens, BatchLayout.of(input_tokens), e, BatchLayout.of(e))
    if h.dim() == 4:
        h = h.squeeze(1)
    logits = sdm.decoder.final_proj(h)[0, -1, :]

    target = torch.tensor([target_token_id], device=dev, dtype=torch.long)
    loss = F.cross_entropy(logits.unsqueeze(0), target)

    pred = logits.argmax().item()
    return loss, pred


# ============================================================================
# Main
# ============================================================================
print("\n" + "=" * 80)
print("CONDITIONED SONAR OPTIMIZATION")
print("=" * 80)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

seed = "The opposite of hot is cold. The opposite of big is small."
print(f"\nSeed: '{seed}'")

with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.01)

print("\n" + "-" * 80)
print(f"{'Step':<6} {'Loss':<10} {'Acc':<6} {'Prompt':<50}")
print("-" * 80)

for step in range(51):
    optimizer.zero_grad()

    # Generate prompt
    prompt_tokens = generate_prompt(z)
    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    total_loss = 0.0
    n_correct = 0

    for input_text, target_text in examples:
        # Get target token
        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
        target_token_id = target_content[0].item()

        # Build full sequence
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full_tokens = prompt_tokens + task_content.tolist() + [target_token_id]

        # Forward and loss
        loss, pred = forward_and_loss(z, full_tokens, target_token_id)
        total_loss = total_loss + loss

        if pred == target_token_id:
            n_correct += 1

    avg_loss = total_loss / len(examples)
    accuracy = n_correct / len(examples)

    if step > 0:
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()

    if step % 5 == 0:
        display = prompt_text[:47] + "..." if len(prompt_text) > 50 else prompt_text
        print(f"{step:<6} {avg_loss.item():<10.4f} {accuracy:<6.2f} {display:<50}")

# ============================================================================
# Final Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

prompt_tokens = generate_prompt(z)
prompt_text = tokens_to_text(torch.tensor(prompt_tokens))
print(f"\nOptimized prompt: '{prompt_text}'")

print("\nTask performance (conditioned generation with z):")
for input_text, target in examples:
    with torch.no_grad():
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full = prompt_tokens + task_content.tolist()

        e = z.unsqueeze(1)
        for _ in range(5):
            di = torch.tensor([full], device=dev)
            h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
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

    correct = answer.lower().startswith(target.lower())
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:25]}' (want: {target}) {mark}")
