"""
Optimize SONAR z vector using unconditioned decoder for task loss - v4.

Use REINFORCE/policy gradient to get gradients through discrete token selection.

Approach:
1. Sample multiple prompt variations from z (using temperature)
2. For each prompt, evaluate task performance with z=0 decoder
3. Use REINFORCE to push z toward prompts that worked better

This avoids the gradient flow problem entirely by using reward-based learning.
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
sonar_embeds = sdm.decoder.decoder_frontend.embed.weight.data

# Zero z for unconditioned generation
z_zero = torch.zeros(1, 1, 1024, device=dev)


def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def text_to_tokens(text):
    tokens = sonar_enc(text)
    return tokens.to(dev)


def decode_conditioned_sample(z, max_len=20, temperature=1.0):
    """
    Decode z to tokens with sampling (for exploration).
    Returns tokens and log probabilities for REINFORCE.
    """
    e = z.unsqueeze(0) if z.dim() == 1 else z
    eo = e.unsqueeze(1)

    generated = [3, 256047]  # BOS, eng_Latn
    log_probs = []

    for _ in range(max_len):
        di = torch.tensor([generated], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), eo, BatchLayout.of(eo))
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        # Sample with temperature
        probs = F.softmax(logits / temperature, dim=0)
        dist = torch.distributions.Categorical(probs)
        next_token = dist.sample()

        log_probs.append(dist.log_prob(next_token))
        generated.append(next_token.item())

        if next_token.item() == 3:  # EOS
            break

    tokens = torch.tensor(generated, device=dev)
    total_log_prob = sum(log_probs) if log_probs else torch.tensor(0.0, device=dev)

    return tokens, total_log_prob


def decode_conditioned_greedy(z, max_len=30):
    """Greedy decode for evaluation."""
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


def evaluate_prompt_on_task(prompt_tokens, examples):
    """
    Evaluate a prompt on the antonym task using z=0 decoder.
    Returns reward (0 to 1 = accuracy) and list of results.
    """
    prompt_text = tokens_to_text(prompt_tokens)
    if not prompt_text:
        return 0.0, []

    n_correct = 0
    results = []

    with torch.no_grad():
        for input_text, target in examples:
            full_text = prompt_text + " " + input_text
            full_tokens = text_to_tokens(full_text)

            # Remove EOS for continuation
            full_input = full_tokens[:-1] if full_tokens[-1] == 3 else full_tokens
            full_input = full_input.unsqueeze(0)

            # Generate with z=0
            generated = full_tokens.tolist()
            for _ in range(8):
                di = torch.tensor([generated], device=dev)
                h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
                logits = sdm.decoder.final_proj(h)[0, -1, :]
                next_tok = logits.argmax().item()
                generated.append(next_tok)
                if next_tok == 3:
                    break

            gen_tokens = torch.tensor(generated, device=dev)
            gen_text = tokens_to_text(gen_tokens)

            # Extract answer
            if len(gen_text) > len(tokens_to_text(full_tokens)):
                answer = gen_text[len(tokens_to_text(full_tokens)):].strip()
            else:
                answer = ""

            correct = answer.lower().startswith(target.lower())
            if correct:
                n_correct += 1
            results.append((input_text, answer, target, correct))

    return n_correct / len(examples), results


# ============================================================================
# Main optimization loop using REINFORCE
# ============================================================================
print("\n" + "=" * 80)
print("SONAR UNCONDITIONED OPTIMIZATION v4 - REINFORCE")
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

# Baseline
print("\nBaseline with seed prompt:")
seed_tokens = text_to_tokens(seed)
seed_tokens = torch.cat([torch.tensor([3, 256047], device=dev), seed_tokens[2:]])
acc, results = evaluate_prompt_on_task(seed_tokens, examples)
print(f"  Accuracy: {acc:.2%}")
for inp, ans, tgt, cor in results:
    mark = "OK" if cor else "X"
    print(f"    {inp} -> '{ans[:20]}' (want: {tgt}) {mark}")

# Encode seed
with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.005)

print("\n" + "-" * 80)
print(f"{'Step':<6} {'AvgRwd':<8} {'MaxRwd':<8} {'BestAcc':<8} {'Best Prompt':<40}")
print("-" * 80)

n_steps = 100
n_samples = 8  # Number of prompt samples per step
temperature = 0.8
baseline_ema = 0.0  # EMA baseline for variance reduction

best_acc = 0.0
best_z = z_init.clone()
best_prompt = seed

for step in range(n_steps + 1):
    optimizer.zero_grad()

    rewards = []
    log_probs_list = []
    prompts = []

    # Sample multiple prompts
    for _ in range(n_samples):
        tokens, log_prob = decode_conditioned_sample(z, max_len=25, temperature=temperature)
        prompt_text = tokens_to_text(tokens)

        if not prompt_text or len(tokens) < 4:
            continue

        # Evaluate on task
        reward, _ = evaluate_prompt_on_task(tokens, examples)

        rewards.append(reward)
        log_probs_list.append(log_prob)
        prompts.append(prompt_text)

        # Track best
        if reward > best_acc:
            best_acc = reward
            best_z = z.detach().clone()
            best_prompt = prompt_text

    if len(rewards) == 0:
        print(f"{step:<6} {'N/A':<8} {'N/A':<8} {best_acc:<8.2f} {best_prompt[:40]:<40}")
        continue

    # Compute REINFORCE loss
    rewards_tensor = torch.tensor(rewards, device=dev)
    avg_reward = rewards_tensor.mean().item()
    max_reward = rewards_tensor.max().item()

    # Update baseline (EMA)
    baseline_ema = 0.9 * baseline_ema + 0.1 * avg_reward

    # REINFORCE with baseline
    if step > 0:
        loss = torch.tensor(0.0, device=dev)
        for log_prob, reward in zip(log_probs_list, rewards):
            advantage = reward - baseline_ema
            loss = loss - log_prob * advantage  # Negative because we maximize reward

        loss = loss / len(rewards)

        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optimizer.step()

    display_prompt = best_prompt[:37] + "..." if len(best_prompt) > 40 else best_prompt
    print(f"{step:<6} {avg_reward:<8.3f} {max_reward:<8.3f} {best_acc:<8.2f} {display_prompt:<40}")


# ============================================================================
# Final evaluation
# ============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

z_final = best_z

with torch.no_grad():
    tokens = decode_conditioned_greedy(z_final)
    decoded_text = tokens_to_text(tokens)

print(f"\nBest prompt (acc={best_acc:.2%}): '{decoded_text}'")

print("\nTask performance with best prompt:")
acc, results = evaluate_prompt_on_task(tokens, examples)
for inp, ans, tgt, cor in results:
    mark = "OK" if cor else "X"
    print(f"  {inp} -> '{ans[:30]}' (want: {tgt}) {mark}")

# Also try the original seed for comparison
print("\nFor comparison, seed prompt performance:")
seed_tokens = text_to_tokens(seed)
seed_tokens = torch.cat([torch.tensor([3, 256047], device=dev), seed_tokens[2:]])
acc, results = evaluate_prompt_on_task(seed_tokens, examples)
for inp, ans, tgt, cor in results:
    mark = "OK" if cor else "X"
    print(f"  {inp} -> '{ans[:30]}' (want: {tgt}) {mark}")
