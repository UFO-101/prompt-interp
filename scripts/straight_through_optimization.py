"""
Straight-through estimator for learning text prompts.

Two-stage approach:
1. Stage 1: z → decoder(z) → logits → argmax → tokens (with gradient bridge)
2. Stage 2: tokens → embed → decoder(z=0) → task loss

Gradient bridge: Use embedding geometry to convert ∂L/∂embedding to ∂L/∂logits
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

# Get the embedding matrix
embed_layer = sdm.decoder.decoder_frontend.embed
embed_matrix = embed_layer.weight.detach()  # [vocab_size, dim]
vocab_size, embed_dim = embed_matrix.shape
print(f"Embedding matrix: {embed_matrix.shape}")

z_zero = torch.zeros(1, 1, 1024, device=dev)


def tokens_to_text(tokens):
    content = tokens[2:-1] if tokens[-1] == 3 else tokens[2:]
    if len(content) == 0:
        return ""
    return sonar_dec(content.cpu())


def text_to_tokens(text):
    return sonar_enc(text).to(dev)


def generate_prompt_with_grads(z, max_len=12):
    """
    Stage 1: Generate prompt tokens using z, keeping track of logits for gradient bridge.

    Returns:
        tokens: list of token ids
        all_logits: list of logit tensors (one per generated token)
    """
    tokens = [3, 256047]  # BOS, lang
    all_logits = []

    e = z.unsqueeze(1)

    for _ in range(max_len):
        di = torch.tensor([tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]  # [vocab_size]

        # Store logits for gradient bridge
        all_logits.append(logits)

        # Discrete selection
        next_tok = logits.argmax().item()
        tokens.append(next_tok)

        if next_tok == 3:  # EOS
            break

    # Remove EOS if present
    if tokens[-1] == 3:
        tokens = tokens[:-1]
        all_logits = all_logits[:-1]

    return tokens, all_logits


def forward_stage2_with_embedding_grads(prompt_tokens, task_tokens, target_id):
    """
    Stage 2: Forward pass with z=0, computing loss and getting gradients w.r.t. embeddings.

    Returns:
        loss: scalar loss
        grad_embeddings: gradients w.r.t. prompt token embeddings [prompt_len, dim]
        correct: whether prediction is correct
    """
    # Get task content tokens
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]

    # Full sequence: prompt + task + target
    full_tokens = prompt_tokens + task_content.tolist() + [target_id]
    input_tokens = full_tokens[:-1]

    # Create embedding tensor with gradients for prompt positions
    prompt_len = len(prompt_tokens)
    input_len = len(input_tokens)

    # Get embeddings - we need gradients for prompt positions
    all_token_ids = torch.tensor(input_tokens, device=dev)

    # Embed all tokens
    with torch.no_grad():
        all_embeddings = sdm.decoder.embed(all_token_ids)  # [input_len, dim]

    # Make prompt embeddings require grad
    prompt_embeddings = all_embeddings[:prompt_len].clone().requires_grad_(True)
    task_embeddings = all_embeddings[prompt_len:].clone()

    # Concatenate
    input_embeddings = torch.cat([prompt_embeddings, task_embeddings], dim=0)  # [input_len, dim]
    input_embeddings = input_embeddings.unsqueeze(0)  # [1, input_len, dim]

    # Forward through decoder with z=0
    # We need to bypass the embedding layer and feed embeddings directly
    # This requires accessing the decoder internals

    # Actually, let's use a different approach:
    # We'll compute the forward pass and use autograd to get gradients

    # For now, let's use the standard forward and compute an approximate gradient
    di = torch.tensor([input_tokens], device=dev)
    h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
    if h.dim() == 4:
        h = h.squeeze(1)
    logits = sdm.decoder.final_proj(h)[0, -1, :]  # [vocab_size]

    # Loss
    target = torch.tensor([target_id], device=dev, dtype=torch.long)
    loss = F.cross_entropy(logits.unsqueeze(0), target)

    # Prediction
    pred = logits.argmax().item()
    correct = (pred == target_id)

    # Now we need gradient w.r.t. embeddings
    # The issue is that we used the standard forward which goes through embed layer
    #
    # Let's compute gradient using the embedding matrix as a proxy:
    # ∂L/∂embed[i] ≈ contribution of position i to the loss
    #
    # Actually, we need to do this properly. Let me restructure.

    return loss, pred, correct, logits


def compute_embedding_gradient(prompt_tokens, task_tokens, target_id):
    """
    Compute gradient of loss w.r.t. prompt token embeddings.

    We need to:
    1. Get the embeddings for all tokens
    2. Forward through decoder with z=0
    3. Compute loss
    4. Backprop to get gradients w.r.t. prompt embeddings
    """
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full_tokens = prompt_tokens + task_content.tolist() + [target_id]
    input_tokens = full_tokens[:-1]

    prompt_len = len(prompt_tokens)

    # Get embeddings with gradient tracking for prompt positions
    token_ids = torch.tensor(input_tokens, device=dev)

    # Get the raw embeddings
    raw_embeds = sdm.decoder.embed(token_ids)  # [seq_len, embed_dim]

    # Create a copy where prompt embeddings require grad
    prompt_embeds = raw_embeds[:prompt_len].detach().clone().requires_grad_(True)
    task_embeds = raw_embeds[prompt_len:].detach()

    # We need to forward through the decoder using these embeddings directly
    # The decoder's forward typically does: embed -> layers -> final_proj
    # We need to skip the embed step

    # Concatenate embeddings
    all_embeds = torch.cat([prompt_embeds, task_embeds], dim=0)  # [seq_len, embed_dim]
    all_embeds = all_embeds.unsqueeze(0)  # [1, seq_len, embed_dim]

    # Now we need to forward through the decoder layers
    # Looking at SONAR decoder structure...
    # The decoder has: embed, layers (transformer layers), final_proj

    # Let's check if we can access the internal forward
    decoder = sdm.decoder

    # Add positional embeddings if needed
    # SONAR uses learned positional embeddings
    seq_len = all_embeds.shape[1]
    positions = torch.arange(seq_len, device=dev)

    # Check if there's a position embedding
    if hasattr(decoder, 'pos_embed'):
        pos_embeds = decoder.pos_embed(positions)
        all_embeds = all_embeds + pos_embeds.unsqueeze(0)

    # Forward through transformer layers
    h = all_embeds

    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=dev), diagonal=1).bool()

    # We need to pass through the decoder properly
    # This is getting complex - let me try a simpler approach using hooks

    # Actually, let's use a gradient computation trick:
    # Run the standard forward, but hook into the embedding layer to capture gradients

    return prompt_embeds, None  # Placeholder - need to implement properly


class EmbeddingGradientCapture:
    """Hook to capture gradients w.r.t. embeddings."""

    def __init__(self, embed_layer):
        self.embed_layer = embed_layer
        self.saved_input = None
        self.saved_output = None
        self.grad_output = None

    def forward_hook(self, module, input, output):
        self.saved_input = input
        self.saved_output = output.clone()
        output.retain_grad()
        return output

    def get_grad(self):
        if self.saved_output is not None and self.saved_output.grad is not None:
            return self.saved_output.grad
        return None


def train_step(z, examples):
    """
    One training step:
    1. Generate prompt with z (Stage 1)
    2. For each example, compute loss with z=0 (Stage 2)
    3. Get embedding gradients and bridge to logits gradients
    4. Backprop to z
    """
    # Stage 1: Generate prompt
    prompt_tokens, stage1_logits = generate_prompt_with_grads(z, max_len=12)
    prompt_len = len(prompt_tokens) - 2  # Exclude BOS and lang tokens

    prompt_text = tokens_to_text(torch.tensor(prompt_tokens))

    total_loss = 0.0
    n_correct = 0
    all_logit_grads = []  # Gradients to backprop to Stage 1 logits

    for input_text, target_text in examples:
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]

        target_tokens = text_to_tokens(target_text)
        target_content = target_tokens[2:-1] if target_tokens[-1] == 3 else target_tokens[2:]
        target_id = target_content[0].item()

        # Stage 2: Forward with z=0
        full_tokens = prompt_tokens + task_content.tolist() + [target_id]
        input_tokens = full_tokens[:-1]

        # Set up gradient capture on embedding layer
        embed_hook_layer = sdm.decoder.decoder_frontend.embed

        # Forward pass with gradient tracking
        di = torch.tensor([input_tokens], device=dev)

        # We need to compute gradients w.r.t. the embeddings
        # Use a hook to capture the embedding output and its gradient
        embedding_output = []

        def embed_hook(module, input, output):
            output.retain_grad()
            embedding_output.append(output)
            return output

        handle = embed_hook_layer.register_forward_hook(embed_hook)

        # Forward
        h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]

        handle.remove()

        # Loss
        target = torch.tensor([target_id], device=dev, dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target)
        total_loss = total_loss + loss.item()

        if logits.argmax().item() == target_id:
            n_correct += 1

        # Backward to get embedding gradients
        loss.backward(retain_graph=False)

        # Get embedding gradients for prompt positions
        if len(embedding_output) > 0 and embedding_output[0].grad is not None:
            embed_grad = embedding_output[0].grad[0]  # [seq_len, dim]
            prompt_embed_grad = embed_grad[:len(prompt_tokens)]  # [prompt_len, dim]

            # Convert to logit gradients using embedding matrix
            # ∂L/∂logits[j] = ⟨∂L/∂embed, E[j]⟩ for each token j
            # We do this for each prompt position
            logit_grads = prompt_embed_grad @ embed_matrix.T  # [prompt_len, vocab_size]

            all_logit_grads.append(logit_grads)

        # Clear gradients
        sdm.zero_grad()

    # Average the logit gradients across examples
    if len(all_logit_grads) > 0:
        avg_logit_grad = torch.stack(all_logit_grads).mean(dim=0)  # [prompt_len, vocab_size]

        # Now backprop through Stage 1
        # We have logits from Stage 1 (stage1_logits) - one per generated token
        # We have gradients for those logits (avg_logit_grad)
        #
        # The stage1_logits are for tokens at positions 2, 3, 4, ... (after BOS, lang)
        # We need to match them up

        # stage1_logits has len = prompt_len - 2 (we generate starting from position 2)
        # avg_logit_grad has len = prompt_len (includes BOS, lang)
        # So we need gradients for positions 2 onwards

        if len(stage1_logits) > 0 and avg_logit_grad.shape[0] > 2:
            grad_for_stage1 = avg_logit_grad[2:2+len(stage1_logits)]  # [num_generated, vocab_size]

            # Backprop through Stage 1
            # Each stage1_logits[i] was used to select token prompt_tokens[2+i]
            # We need to compute: sum_i (stage1_logits[i] @ grad_for_stage1[i])

            total_surrogate_loss = 0.0
            for i, (logits_i, grad_i) in enumerate(zip(stage1_logits, grad_for_stage1)):
                # Surrogate loss: logits @ grad gives a scalar that, when backpropped,
                # gives gradient grad to the logits
                surrogate = (logits_i * grad_i).sum()
                total_surrogate_loss = total_surrogate_loss + surrogate

            # Now backprop the surrogate loss to z
            total_surrogate_loss.backward()

    return total_loss / len(examples), n_correct / len(examples), prompt_text


def evaluate_with_z0(prompt_tokens, examples):
    """Evaluate using only the prompt text with z=0."""
    n_correct = 0
    results = []

    for input_text, target_text in examples:
        task_tokens = text_to_tokens(input_text)
        task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
        full = prompt_tokens + task_content.tolist()

        with torch.no_grad():
            for _ in range(5):
                di = torch.tensor([full], device=dev)
                h = sdm.decode(di, BatchLayout.of(di), z_zero, BatchLayout.of(z_zero))
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

        correct = answer.lower().startswith(target_text.lower())
        if correct:
            n_correct += 1
        results.append((input_text, answer[:20], target_text, correct))

    return n_correct / len(examples), results


# ============================================================================
# Main optimization loop
# ============================================================================
print("\n" + "=" * 80)
print("STRAIGHT-THROUGH OPTIMIZATION")
print("=" * 80)

examples = [
    ("hot -> ", "cold"),
    ("big -> ", "small"),
    ("fast -> ", "slow"),
    ("up -> ", "down"),
    ("happy -> ", "sad"),
    ("light -> ", "dark"),
]

seed = "The opposite of hot is cold."
print(f"Seed: '{seed}'")

with torch.no_grad():
    z_init = se.predict([seed], source_lang='eng_Latn').to(dev)

z = nn.Parameter(z_init.clone())
optimizer = torch.optim.Adam([z], lr=0.01)

print("\nOptimizing...")
for step in range(31):
    optimizer.zero_grad()

    loss, acc, prompt = train_step(z, examples)

    # Check if gradients flowed to z
    if z.grad is not None:
        grad_norm = z.grad.norm().item()
        optimizer.step()
    else:
        grad_norm = 0.0

    if step % 5 == 0:
        print(f"Step {step}: loss={loss:.4f}, acc={acc:.0%}, grad_norm={grad_norm:.4f}")
        print(f"  Prompt: '{prompt[:50]}...'")

# Final evaluation with z=0
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

# Generate final prompt
with torch.no_grad():
    final_tokens = [3, 256047]
    e = z.unsqueeze(1)
    for _ in range(12):
        di = torch.tensor([final_tokens], device=dev)
        h = sdm.decode(di, BatchLayout.of(di), e, BatchLayout.of(e))
        if h.dim() == 4:
            h = h.squeeze(1)
        logits = sdm.decoder.final_proj(h)[0, -1, :]
        next_tok = logits.argmax().item()
        final_tokens.append(next_tok)
        if next_tok == 3:
            break
    if final_tokens[-1] == 3:
        final_tokens = final_tokens[:-1]

final_prompt = tokens_to_text(torch.tensor(final_tokens))
print(f"\nFinal prompt: '{final_prompt}'")

print("\nEvaluation with z=0 (prompt TEXT only):")
acc, results = evaluate_with_z0(final_tokens, examples)
for inp, ans, tgt, cor in results:
    mark = "OK" if cor else "X"
    print(f"  {inp} -> '{ans}' (want: {tgt}) {mark}")
print(f"Accuracy: {acc:.0%}")

# Compare with z
print("\nEvaluation with z (soft prompt):")
n_correct = 0
for input_text, target_text in examples:
    task_tokens = text_to_tokens(input_text)
    task_content = task_tokens[2:-1] if task_tokens[-1] == 3 else task_tokens[2:]
    full = final_tokens + task_content.tolist()

    with torch.no_grad():
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
    base_text = tokens_to_text(torch.tensor(final_tokens + task_content.tolist()))
    answer = full_text[len(base_text):].strip()

    correct = answer.lower().startswith(target_text.lower())
    if correct:
        n_correct += 1
    mark = "OK" if correct else "X"
    print(f"  {input_text} -> '{answer[:20]}' (want: {target_text}) {mark}")
print(f"Accuracy: {n_correct/len(examples):.0%}")
