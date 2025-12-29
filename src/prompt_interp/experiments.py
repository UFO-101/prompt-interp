#%%
"""Experiments for finding embeddings that elicit specific next-sentence predictions."""

import torch
import torch.nn.functional as F

from prompt_interp.sonar_wrapper import SonarWrapper
from prompt_interp.generator import SonarLLMGenerator
from prompt_interp.optimize import (
    project_to_norm,
    predict_next_embedding,
    decoder_ce_loss,
    tokenize_for_decoder,
    LILY_STORY,
)


def add_noise_with_projection(z: torch.Tensor, noise_level: float) -> torch.Tensor:
    """Add Gaussian noise scaled by norm, then project back to original norm."""
    orig_norm = z.norm(dim=-1, keepdim=True)
    noise = torch.randn_like(z) * noise_level * orig_norm
    z_noisy = z + noise
    return z_noisy * (orig_norm / (z_noisy.norm(dim=-1, keepdim=True) + 1e-8))


def eval_two_pass(
    z: torch.Tensor,
    context_emb: torch.Tensor | None,
    target_emb: torch.Tensor,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
) -> tuple[str, str, str, float]:
    """Eval pass: z -> pred_z1 -> re-encode -> pred_z2. Returns (decoded_z, decoded_pred_z1, decoded_pred_z2, cos_sim)."""
    with torch.no_grad():
        decoded_z: str = sonar_wrapper.decode(z.squeeze(1))[0]

        # First pass: z -> SONAR-LLM -> pred_z1
        pred_z1 = predict_next_embedding(z, generator)[:, -1:, :]
        decoded_pred_z1: str = sonar_wrapper.decode(pred_z1.squeeze(1))[0]

        # Re-encode pred_z1 to get it on the sentence manifold
        pred_z1_reenc = sonar_wrapper.encode([decoded_pred_z1]).unsqueeze(1)

        # Second pass: [pred_z1_reenc, context] -> SONAR-LLM -> pred_z2
        if context_emb is not None:
            seq = torch.cat([pred_z1_reenc, context_emb], dim=1)
        else:
            seq = pred_z1_reenc
        pred_z2 = predict_next_embedding(seq, generator)[:, -1:, :]
        decoded_pred_z2: str = sonar_wrapper.decode(pred_z2.squeeze(1))[0]
        cos_sim: float = F.cosine_similarity(pred_z2.view(-1), target_emb.view(-1), dim=0).item()

    return decoded_z, decoded_pred_z1, decoded_pred_z2, cos_sim


def log_z_state(
    z: torch.Tensor,
    context_emb: torch.Tensor | None,
    target_emb: torch.Tensor,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    label: str,
    verbose: bool,
) -> tuple[str, str, str, str, str, float, float]:
    """Two-pass prediction (train + eval). Returns (decoded_z, decoded_pred_z1, decoded_pred_z1_reenc, train_pred_z2, eval_pred_z2, train_sim, eval_sim)."""
    with torch.no_grad():
        decoded_z: str = sonar_wrapper.decode(z.squeeze(1))[0]

        # First pass: z -> SONAR-LLM -> pred_z1
        pred_z1 = predict_next_embedding(z, generator)[:, -1:, :]
        decoded_pred_z1: str = sonar_wrapper.decode(pred_z1.squeeze(1))[0]

        # Train pass: [pred_z1, context] -> SONAR-LLM -> pred_z2 (no re-encoding)
        if context_emb is not None:
            seq_train = torch.cat([pred_z1, context_emb], dim=1)
        else:
            seq_train = pred_z1
        train_pred_z2 = predict_next_embedding(seq_train, generator)[:, -1:, :]
        decoded_train_pred_z2: str = sonar_wrapper.decode(train_pred_z2.squeeze(1))[0]
        train_sim: float = F.cosine_similarity(train_pred_z2.view(-1), target_emb.view(-1), dim=0).item()

        # Eval pass: re-encode pred_z1, then predict
        pred_z1_reenc = sonar_wrapper.encode([decoded_pred_z1]).unsqueeze(1)
        decoded_pred_z1_reenc: str = sonar_wrapper.decode(pred_z1_reenc.squeeze(1))[0]
        if context_emb is not None:
            seq_eval = torch.cat([pred_z1_reenc, context_emb], dim=1)
        else:
            seq_eval = pred_z1_reenc
        eval_pred_z2 = predict_next_embedding(seq_eval, generator)[:, -1:, :]
        decoded_eval_pred_z2: str = sonar_wrapper.decode(eval_pred_z2.squeeze(1))[0]
        eval_sim: float = F.cosine_similarity(eval_pred_z2.view(-1), target_emb.view(-1), dim=0).item()

    if verbose:
        print(f"{label} | train_sim={train_sim:.3f} | eval_sim={eval_sim:.3f}")
        print(f"    z decodes to:     \"{decoded_z}\"")
        print(f"    pred_z1:          \"{decoded_pred_z1}\"")
        print(f"    pred_z1 (reenc):  \"{decoded_pred_z1_reenc}\"")
        print(f"    train pred_z2:    \"{decoded_train_pred_z2}\"")
        print(f"    eval pred_z2:     \"{decoded_eval_pred_z2}\"\n")

    return decoded_z, decoded_pred_z1, decoded_pred_z1_reenc, decoded_train_pred_z2, decoded_eval_pred_z2, train_sim, eval_sim


def run_next_sentence_experiment(
    init_text: str,
    target_text: str,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    context_text: str | None = None,
    n_steps: int = 100,
    lr: float = 0.01,
    log_every: int = 10,
    verbose: bool = True,
    n_noise_samples: int = 7,
    noise_level: float = 0.03,
    perplexity_weight: float = 0.0,
    accum_steps: int = 1,
) -> dict:
    """
    Find z such that SONAR-LLM([z, context...]) predicts target_text at the final position.

    Sequence structure: [z (optimized), context_0, context_1, ...] -> SONAR-LLM -> predictions
    We optimize z so that the prediction at the LAST position decodes to target_text.
    """
    context_sents = sonar_wrapper.segment(context_text) if context_text else []

    # Encode embeddings
    init_emb = sonar_wrapper.encode([init_text]).unsqueeze(1)  # (1, 1, 1024)
    target_emb = sonar_wrapper.encode([target_text]).unsqueeze(1)  # (1, 1, 1024)
    target_tokens = tokenize_for_decoder(target_text, sonar_wrapper).unsqueeze(0)  # (1, seq_len)
    target_norm = init_emb.norm(dim=-1).mean().item()

    # Encode fixed context (frozen)
    if context_sents:
        context_emb = sonar_wrapper.encode(context_sents).unsqueeze(0)  # (1, n_context, 1024)
    else:
        context_emb = None
    seq_len = 1 + len(context_sents)

    # Print sequence structure
    if verbose:
        print("=" * 70)
        print("SEQUENCE STRUCTURE:")
        print(f"  [0] z (optimized) <- init: \"{init_text}\"")
        for i, sent in enumerate(context_sents):
            print(f"  [{i+1}] context (fixed): \"{sent}\"")
        print(f"  -> predict at position {seq_len - 1} -> target: \"{target_text}\"")
        print("=" * 70 + "\n")

    # Optimization
    z = init_emb.clone().requires_grad_(True)  # (1, 1, 1024)
    optimizer = torch.optim.Adam([z], lr=lr)
    trajectory: list[dict] = []
    samples_per_accum = n_noise_samples // accum_steps
    if samples_per_accum < 1:
        raise ValueError(f"n_noise_samples ({n_noise_samples}) must be >= accum_steps ({accum_steps})")

    # Log and initialize decoded_z (z is already a real sentence embedding with natural norm)
    decoded_z, _, _, _, _, _, _ = log_z_state(z, context_emb, target_emb, sonar_wrapper, generator, "Init", verbose)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Compute perplexity loss (using decoded_z from previous roundtrip, or init)
        z_tokens = tokenize_for_decoder(decoded_z, sonar_wrapper).unsqueeze(0)
        z_ppl_loss = decoder_ce_loss(z, z_tokens, sonar_wrapper)
        ppl_weight = perplexity_weight * (step / (n_steps - 1)) if n_steps > 1 else perplexity_weight
        (ppl_weight * z_ppl_loss).backward(retain_graph=True)

        # Pre-project context to hidden space (if any)
        if context_emb is not None:
            context_hidden = generator.forward_proj(context_emb)
        else:
            context_hidden = None

        # Accumulate gradients over multiple forward passes
        total_pred_loss = 0.0
        for accum_idx in range(accum_steps):
            # Add noise to z and carry through both passes
            z_noisy = add_noise_with_projection(z.expand(samples_per_accum, -1, -1), noise_level)

            # First pass: z_noisy -> forward_proj -> llama -> hidden1 (stay in hidden space)
            hidden_z = generator.forward_proj(z_noisy)
            out1 = generator.llama_model(inputs_embeds=hidden_z, output_hidden_states=True)
            hidden1 = out1.hidden_states[-1][:, -1:, :]  # Last position only

            # Second pass: [hidden1, context_hidden] -> llama -> reverse_proj -> pred_z2
            if context_hidden is not None:
                context_hidden_batch = context_hidden.expand(samples_per_accum, -1, -1)
                hidden_seq = torch.cat([hidden1, context_hidden_batch], dim=1)
            else:
                hidden_seq = hidden1
            out2 = generator.llama_model(inputs_embeds=hidden_seq, output_hidden_states=True)
            pred_z2 = generator.reverse_proj(out2.hidden_states[-1][:, -1:, :])

            # Decoder CE loss (scaled for accumulation)
            target_tokens_accum = target_tokens.expand(samples_per_accum, -1)
            pred_loss = decoder_ce_loss(pred_z2, target_tokens_accum, sonar_wrapper) / accum_steps
            pred_loss.backward(retain_graph=(accum_idx < accum_steps - 1))
            total_pred_loss += pred_loss.item()

        # Update z: project after gradient update, then roundtrip to stay on sentence manifold
        optimizer.step()
        should_log = step % log_every == 0 or step == n_steps - 1
        with torch.no_grad():
            # After optimizer.step()
            if should_log:
                decoded_after_opt: str = sonar_wrapper.decode(z.squeeze(1))[0]

            # After projection
            z.data = project_to_norm(z, target_norm).data
            if should_log:
                decoded_after_proj: str = sonar_wrapper.decode(z.squeeze(1))[0]

            # Roundtrip: decode then encode
            decoded_z = sonar_wrapper.decode(z.squeeze(1))[0]
            z.data = sonar_wrapper.encode([decoded_z]).unsqueeze(1)

        # Log state after update
        if should_log:
            decoded_after_enc, decoded_pred_z1, decoded_pred_z1_reenc, train_pred_z2, eval_pred_z2, train_sim, eval_sim = log_z_state(
                z, context_emb, target_emb, sonar_wrapper, generator,
                label=f"Step {step:3d} | pred_loss={total_pred_loss:.3f}",
                verbose=False,  # We'll print manually to include intermediate stages
            )
            z_perplexity: float = torch.exp(z_ppl_loss).item()
            total_loss = total_pred_loss + ppl_weight * z_ppl_loss.item()

            trajectory.append({
                "step": step,
                "loss": total_loss,
                "pred_loss": total_pred_loss,
                "z_ppl_loss": z_ppl_loss.item(),
                "train_similarity": train_sim,
                "eval_similarity": eval_sim,
                "decoded_z": decoded_after_enc,
                "decoded_pred_z1": decoded_pred_z1,
                "decoded_pred_z1_reenc": decoded_pred_z1_reenc,
                "train_pred_z2": train_pred_z2,
                "eval_pred_z2": eval_pred_z2,
                "z_perplexity": z_perplexity,
            })

            if verbose:
                print(f"Step {step:3d} | pred_loss={total_pred_loss:.3f} | z_ppl={z_perplexity:.1f} | train_sim={train_sim:.3f} | eval_sim={eval_sim:.3f}")
                print(f"    after opt:       \"{decoded_after_opt}\"")
                print(f"    after proj:      \"{decoded_after_proj}\"")
                print(f"    after re-enc:    \"{decoded_after_enc}\"")
                print(f"    pred_z1:         \"{decoded_pred_z1}\"")
                print(f"    pred_z1 (reenc): \"{decoded_pred_z1_reenc}\"")
                print(f"    train pred_z2:   \"{train_pred_z2}\"")
                print(f"    eval pred_z2:    \"{eval_pred_z2}\"\n")

    # Final evaluation
    if verbose:
        print("=" * 70)
        print("FINAL RESULT:")
    final_decoded_z, final_pred_z1, final_pred_z1_reenc, final_train_pred_z2, final_eval_pred_z2, final_train_sim, final_eval_sim = log_z_state(
        z, context_emb, target_emb, sonar_wrapper, generator, "Final", verbose
    )
    if verbose:
        print(f"  target:           \"{target_text}\"")
        print(f"  train match: {final_train_pred_z2.strip() == target_text.strip()}")
        print(f"  eval match:  {final_eval_pred_z2.strip() == target_text.strip()}")
        print("=" * 70)

    return {
        "init_text": init_text,
        "context_sents": context_sents,
        "target_text": target_text,
        "final_z": final_decoded_z,
        "final_pred_z1": final_pred_z1,
        "final_pred_z1_reenc": final_pred_z1_reenc,
        "final_train_pred_z2": final_train_pred_z2,
        "final_eval_pred_z2": final_eval_pred_z2,
        "final_loss": trajectory[-1]["loss"],
        "final_train_similarity": final_train_sim,
        "final_eval_similarity": final_eval_sim,
        "train_success": final_train_pred_z2.strip() == target_text.strip(),
        "eval_success": final_eval_pred_z2.strip() == target_text.strip(),
    }

#%%
sonar_wrapper = SonarWrapper()
for p in sonar_wrapper.decoder.model.parameters():
    p.requires_grad = False

generator = SonarLLMGenerator.from_pretrained("raxtemur/sonar-llm-900m")
for p in generator.parameters():
    p.requires_grad = False

#%%
run_next_sentence_experiment(
    init_text="I like cheese.",
    # context_text="She is going to the shop to buy eggs",
    target_text="She went to the shop to buy eggs",
    # target_text="She asked her mom if she could have a new toy.",
    sonar_wrapper=sonar_wrapper,
    generator=generator,
    n_steps=40,
    lr=0.1,
    log_every=2,
    n_noise_samples=64,
    noise_level=0.05,
    perplexity_weight=0.00,
    accum_steps=1,
    verbose=True,
)

#%%
def run_multi_init_experiments(
    target_sentences: list[str],
    init_sentences: list[str],
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    n_steps: int = 10,
    lr: float = 0.01,
    verbose: bool = True,
) -> list[dict]:
    """Run experiments for each (target, init) pair."""
    results = []
    for ti, target in enumerate(target_sentences):
        for ii, init in enumerate(init_sentences):
            if verbose:
                print("=" * 70)
                print(f"Target {ti}: {target}")
                print(f"Init {ii}: {init}")
                print("=" * 70)

            result = run_next_sentence_experiment(
                init, target, sonar_wrapper, generator, n_steps, lr, log_every=1, verbose=verbose
            )
            result.update({"target_idx": ti, "init_idx": ii})
            results.append(result)

            if verbose:
                print(f"SUCCESS: {result['success']}\n")
    return results

