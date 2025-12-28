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


def log_z_state(
    z: torch.Tensor,
    context_emb: torch.Tensor | None,
    target_emb: torch.Tensor,
    sonar_wrapper: SonarWrapper,
    generator: SonarLLMGenerator,
    label: str,
    verbose: bool,
) -> tuple[str, str, float]:
    """Decode z, predict from it, compute similarity. Returns (decoded_z, decoded_pred, cos_sim)."""
    with torch.no_grad():
        decoded_z: str = sonar_wrapper.decode(z.squeeze(1))[0]
        if context_emb is not None:
            seq = torch.cat([z, context_emb], dim=1)
        else:
            seq = z
        pred_emb = predict_next_embedding(seq, generator)[:, -1:, :]
        decoded_pred: str = sonar_wrapper.decode(pred_emb.squeeze(1))[0]
        cos_sim: float = F.cosine_similarity(pred_emb.view(-1), target_emb.view(-1), dim=0).item()

    if verbose:
        print(f"{label} | sim={cos_sim:.3f}")
        print(f"    z decodes to:  \"{decoded_z}\"")
        print(f"    prediction:    \"{decoded_pred}\"\n")

    return decoded_z, decoded_pred, cos_sim


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
    decoded_z, _, _ = log_z_state(z, context_emb, target_emb, sonar_wrapper, generator, "Init", verbose)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Compute perplexity loss (using decoded_z from previous roundtrip, or init)
        z_tokens = tokenize_for_decoder(decoded_z, sonar_wrapper).unsqueeze(0)
        z_ppl_loss = decoder_ce_loss(z, z_tokens, sonar_wrapper)
        ppl_weight = perplexity_weight * (step / (n_steps - 1)) if n_steps > 1 else perplexity_weight
        (ppl_weight * z_ppl_loss).backward(retain_graph=True)

        # Accumulate gradients over multiple forward passes
        total_pred_loss = 0.0
        for accum_idx in range(accum_steps):
            z_batch = add_noise_with_projection(z.expand(samples_per_accum, -1, -1), noise_level)

            # Concatenate with context to form full sequence
            if context_emb is not None:
                context_batch = context_emb.expand(samples_per_accum, -1, -1)
                seq_batch = torch.cat([z_batch, context_batch], dim=1)
            else:
                seq_batch = z_batch

            # Forward through SONAR-LLM
            pred_emb_batch = predict_next_embedding(seq_batch, generator)
            pred_emb_last = pred_emb_batch[:, -1:, :]

            # Decoder CE loss (scaled for accumulation)
            target_tokens_accum = target_tokens.expand(samples_per_accum, -1)
            pred_loss = decoder_ce_loss(pred_emb_last, target_tokens_accum, sonar_wrapper) / accum_steps
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
            decoded_after_enc, decoded_pred, cos_sim = log_z_state(
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
                "similarity": cos_sim,
                "decoded_z": decoded_after_enc,
                "decoded_pred": decoded_pred,
                "z_perplexity": z_perplexity,
            })

            if verbose:
                print(f"Step {step:3d} | pred_loss={total_pred_loss:.3f} | z_ppl={z_perplexity:.1f} | sim={cos_sim:.3f}")
                print(f"    after opt:     \"{decoded_after_opt}\"")
                print(f"    after proj:    \"{decoded_after_proj}\"")
                print(f"    after re-enc:  \"{decoded_after_enc}\"")
                print(f"    prediction:    \"{decoded_pred}\"\n")

    # Final evaluation
    if verbose:
        print("=" * 70)
        print("FINAL RESULT:")
    final_decoded_z, decoded_pred_final, final_sim = log_z_state(
        z, context_emb, target_emb, sonar_wrapper, generator, "Final", verbose
    )
    if verbose:
        print(f"  target:        \"{target_text}\"")
        print(f"  match: {decoded_pred_final.strip() == target_text.strip()}")
        print("=" * 70)

    return {
        "init_text": init_text,
        "context_sents": context_sents,
        "target_text": target_text,
        "final_z": final_decoded_z,
        "final_pred": decoded_pred_final,
        "final_loss": trajectory[-1]["loss"],
        "final_similarity": final_sim,
        "success": decoded_pred_final.strip() == target_text.strip(),
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
    # target_text="She went to the shop to buy eggs",
    target_text="She asked her mom if she could have a new toy.",
    sonar_wrapper=sonar_wrapper,
    generator=generator,
    n_steps=60,
    lr=0.02,
    log_every=2,
    n_noise_samples=64,
    # noise_level=0.06,
    noise_level=0.05,
    perplexity_weight=0.01,
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

