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
    trajectory = []
    samples_per_accum = n_noise_samples // accum_steps

    for step in range(n_steps):
        optimizer.zero_grad()
        z_proj = project_to_norm(z, target_norm)

        # Compute perplexity loss and roundtrip error
        decoded_z = sonar_wrapper.decode(z_proj.squeeze(1))[0]
        z_tokens = tokenize_for_decoder(decoded_z, sonar_wrapper).unsqueeze(0)
        z_ppl_loss = decoder_ce_loss(z_proj, z_tokens, sonar_wrapper)
        ppl_weight = perplexity_weight * (step / (n_steps - 1)) if n_steps > 1 else perplexity_weight
        (ppl_weight * z_ppl_loss).backward(retain_graph=True)
        z_reencoded = sonar_wrapper.encode([decoded_z]).unsqueeze(1)
        roundtrip_l2 = (z_proj - z_reencoded).norm().item()
        roundtrip_cos = F.cosine_similarity(z_proj.view(-1), z_reencoded.view(-1), dim=0).item()

        # Accumulate gradients over multiple forward passes
        total_pred_loss = 0.0
        for accum_idx in range(accum_steps):
            z_batch = add_noise_with_projection(z_proj.expand(samples_per_accum, -1, -1), noise_level)

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

        optimizer.step()

        # Predict from unnoised z for logging
        with torch.no_grad():
            if context_emb is not None:
                seq_unnoised = torch.cat([z_proj, context_emb], dim=1)
            else:
                seq_unnoised = z_proj
            pred_emb = predict_next_embedding(seq_unnoised, generator)[:, -1:, :]

        # Cosine similarity for logging
        cos_sim = F.cosine_similarity(pred_emb.view(-1), target_emb.view(-1), dim=0).item()
        total_loss = total_pred_loss + ppl_weight * z_ppl_loss.item()


        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                decoded_pred = sonar_wrapper.decode(pred_emb.squeeze(1))[0]
                z_perplexity = torch.exp(z_ppl_loss).item()

                # Roundtrip prediction: z_reencoded -> SONAR-LLM -> decode
                if context_emb is not None:
                    rt_seq = torch.cat([z_reencoded, context_emb], dim=1)
                else:
                    rt_seq = z_reencoded
                rt_pred_emb = predict_next_embedding(rt_seq, generator)[:, -1:, :]
                decoded_rt_pred = sonar_wrapper.decode(rt_pred_emb.squeeze(1))[0]

            trajectory.append({
                "step": step,
                "loss": total_loss,
                "pred_loss": total_pred_loss,
                "z_ppl_loss": z_ppl_loss.item(),
                "similarity": cos_sim,
                "decoded_z": decoded_z,
                "decoded_pred": decoded_pred,
                "decoded_rt_pred": decoded_rt_pred,
                "z_perplexity": z_perplexity,
                "roundtrip_l2": roundtrip_l2,
                "roundtrip_cos": roundtrip_cos,
            })

            if verbose:
                print(f"Step {step:3d} | pred_loss={total_pred_loss:.3f} | z_ppl={z_perplexity:.1f} | sim={cos_sim:.3f} | rt_l2={roundtrip_l2:.3f} | rt_cos={roundtrip_cos:.3f}")
                print(f"    z decodes to:  \"{decoded_z}\"")
                print(f"    prediction:    \"{decoded_pred}\"")
                print(f"    rt prediction: \"{decoded_rt_pred}\"\n")

    # Final evaluation: decode z -> re-encode -> run SONAR-LLM -> decode prediction
    with torch.no_grad():
        z_final = project_to_norm(z, target_norm)
        decoded_z_final = sonar_wrapper.decode(z_final.squeeze(1))[0]
        z_reencoded = sonar_wrapper.encode([decoded_z_final]).unsqueeze(1)  # (1, 1, 1024)
        final_roundtrip_l2 = (z_final - z_reencoded).norm().item()
        final_roundtrip_cos = F.cosine_similarity(z_final.view(-1), z_reencoded.view(-1), dim=0).item()
        if context_emb is not None:
            seq_final = torch.cat([z_reencoded, context_emb], dim=1)
        else:
            seq_final = z_reencoded
        pred_final = predict_next_embedding(seq_final, generator)[:, -1:, :]
        decoded_pred_final = sonar_wrapper.decode(pred_final.squeeze(1))[0]

    if verbose:
        print("=" * 70)
        print("FINAL RESULT (z decoded -> re-encoded -> predict):")
        print(f"  z decodes to:  \"{decoded_z_final}\"")
        print(f"  roundtrip:     L2={final_roundtrip_l2:.3f}, cos={final_roundtrip_cos:.3f}")
        print(f"  RT prediction:    \"{decoded_pred_final}\"")
        print(f"  target:        \"{target_text}\"")
        print(f"  match: {decoded_pred_final.strip() == target_text.strip()}")
        print("=" * 70)

    return {
        "init_text": init_text,
        "context_sents": context_sents,
        "target_text": target_text,
        "final_z": decoded_z_final,
        "final_pred": decoded_pred_final,
        "final_loss": trajectory[-1]["loss"],
        "final_similarity": trajectory[-1]["similarity"],
        "final_roundtrip_l2": final_roundtrip_l2,
        "final_roundtrip_cos": final_roundtrip_cos,
        # "trajectory": trajectory,
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
    target_text="She went to the shop to buy eggs",
    # target_text="She asked her mom if she could have a new toy.",
    sonar_wrapper=sonar_wrapper,
    generator=generator,
    n_steps=20,
    lr=0.02,
    log_every=2,
    n_noise_samples=64,
    # noise_level=0.06,
    noise_level=0.09,
    perplexity_weight=0.0,
    accum_steps=10,
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

