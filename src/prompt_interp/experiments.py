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
    n_steps: int = 100,
    lr: float = 0.01,
    log_every: int = 10,
    verbose: bool = True,
    n_noise_samples: int = 7,
    noise_level: float = 0.03,
) -> dict:
    """
    Find z such that SONAR-LLM(z) decodes to target_text.

    Optimizes z using decoder cross-entropy loss so that the model's
    predicted next-sentence embedding decodes to the target sentence.
    Uses gradient averaging over noised copies of z for robustness.
    """
    # Encode init and target: (1, 1, 1024) for (batch, seq, embed_dim)
    init_emb = sonar_wrapper.encode([init_text]).unsqueeze(1)  # (1, 1, 1024)
    target_emb = sonar_wrapper.encode([target_text]).unsqueeze(1)  # (1, 1, 1024)
    target_tokens = tokenize_for_decoder(target_text, sonar_wrapper).unsqueeze(0)  # (1, seq_len)
    target_norm = init_emb.norm(dim=-1).mean().item()

    # Optimization
    z = init_emb.clone().requires_grad_(True)  # (1, 1, 1024)
    optimizer = torch.optim.Adam([z], lr=lr)
    trajectory = []
    batch_size = 1 + n_noise_samples
    target_tokens_batch = target_tokens.expand(batch_size, -1)  # (8, seq_len)

    for step in range(n_steps):
        optimizer.zero_grad()
        z_proj = project_to_norm(z, target_norm)

        # Create batch: original + n_noise_samples noised copies
        z_expanded = z_proj.expand(n_noise_samples, -1, -1)  # (7, 1, 1024)
        z_noisy = add_noise_with_projection(z_expanded, noise_level)
        z_batch = torch.cat([z_proj, z_noisy], dim=0)  # (8, 1, 1024)

        # Forward: z_batch -> SONAR-LLM -> pred_emb_batch
        pred_emb_batch = predict_next_embedding(z_batch, generator)  # (8, 1, 1024)

        # Decoder CE loss averaged over batch
        loss = decoder_ce_loss(pred_emb_batch, target_tokens_batch, sonar_wrapper)

        # Cosine similarity for logging (use original z, not noised)
        pred_emb = pred_emb_batch[0:1]  # (1, 1, 1024)
        cos_sim = F.cosine_similarity(pred_emb.view(-1), target_emb.view(-1), dim=0).item()

        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                decoded_z = sonar_wrapper.decode(project_to_norm(z, target_norm).squeeze(1))[0]
                decoded_pred = sonar_wrapper.decode(pred_emb.squeeze(1))[0]

            trajectory.append({
                "step": step,
                "loss": loss.item(),
                "similarity": cos_sim,
                "decoded_z": decoded_z,
                "decoded_pred": decoded_pred,
            })

            if verbose:
                print(f"Step {step:2d} | loss={loss.item():.3f} | sim={cos_sim:.3f}")
                print(f"  z: {decoded_z[:70]}")
                print(f"  pred: {decoded_pred[:70]}\n")

    # Final evaluation
    with torch.no_grad():
        z_final = project_to_norm(z, target_norm)
        pred_final = predict_next_embedding(z_final, generator)
        decoded_z_final = sonar_wrapper.decode(z_final.squeeze(1))[0]
        decoded_pred_final = sonar_wrapper.decode(pred_final.squeeze(1))[0]

    return {
        "init_text": init_text,
        "target_text": target_text,
        "final_z": decoded_z_final,
        "final_pred": decoded_pred_final,
        "final_loss": trajectory[-1]["loss"],
        "final_similarity": trajectory[-1]["similarity"],
        "trajectory": trajectory,
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
    # target_text="Then she went to the shop to buy eggs",
    target_text="She asked her mom if she could have a new toy.",
    sonar_wrapper=sonar_wrapper,
    generator=generator,
    n_steps=20,
    lr=0.01,
    log_every=2,
    n_noise_samples=63,
    # noise_level=0.06,
    noise_level=0.09,

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

