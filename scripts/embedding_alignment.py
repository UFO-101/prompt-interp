"""
Embedding Alignment: NLLB/SONAR → Qwen using Procrustes Analysis

This script learns a mapping between NLLB token embeddings (same tokenizer as SONAR)
and Qwen token embeddings using shared vocabulary tokens as anchor points.

We use Full Procrustes Analysis with:
- Orthogonal transformation (rotation/reflection)
- Uniform scaling
- Translation (centering)

The goal is to enable mapping SONAR decoder outputs to Qwen's embedding space.
"""

import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM


@dataclass
class AlignmentResult:
    """Result of Procrustes alignment."""
    W: np.ndarray          # Full transformation matrix (includes scale)
    R: np.ndarray          # Orthogonal component
    scale: float           # Scale factor
    src_mean: np.ndarray   # Source centroid
    tgt_mean: np.ndarray   # Target centroid
    train_mse: float
    test_mse: float
    test_cos_sim: float


def normalize_token(tok: str) -> str:
    """Remove word boundary markers for cross-tokenizer matching."""
    return tok.replace("Ġ", "").replace("▁", "").replace("##", "")


def find_shared_vocabulary(qwen_tokenizer, nllb_tokenizer, min_len=2, max_len=20, ascii_only=True):
    """Find tokens that exist in both vocabularies."""
    qwen_vocab = qwen_tokenizer.get_vocab()
    nllb_vocab = nllb_tokenizer.get_vocab()

    # Build normalized lookup tables
    qwen_normalized = {}
    for tok, idx in qwen_vocab.items():
        norm = normalize_token(tok)
        if norm and len(norm) >= 1:
            if norm not in qwen_normalized:
                qwen_normalized[norm] = (tok, idx)

    nllb_normalized = {}
    for tok, idx in nllb_vocab.items():
        norm = normalize_token(tok)
        if norm and len(norm) >= 1:
            if norm not in nllb_normalized:
                nllb_normalized[norm] = (tok, idx)

    # Find intersection
    shared = set(qwen_normalized.keys()) & set(nllb_normalized.keys())

    # Filter by quality criteria
    pairs = []
    for norm_tok in shared:
        if ascii_only and not norm_tok.isascii():
            continue
        if not (min_len <= len(norm_tok) <= max_len):
            continue

        qwen_tok, qwen_idx = qwen_normalized[norm_tok]
        nllb_tok, nllb_idx = nllb_normalized[norm_tok]
        pairs.append({
            'norm': norm_tok,
            'qwen_tok': qwen_tok,
            'qwen_idx': qwen_idx,
            'nllb_tok': nllb_tok,
            'nllb_idx': nllb_idx
        })

    return pairs


def procrustes_with_scaling(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Full Procrustes analysis: find R (orthogonal) and s (scale) such that
    ||s * A @ R - B||^2 is minimized.

    Args:
        A: Source matrix (n_samples, d_src) - will be projected if d_src != d_tgt
        B: Target matrix (n_samples, d_tgt)

    Returns:
        R: Orthogonal transformation matrix
        scale: Optimal scaling factor
        W: Combined transformation (scale * R), potentially with projection
    """
    n = A.shape[0]
    d_src, d_tgt = A.shape[1], B.shape[1]

    if d_src != d_tgt:
        # Dimensions differ - use least squares to find best linear map
        # Then extract orthogonal component via polar decomposition
        print(f"  Dimension mismatch ({d_src} → {d_tgt}), using linear projection + Procrustes")

        # First, find best linear map: W_linear = argmin ||A @ W - B||
        W_linear, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        # Project A to target dimension
        A_proj = A @ W_linear

        # Now do Procrustes on same-dimensional spaces
        # SVD of B^T @ A_proj
        M = B.T @ A_proj
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt

        # Optimal scale: s = trace(B^T @ A_proj @ R) / trace(A_proj^T @ A_proj)
        scale = np.trace(B.T @ A_proj @ R) / np.trace(A_proj.T @ A_proj)

        # Combined transformation
        W = scale * W_linear @ R

        return R, scale, W

    else:
        # Same dimensions - standard Procrustes
        # SVD of B^T @ A
        M = B.T @ A
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt

        # Optimal scale
        scale = np.trace(B.T @ A @ R) / np.trace(A.T @ A)

        W = scale * R
        return R, scale, W


def align_embeddings(
    nllb_embed: torch.Tensor,
    qwen_embed: torch.Tensor,
    pairs: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> AlignmentResult:
    """
    Learn alignment from NLLB to Qwen embedding space.

    Args:
        nllb_embed: NLLB embedding matrix [vocab_size, nllb_dim]
        qwen_embed: Qwen embedding matrix [vocab_size, qwen_dim]
        pairs: List of shared vocabulary pairs
        train_ratio: Fraction of pairs to use for training
        seed: Random seed for train/test split

    Returns:
        AlignmentResult with transformation parameters and metrics
    """
    np.random.seed(seed)

    # Extract embeddings for shared tokens
    nllb_indices = torch.tensor([p['nllb_idx'] for p in pairs])
    qwen_indices = torch.tensor([p['qwen_idx'] for p in pairs])

    A = nllb_embed[nllb_indices].numpy()  # Source
    B = qwen_embed[qwen_indices].numpy()  # Target

    # Train/test split
    n = len(pairs)
    n_train = int(train_ratio * n)
    indices = np.random.permutation(n)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    A_train, A_test = A[train_idx], A[test_idx]
    B_train, B_test = B[train_idx], B[test_idx]

    # Center the data
    A_mean = A_train.mean(axis=0)
    B_mean = B_train.mean(axis=0)

    A_train_centered = A_train - A_mean
    A_test_centered = A_test - A_mean
    B_train_centered = B_train - B_mean
    B_test_centered = B_test - B_mean

    # Procrustes with scaling
    R, scale, W = procrustes_with_scaling(A_train_centered, B_train_centered)

    # Evaluate
    B_train_pred = A_train_centered @ W + B_mean
    B_test_pred = A_test_centered @ W + B_mean

    train_mse = np.mean((B_train_pred - B_train) ** 2)
    test_mse = np.mean((B_test_pred - B_test) ** 2)

    # Cosine similarity on test set
    cos_sims = np.sum(B_test_pred * B_test, axis=1) / (
        np.linalg.norm(B_test_pred, axis=1) * np.linalg.norm(B_test, axis=1) + 1e-8
    )
    test_cos_sim = np.mean(cos_sims)

    return AlignmentResult(
        W=W,
        R=R,
        scale=scale,
        src_mean=A_mean,
        tgt_mean=B_mean,
        train_mse=train_mse,
        test_mse=test_mse,
        test_cos_sim=test_cos_sim
    )


class EmbeddingAligner:
    """Applies learned alignment to map NLLB embeddings to Qwen space."""

    def __init__(self, result: AlignmentResult):
        self.W = torch.tensor(result.W, dtype=torch.float32)
        self.src_mean = torch.tensor(result.src_mean, dtype=torch.float32)
        self.tgt_mean = torch.tensor(result.tgt_mean, dtype=torch.float32)

    def transform(self, nllb_embeddings: torch.Tensor) -> torch.Tensor:
        """Transform NLLB embeddings to Qwen space."""
        centered = nllb_embeddings - self.src_mean
        return centered @ self.W + self.tgt_mean

    def to(self, device):
        """Move aligner to device."""
        self.W = self.W.to(device)
        self.src_mean = self.src_mean.to(device)
        self.tgt_mean = self.tgt_mean.to(device)
        return self


def main():
    print("=" * 70)
    print("EMBEDDING ALIGNMENT: NLLB/SONAR → Qwen (Full Procrustes + Scaling)")
    print("=" * 70)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Load models
    print("\n1. Loading Qwen model...")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    qwen_model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    qwen_embed = qwen_model.embed_tokens.weight.detach()
    print(f"   Shape: {qwen_embed.shape}")

    print("\n2. Loading NLLB model...")
    nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    nllb_embed = nllb_model.model.shared.weight.detach()
    print(f"   Shape: {nllb_embed.shape}")

    # Find shared vocabulary
    print("\n3. Finding shared vocabulary...")
    pairs = find_shared_vocabulary(qwen_tokenizer, nllb_tokenizer)
    print(f"   Found {len(pairs)} shared tokens")

    # Show samples
    print("\n   Sample pairs:")
    import random
    random.seed(42)
    for p in random.sample(pairs, min(8, len(pairs))):
        print(f"      '{p['norm']}': NLLB[{p['nllb_idx']}] → Qwen[{p['qwen_idx']}]")

    # Align embeddings
    print("\n4. Learning alignment (Full Procrustes with scaling)...")
    result = align_embeddings(nllb_embed, qwen_embed, pairs)

    print(f"\n5. Results:")
    print(f"   Scale factor: {result.scale:.4f}")
    print(f"   Train MSE: {result.train_mse:.6f}")
    print(f"   Test MSE: {result.test_mse:.6f}")
    print(f"   Test cosine similarity: {result.test_cos_sim:.4f}")

    # Detailed cosine sim analysis
    print("\n6. Evaluating on test set in detail...")
    np.random.seed(42)
    n = len(pairs)
    n_train = int(0.8 * n)
    indices = np.random.permutation(n)
    test_idx = indices[n_train:]

    nllb_indices = torch.tensor([pairs[i]['nllb_idx'] for i in test_idx])
    qwen_indices = torch.tensor([pairs[i]['qwen_idx'] for i in test_idx])

    A_test = nllb_embed[nllb_indices].numpy()
    B_test = qwen_embed[qwen_indices].numpy()

    A_test_centered = A_test - result.src_mean
    B_test_pred = A_test_centered @ result.W + result.tgt_mean

    cos_sims = np.sum(B_test_pred * B_test, axis=1) / (
        np.linalg.norm(B_test_pred, axis=1) * np.linalg.norm(B_test, axis=1) + 1e-8
    )

    print(f"   Cosine sim > 0.9: {100*np.mean(cos_sims > 0.9):.1f}%")
    print(f"   Cosine sim > 0.8: {100*np.mean(cos_sims > 0.8):.1f}%")
    print(f"   Cosine sim > 0.5: {100*np.mean(cos_sims > 0.5):.1f}%")
    print(f"   Cosine sim > 0.0: {100*np.mean(cos_sims > 0.0):.1f}%")

    # Save results
    print("\n7. Saving alignment...")
    np.savez(
        results_dir / "nllb_qwen_alignment.npz",
        W=result.W,
        R=result.R,
        scale=result.scale,
        src_mean=result.src_mean,
        tgt_mean=result.tgt_mean
    )
    print(f"   Saved to {results_dir / 'nllb_qwen_alignment.npz'}")

    # Save vocabulary mapping
    vocab_map = {p['norm']: {'qwen_idx': p['qwen_idx'], 'nllb_idx': p['nllb_idx']}
                 for p in pairs}
    with open(results_dir / "shared_vocab_map.json", 'w') as f:
        json.dump(vocab_map, f)
    print(f"   Saved vocab map to {results_dir / 'shared_vocab_map.json'}")

    # Semantic preservation check
    print("\n8. Checking if semantic structure is preserved...")

    # For a few test words, check if nearest neighbors are similar before/after mapping
    test_words = ["king", "queen", "man", "woman", "dog", "cat", "one", "two", "three"]

    def get_embedding(word, tokenizer, embed_matrix, normalized_pairs):
        """Get embedding for a word if it's in shared vocab."""
        for p in pairs:
            if p['norm'].lower() == word.lower():
                if 'nllb' in str(type(tokenizer)).lower() or 'nllb' in str(tokenizer):
                    return embed_matrix[p['nllb_idx']].numpy()
                else:
                    return embed_matrix[p['qwen_idx']].numpy()
        return None

    # Get embeddings for test words
    nllb_test_embeds = {}
    qwen_test_embeds = {}
    for word in test_words:
        for p in pairs:
            if p['norm'].lower() == word.lower():
                nllb_test_embeds[word] = nllb_embed[p['nllb_idx']].numpy()
                qwen_test_embeds[word] = qwen_embed[p['qwen_idx']].numpy()
                break

    found_words = list(nllb_test_embeds.keys())
    print(f"   Found {len(found_words)} test words in shared vocab: {found_words}")

    if len(found_words) >= 2:
        # Map NLLB embeddings to Qwen space
        mapped_embeds = {}
        for word, emb in nllb_test_embeds.items():
            centered = emb - result.src_mean
            mapped_embeds[word] = centered @ result.W + result.tgt_mean

        # Compare similarity matrices
        print("\n   Similarity matrix in original Qwen space:")
        for w1 in found_words[:5]:
            sims = []
            for w2 in found_words[:5]:
                e1, e2 = qwen_test_embeds[w1], qwen_test_embeds[w2]
                sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                sims.append(f"{sim:.2f}")
            print(f"      {w1:8s}: {' '.join(sims)}")

        print("\n   Similarity matrix after mapping NLLB → Qwen:")
        for w1 in found_words[:5]:
            sims = []
            for w2 in found_words[:5]:
                e1, e2 = mapped_embeds[w1], mapped_embeds[w2]
                sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                sims.append(f"{sim:.2f}")
            print(f"      {w1:8s}: {' '.join(sims)}")

        # Check correlation of pairwise similarities
        qwen_sims = []
        mapped_sims = []
        for i, w1 in enumerate(found_words):
            for j, w2 in enumerate(found_words):
                if i < j:
                    e1_q, e2_q = qwen_test_embeds[w1], qwen_test_embeds[w2]
                    e1_m, e2_m = mapped_embeds[w1], mapped_embeds[w2]
                    qwen_sims.append(np.dot(e1_q, e2_q) / (np.linalg.norm(e1_q) * np.linalg.norm(e2_q)))
                    mapped_sims.append(np.dot(e1_m, e2_m) / (np.linalg.norm(e1_m) * np.linalg.norm(e2_m)))

        corr = np.corrcoef(qwen_sims, mapped_sims)[0, 1]
        print(f"\n   Correlation of pairwise similarities: {corr:.3f}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)

    return result, pairs


if __name__ == "__main__":
    main()
