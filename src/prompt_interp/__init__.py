"""SONAR-LLM: Sentence-level language model using SONAR embeddings."""
from pathlib import Path

import torch as t


def get_repo_root() -> Path:
    """
    Find the repository root by looking for pyproject.toml.

    Walks up from this file's directory until it finds pyproject.toml.
    This is more robust than counting parent directories.
    """
    current = Path(__file__).parent
    for _ in range(10):  # Limit search depth
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repo root (no pyproject.toml found)")


REPO_ROOT = get_repo_root()

default_tensor_repr = t.Tensor.__repr__


def custom_tensor_repr(self: t.Tensor) -> str:
    return f"{list(self.shape)}, {default_tensor_repr(self)}"


t.Tensor.__repr__ = custom_tensor_repr  # type: ignore