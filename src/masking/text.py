from __future__ import annotations

import torch

from .base import AbstractMasker


"""
Text masking utilities for GXI-Shapley.

This module defines a masker for token-level text representations, where
each Shapley player corresponds to a single token embedding. Masking is
performed by replacing the embedding of a masked token with a fixed
baseline embedding.
"""


class TextMasker(AbstractMasker):
    """
    Masker for token-level text representations.

    Each player corresponds to one token in the input sequence. When a
    player is masked, the corresponding token embedding is replaced by
    a predefined baseline embedding.

    This masker operates directly on embedding-level representations
    and is therefore compatible with models that expose or accept
    precomputed embeddings.
    """

    def __init__(self, baseline_embedding: torch.Tensor) -> None:
        """
        Parameters
        ----------
        baseline_embedding:
            Tensor of shape (embedding_dim,) used to replace masked tokens.
            This embedding typically represents a neutral or reference token
            (e.g. zero vector, [PAD], or [MASK] embedding).
        """
        self.baseline = baseline_embedding

    def mask(self, x: torch.Tensor, coalition: list[bool]) -> torch.Tensor:
        """
        Apply a token-level mask to the input embedding sequence.

        Parameters
        ----------
        x:
            Tensor of shape (seq_len, embedding_dim) representing a single
            text instance (no batch dimension).

        coalition:
            Boolean list of length seq_len indicating which tokens are present
            in the coalition.

        Returns
        -------
        masked_x:
            Tensor of shape (seq_len, embedding_dim) where masked tokens
            have been replaced by the baseline embedding.
        """
        # Work on a copy to avoid modifying the original input
        x = x.clone()

        for i, keep in enumerate(coalition):
            if not keep:
                x[i] = self.baseline

        return x
