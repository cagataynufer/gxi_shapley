from __future__ import annotations

import torch

from .base import AbstractMasker


"""
Tabular masking utilities for GXI-Shapley.

This module defines a masker for tabular inputs, where each Shapley player
corresponds to one feature. Masking is performed by replacing the value of
a masked feature with its baseline value.
"""


class TabularMasker(AbstractMasker):
    """
    Masker for tabular input representations.

    Each player corresponds to a single feature in the input vector.
    When a player is masked, the feature value is replaced by a
    predefined baseline value.
    """

    def __init__(self, baseline: torch.Tensor) -> None:
        """
        Parameters
        ----------
        baseline:
            Tensor of shape (num_features,) specifying the baseline value
            for each feature.
        """
        self.baseline = baseline

    def mask(self, x: torch.Tensor, coalition: list[bool]) -> torch.Tensor:
        """
        Apply a feature-level mask to the tabular input.

        Parameters
        ----------
        x:
            Tensor of shape (num_features,) representing a single instance
            (no batch dimension).

        coalition:
            Boolean list of length num_features indicating which features
            are present in the coalition.

        Returns
        -------
        masked_x:
            Tensor of shape (num_features,) where masked features have been
            replaced by their corresponding baseline values.
        """
        # Work on a copy to avoid modifying the original input
        x = x.clone()

        for i, keep in enumerate(coalition):
            if not keep:
                x[i] = self.baseline[i]

        return x
