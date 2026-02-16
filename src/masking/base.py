from __future__ import annotations

import torch


"""
Abstract masking interface for GXI-Shapley.

A masker defines how individual Shapley players correspond to parts of an
input representation and how those parts are removed or replaced when
forming a coalition.

Maskers are deliberately kept independent of:
- the model architecture,
- the attribution method,
- the cooperative game definition.

Their sole responsibility is to map (x, coalition) -> masked_x.
"""


class AbstractMasker:
    """
    Abstract interface for masking input representations.

    A masker specifies how individual players (maskable units) are
    removed or replaced in the input when forming a coalition S.

    Concrete subclasses define modality-specific masking behavior
    (e.g. image superpixels, text tokens, tabular features).
    """

    def mask(self, x: torch.Tensor, coalition: list[bool]) -> torch.Tensor:
        """
        Apply a coalition mask to the input representation.

        Parameters
        ----------
        x:
            Input tensor representing a single instance (no batch dimension).

        coalition:
            Boolean list of length n_players, where coalition[j] = True
            indicates that player j is present in the coalition, and
            coalition[j] = False indicates that player j is masked.

        Returns
        -------
        masked_x:
            Tensor of the same shape as x, where masked players have been
            replaced according to the modality-specific baseline.

        Notes
        -----
        This method must be side-effect free and should not modify x in place.
        """
        raise NotImplementedError("Masker subclasses must implement mask().")
