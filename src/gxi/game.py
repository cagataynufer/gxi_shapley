from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from shapiq import Game
from captum.attr import InputXGradient
from shapiq import Game
from shapiq.approximator import PermutationSamplingSV
from shapiq.game_theory import ExactComputer

from .model_adapter import ModelAdapter
from src.masking.base import AbstractMasker
from src.masking.image import ImageMasker
from src.masking.text import TextMasker
from src.masking.tabular import TabularMasker


"""
Core implementation of the Gradient × Input Shapley (GXI-Shapley) game.

This module defines a cooperative game compatible with the ShapIQ framework,
in which the payoff of a coalition is defined via Input × Gradient attribution
evaluated on a masked version of the input.

Conceptually, each GXIShapleyGame instance corresponds to exactly one target
feature i. Shapley values are then computed over the player dimension j,
yielding one row of a matrix-valued attribution representation.

This file contains only game-level logic:
- no dataset loading
- no model training
- no experiment orchestration

The goal is to provide a transparent, minimal, and reproducible definition
of the cooperative game used throughout the thesis.
"""


class GXIShapleyGame(Game):
    """
    Gradient × Input Shapley game for a single target feature.

    Players correspond to maskable input units (e.g. superpixels, tokens,
    or tabular features). For a fixed target feature i, the value function
    v(S) assigns a scalar payoff to each coalition S by:

        1. Masking the input according to S,
        2. Computing Input × Gradient with respect to the model input,
        3. Reducing the resulting attribution tensor to a scalar contribution
           associated with feature i.

    To obtain a full attribution matrix, one GXIShapleyGame is instantiated
    per target feature, and Shapley values are computed over the player axis.
    """

    def __init__(
        self,
        n_players: int,
        model: nn.Module,
        x: torch.Tensor,
        masker: AbstractMasker,
        device: Optional[torch.device] = None,
        *,
        aggregation: str,
        normalize: bool = False,
        forward_args: Optional[tuple[Any, ...]] = None,
        forward_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        n_players:
            Number of Shapley players, corresponding to maskable input units.

        model:
            A trained PyTorch model whose predictions are to be explained.

        x:
            A single input instance (without batch dimension) serving as the
            reference input for attribution.

        masker:
            An instance of AbstractMasker defining how players are masked
            for a given coalition.

        device:
            Torch device. If None, the device is inferred from the model.

        aggregation:
            Specifies how Input × Gradient values are reduced to a scalar.
            Supported options are "signed" and "absolute".

        normalize:
            Passed to the ShapIQ Game base class. Typically set to False
            for GXI-Shapley games.

        forward_args / forward_kwargs:
            Optional additional arguments forwarded to the model during
            prediction and gradient computation.

        Notes
        -----
        A GXIShapleyGame instance is intended to be reused only for a single
        target feature, specified via the attribute `feature_index`.
        """

        if aggregation not in ("signed", "absolute"):
            raise ValueError("aggregation must be either 'signed' or 'absolute'.")

        self.aggregation = aggregation

        # Initialize ShapIQ Game base class
        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=0.0,
            **kwargs,
        )

        # Adapter responsible for prediction and gradient computation
        self.model_adapter = ModelAdapter(model, device=device)

        # Store input instance on the correct device (no batch dimension)
        self.device = self.model_adapter.device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        self.x = x.to(self.device)

        # Masker defining player semantics
        self.masker = masker

        # Additional forward arguments passed to the model
        self._forward_args = forward_args or ()
        self._forward_kwargs = forward_kwargs or {}

        # Target feature configuration
        self.feature_index: Optional[int] = None
        self.target_index: Optional[int] = None


    # Internal helper: masking + batching
    def _apply_mask(self, coalition: list[bool]) -> torch.Tensor:
        """
        Apply the coalition mask to the stored input and ensure the result
        has a batch dimension suitable for model evaluation.

        The masking operation itself is delegated to the masker instance.
        """
        masked = self.masker.mask(self.x, coalition)

        # Add batch dimension depending on input modality
        if masked.ndim == 1:      # tabular
            masked = masked.unsqueeze(0)
        elif masked.ndim == 2:    # text
            masked = masked.unsqueeze(0)
        elif masked.ndim == 3:    # image
            masked = masked.unsqueeze(0)

        return masked


    # Cooperative game value function v(S)
    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """
        Evaluate the cooperative game payoff v(S) for one or more coalitions.

        For each coalition S, the payoff is defined as the Input × Gradient
        attribution of the selected target feature, evaluated on the input
        masked according to S.

        This method is repeatedly called by ShapIQ solvers and must therefore
        be free of side effects.
        """

        if self.feature_index is None:
            raise ValueError("feature_index must be set before calling value_function().")

        # Standardize coalition representation
        coalitions = np.asarray(coalitions, dtype=bool)
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        n_coalitions = coalitions.shape[0]
        values = np.empty(n_coalitions, dtype=float)

        # Iterate over coalitions
        for i, coalition_bool in enumerate(coalitions):
            coalition = coalition_bool.tolist()

            # Apply mask to input
            masked_x = self._apply_mask(coalition)

            # Forward pass without gradients to infer target if needed
            with torch.no_grad():
                output = self.model_adapter.predict(
                    masked_x, *self._forward_args, **self._forward_kwargs
                )

            # Determine target output dimension
            if self.target_index is None:
                if output.ndim == 1 or output.shape[-1] == 1:
                    target = None  # regression
                else:
                    target = output.argmax(dim=-1).item()
            else:
                target = self.target_index

            # Build Captum-compatible forward arguments
            if self._forward_kwargs:
                additional_forward_args = (*self._forward_args, self._forward_kwargs)
            elif self._forward_args:
                additional_forward_args = self._forward_args
            else:
                additional_forward_args = None

            # Input × Gradient attribution
            gxi = self.model_adapter.input_x_gradient(
                masked_x,
                target_index=target,
                additional_forward_args=additional_forward_args,
            )


            # Reduce attribution tensor to scalar payoff
            agg = self.aggregation

            if isinstance(self.masker, ImageMasker):
                h0, h1, w0, w1 = self.masker.pixel_slices[self.feature_index]
                block = gxi[0, :, h0:h1, w0:w1]
                scalar = block.abs().sum().item() if agg == "absolute" else block.sum().item()

            elif isinstance(self.masker, TextMasker):
                vec = gxi[0, self.feature_index]
                scalar = vec.abs().sum().item() if agg == "absolute" else vec.sum().item()

            elif isinstance(self.masker, TabularMasker):
                scalar = gxi.view(-1)[self.feature_index].item()

            else:
                raise RuntimeError("Unknown masker type.")

            values[i] = scalar

        return values
