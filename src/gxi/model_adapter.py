from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from captum.attr import InputXGradient


"""
Model adapter utilities for GXI-Shapley.

This module defines a lightweight adapter that standardizes interaction
with differentiable PyTorch models. Its role is to isolate model-specific
operations (forward passes and gradient computation) from the cooperative
game logic defined in GXIShapleyGame.

The adapter intentionally avoids any assumptions about:
- model architecture,
- input modality,
- task type (classification or regression).

This separation ensures that the GXI-Shapley game definition remains
model-agnostic and conceptually transparent.
"""


class ModelAdapter:
    """
    Lightweight adapter for differentiable PyTorch models.

    The adapter provides a minimal interface for:
    - performing forward passes in evaluation mode,
    - computing gradients with respect to the input tensor,
    - delegating Input × Gradient attribution to Captum.

    All model interaction required by the GXI-Shapley framework is funneled
    through this class, allowing the game logic to remain independent of
    specific model implementations.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Parameters
        ----------
        model:
            A trained PyTorch model whose predictions are to be explained.

        device:
            Torch device on which computations are performed. If None,
            the device is inferred from the model parameters.
        """
        self.model = model
        self.model.eval()


        # Device inference
        # If no device is provided explicitly, infer it from the
        # model parameters. This ensures consistency between the
        # model, inputs, and gradient computations.
        if device is None:
            try:
                param = next(model.parameters())
                device = param.device
            except StopIteration:
                # Models without parameters (rare but possible)
                device = torch.device("cpu")

        self.device = device

        # Captum Input × Gradient
        # Captum is used as the backend for gradient-based attribution.
        # The adapter does not wrap or modify Captum behavior; it merely
        # exposes a consistent interface to the GXI-Shapley game.
        if InputXGradient is not None:
            self._ixg = InputXGradient(self.model)
        else:
            self._ixg = None


    # Internal utility: move input to correct device
    def _move_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure that the input is a torch.Tensor and resides on the
        correct device.

        This method does not modify the input semantics and is used
        only to enforce device consistency.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.to(self.device)

    # Forward pass (no gradients)
    def predict(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Perform a forward pass without tracking gradients.

        This method is used exclusively to obtain model outputs for
        determining the target index (e.g. predicted class) prior to
        gradient-based attribution.

        Parameters
        ----------
        x:
            Input tensor (with batch dimension).

        Returns
        -------
        output:
            Model output tensor.
        """
        x = self._move_input(x)
        self.model.eval()

        with torch.no_grad():
            return self.model(x, *args, **kwargs)

    # Input × Gradient attribution
    def input_x_gradient(
        self,
        x: torch.Tensor,
        target_index: Optional[int] = None,
        additional_forward_args: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Compute Input × Gradient attributions using Captum.

        Parameters
        ----------
        x:
            Input tensor with respect to which gradients are computed.
            The tensor must include a batch dimension.

        target_index:
            Index of the output component to attribute. For regression
            models, this is typically None.

        additional_forward_args:
            Optional additional arguments forwarded to the model's
            forward method, as expected by Captum.

        Returns
        -------
        attributions:
            Tensor of the same shape as x containing Input × Gradient
            attributions.
        """
        if self._ixg is None:
            raise RuntimeError(
                "Captum is not available. Install `captum` to use "
                "Input × Gradient attribution."
            )

        # Prepare input for gradient computation
        x = self._move_input(x).clone().detach()
        x.requires_grad_(True)

        # Delegate attribution computation to Captum
        attributions = self._ixg.attribute(
            x,
            target=target_index,
            additional_forward_args=additional_forward_args,
        )

        return attributions
