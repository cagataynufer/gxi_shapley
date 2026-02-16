from __future__ import annotations

import torch
import torch.nn as nn

"""
Nonlinear MLP model used in the second interaction experiment.

This model is intentionally simple and overparameterized relative to the
synthetic data-generating process. Its purpose is not predictive performance,
but to induce internal representation entanglement between input features.

In particular:
- The ground-truth data-generating process contains interactions only between
  a subset of features.
- This MLP is trained on that data and may learn internal dependencies that do
  not correspond to explicit functional interactions.
- This makes it suitable for demonstrating that GXI-Shapley captures
  model-induced dependencies beyond the data-generating equation.

No training, data loading, or attribution logic is included in this file.
"""


class MLPInteractionExperiment2(nn.Module):
    """
    Multi-layer perceptron used for coalition-path analysis.

    Architecture:
    - Input layer: D features
    - Two hidden layers with ReLU activations
    - Scalar output

    Notes:
    - The depth and width are chosen to allow nonlinear feature mixing.
    - No regularization is applied; this is intentional to encourage
      representation entanglement.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Number of input features D.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, D).

        Returns
        -------
        torch.Tensor
            Model output of shape (batch_size, 1).
        """
        return self.net(x)
