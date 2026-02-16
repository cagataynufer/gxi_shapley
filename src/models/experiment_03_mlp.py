"""
Multilayer Perceptron used for black-box faithfulness experiments.

This model is intentionally simple: a fully connected feed-forward network
with ReLU activations. Nonlinear hidden layers allow the model to represent
feature interactions implicitly, while the architecture itself does not
encode any explicit interaction structure.

The purpose of this model is not performance optimization, but to serve as
a generic differentiable black-box function for evaluating attribution
faithfulness.
"""

import torch
import torch.nn as nn


class MLPFaithfulness(nn.Module):
    """
    Fully connected multilayer perceptron for regression.

    Architecture:
        input (d)
          → Linear(d, hidden_dim)
          → ReLU
          → Linear(hidden_dim, hidden_dim)
          → ReLU
          → Linear(hidden_dim, 1)

    The model outputs a single scalar per input, making it suitable for
    regression tasks such as California Housing.

    Notes
    -----
    - Hidden layers introduce nonlinear feature interactions implicitly.
    - No regularization or architectural bias toward additivity is imposed.
    - The model is deliberately kept fixed across experiments to isolate
      the effect of the attribution method.
    """

    def __init__(self, d: int, hidden_dim: int = 64):
        """
        Parameters
        ----------
        d : int
            Number of input features.

        hidden_dim : int, default=64
            Width of the hidden layers.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, d).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size,), containing one scalar
            prediction per input.
        """
        return self.net(x).squeeze(-1)
