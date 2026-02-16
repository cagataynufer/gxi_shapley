from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.experiment_02_synthetic_interaction_data import (
    generate_synthetic_interaction_data,
)
from src.models.experiment_02_mlp_model import (
    MLPInteractionExperiment2,
)

"""
Training script for the nonlinear MLP used in Experiment 2.

This script trains a deliberately overparameterized MLP on a synthetic dataset
with controlled feature interactions. The trained model is used immediately
for coalition-path analysis using the GXI-Shapley framework.

The model is not saved to disk. Training and attribution are treated as a
single experimental pipeline, which avoids unnecessary artifact management
and keeps the experiment self-contained.
"""


def train_mlp_interaction_experiment2(
    *,
    n_epochs: int = 500,
    learning_rate: float = 1e-2,
    seed: int = 0,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Train the MLP model for the coalition-path experiment.

    Parameters
    ----------
    n_epochs : int, default=500
        Number of training epochs.
    learning_rate : float, default=1e-2
        Learning rate for Adam optimizer.
    seed : int, default=0
        Random seed for reproducibility.
    device : torch.device or None
        Device on which to run training.
        If None, inferred automatically.

    Returns
    -------
    nn.Module
        Trained MLP model, ready for attribution analysis.
    """


    # Reproducibility
    torch.manual_seed(seed)

    # Device inference
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load synthetic data
    data = generate_synthetic_interaction_data(seed=seed)

    X_train = data["X_train"].to(device)
    y_train = data["y_train"].to(device)

    input_dim = X_train.shape[1]

    # Model, optimizer, loss
    model = MLPInteractionExperiment2(input_dim=input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()


    # Training loop
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        preds = model(X_train)
        loss = loss_fn(preds, y_train)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"[Epoch {epoch+1:04d}/{n_epochs}] "
                f"Training loss: {loss.item():.6f}"
            )

    # Switch to evaluation mode before returning
    model.eval()
    return model

