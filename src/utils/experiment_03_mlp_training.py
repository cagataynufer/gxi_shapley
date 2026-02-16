"""
Training utility for the MLP used in black-box faithfulness experiments.

This module provides a deterministic training routine for the
MLPFaithfulness model on the California Housing dataset. The goal is
not to optimize predictive performance, but to obtain a stable,
nonlinear black-box model for attribution evaluation.
"""

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.experiment_03_mlp import MLPFaithfulness


def set_seed(seed: int) -> None:
    """
    Fix all relevant random seeds to ensure reproducible training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mlp_faithfulness(
    train_dataset,
    test_dataset,
    d: int,
    *,
    hidden_dim: int = 64,
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 300,
    seed: int = 0,
    device: torch.device | None = None,
    save_path: str | Path | None = None,
):
    """
    Train an MLPFaithfulness model on the California Housing dataset.

    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        Training dataset containing standardized inputs and targets.

    test_dataset : torch.utils.data.Dataset
        Test dataset containing standardized inputs and targets.

    d : int
        Number of input features.

    hidden_dim : int, default=64
        Width of the hidden layers.

    batch_size : int, default=128
        Mini-batch size used during training.

    lr : float, default=1e-3
        Learning rate for Adam optimization.

    epochs : int, default=300
        Number of training epochs.

    seed : int, default=0
        Random seed controlling model initialization and data shuffling.

    device : torch.device or None, default=None
        Device on which training is performed.

    save_path : str or Path or None, default=None
        If provided, the trained model state_dict is saved to this path.

    Returns
    -------
    model : MLPFaithfulness
        Trained model.

    train_mse : float
        Final mean squared error on the training set.

    test_mse : float
        Mean squared error on the test set.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

 
    # Model, optimizer, and loss
    model = MLPFaithfulness(d=d, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Training loop
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

        if epoch % 20 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / total_n
            print(f"[Epoch {epoch:03d}] Train MSE = {avg_loss:.6f}")


    # Evaluation
    model.eval()

    def compute_mse(loader):
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                loss = criterion(preds, y)
                total_loss += loss.item() * x.size(0)
                total_n += x.size(0)
        return total_loss / total_n

    train_mse = compute_mse(train_loader)
    test_mse = compute_mse(test_loader)


    # Optional checkpointing
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    return model, train_mse, test_mse
