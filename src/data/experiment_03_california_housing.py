"""
California Housing dataset loader.

This module provides a deterministic interface for
loading the California Housing dataset using scikit-learn.

Design principles:
- Standard, widely accepted data source
- Fully reproducible train–test split
- No external services or authentication
- Explicit preprocessing with no hidden state
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset


def load_california_housing(
    test_size: float = 0.2,
    random_state: int = 42,
    device: torch.device | None = None,
):
    """
    Load and preprocess the California Housing dataset.

    The dataset is obtained from scikit-learn, split into training and
    test sets using a fixed random seed, and standardized using statistics
    computed on the training data only.

    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of the dataset used for the test split.

    random_state : int, default=42
        Seed controlling the train–test split for reproducibility.

    device : torch.device or None, default=None
        If provided, returned tensors are moved to this device.

    Returns
    -------
    train_dataset : torch.utils.data.TensorDataset
        Training dataset containing standardized features and targets.

    test_dataset : torch.utils.data.TensorDataset
        Test dataset containing standardized features and targets.

    d : int
        Number of input features.
    """

  
    # Fetch dataset
    # This call is deterministic and cached locally by scikit-learn
    data = fetch_california_housing(as_frame=False)

    X = data.data      # shape (n_samples, d)
    y = data.target   # shape (n_samples,)

    # Train–test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


    # Feature standardization
    # The scaler is fit on the training data only to avoid information leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)


    # Conversion to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    if device is not None:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test  = X_test.to(device)
        y_test  = y_test.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset  = TensorDataset(X_test, y_test)

    d = X_train.shape[1]

    return train_dataset, test_dataset, d
