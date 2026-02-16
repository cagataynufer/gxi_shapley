from __future__ import annotations

import numpy as np
import torch

"""
Synthetic dataset for the coalition-path interaction experiment (Experiment 2).

The data-generating process is explicitly constructed to include interactions
between a subset of features, while leaving others independent.

Ground-truth structure:
- x1 and x2 interact
- x2 and x3 interact
- x4 has no interaction with any other feature

No higher-order self-interaction terms are included.

This dataset is used to demonstrate that GXI-Shapley captures not only
functional dependencies imposed by the data-generating process, but also
model-induced dependencies arising from internal representation entanglement.
"""


def generate_synthetic_interaction_data(
    *,
    n_samples: int = 400,
    noise_std: float = 0.1,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Generate a nonlinear synthetic regression dataset with controlled interactions.

    Parameters
    ----------
    n_samples : int, default=400
        Total number of samples.
    noise_std : float, default=0.1
        Standard deviation of additive Gaussian noise.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - X_train : (N_train, 4)
        - y_train : (N_train, 1)
        - X_test  : (N_test, 4)
        - y_test  : (N_test, 1)
    """

    rng = np.random.default_rng(seed)


    # Generate input features
    D = 4
    X = rng.standard_normal(size=(n_samples, D)).astype(np.float32)

    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    x4 = X[:, 3]  # explicitly unused in the target


    # Data-generating process
    # True interactions:
    #   - x1 × x2
    #   - x2 × x3
    # Feature x4 does not participate in any interaction.

    y = (
        2.0 * x1 * x2
        - 3.0 * x2 * x3
        + noise_std * rng.standard_normal(n_samples).astype(np.float32)
    )

    # -----------------------------------------------------
    # Train / test split
    # -----------------------------------------------------
    perm = rng.permutation(n_samples)
    train_idx = perm[: int(0.75 * n_samples)]
    test_idx  = perm[int(0.75 * n_samples):]

    X_train = torch.tensor(X[train_idx])
    y_train = torch.tensor(y[train_idx]).unsqueeze(1)

    X_test = torch.tensor(X[test_idx])
    y_test = torch.tensor(y[test_idx]).unsqueeze(1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test":  X_test,
        "y_test":  y_test,
    }
