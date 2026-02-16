from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from shapiq.approximator import PermutationSamplingSV
from shapiq.game_theory import ExactComputer
from .game import GXIShapleyGame


def compute_shapley_matrix(
    model,
    x,
    masker,
    n_players,
    *,
    method="sv",
    aggregation="absolute",
    normalize=False,
    budget=1000,
    feature_indices=None,
    device=None,
    forward_args=None,
    forward_kwargs=None,
    batch_size=10,
    random_state=0,
):
    """
    Compute the matrix-valued GXI-Shapley attributions for all target features.

    For each target feature i, this function instantiates a separate
    GXIShapleyGame and computes Shapley-style attributions over all players j.
    The result is a matrix where rows correspond to target features and columns
    correspond to players (plus the null player term).

    Depending on the selected method, values are computed using:
    - permutation-based Shapley value approximation ("sv"),
    - Shapley interaction indices ("sii"),
    - or exact Shapley values ("exact").

    This function is the primary interface for producing the attribution
    matrices analyzed in the thesis.
    """

    # The device is inferred from the model parameters to ensure that
    # all computations (masking, forward pass, and gradients) occur on the same device.
    # Device inference
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    # Ensure the input sample is a tensor on the correct device.
    # The input x represents a single instance to be explained.
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)


    # Determine feature indices
    # Feature indices determine which target features are explained.
    # If not specified, all players are treated as potential targets, resulting in a full square attribution matrix.

    if feature_indices is None:
        feature_indices = list(range(n_players))

    # Number of target features (rows of the attribution matrix).
    n_features = len(feature_indices)


    # allocate with +1 for null player
    # Allocate the attribution matrix.
    # The extra column (and row, for SII) corresponds to the empty coalition,
    # which is included by ShapIQ in its value representation.

    if method.lower() == "sii":
        shap_matrix = np.zeros((n_features, n_players + 1, n_players + 1))
    else:
        shap_matrix = np.zeros((n_features, n_players + 1))


    # Choose approximator
    # Select the ShapIQ approximator or exact solver.
    # The choice determines how Shapley values or interactions are estimated.

    method_lower = method.lower()

    if method_lower == "sv":
        approximator = PermutationSamplingSV(n=n_players, random_state=random_state)

    elif method_lower == "sii":
        from shapiq.approximator.permutation.sii import PermutationSamplingSII
        approximator = PermutationSamplingSII(n=n_players, random_state=random_state)


    elif method_lower == "exact":
        approximator = None

    else:
        raise ValueError("method must be one of {'sv','sii','exact'}")


    # MAIN LOOP
    # Main loop over target features.
    # Each iteration constructs a separate cooperative game in which
    # the selected feature acts as the target of attribution.

    for out_i, feat_idx in enumerate(feature_indices):

        # Build one game per feature
        # Instantiate a new GXI-Shapley game for the current target feature.
        # This enforces the one-game-per-feature design of the framework.
        game = GXIShapleyGame(
            n_players=n_players,
            model=model,
            x=x,
            masker=masker,
            device=device,
            aggregation=aggregation,
            normalize=normalize,
            forward_args=forward_args,
            forward_kwargs=forward_kwargs,
        )
        # Specify which feature index defines the payoff of the game.
        game.feature_index = feat_idx
        game.target_index  = None

        # Compute
        # Compute Shapley values for the current game using the selected method.
        # For exact computation, ShapIQ evaluates all coalitions explicitly.
        if method_lower == "exact":
            ec = ExactComputer(game=game, n_players=n_players, evaluate_game=False)
            ivs = ec("SV")
        else:
            # For approximate computation, ShapIQ samples permutations up to the specified computational budget.
            ivs = approximator.approximate(
                budget=budget,
                game=game,
                batch_size=batch_size,
            )
        # Extract the ShapIQ value vector.
        # By convention, index 0 corresponds to the null player.
        vals = ivs.values  # <-- This now includes null player at index 0!


        # Store full vector (n_players + 1)
        # Store the full Shapley value vector for the current target feature as one row of the attribution matrix.
        shap_matrix[out_i, :len(vals)] = vals

        # Progress logging for long-running computations.
        print(f"[{out_i+1}/{n_features}] Completed feature {feat_idx}")

    # Return the matrix-valued GXI-Shapley attributions.
    return shap_matrix