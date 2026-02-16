"""
GXI coalition-path analysis for a nonlinear MLP with controlled interactions.

This experiment evaluates the Gradient × Input (GXI) game payoff directly
along structured coalition paths, WITHOUT Shapley aggregation.

Objective
---------
Analyze how the GXI payoff assigned to a target feature changes as other
features enter the coalition. This exposes contextual / representation-induced
dependencies that are not visible from aggregated Shapley values alone.

Payoff definition (per target feature i)
----------------------------------------
For a coalition S (set of unmasked features) and baseline-masked input x_S:
    v_i(S) = e_i^T ( ∇F(x_S) ⊙ x_S ) = x_{S,i} * ∂F(x_S)/∂x_i.

Baseline masking uses b = 0 (zero baseline) for all absent features.

Coalition-path ordering
------------------------
Let D be the number of features (here D = 4).
We enumerate all 2^D binary coalitions c ∈ {0,1}^D.

For a fixed target feature i, we order coalitions by:
    (1) c[i]          : target absent (0) before target present (1)
    (2) sum(c)        : increasing coalition size
    (3) c (tuple)     : lexicographic tie-breaker

Outputs
-------
Figures saved under:
    results/02_gxi_coalition_paths/

For each target feature i in {0,1,2,3}:
- gxi_path_testset_mean_std_target_i.png :
    mean GXI payoff along the ordered coalition path, with ±1 std shading
    across test samples.
"""

import itertools
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils.experiment_02_mlp_training import train_mlp_interaction_experiment2
from src.data.experiment_02_synthetic_interaction_data import generate_synthetic_interaction_data


# Masking and GXI payoff definition
def mask_input(
    x: torch.Tensor,
    coalition_keep: List[bool],
    baseline: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a coalition mask to the input vector x.

    For each feature m:
      if coalition_keep[m] is False, set x_m to baseline_m.

    Returns x_S in x-space.
    """
    xm = x.clone()
    for m, keep in enumerate(coalition_keep):
        if not keep:
            xm[m] = baseline[m]
    return xm


def payoff_gxi(
    x: torch.Tensor,
    coalition_keep: List[bool],
    target_feature: int,
    model: nn.Module,
    baseline: torch.Tensor,
) -> float:
    """
    Compute GXI payoff for target_feature i under coalition S:

        v_i(S) = x_{S,i} * ∂F(x_S) / ∂x_i

    where x_S is the baseline-masked input for coalition S.
    """
    xm = mask_input(x, coalition_keep, baseline)
    xm = xm.clone().detach().requires_grad_(True)

    model.zero_grad()
    out = model(xm).squeeze()
    out.backward()

    grad = xm.grad
    gxi = xm * grad
    return float(gxi[target_feature].item())


def compute_gxi_for_coalition(
    x: torch.Tensor,
    coalition_binary: Tuple[int, ...],
    target_feature: int,
    model: nn.Module,
    baseline: torch.Tensor,
) -> float:
    """
    Wrapper: coalition_binary is a tuple in {0,1}^D.
    """
    coalition_keep = [bool(v) for v in coalition_binary]
    return payoff_gxi(x, coalition_keep, target_feature, model, baseline)


# Coalition ordering (path)
def ordered_coalitions(D: int, target_feature: int) -> List[Tuple[int, ...]]:
    """
    Generate all 2^D coalitions, ordered as:
        (c[target_feature], sum(c), c)
    """
    coalitions = list(itertools.product([0, 1], repeat=D))
    coalitions.sort(key=lambda c: (c[target_feature], sum(c), c))
    return coalitions



# Reproducibility helper
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Plotting: test-set mean ± std (only)
def plot_gxi_paths_testset_mean_std(
    X_test: torch.Tensor,
    model: nn.Module,
    baseline: torch.Tensor,
    results_dir: Path,
) -> None:
    """
    For each target feature i, compute v_i(S) for all coalitions S and all test
    samples, then plot the test-set mean with ±1 std shading along the ordered
    coalition path.

    Important plotting choice:
    - The y-axis limits are set based on the mean curve (plus padding),
      not on the full std envelope, to avoid the std band dominating the scale.
    """
    n_samples, D = X_test.shape
    assert D == 4, f"Expected D=4 features for this experiment, got D={D}."

    for target_feature in range(D):
        coalitions = ordered_coalitions(D, target_feature)
        nC = len(coalitions)

        # all_vals[s_idx, c_idx] = v_i(S) for sample s_idx and coalition c_idx
        all_vals = np.zeros((n_samples, nC), dtype=float)

        for s_idx, x in enumerate(X_test):
            for c_idx, c in enumerate(coalitions):
                all_vals[s_idx, c_idx] = compute_gxi_for_coalition(
                    x=x,
                    coalition_binary=c,
                    target_feature=target_feature,
                    model=model,
                    baseline=baseline,
                )

        mean_vals = all_vals.mean(axis=0)
        std_vals = all_vals.std(axis=0, ddof=0)  # population std across test inputs

        labels = ["".join(map(str, c)) for c in coalitions]
        x_axis = np.arange(nC, dtype=int)

        fig, ax = plt.subplots(figsize=(18, 4))

        # Mean curve
        ax.plot(
            x_axis,
            mean_vals,
            marker="o",
            linewidth=2,
            label="Mean GXI payoff (test set)",
        )

        # Std band (±1 std across test inputs)
        ax.fill_between(
            x_axis,
            mean_vals - std_vals,
            mean_vals + std_vals,
            alpha=0.15,
            label="± 1 std across test inputs",
        )

        ax.set_xticks(x_axis)
        ax.set_xticklabels(labels, rotation=90)

        # titles/labels 
        ax.set_title(f"GXI coalition path (test-set mean ± 1 std) of the game defined on feature i = {target_feature}")
        ax.set_xlabel("Coalition (binary mask, ordered)")
        ax.set_ylabel("GXI payoff")

        ax.grid(True)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

        # --- y-axis follows the mean curve, not the std envelope ---
        mean_min = float(np.min(mean_vals))
        mean_max = float(np.max(mean_vals))
        pad = 0.15 * (mean_max - mean_min + 1e-12)

        if (mean_max - mean_min) < 1e-6:
            # Degenerate mean curve: enforce a small symmetric window
            y1 = max(1e-3, abs(mean_max) + 1e-3)
            ax.set_ylim(-y1, y1)
        else:
            ax.set_ylim(mean_min - pad, mean_max + pad)

        # Small note
        ax.text(
            0.01, 0.02,
            "Shaded band: ±1 std across test inputs.",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.08, linewidth=0.6),
        )

        fig.tight_layout()

        fname = results_dir / f"gxi_path_testset_mean_std_target_{target_feature}.pdf"
        fig.savefig(fname, bbox_inches="tight")

        plt.close(fig)



# Experiment execution
def main() -> None:
    results_dir = Path("results") / "02_gxi_coalition_paths"
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    device = torch.device("cpu")
    set_seed(seed)

    # Train the nonlinear model
    model_nl = train_mlp_interaction_experiment2(
        n_epochs=500,
        learning_rate=1e-2,
        seed=seed,
        device=device,
    )
    model_nl.eval()

    # Load synthetic data for attribution analysis
    data = generate_synthetic_interaction_data(seed=seed)
    X_test = data["X_test"].to(device)

    # Baseline input used for masking (zero baseline)
    baseline = torch.zeros(X_test.shape[1], device=device)

    # Run ONLY the test-set mean ± std coalition-path analysis
    plot_gxi_paths_testset_mean_std(
        X_test=X_test,
        model=model_nl,
        baseline=baseline,
        results_dir=results_dir,
    )

    print(f"Saved GXI coalition-path mean±std figures to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
