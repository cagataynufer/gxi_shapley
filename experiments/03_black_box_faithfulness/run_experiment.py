"""
Multi-input faithfulness evaluation (black-box model).

This script evaluates the empirical faithfulness of matrix-valued GXI-Shapley
attributions across multiple inputs from the California Housing dataset.

Summary of what is evaluated
------------------------------------
We evaluate surrogate predictors derived from the attribution matrix Φ(x)
against the true masked-model outputs F(x_S), where S is a coalition of
UNMASKED features and x_S is constructed by baseline masking.

Two surrogate predictors (both derived from Φ for a fixed input) are compared:

(1) Main-effects-only surrogate predictor:
    \\hat{F}_{main}(x_S)
    = F(b) + \\sum_{i\\in S} \\Big( \\phi_{i,i}(x) - \\sum_{j\\in S\\setminus\\{i\\}} \\phi_{i,j}(x) \\Big)

(2) Main-effects + pairwise-interactions surrogate predictor:
    \\hat{F}_{main+int}(x_S)
    = \\hat{F}_{main}(x_S) + \\sum_{i<j,\\ i,j\\in S} 2\\,\\phi_{i,j}(x)

Faithfulness error metric (stabilized relative error)
-----------------------------------------------------
For each coalition S we compute:
    Err(S) = |F(x_S) - \\hat{F}(x_S)| / (|F(x_S)| + ε(S)),
with
    ε(S) = 0.01 * |F(x_S)| + 1e-12.

Aggregation structure (what means and std dev refer to)
-------------------------------------------------------
For a fixed input x and coalition size k = |S|:
- We evaluate Err(S) for all coalitions with |S| = k.
- We average over those coalitions to get a per-input, per-k mean error:
    E_x(k) = mean_{|S|=k} Err(S).

Across inputs, for each k:
- We compute the across-input mean and standard deviation:
    mean_k = mean_x E_x(k),
    std_k  = std_x  E_x(k).
The faithfulness plot shows mean_k with a shaded band of ±1 std_k.

Win rate definition (input-level, not coalition-level)
------------------------------------------------------
For each k, define per-input mean errors:
    E_x_main(k), E_x_main+int(k).
Then input-level win rate at k is:
    WinRate(k) = fraction of inputs x such that E_x_main+int(k) < E_x_main(k).

Relative improvement (per input)
--------------------------------
Let
    E_x_main     = mean_{S != empty} Err_main(S),
    E_x_main+int = mean_{S != empty} Err_main+int(S).
Define per-input relative improvement:
    RI_x = (E_x_main - E_x_main+int) / (E_x_main + 1e-12).
The histogram visualizes RI_x over inputs.

Trials (baseline + coordinate system + model identity)
------------------------------------------------------
(A) Zero-baseline trial:
    - Baseline in x-space: b = 0.
    - Φ is computed for the original model F(x).
    - Masking is applied in x-space.

(B) Median-baseline centered trial (baseline-centered formulation):
    - Baseline in x-space: b = coordinate-wise median over X_train.
    - Centered coordinates: δ = x - b.
    - Shifted model: G(δ) = F(b + δ).
    - Φ is computed for G (in δ-space) under a zero masking baseline in δ-space.
    - Coalitions are masked in δ-space (δ_m = 0 for masked features), then mapped back:
        x_S = b + δ_S,
      and the truth is always evaluated with F(x_S).

Outputs (per trial)
-------------------
Saved under:
    results/03_black_box_faithfulness/<trial_name>/

Artifacts include:
- per_input.csv                 : per-input mean errors & per-input relative improvement
- per_k.csv                     : across-input mean/std per k + input-level win rate per k
- faithfulness_vs_k.png         : mean ± 1 std across inputs vs k (two surrogate curves)
- interaction_winrate.png       : input-level win rate vs k
- rel_improvement_hist.png      : histogram of per-input relative improvements RI_x
- Phi_all.npy                   : stacked Φ matrices, shape (n_inputs, d, d)
- Phi_all.txt                   : human-readable dump of Φ matrices
- trial_info.txt                : ultra-pedantic trial metadata (baseline, model identity, definitions)
"""

import itertools
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data.experiment_03_california_housing import load_california_housing
from src.models.experiment_03_mlp import MLPFaithfulness
from src.gxi.attribution_matrix_computer import compute_shapley_matrix
from src.masking.tabular import TabularMasker



# Plot helper
def save_figure(fig: plt.Figure, path: Path, description: str) -> None:
    """
    Save figure as PDF only (vector format, thesis-ready).
    """

    pdf_path = path.with_suffix(".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {description}")
    print(f"           → {pdf_path.resolve()}")




# Masking helpers
# Note: mask_fn(effective_coords, S, baseline_ref) MUST return x_S in x-space
def mask_input_baseline_xspace(
    x_eff: torch.Tensor,
    S: List[int],
    baseline_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Baseline masking in x-space for Trial A.

    Here x_eff is x in x-space, and baseline_ref is b in x-space.
    Returns x_S in x-space.
    """
    x = x_eff
    x_masked = baseline_ref.clone()
    if len(S) > 0:
        x_masked[list(S)] = x[list(S)]
    return x_masked


def mask_input_centered_then_map_back(
    x_eff: torch.Tensor,
    S: List[int],
    baseline_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Masking in centered coordinates δ = x - b (Trial B), then map back to x-space.

    Here x_eff is δ in δ-space and baseline_ref is b in x-space.
    Coalition masking uses δ_m = 0 for masked features.
    Returns x_S = b + δ_S in x-space.
    """
    delta = x_eff
    delta_masked = torch.zeros_like(delta)
    if len(S) > 0:
        delta_masked[list(S)] = delta[list(S)]
    return baseline_ref + delta_masked


# Surrogate helpers (derived from Φ)
def main_effect_in_coalition(i: int, S: List[int], Phi: torch.Tensor) -> float:
    """
    main_i(S) = φ_{i,i} - sum_{j in S, j != i} φ_{i,j}.
    """
    val = float(Phi[i, i].item())
    for j in S:
        if j != i:
            val -= float(Phi[i, j].item())
    return float(val)


def surrogate_main_effects_only(S: List[int], Phi: torch.Tensor, y_base: float) -> float:
    """
    \\hat{F}_{main}(x_S) = F(b) + sum_{i in S} main_i(S).
    """
    return float(y_base + sum(main_effect_in_coalition(i, S, Phi) for i in S))


def surrogate_main_plus_pairwise_interactions(S: List[int], Phi: torch.Tensor, y_base: float) -> float:
    """
    \\hat{F}_{main+int}(x_S) = \\hat{F}_{main}(x_S) + sum_{i<j in S} 2 φ_{i,j}.
    """
    main_term = sum(main_effect_in_coalition(i, S, Phi) for i in S)

    interaction_term = 0.0
    for i in S:
        for j in S:
            if j > i:
                interaction_term += 2.0 * float(Phi[i, j].item())

    return float(y_base + main_term + interaction_term)


def stabilized_relative_error(y_true: float, y_hat: float) -> float:
    """
    Err(S) = |y_true - y_hat| / (|y_true| + ε), with ε = 0.01|y_true| + 1e-12.
    """
    eps = float(0.01 * abs(y_true) + 1e-12)
    return float(abs(y_true - y_hat) / (abs(y_true) + eps))



# Wrapper for baseline-centered forward pass (δ -> x = b + δ)
class WrappedShiftedModel(torch.nn.Module):
    """
    Shifted model wrapper: G(δ) = F(b + δ).
    Required for Trial B so Φ is computed for the same model identity used
    by δ-space coalitions.
    """

    def __init__(self, base_model: torch.nn.Module, baseline_ref: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("baseline_ref", baseline_ref)

    def forward(self, delta_batch: torch.Tensor) -> torch.Tensor:
        x_batch = self.baseline_ref.unsqueeze(0) + delta_batch
        return self.base_model(x_batch)



# Trial runner
def run_trial(
    *,
    trial_name: str,
    trial_long_name: str,
    model_identity_str: str,
    baseline_identity_str: str,
    x_transform_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mask_fn: Callable[[torch.Tensor, List[int], torch.Tensor], torch.Tensor],
    baseline_fn: Callable,
    train_dataset,
    model_F: torch.nn.Module,
    d: int,
    coalitions_by_k: Dict[int, List[Tuple[int, ...]]],
    results_dir: Path,
    n_inputs: int,
    device: torch.device,
    compute_phi_on_shifted_model: bool,
) -> None:
    """
    Run one trial of faithfulness evaluation.

    Notes:
    - x_transform_fn(x, b) defines "effective coordinates" x_eff in which Φ is computed.
      * Trial A: x_eff = x in x-space
      * Trial B: x_eff = δ = x - b in δ-space
    - mask_fn(x_eff, S, b) returns x_S in x-space (always), used for truth evaluation F(x_S).
    - y_base is always F(b) (evaluated in x-space).
    - Φ is computed either for:
      * Trial A: original model F(x)
      * Trial B: shifted model G(δ)=F(b+δ)
    """
    print(f"\n{'='*88}")
    print(f"Starting trial: {trial_name}")
    print(f"{trial_long_name}")
    print(f"{'='*88}\n")

    trial_dir = results_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Compute baseline b (x-space)
    baseline_ref = baseline_fn(train_dataset).to(device)

    # Cache F(b) once per trial 
    with torch.no_grad():
        y_base = float(model_F(baseline_ref.unsqueeze(0)).item())

    print(f"[{trial_name}] baseline_ref ||b||_2 = {baseline_ref.norm().item():.6f}")
    print(f"[{trial_name}] baseline_ref first 8 coords = {baseline_ref[:8].detach().cpu().numpy()}")
    print(f"[{trial_name}] y_base = F(b) = {y_base:.8f}")

    # Save trial info
    info_path = trial_dir / "trial_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"trial_name: {trial_name}\n")
        f.write(f"trial_long_name: {trial_long_name}\n")
        f.write(f"baseline_identity: {baseline_identity_str}\n")
        f.write(f"model_identity_for_phi: {model_identity_str}\n")
        f.write("\nDefinitions:\n")
        f.write("Err(S) = |F(x_S) - Fhat(x_S)| / (|F(x_S)| + eps(S))\n")
        f.write("eps(S) = 0.01 * |F(x_S)| + 1e-12\n")
        f.write("Per-input, per-k mean error E_x(k) = mean_{|S|=k} Err(S)\n")
        f.write("Across-input mean/std per k computed over E_x(k)\n")
        f.write("Input-level WinRate(k) = frac_x [E_x_main+int(k) < E_x_main(k)]\n")
        f.write("Per-input relative improvement RI_x = (E_main - E_main+int) / (E_main + 1e-12)\n")
    print(f"[{trial_name}] Saved trial metadata → {info_path.resolve()}")

    # Choose model identity used for Φ computation
    if compute_phi_on_shifted_model:
        model_for_phi = WrappedShiftedModel(model_F, baseline_ref).to(device)
        model_for_phi.eval()
    else:
        model_for_phi = model_F

    per_input_records = []
    per_input_per_k_records = []

    # Collect Φ matrices
    Phi_all: List[np.ndarray] = []

    # Loop over inputs
    n_eval = min(n_inputs, len(train_dataset))
    for idx in range(n_eval):
        print(f"[{trial_name}] Processing input {idx + 1}/{n_eval}")

        x, _ = train_dataset[idx]
        x = x.to(device)

        # Effective coordinates for Φ computation (x-space or δ-space)
        x_eff = x_transform_fn(x, baseline_ref)
        assert x_eff.shape == baseline_ref.shape, "Shape mismatch: x_eff vs baseline_ref"

        # Shapley masking baseline is always zero in effective coordinates:
        # Trial A: x_eff = x, baseline is 0 => corresponds to b=0 in x-space
        # Trial B: x_eff = δ, baseline is 0 => corresponds to δ=0 in δ-space
        masker = TabularMasker(baseline=torch.zeros_like(x_eff))

        # Compute Φ (exact Shapley: full coalition enumeration)
        Phi_np = compute_shapley_matrix(
            model=model_for_phi,
            x=x_eff,
            masker=masker,
            n_players=d,
            method="exact",          # exact Shapley enumeration 
            aggregation="signed",
        )
        Phi = torch.as_tensor(Phi_np, device=device)

        Phi_all.append(Phi.detach().cpu().numpy().astype(np.float64))


        # Evaluate errors over all coalitions, grouped by k
        sum_err_main_all = 0.0
        sum_err_int_all = 0.0
        total_coalitions = int(sum(len(v) for v in coalitions_by_k.values()))

        for k, S_list in coalitions_by_k.items():
            err_main_list = []
            err_int_list = []

            for S_tuple in S_list:
                S = list(S_tuple)

                # Apply masking in effective coordinates, but ALWAYS return x_S in x-space for truth eval
                x_masked = mask_fn(x_eff, S, baseline_ref)

                with torch.no_grad():
                    y_true = float(model_F(x_masked.unsqueeze(0)).item())

                y_hat_main = surrogate_main_effects_only(S, Phi, y_base)
                y_hat_int = surrogate_main_plus_pairwise_interactions(S, Phi, y_base)

                err_main = stabilized_relative_error(y_true, y_hat_main)
                err_int = stabilized_relative_error(y_true, y_hat_int)

                err_main_list.append(err_main)
                err_int_list.append(err_int)

            mean_err_main_k = float(np.mean(err_main_list))
            mean_err_int_k = float(np.mean(err_int_list))

            per_input_per_k_records.append({
                "idx": int(idx),
                "trial": trial_name,
                "k": int(k),
                "E_x_main_k": mean_err_main_k,
                "E_x_main_int_k": mean_err_int_k,
                "win_input_k": float(mean_err_int_k < mean_err_main_k),  # input-level win at this k
            })

            # Also accumulate for per-input global means across all nonempty coalitions
            sum_err_main_all += float(np.sum(err_main_list))
            sum_err_int_all += float(np.sum(err_int_list))

        # Per-input global mean errors (averaged over all nonempty coalitions)
        E_main = float(sum_err_main_all / total_coalitions)
        E_int = float(sum_err_int_all / total_coalitions)

        per_input_records.append({
            "idx": int(idx),
            "trial": trial_name,
            "E_main": E_main,
            "E_main_int": E_int,
            "win_input_all": bool(E_int < E_main),
            # Per-input relative improvement:
            # RI_x = (E_main - E_main_int) / (E_main + 1e-12)
            "rel_improvement": float((E_main - E_int) / (E_main + 1e-12)),
        })

    # Save Φ matrices for this trial
    Phi_all_np = np.stack(Phi_all, axis=0)  # (n_inputs, d, d)
    np.save(trial_dir / "Phi_all.npy", Phi_all_np)

    txt_path = trial_dir / "Phi_all.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Trial: {trial_name}\n")
        f.write(f"Trial long name: {trial_long_name}\n")
        f.write(f"Baseline identity: {baseline_identity_str}\n")
        f.write(f"Model identity for Phi: {model_identity_str}\n")
        f.write(f"Phi_all shape: {Phi_all_np.shape}\n")
        f.write("=" * 100 + "\n\n")
        for i in range(Phi_all_np.shape[0]):
            f.write(f"[Phi] input_idx={i}\n")
            f.write(np.array2string(Phi_all_np[i], precision=6, suppress_small=False, max_line_width=200))
            f.write("\n\n")

    print(f"[{trial_name}] Saved Phi_all.npy and Phi_all.txt → {trial_dir.resolve()}")

    # Save CSVs
    df_input = pd.DataFrame(per_input_records)
    df_input_k = pd.DataFrame(per_input_per_k_records)

    df_input.to_csv(trial_dir / "per_input.csv", index=False)

    # Across-input mean/std per k computed over per-input-per-k errors E_x(k)
    def pop_std(x: pd.Series) -> float:
        return float(np.std(x.to_numpy(dtype=float), ddof=0))

    def sem_pop(x: pd.Series) -> float:
        arr = x.to_numpy(dtype=float)
        n = arr.size
        if n <= 1:
            return 0.0
        return float(np.std(arr, ddof=0) / np.sqrt(n))


    df_k = (
        df_input_k
        .groupby("k", as_index=True)
        .agg(
            err_main_mean=("E_x_main_k", "mean"),
            err_main_sem=("E_x_main_k", sem_pop),

            err_main_int_mean=("E_x_main_int_k", "mean"),
            err_main_int_sem=("E_x_main_int_k", sem_pop),

            input_winrate_int=("win_input_k", "mean"),
        )
        .sort_index()
    )


    df_k.to_csv(trial_dir / "per_k.csv")

    # Plot: Faithfulness vs coalition size (mean ± SEM across inputs)
    x_axis = df_k.index.to_numpy(dtype=float)

    y_main = df_k["err_main_mean"].to_numpy(dtype=float)
    s_main = df_k["err_main_sem"].to_numpy(dtype=float)   # SEM

    y_int  = df_k["err_main_int_mean"].to_numpy(dtype=float)
    s_int  = df_k["err_main_int_sem"].to_numpy(dtype=float)  # SEM

    # SEM bounds 
    main_lo = np.maximum(y_main - s_main, 0.0)
    main_hi = y_main + s_main
    int_lo  = np.maximum(y_int - s_int, 0.0)
    int_hi  = y_int + s_int

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Mean curves (capture line objects) 
    line_main, = ax.plot(
        x_axis,
        y_main,
        marker="o",
        label="Main-effects surrogate (mean)"
    )

    line_int, = ax.plot(
        x_axis,
        y_int,
        marker="o",
        label="Main+interactions surrogate (mean)"
    )

    # Extract their colors
    c_main = line_main.get_color()
    c_int  = line_int.get_color()

    # SEM bounds (color-matched, dashed, no shading)
    ax.plot(x_axis, main_lo, linestyle="--", linewidth=1.0, color=c_main)
    ax.plot(x_axis, main_hi, linestyle="--", linewidth=1.0, color=c_main)

    ax.plot(x_axis, int_lo, linestyle="--", linewidth=1.0, color=c_int)
    ax.plot(x_axis, int_hi, linestyle="--", linewidth=1.0, color=c_int)

    ax.set_xlabel(r"Number of unmasked features $|S|$")
    ax.set_ylabel("Mean stabilized relative error")
    ax.set_title(f"{trial_long_name}")
    ax.grid(True)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Force y-axis to reflect mean curves (NOT the full SEM envelope)
    mean_min = float(np.min([y_main.min(), y_int.min()]))
    mean_max = float(np.max([y_main.max(), y_int.max()]))

    pad = 0.08 * (mean_max - mean_min + 1e-12)
    y0 = max(0.0, mean_min - pad)
    y1 = mean_max + pad
    ax.set_ylim(y0, y1)

    ax.text(
        0.02, 0.02,
        "Dashed lines: mean ± SEM across inputs.\n"
        "Each point: per-input mean over all coalitions with that |S|.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.10, linewidth=0.6),
    )

    save_figure(
        fig,
        trial_dir / "faithfulness_vs_k.png",
        "Faithfulness vs k (mean ± SEM across inputs)"
    )

    # Plot: Input-level win rate vs coalition size
    y_win = df_k["input_winrate_int"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.plot(
        x_axis,
        y_win,
        marker="o",
        label="Interaction-aware surrogate win rate",
    )
    ax.axhline(
        0.5,
        linestyle="--",
        color="gray",
        label="Chance level (0.5)",
    )

    ax.set_xlabel(r"Number of unmasked features $|S|$")
    ax.set_ylabel("Input-level win rate")
    ax.set_title(f"Win rate across inputs — {trial_name}")

    ax.grid(True)

    # Legend with definition embedded (no axis text)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="C0",
            label="Interaction-aware surrogate win rate"),
        Line2D([0], [0], linestyle="--", color="gray",
            label="Chance level (0.5)"),
        Line2D([0], [0], color="none",
            label=r"Win rate: $\Pr[\mathrm{Err}_{\text{main+int}} < \mathrm{Err}_{\text{main}}]$")
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    save_figure(
        fig,
        trial_dir / "interaction_winrate.png",
        "Input-level interaction-aware surrogate win rate vs coalition size",
    )

    # Plot: Histogram of per-input relative improvements
    vals = pd.to_numeric(
        df_input["rel_improvement"],
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    ax.hist(vals, bins=30, edgecolor="black")
    ax.axvline(0.0, linestyle="--", color="black")

    ax.set_xlabel("Relative improvement in mean stabilized error (per input)")
    ax.set_ylabel("Number of inputs")
    ax.set_title(f"Interaction benefit distribution — {trial_name}")

    ax.grid(True)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    save_figure(
        fig,
        trial_dir / "rel_improvement_hist.png",
        "Relative improvement histogram (per input)",
    )




# Entry point
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results/03_black_box_faithfulness")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Exact Shapley per input: keep moderate
    n_inputs = 100

    train_dataset, _, d = load_california_housing(device=device)

    model = MLPFaithfulness(d=d, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(results_dir / "mlp_california_housing.pt", map_location=device))
    model.eval()

    # Enumerate all non-empty coalitions grouped by size k
    coalitions_by_k = {
        k: list(itertools.combinations(range(d), k))
        for k in range(1, d + 1)
    }


    # Trial A: Zero baseline (b = 0 in x-space), Φ computed for F
    run_trial(
        trial_name="zero_baseline",
        trial_long_name="Faithfulness vs coalition size (zero baseline, original model F)",
        model_identity_str="original model F(x)",
        baseline_identity_str="b = 0 (x-space)",
        x_transform_fn=lambda x, b: x,  # x_eff = x
        mask_fn=mask_input_baseline_xspace,  # masking in x-space → returns x_S
        baseline_fn=lambda ds: torch.zeros_like(ds[0][0]),
        train_dataset=train_dataset,
        model_F=model,
        d=d,
        coalitions_by_k=coalitions_by_k,
        results_dir=results_dir,
        n_inputs=n_inputs,
        device=device,
        compute_phi_on_shifted_model=False,
    )

    # Trial B: Median baseline (b = median(X_train) in x-space),
    #          centered coordinates δ = x - b,
    #          Φ computed for shifted model G(δ)=F(b+δ)
    def median_baseline(ds) -> torch.Tensor:
        """
        Compute coordinate-wise median baseline b from the training dataset (x-space).
        """
        X = torch.stack([ds[i][0] for i in range(len(ds))])  # (n, d)
        return X.median(dim=0).values

    run_trial(
        trial_name="median_baseline_centered",
        trial_long_name="Faithfulness vs coalition size (median-centered baseline, shifted model G(δ)=F(b+δ))",
        model_identity_str=r"shifted model $G(\delta)=F(b+\delta)$",
        baseline_identity_str=r"b = median(X_train) (x-space), with centered coordinates $\delta=x-b$",
        x_transform_fn=lambda x, b: x - b,  # x_eff = δ
        mask_fn=mask_input_centered_then_map_back,  # mask in δ-space, map back → returns x_S
        baseline_fn=median_baseline,
        train_dataset=train_dataset,
        model_F=model,
        d=d,
        coalitions_by_k=coalitions_by_k,
        results_dir=results_dir,
        n_inputs=n_inputs,
        device=device,
        compute_phi_on_shifted_model=True,
    )


if __name__ == "__main__":
    main()
