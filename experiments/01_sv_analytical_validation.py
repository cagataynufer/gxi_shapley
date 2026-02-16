"""
Analytical validation of matrix-valued Shapley attributions
for a quadratic model with closed-form ground truth.

This experiment compares:
- exact (permutation-based) Shapley values, and
- analytically derived Shapley values,

and reports their agreement for a single sample as well as
aggregated validation over multiple samples.

All outputs are written to disk for reproducibility.
"""

from __future__ import annotations

import sys
import itertools
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.experiment_01_quadratic_model import QuadraticModel


# Synthetic data
def generate_data(n: int, d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)



# Masking operator
def masked_x(x, baseline, S):
    xS = baseline.clone()
    for i in S:
        xS[i] = x[i]
    return xS


# Gradient of the quadratic model
def grad_f(x, model):
    g = torch.zeros_like(x)
    g[0] = model.a0 + model.b01 * x[1] + model.b03 * x[3]
    g[1] = model.a1 + model.b01 * x[0]
    g[2] = model.a2
    g[3] = model.a3 + model.b03 * x[0]
    g[4] = model.a4
    return g



# GXI game payoff
def nu_i(i, S, x, baseline, model):
    if i not in S:
        return 0.0
    xS = masked_x(x, baseline, S)
    return (xS[i] * grad_f(xS, model)[i]).item()



# Exact Shapley matrix (permutation definition)
def shapley_matrix(x, baseline, model):
    d = x.numel()
    players = list(range(d))
    perms = list(itertools.permutations(players))
    Phi = torch.zeros(d, d)

    for i in players:
        for j in players:
            acc = 0.0
            for pi in perms:
                S = set()
                for p in pi:
                    if p == j:
                        break
                    S.add(p)
                acc += nu_i(i, S | {j}, x, baseline, model) - nu_i(i, S, x, baseline, model)
            Phi[i, j] = acc / len(perms)

    return Phi



# Analytical Shapley matrix (closed form)
def analytical_phi(x, baseline, model):
    d = x.numel()

    beta = torch.tensor([model.a0, model.a1, model.a2, model.a3, model.a4])
    B = torch.zeros(d, d)
    B[0, 1] = model.b01
    B[1, 0] = model.b01
    B[0, 3] = model.b03
    B[3, 0] = model.b03

    Phi = torch.zeros(d, d)

    for i in range(d):
        s = sum(B[k, i] * (x[k] + baseline[k]) / 2 for k in range(d) if k != i)
        Phi[i, i] = (x[i] - baseline[i]) * (beta[i] + s)

    for i in range(d):
        for j in range(d):
            if i != j:
                Phi[i, j] = B[j, i] * (x[j] - baseline[j]) * (x[i] + baseline[i]) / 2

    return Phi



# Experiment Execution
# --------------------------------------------------
# Exact recovery validation statistics
#
# For each input x^(k), we compute the error matrix
#
#     E^(k) = Phi_num(x^(k)) - Phi_ana(x^(k)),
#
# and summarize agreement using supremum-type norms.
# Since the analytical and numeric attributions are
# theoretically identical, any discrepancy is due
# solely to floating-point precision.
#
# The reported quantities are:
#
# - Global max error:
#       max_{k,i,j} |E^(k)_{ij}|
#
# - Diagonal max error (main effects):
#       max_{k,i} |E^(k)_{ii}|
#
# - Off-diagonal max error (interactions):
#       max_{k,i≠j} |E^(k)_{ij}|
#
# - Frobenius norm max:
#       max_k ||E^(k)||_F
#
# These diagnostics verify exact recovery uniformly
# over all inputs, up to numerical precision.
# --------------------------------------------------

def main():
    results_dir = REPO_ROOT / "results" / "01_sv_analytical_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    model = QuadraticModel().eval()
    baseline = torch.zeros(5)

    X = generate_data(50, 5, seed=0)
    x0 = X[0]

    # --- single + mean matrices
    Phi_single = shapley_matrix(x0, baseline, model)
    Phi_mean = torch.stack(
        [shapley_matrix(x, baseline, model) for x in X]
    ).mean(dim=0)
    Phi_ana = analytical_phi(x0, baseline, model)


    # Exact recovery validation (max-based)
    errors = []
    for x in X:
        Phi_num = shapley_matrix(x, baseline, model)
        Phi_ana_x = analytical_phi(x, baseline, model)
        errors.append(Phi_num - Phi_ana_x)

    E = torch.stack(errors)
    abs_E = E.abs()

    d = E.shape[-1]
    diag_mask = torch.eye(d, dtype=bool)
    off_mask = ~diag_mask

    global_max_err = abs_E.max().item()
    diag_max_err = abs_E[:, diag_mask].max().item()
    off_max_err = abs_E[:, off_mask].max().item()
    fro_max_err = torch.linalg.norm(E, dim=(1, 2)).max().item()


    # TXT output
    out = []
    out.append("Matrix-Valued Shapley Attribution — Quadratic Model\n\n")
    out.append("Shapley attribution matrix Φ (single sample)\n")
    out.append(str(Phi_single.numpy()) + "\n\n")
    out.append("Mean Shapley attribution matrix Φ (50 samples)\n")
    out.append(str(Phi_mean.numpy()) + "\n\n")

    out.append("Analytical vs Numeric Φ (single sample)\n")
    out.append("i  j   numeric_phi   analytical_phi   abs_error\n")

    for i in range(d):
        for j in range(d):
            num = Phi_single[i, j].item()
            ana = Phi_ana[i, j].item()
            out.append(
                f"{i}  {j}  {num: .6f}    {ana: .6f}    {abs(num-ana):.2e}\n"
            )

    out.append("\n")
    out.append("Exact Recovery Validation (50 random inputs)\n")
    out.append(f"Global max |Φ_num − Φ_ana|:        {global_max_err:.2e}\n")
    out.append(f"Diagonal max error:               {diag_max_err:.2e}\n")
    out.append(f"Off-diagonal max error:           {off_max_err:.2e}\n")
    out.append(f"Max Frobenius norm deviation:     {fro_max_err:.2e}\n")

    out_path = results_dir / "phi_results.txt"
    out_path.write_text("".join(out))


    # LaTeX output
    latex_table = r"""
    \begin{table}[t]
    \centering
    \caption{Exact recovery of matrix-valued Shapley attributions for the quadratic model (50 random inputs).}
    \label{tab:shapley_exact_validation}
    \begin{tabular}{l c}
    \toprule
    Quantity & Max error \\
    \midrule
    Global $\max |\Phi_{\mathrm{num}} - \Phi_{\mathrm{ana}}|$ & %.2e \\
    Diagonal entries & %.2e \\
    Off-diagonal entries & %.2e \\
    Frobenius norm & %.2e \\
    \bottomrule
    \end{tabular}
    \end{table}
    """ % (
        global_max_err,
        diag_max_err,
        off_max_err,
        fro_max_err,
    )

    latex_path = results_dir / "exact_validation_table.tex"
    latex_path.write_text(latex_table)

    print(f"Saved TXT results to:   {out_path.resolve()}")
    print(f"Saved LaTeX table to:  {latex_path.resolve()}")

if __name__ == "__main__":
    main()
