# Bridging Gradient and Perturbation-Based Explainability

### A Cooperative Game Approach

This repository contains the official implementation accompanying the Master's thesis:

**“Bridging Gradient and Perturbation-Based Explainability: A Cooperative Game Approach.”**

The work introduces a derivative-informed Shapley framework that integrates coalition-based feature masking with Gradient × Input attributions to produce matrix-valued explanations capturing cross-feature sensitivity structure.

---

## Repository Structure

```
experiments/    → Quantitative empirical studies (faithfulness, surrogate evaluation)
illustration/   → Image and text modality demonstrations
src/            → Core Gradient×Input (GXI) Shapley framework implementation

### src/ overview
src/gxi/        → GXI game definition + attribution computation
                 - game.py: feature-specific GXI Shapley game payoff
                 - attribution_matrix_computer.py: computes Φ 
                 - model_adapter.py: wraps black-box models for differentiable evaluation

src/masking/    → Baseline masking + modality-specific masking utilities
                 - base.py: shared masking interface
                 - image.py / text.py / tabular.py: modality-specific masking

src/data/       → Data loading / generation utilities used by experiments
src/models/     → Model definitions used in experiments (MLP, CNN, etc.)
src/utils/      → Training and experiment utilities (training scripts, helpers)
```

All commands below are executed **from the repository root**.

---

# Environment Setup

Using pip:

```
pip install -r requirements.txt
```

Using conda:

```
conda env create -f environment.yml
conda activate gxi-thesis
```

---

# Execution Order

The experiments and illustrations are independent but follow the logical order of the thesis.

---

## 1. Analytical Validation (Quadratic Reference Model)

Exact recovery of main and pairwise interaction effects.

```
python -m experiments.01_sv_analytical_validation
```

---

## 2. Coalition-Path Analysis (Synthetic Nonlinear Model)

Evaluation of GXI payoffs along structured coalition paths in a trained MLP.

```
python -m experiments.02_gxi_coalition_paths
```

---

## 3. Faithfulness Evaluation (Black-Box Model)

This experiment requires two stages.

### Step 1 — Train the MLP

```
python -m experiments.03_black_box_faithfulness.train_mlp
```

### Step 2 — Run faithfulness analysis

```
python -m experiments.03_black_box_faithfulness.run_experiment
```

This produces coalition-based surrogate faithfulness evaluations.

---

# Illustrations

These demonstrations visualize the attribution matrix in structured modalities.

---

## Image Illustration (CNN on CIFAR-10)

### Step 1 — Train CNN

```
python -m illustration.image.train_cnn_cifar10
```

### Step 2 — Generate attribution visualization

```
python -m illustration.image.image_illustration
```

---

## Text Illustration

Token-level attribution analysis using the GXI Shapley framework.

```
python -m illustration.text.illustration_text
```

---

# Notes

* All commands assume execution from the repository root.
* Experiments are deterministic where applicable (fixed seeds).
* Outputs (figures, matrices, trained models) are provided in the
`results` branch of this repository.


---

# Contact

Çağatay Nüfer
C.Nuefer@campus.lmu.de
M.Sc. Statistics & Data Science
Ludwig-Maximilians-Universität München

---
