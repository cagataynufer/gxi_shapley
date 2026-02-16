"""
Train the MLP black-box model for the faithfulness experiment.

This script trains a fixed multilayer perceptron on the California Housing
dataset and saves the trained model to disk. The trained model serves as
the black-box function whose explanations are evaluated in subsequent
experiment stages.

This script is intentionally self-contained and performs no attribution
or evaluation beyond reporting final train and test error.
"""

from pathlib import Path
import torch

from src.data.experiment_03_california_housing import load_california_housing
from src.utils.experiment_03_mlp_training import train_mlp_faithfulness


def main() -> None:
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path("results/03_black_box_faithfulness")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "mlp_california_housing.pt"

    # Load data
    train_dataset, test_dataset, d = load_california_housing(device=device)

    # Train model
    model, train_mse, test_mse = train_mlp_faithfulness(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        d=d,
        hidden_dim=64,
        batch_size=128,
        lr=1e-3,
        epochs=300,
        seed=0,
        device=device,
        save_path=model_path,
    )

    # Report results
    print("\nTraining completed.")
    print(f"Final train MSE: {train_mse:.6f}")
    print(f"Final test  MSE: {test_mse:.6f}")
    print(f"Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
