from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
    device: torch.device,
    seed: int = 0,
    save_path: Optional[Path] = None,
) -> Tuple[nn.Module, float, float]:
    """
    Train a convolutional neural network for classification.

    This function is intentionally generic and model-agnostic.
    It performs standard supervised training and reports final
    train loss and test accuracy.

    Parameters
    ----------
    model : nn.Module
        CNN model to train.
    train_loader : DataLoader
        Training data loader.
    test_loader : DataLoader
        Test data loader.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    momentum : float
        SGD momentum.
    weight_decay : float
        Weight decay.
    step_size : int
        Step size for learning rate scheduler.
    gamma : float
        Learning rate decay factor.
    device : torch.device
        Device to run training on.
    seed : int, optional
        Random seed for reproducibility.
    save_path : Path, optional
        If provided, saves the trained model state_dict.

    Returns
    -------
    model : nn.Module
        Trained model.
    final_train_loss : float
        Loss from the last training batch.
    test_accuracy : float
        Final test accuracy in [0, 1].
    """

    # Reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    criterion = nn.CrossEntropyLoss()


    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()

        scheduler.step()

        print(
            f"Epoch {epoch + 1:03d}/{epochs} "
            f"- train loss: {running_loss:.4f}"
        )

    final_train_loss = running_loss


    # Evaluation loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    test_accuracy = correct / total if total > 0 else 0.0

    print(f"Final test accuracy: {test_accuracy * 100:.2f}%")


    # Save model (optional)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to: {save_path.resolve()}")

    return model, final_train_loss, test_accuracy
