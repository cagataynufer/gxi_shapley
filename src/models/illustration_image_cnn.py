import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    Simple convolutional neural network for CIFAR-10.

    This architecture is intentionally lightweight and mirrors the model
    used in the original notebook. It is used to obtain a non-degenerate
    image model for illustration purposes.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 3, 32, 32)

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
