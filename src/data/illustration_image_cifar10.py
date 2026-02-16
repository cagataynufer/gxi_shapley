from pathlib import Path
from typing import Tuple, Literal

import torch
from torchvision import datasets, transforms



# CIFAR-10 normalization constants (canonical)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)



# Deterministic transform (no augmentation)
CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])



# Dataset loader
def load_cifar10(
    split: Literal["train", "test"],
    root: str | Path = "./data",
    download: bool = True,
) -> datasets.CIFAR10:
    """
    Load CIFAR-10 with deterministic preprocessing.

    Parameters
    ----------
    split : {"train", "test"}
        Dataset split to load.
    root : str or Path
        Root directory for dataset storage.
    download : bool
        Whether to download the dataset if not present.

    Returns
    -------
    torchvision.datasets.CIFAR10
        CIFAR-10 dataset instance.
    """
    return datasets.CIFAR10(
        root=root,
        train=(split == "train"),
        transform=CIFAR10_TRANSFORM,
        download=download,
    )



# Convenience helper: single sample (for illustration)
def get_cifar10_sample(
    index: int,
    split: Literal["train", "test"] = "test",
    root: str | Path = "./data",
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, int]:
    """
    Return a single CIFAR-10 sample without batch dimension.

    Intended for illustration and attribution analysis.

    Returns
    -------
    x : torch.Tensor
        Image tensor of shape (3, 32, 32)
    y : int
        Class label
    """
    dataset = load_cifar10(
        split=split,
        root=root,
        download=True,
    )

    x, y = dataset[index]

    if device is not None:
        x = x.to(device)

    return x, y
