from __future__ import annotations

import torch

from .base import AbstractMasker


"""
Image masking utilities for GXI-Shapley.

This module defines a masker for image inputs based on fixed superpixel
partitions. Each Shapley player corresponds to one rectangular block
(superpixel) in the image, which is replaced by a baseline patch when masked.
"""


class ImageMasker(AbstractMasker):
    """
    Masker for image inputs using fixed superpixel partitions.

    Each player corresponds to one superpixel block of size
    (sp_height × sp_width) across all channels. When a player is masked,
    the corresponding image region is replaced by a baseline patch.

    This design supports attribution over spatial regions while remaining
    fully model-agnostic.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        channels: int,
        sp_height: int,
        sp_width: int,
        baseline: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        image_height, image_width:
            Spatial dimensions of the input image.

        channels:
            Number of image channels.

        sp_height, sp_width:
            Height and width of each superpixel block.

        baseline:
            Baseline image tensor of shape (C, H, W), used to replace
            masked superpixels.
        """
        self.H = image_height
        self.W = image_width
        self.C = channels
        self.sp_h = sp_height
        self.sp_w = sp_width
        self.baseline = baseline

        # Number of superpixels along each spatial dimension
        self.n_sp_h = self.H // self.sp_h
        self.n_sp_w = self.W // self.sp_w

        # Total number of Shapley players
        self.n_players = self.n_sp_h * self.n_sp_w


        # Precompute spatial slices for each superpixel
        # Each entry in pixel_slices corresponds to one player
        # and specifies the spatial region it controls.
        self.pixel_slices = []
        for i in range(self.n_sp_h):
            for j in range(self.n_sp_w):
                h_start = i * self.sp_h
                h_end = h_start + self.sp_h
                w_start = j * self.sp_w
                w_end = w_start + self.sp_w
                self.pixel_slices.append(
                    (h_start, h_end, w_start, w_end)
                )

    def mask(self, x: torch.Tensor, coalition: list[bool]) -> torch.Tensor:
        """
        Apply a superpixel-based mask to the input image.

        Parameters
        ----------
        x:
            Image tensor of shape (C, H, W) representing a single instance.

        coalition:
            Boolean list indicating which superpixels are present.

        Returns
        -------
        masked_x:
            Image tensor of shape (C, H, W) with masked superpixels replaced
            by the corresponding baseline regions.
        """
        # Work on a copy to avoid modifying the original input
        x = x.clone()

        for idx, keep in enumerate(coalition):
            if not keep:
                h_start, h_end, w_start, w_end = self.pixel_slices[idx]
                x[:, h_start:h_end, w_start:w_end] = (
                    self.baseline[:, h_start:h_end, w_start:w_end]
                )

        return x
