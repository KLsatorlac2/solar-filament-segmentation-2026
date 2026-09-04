"""
U-Net++ for binary solar filament segmentation.

Input:
    [B, 3, H, W]

Output:
    [B, 1, H, W]

The model uses nested dense skip connections to improve the fusion
of low-level spatial information and high-level semantic features.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolution blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double convolution block."""
        return self.block(x)


class UNetPlusPlus(nn.Module):
    """
    Lightweight U-Net++ for binary image segmentation.

    Parameters
    ----------
    in_channels:
        Number of input image channels.

    out_channels:
        Number of output segmentation channels.

    features:
        Number of channels at each encoder level.
        The default configuration is (32, 64, 128, 256).

    Notes
    -----
    The model outputs raw logits. Apply sigmoid externally when
    probabilities are required.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()

        if len(features) != 4:
            raise ValueError(
                "UNetPlusPlus requires exactly four feature levels."
            )

        f0, f1, f2, f3 = features

        # =========================================================
        # Encoder
        # =========================================================

        self.encoder0 = DoubleConv(in_channels, f0,)
        self.encoder1 = DoubleConv(f0, f1,)
        self.encoder2 = DoubleConv(f1, f2,)
        self.encoder3 = DoubleConv(f2, f3,)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2,)

        # =========================================================
        # Bottleneck
        # =========================================================

        self.bottleneck = DoubleConv(f3, f3 * 2,)

        # =========================================================
        # Nested skip connections
        #
        # x_{i,j}:
        #   i -> encoder depth
        #   j -> nested decoder depth
        #
        # Example:
        #
        #   x00 -> x01 -> x02 -> x03 -> x04
        #     |      |      |      |
        #   x10 -> x11 -> x12 -> x13
        #     |      |      |
        #   x20 -> x21 -> x22
        #     |      |
        #   x30 -> x31
        #     |
        #   x40
        # =========================================================

        self.conv01 = DoubleConv(f0 + f1, f0,)
        self.conv11 = DoubleConv(f1 + f2, f1,)
        self.conv21 = DoubleConv(f2 + f3, f2,)
        self.conv31 = DoubleConv(f3 + f3 * 2, f3,)

        self.conv02 = DoubleConv(f0 * 2 + f1, f0,)
        self.conv12 = DoubleConv(f1 * 2 + f2, f1,)
        self.conv22 = DoubleConv(f2 * 2 + f3, f2,)

        self.conv03 = DoubleConv(f0 * 3 + f1, f0,)
        self.conv13 = DoubleConv(f1 * 3 + f2, f1,)

        self.conv04 = DoubleConv(f0 * 4 + f1, f0,)

        # =========================================================
        # Segmentation head
        # =========================================================

        self.head = nn.Conv2d(f0, out_channels, kernel_size=1,)

    @staticmethod
    def _upsample(
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Resize x to the spatial resolution of target.
        """

        return F.interpolate(
            x,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _concat(
        *tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate feature maps along the channel dimension."""

        return torch.cat(tensors, dim=1,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor with shape [B, C, H, W].

        Returns
        -------
        torch.Tensor
            Segmentation logits with shape [B, 1, H, W].
        """

        # =========================================================
        # Encoder
        # =========================================================

        x00 = self.encoder0(x)
        x10 = self.encoder1(self.pool(x00))
        x20 = self.encoder2(self.pool(x10))
        x30 = self.encoder3(self.pool(x20))
        x40 = self.bottleneck(self.pool(x30))

        # =========================================================
        # Nested decoder
        # =========================================================

        # ---------------------------------------------------------
        # Level 3
        # ---------------------------------------------------------

        x31 = self.conv31(
            self._concat(
                x30,
                self._upsample(x40, x30),
            )
        )

        # ---------------------------------------------------------
        # Level 2
        # ---------------------------------------------------------

        x21 = self.conv21(
            self._concat(
                x20,
                self._upsample(x30, x20),
            )
        )

        x22 = self.conv22(
            self._concat(
                x20,
                x21,
                self._upsample(x31, x20),
            )
        )

        # ---------------------------------------------------------
        # Level 1
        # ---------------------------------------------------------

        x11 = self.conv11(
            self._concat(
                x10,
                self._upsample(x20, x10),
            )
        )

        x12 = self.conv12(
            self._concat(
                x10,
                x11,
                self._upsample(x21, x10),
            )
        )

        x13 = self.conv13(
            self._concat(
                x10,
                x11,
                x12,
                self._upsample(x22, x10),
            )
        )

        # ---------------------------------------------------------
        # Level 0
        # ---------------------------------------------------------

        x01 = self.conv01(
            self._concat(
                x00,
                self._upsample(x10, x00),
            )
        )

        x02 = self.conv02(
            self._concat(
                x00,
                x01,
                self._upsample(x11, x00),
            )
        )

        x03 = self.conv03(
            self._concat(
                x00,
                x01,
                x02,
                self._upsample(x12, x00),
            )
        )

        x04 = self.conv04(
            self._concat(
                x00,
                x01,
                x02,
                x03,
                self._upsample(x13, x00),
            )
        )

        # =========================================================
        # Segmentation output
        # =========================================================

        return self.head(x04)
