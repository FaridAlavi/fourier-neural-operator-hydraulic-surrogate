"""
2D Spatio-Temporal Fourier Neural Operator (FNO) Architecture for 1D Saint-Venant PDE.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """
    2D Fourier Spectral Convolution Layer.
    Computes FFT2, multiplies lower modes by learnable complex weights, and performs IFFT2.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Spatial modes (x)
        self.modes2 = modes2  # Temporal modes (t)

        # Scale weights initialization
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Complex multiplication: (batch, in_channel, x, y) * (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_channels, size_x, size_y)
        batchsize = x.shape[0]
        size_x = x.shape[2]
        size_y = x.shape[3]

        # Compute 2D Real-to-Complex FFT
        x_ft = torch.fft.rfft2(x)

        # Truncated frequency output tensor
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            size_x,
            size_y // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        # Multiply relevant Fourier modes
        m1 = min(self.modes1, size_x // 2)
        m2 = min(self.modes2, size_y // 2 + 1)

        # Top-left corner (positive frequencies)
        out_ft[:, :, :m1, :m2] = self._compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
        )
        # Bottom-left corner (negative spatial frequencies)
        out_ft[:, :, -m1:, :m2] = self._compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )

        # Inverse 2D Real FFT
        x_out = torch.fft.irfft2(out_ft, s=(size_x, size_y))
        return x_out


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for spatio-temporal dynamics (x, t).
    
    Structure:
      - Lifting Layer: Maps (in_channels) -> (width)
      - Domain padding: extends x and t so the FFT is not forced to treat a
        non-periodic reach and a non-periodic hydrograph as periodic
      - Spectral Conv Blocks: 4 x (SpectralConv2d + 1x1 Conv skip + GELU)
      - Projection Head: Maps (width) -> 128 -> (out_channels)
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 2,
        modes_x: int = 16,
        modes_t: int = 24,
        width: int = 64,
        num_layers: int = 4,
        padding_frac: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_t = modes_t
        self.width = width
        self.num_layers = num_layers
        # Fraction of the grid appended along each axis; scales with resolution so
        # the padded physical extent is identical at every evaluation mesh.
        self.padding_frac = padding_frac

        # Lifting layer
        self.fc0 = nn.Linear(in_channels, width)

        # Spectral convolution & parallel bypass layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes_x, modes_t) for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)
        ])

        # Projection head
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    @staticmethod
    def _smooth_periodic_pad_2d(x: torch.Tensor, pad_x: int, pad_t: int) -> torch.Tensor:
        """
        Right-only smooth cosine periodic bridge along spatial (-2) and temporal (-1) axes.
        Smoothly transitions from the end of the domain (x=L, t=T) back to the start (x=0, t=0)
        so the FFT sees a continuous periodic signal without a boundary cliff.
        """
        if pad_x > 0:
            alpha_x = torch.linspace(
                1.0 / (pad_x + 1), pad_x / (pad_x + 1), pad_x, device=x.device, dtype=x.dtype
            )
            wx = 0.5 * (1.0 - torch.cos(math.pi * alpha_x)).view(1, 1, pad_x, 1)
            pad_block_x = (1.0 - wx) * x[:, :, -1:, :] + wx * x[:, :, :1, :]
            x = torch.cat([x, pad_block_x], dim=-2)

        if pad_t > 0:
            alpha_t = torch.linspace(
                1.0 / (pad_t + 1), pad_t / (pad_t + 1), pad_t, device=x.device, dtype=x.dtype
            )
            wt = 0.5 * (1.0 - torch.cos(math.pi * alpha_t)).view(1, 1, 1, pad_t)
            pad_block_t = (1.0 - wt) * x[:, :, :, -1:] + wt * x[:, :, :, :1]
            x = torch.cat([x, pad_block_t], dim=-1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, nx, nt, in_channels).

        Returns
        -------
        torch.Tensor
            Predicted state tensor of shape (batch, nx, nt, out_channels).
        """
        # Lifting: (B, nx, nt, in_channels) -> (B, nx, nt, width)
        x = self.fc0(x)
        # Permute to (B, width, nx, nt) for 2D convolutions & FFT2
        x = x.permute(0, 3, 1, 2)

        nx, nt = x.shape[-2], x.shape[-1]
        # (nx - 1) physical intervals + (pad_x + 1) bridge intervals = (nx + pad_x) total DFT intervals.
        # Setting pad_x = round(padding_frac * (nx - 1)) - 1 keeps the total DFT physical period
        # L_dft = (nx + pad_x) * dx = (1 + padding_frac) * L strictly invariant across spatial meshes.
        pad_x = max(0, round(self.padding_frac * (nx - 1)) - 1) if self.padding_frac > 0 else 0
        pad_t = max(0, round(self.padding_frac * nt)) if self.padding_frac > 0 else 0
        if pad_x > 0 or pad_t > 0:
            x = self._smooth_periodic_pad_2d(x, pad_x, pad_t)

        # Fourier Layers with Residual Skips
        for i in range(self.num_layers):
            x1 = self.spectral_layers[i](x)
            x2 = self.w_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)

        if pad_x > 0 or pad_t > 0:
            x = x[..., :nx, :nt]

        # Permute back to (B, nx, nt, width)
        x = x.permute(0, 2, 3, 1)

        # Projection Head
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
