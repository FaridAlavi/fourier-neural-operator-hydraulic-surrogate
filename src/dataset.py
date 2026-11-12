"""
PyTorch Dataset and Normalization utilities for 1D Saint-Venant FNO surrogate.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class UnitGaussianNormalizer:
    """
    Normalizes a tensor with mean and standard deviation along batch dimension.
    """

    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None, eps: float = 1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def fit(self, x: torch.Tensor) -> "UnitGaussianNormalizer":
        # Compute mean and std over batch (dim 0) and space/time dims, keeping channel dim intact
        # Expected input shape: (N, Nx, Nt, C) or (N, ..., C)
        dims = tuple(range(x.ndim - 1))
        self.mean = torch.mean(x, dim=dims, keepdim=True)
        self.std = torch.std(x, dim=dims, keepdim=True)
        # Avoid division by zero
        self.std = torch.clamp(self.std, min=self.eps)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fit before calling encode.")
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fit before calling decode.")
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std, "eps": torch.tensor(self.eps)}

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state["eps"].item() if isinstance(state["eps"], torch.Tensor) else state["eps"]


class SaintVenantDataset(Dataset):
    """
    Spatio-temporal dataset loader for 1D Saint-Venant trajectories.

    Constructs 2D space-time input feature grid:
      Channels (in_channels = 6):
        0: H0(x) broadcast across time t
        1: Q0(x) broadcast across time t
        2: q_up(t) broadcast across space x
        3: h_down(t) broadcast across space x
        4: Normalized spatial coordinate x / L
        5: Normalized temporal coordinate t / T

    Target:
      Channels (out_channels = 2):
        0: H(x, t) water levels
        1: Q(x, t) discharges
    """

    def __init__(self, x_data: torch.Tensor, y_data: torch.Tensor):
        self.x_data = x_data.float()  # (N, nx, nt, 6)
        self.y_data = y_data.float()  # (N, nx, nt, 2)

    def __len__(self) -> int:
        return self.x_data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[idx], self.y_data[idx]


def build_spatiotemporal_tensors(
    data_path: Union[str, Path]
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray]]:
    """
    Loads raw NPZ file and builds 2D space-time tensors for FNO.
    """
    raw = np.load(data_path)
    H = raw["H"]          # (N, nx, nt)
    Q = raw["Q"]          # (N, nx, nt)
    q_up = raw["q_up"]    # (N, nt)
    h_down = raw["h_down"]# (N, nt)
    H0 = raw["H0"]        # (N, nx)
    Q0 = raw["Q0"]        # (N, nx)
    x = raw["x"]          # (nx,)
    t = raw["t"]          # (nt,)

    n_samples, nx, nt = H.shape
    length = float(raw["channel_length"]) if "channel_length" in raw else float(x[-1])
    total_time = float(t[-1]) if t[-1] > 0 else 1.0

    # Build input tensor of shape (N, nx, nt, 6)
    inputs = np.zeros((n_samples, nx, nt, 6), dtype=np.float32)

    # 0: H0(x) broadcast over t
    inputs[:, :, :, 0] = H0[:, :, None]
    # 1: Q0(x) broadcast over t
    inputs[:, :, :, 1] = Q0[:, :, None]
    # 2: q_up(t) broadcast over x
    inputs[:, :, :, 2] = q_up[:, None, :]
    # 3: h_down(t) broadcast over x
    inputs[:, :, :, 3] = h_down[:, None, :]
    # 4: x coordinate grid [0, 1]
    x_norm = (x / length).reshape(1, nx, 1)
    inputs[:, :, :, 4] = np.repeat(np.repeat(x_norm, n_samples, axis=0), nt, axis=2)
    # 5: t coordinate grid [0, 1]
    t_norm = (t / total_time).reshape(1, 1, nt)
    inputs[:, :, :, 5] = np.repeat(np.repeat(t_norm, n_samples, axis=0), nx, axis=1)

    # Build target tensor of shape (N, nx, nt, 2)
    targets = np.zeros((n_samples, nx, nt, 2), dtype=np.float32)
    targets[:, :, :, 0] = H
    targets[:, :, :, 1] = Q

    meta = {
        "x": x,
        "t": t,
        "length": length,
        "total_time": total_time,
        "nx": nx,
        "nt": nt,
    }

    return torch.from_numpy(inputs), torch.from_numpy(targets), meta


def get_split_indices(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic, mutually exclusive train/validation/test indices."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if train_ratio < 0.0 or val_ratio < 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to less than 1.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def get_dataloaders(
    data_path: Union[str, Path],
    batch_size: int = 16,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, UnitGaussianNormalizer, UnitGaussianNormalizer, Dict]:
    """
    Splits data into train/val/test, computes normalizers, and returns DataLoaders.
    """
    torch.manual_seed(seed)
    inputs, targets, meta = build_spatiotemporal_tensors(data_path)
    n_samples = inputs.shape[0]
    train_idx, val_idx, test_idx = get_split_indices(
        n_samples=n_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    x_train, y_train = inputs[train_idx], targets[train_idx]
    x_val, y_val = inputs[val_idx], targets[val_idx]
    x_test, y_test = inputs[test_idx], targets[test_idx]

    # Fit normalizers only on training partition
    x_normalizer = UnitGaussianNormalizer().fit(x_train)
    y_normalizer = UnitGaussianNormalizer().fit(y_train)

    train_ds = SaintVenantDataset(x_normalizer.encode(x_train), y_normalizer.encode(y_train))
    val_ds = SaintVenantDataset(x_normalizer.encode(x_val), y_normalizer.encode(y_val))
    test_ds = SaintVenantDataset(x_normalizer.encode(x_test), y_normalizer.encode(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, x_normalizer, y_normalizer, meta
