"""Transformer model for vibration-based bearing fault classification (US-03).

Architecture
------------
Input (B, C, W)
  → Input Projection: Linear(C → d_model) per time-step
    (each time-step's C-channel vector is treated as a token)
  → Positional Encoding
  → N × TransformerEncoderLayer (d_model, nhead, dim_feedforward, dropout)
  → Global Average Pooling over sequence dimension
  → Linear(d_model → n_classes)

Hyperparameter defaults (per spec):
  d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1

Input shape convention: (batch, n_channels, window_size)
  - Each of the `window_size` time steps has `n_channels` features.
  - We project each time-step vector (C-dim) → d_model-dim before the encoder.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017).

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length (window_size).
    dropout : float
        Dropout rate applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model) — broadcast over batch
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (seq_len, batch, d_model)

        Returns
        -------
        Tensor shape (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer model
# ---------------------------------------------------------------------------

class VibrationTransformer(nn.Module):
    """Transformer encoder for vibration fault classification.

    Parameters
    ----------
    n_channels : int
        Number of accelerometer channels (4 for EGB-250).
    window_size : int
        Number of time-steps per window (4096).
    n_classes : int
        Number of output classes (4).
    d_model : int
        Embedding dimension (default 128).
    nhead : int
        Number of attention heads (default 8).
    num_layers : int
        Number of TransformerEncoderLayer stacks (default 4).
    dim_feedforward : int
        Hidden size of the feedforward sub-layer (default 256).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        n_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Store spec attributes for inspection / tests
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.window_size = window_size

        # Input projection: map each time-step's C-dim vector → d_model
        self.input_proj = nn.Linear(n_channels, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=window_size, dropout=dropout)

        # Transformer encoder stack — batch_first=True: (B, seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head: global avg pool → linear
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (batch, n_channels, window_size)

        Returns
        -------
        Tensor shape (batch, n_classes) — raw logits
        """
        # x: (B, C, W) → (B, W, C)
        x = x.permute(0, 2, 1)

        # Project: (B, W, C) → (B, W, d_model)
        x = self.input_proj(x)

        # Positional encoding expects (W, B, d_model) → permute, add PE, permute back
        x = x.permute(1, 0, 2)   # (W, B, d_model)
        x = self.pos_enc(x)       # (W, B, d_model)
        x = x.permute(1, 0, 2)   # (B, W, d_model)

        # Transformer encoder: (B, W, d_model)
        x = self.transformer_encoder(x)

        # Global average pooling over sequence dimension: (B, d_model)
        x = x.mean(dim=1)

        # Classification: (B, n_classes)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Monitor validation loss and signal when training should stop.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs with no improvement before stopping (default 10).
    min_delta : float
        Minimum decrease in val_loss to be counted as improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> None:
        """Update state with the latest validation loss."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from numpy arrays.

    Parameters
    ----------
    X_train, X_val : np.ndarray
        Shape (n, C, W), dtype float32 — already normalised.
    y_train, y_val : np.ndarray
        Shape (n,), dtype int64.
    batch_size : int
    num_workers : int
    seed : int
        Generator seed for reproducible train shuffling.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    # Cast float64 → float32 to avoid dtype mismatch with model weights
    if X_train.dtype == np.float64:
        X_train = X_train.astype(np.float32)
    if X_val.dtype == np.float64:
        X_val = X_val.astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Epoch step (train or eval)
# ---------------------------------------------------------------------------

def epoch_step(
    model: VibrationTransformer,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    train: bool,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Run one epoch (train or eval) and return (mean_loss, accuracy).

    Parameters
    ----------
    model : VibrationTransformer
    loader : DataLoader
    loss_fn : nn.Module
        CrossEntropyLoss instance.
    optimizer : Optimizer or None
        Must be provided when train=True.
    train : bool
        If True, update model parameters; else eval mode, no grad.
    device : torch.device or None
        Device to move batches to. Defaults to model's first parameter device.

    Returns
    -------
    Tuple[float, float]
        (mean_loss_over_batches, accuracy)
    """
    if train and optimizer is None:
        raise ValueError("optimizer must be provided when train=True")

    if device is None:
        device = next(model.parameters()).device

    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += x_batch.size(0)

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return mean_loss, accuracy
