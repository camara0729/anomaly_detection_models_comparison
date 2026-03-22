"""Preprocessing utilities for EGB-250 vibration data (US-02).

Pipeline
--------
1. sliding_window        — segment a run into overlapping windows
2. split_runs            — build train/val/test splits from a class directory
3. fit_scaler            — fit StandardScaler on flattened train windows
4. apply_scaler          — apply fitted scaler to any split
5. compute_statistical_features — 54-d feature vector per window (6 × 9 channels)
6. build_knn_graph       — k-NN graph per window using PyTorch Geometric

Feature order per channel (6 features):
    0: RMS, 1: kurtosis, 2: crest_factor, 3: peak_to_peak, 4: skewness, 5: zero_crossing_rate
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew as scipy_skew
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
from torch_geometric.data import Data

from src.data_loader import load_all_runs

RANDOM_SEED = 42
_N_FEATURES_PER_CHANNEL = 6  # RMS, kurtosis, crest_factor, peak_to_peak, skewness, zcr
_N_CHANNELS = 9


# ---------------------------------------------------------------------------
# 1. Sliding window
# ---------------------------------------------------------------------------

def sliding_window(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Segment a multi-channel signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray
        Shape (C, N) — C channels, N samples.
    window_size : int
        Number of samples per window.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    np.ndarray
        Shape (n_windows, C, window_size), dtype same as input.

    Raises
    ------
    ValueError
        If window_size <= 0 or stride <= 0.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    n_channels, n_samples = signal.shape
    if n_samples < window_size:
        return np.empty((0, n_channels, window_size), dtype=signal.dtype)

    n_windows = (n_samples - window_size) // stride + 1
    # Use stride_tricks for zero-copy view; copy to ensure contiguous float32
    shape = (n_windows, n_channels, window_size)
    strides = (
        signal.strides[1] * stride,  # advance stride samples along time axis
        signal.strides[0],            # channel axis
        signal.strides[1],            # sample axis within window
    )
    windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return np.ascontiguousarray(windows, dtype=signal.dtype)


# ---------------------------------------------------------------------------
# 2. Split runs
# ---------------------------------------------------------------------------

# Type alias: maps split names to lists of 1-indexed run numbers
SplitConfig = Dict[str, List[int]]

_SPLIT_SIZES: Tuple[int, int, int] = (10, 2, 3)  # train, val, test


def make_random_split_config(
    n_runs: int = 15,
    sizes: Tuple[int, int, int] = _SPLIT_SIZES,
    seed: int = 42,
) -> SplitConfig:
    """Shuffle runs randomly and assign to train/val/test by sizes.

    Parameters
    ----------
    n_runs : int
        Total number of runs per class (default 15).
    sizes : tuple of (int, int, int)
        Number of runs for (train, val, test). Must sum to n_runs and all > 0.
    seed : int
        Random seed for reproducibility.  Same seed always yields the same
        partition — use different seeds for robustness experiments.

    Returns
    -------
    SplitConfig
        ``{"train": [...], "val": [...], "test": [...]}`` with 1-indexed run numbers.

    Raises
    ------
    ValueError
        If sum(sizes) != n_runs or any size <= 0.
    """
    if any(s <= 0 for s in sizes):
        raise ValueError(f"All sizes must be > 0, got sizes={sizes}")
    if sum(sizes) != n_runs:
        raise ValueError(f"sum(sizes)={sum(sizes)} must equal n_runs={n_runs}, got sizes={sizes}")

    rng = np.random.RandomState(seed)
    shuffled = list(range(1, n_runs + 1))
    rng.shuffle(shuffled)

    n_train, n_val, n_test = sizes
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def split_runs(
    class_dir: str,
    label: int,
    window_size: int = 4096,
    stride: int = 2048,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a class directory, window each run, and split into train/val/test.

    Runs are shuffled with ``seed`` and assigned as 10 train / 2 val / 3 test.
    The same seed always produces the same partition; use different seeds for
    robustness experiments (same seed must be used for both models to ensure
    a fair comparison).

    Parameters
    ----------
    class_dir : str
        Path to a single fault class directory (e.g. ``"data/P1"``).
    label : int
        Integer class label (P1=0, P2=1, P3=2, P4=3).
    window_size : int
    stride : int
    seed : int
        Random seed for run shuffling. Default 42.

    Returns
    -------
    Tuple of (train_X, val_X, test_X, train_y, val_y, test_y) — all np.ndarray.
    X arrays have shape (n_windows, 9, window_size), y arrays shape (n_windows,).
    """
    config = make_random_split_config(seed=seed)
    runs = load_all_runs(class_dir)  # already in numeric order R1..R15

    def _windows_for_run_indices(run_indices: List[int]) -> np.ndarray:
        parts = [sliding_window(runs[i], window_size, stride) for i in run_indices]
        return np.concatenate(parts, axis=0)

    # Convert 1-indexed lists to 0-indexed
    train_X = _windows_for_run_indices([r - 1 for r in config["train"]])
    val_X = _windows_for_run_indices([r - 1 for r in config["val"]])
    test_X = _windows_for_run_indices([r - 1 for r in config["test"]])

    train_y = np.full(train_X.shape[0], label, dtype=np.int64)
    val_y = np.full(val_X.shape[0], label, dtype=np.int64)
    test_y = np.full(test_X.shape[0], label, dtype=np.int64)

    return train_X, val_X, test_X, train_y, val_y, test_y


# ---------------------------------------------------------------------------
# 3 & 4. Scaler (fit on train, apply to all splits)
# ---------------------------------------------------------------------------

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on flattened train windows.

    Parameters
    ----------
    X_train : np.ndarray
        Shape (n_windows, C, W).

    Returns
    -------
    sklearn.preprocessing.StandardScaler
        Fitted scaler. Must be applied with apply_scaler.
    """
    n = X_train.shape[0]
    X_flat = X_train.reshape(n, -1)  # (n_windows, C*W)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply a fitted StandardScaler to windows.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler from fit_scaler.
    X : np.ndarray
        Shape (n_windows, C, W).

    Returns
    -------
    np.ndarray
        Scaled array, same shape as X, dtype float32.
    """
    original_shape = X.shape
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(original_shape).astype(np.float32)


def save_scaler(scaler: StandardScaler, path: str) -> None:
    """Persist scaler to disk via pickle."""
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> StandardScaler:
    """Load a persisted scaler from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 5. Statistical features (54-d per window)
# ---------------------------------------------------------------------------

def _channel_features(ch_signal: np.ndarray) -> np.ndarray:
    """Compute 6 features for a single-channel 1-D signal."""
    rms = float(np.sqrt(np.mean(ch_signal**2)))
    peak = float(np.max(np.abs(ch_signal)))
    crest = peak / rms if rms > 0 else 0.0
    p2p = float(np.max(ch_signal) - np.min(ch_signal))
    zcr = float(np.sum(np.diff(np.sign(ch_signal)) != 0) / (len(ch_signal) - 1))
    # scipy kurtosis/skew return NaN for zero-variance (constant) signals;
    # guard: return 0.0 when the signal has no variation.
    if np.std(ch_signal) == 0:
        kurt, skewness = 0.0, 0.0
    else:
        kurt = float(scipy_kurtosis(ch_signal, fisher=True))
        skewness = float(scipy_skew(ch_signal))
    return np.array([rms, kurt, crest, p2p, skewness, zcr], dtype=np.float32)


def compute_statistical_features(windows: np.ndarray) -> np.ndarray:
    """Compute 54-d statistical feature vector for each window.

    Features per channel (6): RMS, kurtosis, crest_factor, peak_to_peak,
    skewness, zero_crossing_rate. Applied to all 9 channels → 54 features.

    Parameters
    ----------
    windows : np.ndarray
        Shape (n_windows, C, W) — C channels (9 for full Analog50k pipeline).

    Returns
    -------
    np.ndarray
        Shape (n_windows, 54), dtype float32.
    """
    n_windows, n_ch, _ = windows.shape
    features = np.zeros((n_windows, n_ch * _N_FEATURES_PER_CHANNEL), dtype=np.float32)
    for i in range(n_windows):
        for ch in range(n_ch):
            start = ch * _N_FEATURES_PER_CHANNEL
            features[i, start : start + _N_FEATURES_PER_CHANNEL] = _channel_features(windows[i, ch])
    return features


# ---------------------------------------------------------------------------
# 6. k-NN graph construction (PyTorch Geometric)
# ---------------------------------------------------------------------------

def save_split_config(config: SplitConfig, path: Union[str, Path]) -> None:
    """Persist a SplitConfig to JSON for experiment traceability.

    Serialises each split's run indices as a plain list so the file is
    human-readable and can be loaded without importing this module.

    Parameters
    ----------
    config : SplitConfig
        Dict mapping ``{"train": range(...), "val": range(...), "test": range(...)}``.
    path : str or Path
        Destination file (created along with any missing parent directories).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: list(v) for k, v in config.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


def build_knn_graph(features: np.ndarray, k: int = 8) -> List[Data]:
    """Build one k-NN graph per window using statistical features as node context.

    Each window becomes a self-contained graph with k+1 nodes: the central
    window (node 0) plus its k nearest neighbours (nodes 1..k).  Edges run
    from the central node to each neighbour (local indices 0→1, 0→2, …, 0→k).

    This design makes every Data object independent — edge_index uses local
    node indices (0..k), so PyG's Batch collation works correctly without
    index-out-of-range errors.

    Parameters
    ----------
    features : np.ndarray
        Shape (n_windows, F) — F features per window (54 for full Analog50k pipeline).
    k : int
        Number of nearest neighbours per node (default 8).

    Returns
    -------
    List[torch_geometric.data.Data]
        One Data object per window. Each has:
          - x          : (k+1, F) — features of central node + k neighbours
          - edge_index : (2, k)   — edges from node 0 to nodes 1..k (local)
    """
    n = features.shape[0]
    k_actual = min(k, n - 1)  # can't have more neighbours than n-1 nodes

    # algorithm="ball_tree" is deterministic — no random_state needed
    nbrs = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="ball_tree", metric="euclidean")
    nbrs.fit(features)
    _, indices = nbrs.kneighbors(features)
    # indices[:,0] is self — drop it
    indices = indices[:, 1:]  # (n, k_actual)

    graphs = []
    feat_tensor = torch.from_numpy(features.astype(np.float32))  # (n, 24)
    for i in range(n):
        neighbour_idxs = indices[i]  # (k_actual,) — global indices

        # Node features: central node first, then neighbours (local graph)
        node_feats = torch.cat(
            [feat_tensor[i].unsqueeze(0),           # (1, 24)
             feat_tensor[neighbour_idxs]],           # (k_actual, 24)
            dim=0,
        )  # (k_actual+1, 24)

        # Local edge_index: central node (0) → each neighbour (1..k_actual)
        src = torch.zeros(k_actual, dtype=torch.long)
        dst = torch.arange(1, k_actual + 1, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)  # (2, k_actual)

        data = Data(x=node_feats, edge_index=edge_index)
        graphs.append(data)

    return graphs
