"""Evaluation utilities for US-05 comparative analysis (Transformer vs GAT).

Functions
---------
compute_metrics(y_true, y_pred, y_prob)
    Per-class and macro Accuracy, Precision, Recall, F1, AUC-ROC.

dtw_distance(seq_a, seq_b)
    Dynamic Time Warping distance between two integer label sequences.

dtw_consistency(y_true, y_pred, windows_per_run)
    Mean DTW distance across runs — measures temporal prediction consistency.

build_comparison_table(results)
    Build a pandas DataFrame comparing models side-by-side.

run_inference_transformer(model, loader, device, return_timing=False)
    Run a VibrationTransformer on a DataLoader; return labels, preds, probs.

run_inference_gat(model, loader, device, return_timing=False)
    Run a VibrationGAT on a PyG DataLoader; return labels, preds, probs.

extract_gat_embeddings(model, loader, device)
    Extract pre-classifier embeddings from a VibrationGAT via a forward hook.

InferenceResult
    Dataclass holding inference outputs plus per-batch timing information.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# InferenceResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Container for inference outputs with timing information.

    Attributes
    ----------
    y_true : np.ndarray shape (n,)
        Ground-truth integer labels.
    y_pred : np.ndarray shape (n,)
        Predicted integer labels (argmax).
    y_prob : np.ndarray shape (n, n_classes)
        Softmax probabilities.
    batch_times_ms : List[float]
        Wall-clock time per batch in milliseconds.
    total_time_s : float
        Total inference time in seconds (excluding warmup).
    n_samples : int
        Total number of samples processed.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    batch_times_ms: List[float] = field(default_factory=list)
    total_time_s: float = 0.0
    n_samples: int = 0

    @property
    def mean_batch_ms(self) -> float:
        """Mean batch inference time in milliseconds."""
        if not self.batch_times_ms:
            return 0.0
        return float(np.mean(self.batch_times_ms))

    @property
    def std_batch_ms(self) -> float:
        """Standard deviation of batch inference time in milliseconds."""
        if not self.batch_times_ms:
            return 0.0
        return float(np.std(self.batch_times_ms))

    @property
    def latency_per_sample_ms(self) -> float:
        """Average latency per sample in milliseconds."""
        if self.n_samples == 0:
            return 0.0
        return self.total_time_s * 1000.0 / self.n_samples


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics for one model.

    Parameters
    ----------
    y_true : np.ndarray shape (n,)
        Ground-truth integer class labels.
    y_pred : np.ndarray shape (n,)
        Predicted integer class labels.
    y_prob : np.ndarray shape (n, n_classes)
        Predicted probabilities (softmax outputs). Must match n_classes inferred
        from y_true.

    Returns
    -------
    Dict[str, float]
        Keys: accuracy, precision_macro, recall_macro, f1_macro, auc_roc_macro,
        and per-class precision_c{i}, recall_c{i}, f1_c{i} for each class i.

    Raises
    ------
    ValueError
        If y_prob.shape[1] does not match the number of unique classes in y_true.
    """
    n_classes = len(np.unique(y_true))
    if y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        raise ValueError(
            f"y_prob must have shape (n_samples, {n_classes}), got {y_prob.shape}"
        )

    result: Dict[str, float] = {}

    result["accuracy"] = float(accuracy_score(y_true, y_pred))

    result["precision_macro"] = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result["recall_macro"] = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    result["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    if n_classes == 2:
        result["auc_roc_macro"] = float(
            roc_auc_score(y_true, y_prob[:, 1])
        )
    else:
        result["auc_roc_macro"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )

    # Per-class metrics
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)

    for c in range(n_classes):
        result[f"precision_c{c}"] = float(prec_per[c])
        result[f"recall_c{c}"] = float(rec_per[c])
        result[f"f1_c{c}"] = float(f1_per[c])

    return result


def compute_anomaly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute anomaly-detection metrics from continuous scores.

    Parameters
    ----------
    y_true : np.ndarray shape (n,)
        Binary labels where ``0=normal`` and ``1=anomaly``.
    scores : np.ndarray shape (n,)
        Continuous anomaly scores. Higher means more anomalous.
    threshold : float or None
        If provided, compute thresholded classification metrics in addition to
        ROC-AUC and Average Precision.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError(
            f"y_true and scores must have the same length, got {len(y_true)} and {len(scores)}"
        )

    result: Dict[str, float] = {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }

    if threshold is not None:
        y_pred = (scores >= threshold).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        result.update(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp),
            }
        )

    return result


# ---------------------------------------------------------------------------
# dtw_distance
# ---------------------------------------------------------------------------

def dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute DTW distance between two integer label sequences.

    Uses standard dynamic programming with L1 cost |a_i - b_j|.

    Parameters
    ----------
    seq_a, seq_b : np.ndarray shape (n,) / (m,)
        Integer label sequences (e.g., predicted or ground-truth class labels).

    Returns
    -------
    float
        DTW distance (≥ 0). Zero iff sequences are identical.
    """
    n, m = len(seq_a), len(seq_b)
    # Accumulation matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(seq_a[i - 1]) - float(seq_b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


# ---------------------------------------------------------------------------
# dtw_consistency
# ---------------------------------------------------------------------------

def dtw_consistency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    windows_per_run: int,
) -> float:
    """Compute mean DTW distance between predicted and true label sequences per run.

    Each run is a consecutive slice of `windows_per_run` windows. DTW measures
    how closely the prediction sequence tracks the ground-truth pattern over time.

    Parameters
    ----------
    y_true : np.ndarray shape (n_total,)
        Ground-truth labels ordered by run (all windows of run 0, then run 1, …).
    y_pred : np.ndarray shape (n_total,)
        Predicted labels in the same order.
    windows_per_run : int
        Number of windows per run. Must divide len(y_true) evenly.

    Returns
    -------
    float
        Mean DTW distance across runs. Lower = more temporally consistent.

    Raises
    ------
    ValueError
        If len(y_true) != len(y_pred) or len(y_true) % windows_per_run != 0.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}"
        )

    n_total = len(y_true)
    if n_total % windows_per_run != 0:
        raise ValueError(
            f"len(y_true)={n_total} is not divisible by windows_per_run={windows_per_run}"
        )

    n_runs = n_total // windows_per_run
    distances = []
    for r in range(n_runs):
        start = r * windows_per_run
        end = start + windows_per_run
        d = dtw_distance(y_true[start:end], y_pred[start:end])
        distances.append(d)

    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# run_inference_transformer
# ---------------------------------------------------------------------------


def run_inference_transformer(
    model: "torch.nn.Module",
    loader: "torch.utils.data.DataLoader",
    device: "torch.device",
    return_timing: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], InferenceResult]:
    """Run inference with a VibrationTransformer over a DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained VibrationTransformer (or any model accepting (B, C, W) tensors).
        Must already be in eval mode and on *device*.
    loader : DataLoader
        Yields (xb, yb) tuples — xb shape (B, C, W), yb shape (B,).
    device : torch.device
        Device to move input batches to.
    return_timing : bool
        If True, return an ``InferenceResult`` with per-batch timing.  If False
        (default), return the legacy tuple ``(y_true, y_pred, y_prob)``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray] or InferenceResult
    """
    model.eval()
    preds: list = []
    probs: list = []
    labels: list = []
    batch_times: List[float] = []
    use_cuda = device is not None and device.type == "cuda"
    n_samples = 0
    is_warmup = return_timing  # skip timing for first batch if timing requested

    with torch.no_grad():
        for xb, yb in loader:
            if is_warmup:
                # Warmup batch: run without timing (JIT/CUDA cache)
                _ = model(xb.to(device))
                is_warmup = False

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(xb.to(device))

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_times.append((t1 - t0) * 1000.0)
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(yb.numpy())
            n_samples += xb.size(0)

    y_true = np.array(labels)
    y_pred = np.array(preds)
    y_prob = np.array(probs)

    if return_timing:
        return InferenceResult(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            batch_times_ms=batch_times,
            total_time_s=sum(batch_times) / 1000.0,
            n_samples=n_samples,
        )
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# run_inference_gat
# ---------------------------------------------------------------------------


def run_inference_gat(
    model: "torch.nn.Module",
    loader,
    device: "torch.device",
    return_timing: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], InferenceResult]:
    """Run inference with a VibrationGAT over a PyG DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        Trained VibrationGAT (or any model accepting (x, edge_index, batch)).
        Must already be in eval mode and on *device*.
    loader : torch_geometric.loader.DataLoader
        Yields PyG Batch objects. Each batch must have .x, .edge_index, .batch,
        and .y attributes.
    device : torch.device
        Device to move batches to.
    return_timing : bool
        If True, return an ``InferenceResult`` with per-batch timing.  If False
        (default), return the legacy tuple ``(y_true, y_pred, y_prob)``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray] or InferenceResult
    """
    model.eval()
    preds: list = []
    probs: list = []
    labels: list = []
    batch_times: List[float] = []
    use_cuda = device is not None and device.type == "cuda"
    n_samples = 0
    is_warmup = return_timing

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if is_warmup:
                _ = model(batch.x, batch.edge_index, batch.batch)
                is_warmup = False

            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(batch.x, batch.edge_index, batch.batch)

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_times.append((t1 - t0) * 1000.0)
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
            n_samples += batch.num_graphs

    y_true = np.array(labels)
    y_pred = np.array(preds)
    y_prob = np.array(probs)

    if return_timing:
        return InferenceResult(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            batch_times_ms=batch_times,
            total_time_s=sum(batch_times) / 1000.0,
            n_samples=n_samples,
        )
    return y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------


def build_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame from a list of model result dicts.

    Parameters
    ----------
    results : List[Dict]
        Each dict must have at minimum: 'model', 'accuracy', 'f1_macro',
        'auc_roc_macro', 'dtw_mean', 'n_params', 'train_time_s'.

    Returns
    -------
    pd.DataFrame
        One row per model, columns = metric names.

    Raises
    ------
    ValueError
        If results is empty.
    """
    if not results:
        raise ValueError("results list is empty")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# extract_gat_embeddings (P1.5 Item 5)
# ---------------------------------------------------------------------------


def extract_gat_embeddings(
    model: "VibrationGAT",
    loader: "PyGDataLoader",
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract pre-classifier embeddings from a VibrationGAT via a forward hook.

    Registers a hook on the ``global_mean_pool`` output (the last operation
    before the classifier Linear layer).  The hook captures the pooled graph
    representation of shape ``(batch_size, hidden)`` for every batch, then
    is removed immediately after inference completes.

    The model is run in eval mode during extraction regardless of its current
    training state; the original state is restored before returning.

    Parameters
    ----------
    model : VibrationGAT
        Trained GAT model.  Must have a ``classifier`` attribute (nn.Linear)
        and a ``hidden`` attribute (int).
    loader : PyGDataLoader
        PyTorch Geometric DataLoader yielding Batch objects with ``.y`` labels.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(embeddings, y_true, y_pred)`` where:

        - ``embeddings`` : float32 ndarray of shape ``(N, hidden)``
        - ``y_true``     : int64  ndarray of shape ``(N,)``
        - ``y_pred``     : int64  ndarray of shape ``(N,)``
    """
    # global_mean_pool is a function (not nn.Module), so we hook on the classifier's
    # pre-hook instead — its input equals the global_mean_pool output.
    captured: List[torch.Tensor] = []

    def _pre_hook(module: torch.nn.Module, inputs: tuple) -> None:
        captured.append(inputs[0].detach().cpu())

    handle = model.classifier.register_forward_pre_hook(_pre_hook)

    was_training = model.training
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []

    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_labels.extend(batch.y.cpu().numpy().tolist())
                all_preds.extend(preds.tolist())
    finally:
        handle.remove()
        model.train(was_training)

    embeddings = np.concatenate([t.numpy() for t in captured], axis=0).astype(np.float32)
    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    return embeddings, y_true, y_pred
