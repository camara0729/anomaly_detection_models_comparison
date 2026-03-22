"""Attention weight extraction and visualization for US-08.

Functions
---------
extract_transformer_attention(model, x, device)
    Extract per-layer, per-head attention weights from VibrationTransformer.
    Returns List[Tensor] of shape (nhead, seq_len, seq_len) per layer.

extract_gat_attention(model, graph, device)
    Extract per-layer edge attention weights from VibrationGAT.
    Returns List[Tensor] of shape (n_edges, n_heads) per layer.

plot_transformer_attention(attn_weights, save_dir)
    Save one heatmap PNG per layer to save_dir.

plot_gat_attention(attn_weights, graph, save_dir)
    Save one edge-attention bar chart PNG per layer to save_dir.
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")  # non-interactive backend for file saving


# ---------------------------------------------------------------------------
# Transformer attention extraction
# ---------------------------------------------------------------------------


def extract_transformer_attention(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    """Extract per-head attention matrices from every TransformerEncoderLayer.

    Parameters
    ----------
    model : VibrationTransformer
        Must have a `.transformer_encoder` attribute containing
        `TransformerEncoder` with `.layers` (list of `TransformerEncoderLayer`).
    x : Tensor shape (batch, n_channels, window_size)
        Input window(s). If batch > 1, all samples are passed through; the
        returned attention is averaged over the batch dimension.
    device : torch.device

    Returns
    -------
    List[Tensor]
        One tensor per encoder layer, shape (nhead, seq_len, seq_len).
        Values are softmax attention weights (non-negative, summing to 1 along
        the last dimension).
    """
    model.eval()
    x = x.to(device)

    captured: List[torch.Tensor] = []
    hooks = []

    # For each TransformerEncoderLayer, hook its self_attn module
    for layer in model.transformer_encoder.layers:
        mha: nn.MultiheadAttention = layer.self_attn

        def _make_hook(storage: List[torch.Tensor]):
            def _hook(module, inputs, output):
                # MultiheadAttention forward returns (attn_output, attn_weights)
                # when need_weights=True and average_attn_weights=False.
                # However, the standard TransformerEncoderLayer forward calls
                # self_attn with need_weights=False. We capture via a separate
                # manual call inside the hook — see note below.
                pass
            return _hook

        # We cannot rely on the default forward's attn_weights (need_weights=False
        # is hard-coded in TransformerEncoderLayer). Instead we patch by wrapping.
        captured.append(None)  # placeholder

    # Clean up placeholders — use a different approach: manual forward pass
    # through each layer, calling self_attn manually with need_weights=True.

    # Prepare input through projection + positional encoding (same as forward)
    with torch.no_grad():
        # Replicate model.forward up to transformer_encoder, layer by layer
        xi = x.permute(0, 2, 1)                  # (B, W, C)
        xi = model.input_proj(xi)                 # (B, W, d_model)
        xi = xi.permute(1, 0, 2)                  # (W, B, d_model)
        xi = model.pos_enc(xi)                    # (W, B, d_model) — dropout off in eval
        xi = xi.permute(1, 0, 2)                  # (B, W, d_model)

        attn_per_layer: List[torch.Tensor] = []

        for layer in model.transformer_encoder.layers:
            mha: nn.MultiheadAttention = layer.self_attn

            # Call self_attn manually with need_weights=True, average_attn_weights=False
            # TransformerEncoderLayer uses batch_first=True, so query/key/value are (B, W, d_model)
            # nn.MultiheadAttention.forward expects (tgt, src) — here self-attention so all same
            q = k = v = xi
            _, attn_weights = mha(q, k, v, need_weights=True, average_attn_weights=False)
            # attn_weights: (batch, nhead, seq_len, seq_len) — average over batch
            avg_attn = attn_weights.mean(dim=0)  # (nhead, seq_len, seq_len)
            attn_per_layer.append(avg_attn.detach().cpu())

            # Continue the forward pass through this layer normally (for next layer input)
            xi = layer(xi)

    return attn_per_layer


# ---------------------------------------------------------------------------
# GAT attention extraction
# ---------------------------------------------------------------------------


def extract_gat_attention(
    model: nn.Module,
    graph,  # torch_geometric.data.Data
    device: torch.device,
) -> List[torch.Tensor]:
    """Extract per-layer edge attention weights from VibrationGAT.

    Parameters
    ----------
    model : VibrationGAT
        Must have `.convs` attribute (ModuleList of GATConv layers).
    graph : torch_geometric.data.Data
        Single graph with `.x` (n_nodes, n_feat) and `.edge_index` (2, n_edges).
    device : torch.device

    Returns
    -------
    List[Tensor]
        One tensor per GATConv layer, shape (n_edges_actual, n_heads).
        ``n_edges_actual`` includes any self-loops added by GATConv internally
        (``n_edges_actual = n_orig_edges + n_nodes`` when ``add_self_loops=True``).
        For the last layer (concat=False, heads=1): shape (n_edges_actual, 1).
        Values are the softmax attention coefficients (non-negative).

        The matching per-layer edge indices (including self-loops) are stored as
        the ``.edge_index`` attribute on each returned tensor for use in plotting.
    """
    model.eval()
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)

    attn_per_layer: List[torch.Tensor] = []

    with torch.no_grad():
        xi = x
        for conv, act, drop in zip(model.convs, model.acts, model.drops):
            # return_attention_weights=True → returns (out, (edge_index_out, alpha))
            # edge_index_out includes self-loops added by GATConv
            out, (ei_out, alpha) = conv(xi, edge_index, return_attention_weights=True)
            # alpha: (n_edges_actual, heads) — softmax attention coefficients
            alpha_cpu = alpha.detach().cpu().abs()  # abs for safety
            # Store the corresponding edge_index as a tensor attribute for plotting
            alpha_cpu.edge_index = ei_out.detach().cpu()  # type: ignore[attr-defined]
            attn_per_layer.append(alpha_cpu)
            xi = act(out)
            xi = drop(xi)

    return attn_per_layer


# ---------------------------------------------------------------------------
# Visualization: Transformer
# ---------------------------------------------------------------------------


def plot_transformer_attention(
    attn_weights: List[torch.Tensor],
    save_dir: Union[str, Path],
) -> None:
    """Save one heatmap PNG per Transformer encoder layer.

    Each figure shows a grid of (nhead) heatmaps, one per attention head.
    Pixels represent the attention weight from query position (y-axis) to
    key position (x-axis).

    Parameters
    ----------
    attn_weights : List[Tensor]
        Output of `extract_transformer_attention`. Each tensor shape
        (nhead, seq_len, seq_len).
    save_dir : str or Path
        Directory where PNGs will be saved (created if not exists).
        File names: transformer_attention_layer{i}.png
    """
    if not attn_weights:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, attn in enumerate(attn_weights):
        # attn: (nhead, seq_len, seq_len)
        nhead = attn.shape[0]
        ncols = min(nhead, 4)
        nrows = (nhead + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for head_idx in range(nhead):
            row, col = divmod(head_idx, ncols)
            ax = axes[row][col]
            im = ax.imshow(
                attn[head_idx].numpy(),
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
            )
            ax.set_title(f"Head {head_idx}", fontsize=9)
            ax.set_xlabel("Key position", fontsize=7)
            ax.set_ylabel("Query position", fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused subplots
        for idx in range(nhead, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"Transformer Attention — Layer {layer_idx}", fontsize=11)
        plt.tight_layout()
        fig.savefig(save_dir / f"transformer_attention_layer{layer_idx}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: GAT
# ---------------------------------------------------------------------------


def plot_gat_attention(
    attn_weights: List[torch.Tensor],
    graph,  # torch_geometric.data.Data
    save_dir: Union[str, Path],
) -> None:
    """Save one edge-attention bar chart PNG per GAT layer.

    Each figure shows the mean attention coefficient per edge (averaged
    over heads) as a horizontal bar chart. Edge labels show (src→dst).

    Parameters
    ----------
    attn_weights : List[Tensor]
        Output of `extract_gat_attention`. Each tensor shape (n_edges, n_heads).
    graph : torch_geometric.data.Data
        Graph used for extraction (provides `.edge_index` for edge labels).
    save_dir : str or Path
        Directory where PNGs will be saved (created if not exists).
        File names: gat_attention_layer{i}.png
    """
    if not attn_weights:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, attn in enumerate(attn_weights):
        # attn: (n_edges_actual, n_heads) — may include self-loops added by GATConv
        # Use stored edge_index if available, else fall back to graph's edge_index
        if hasattr(attn, "edge_index"):
            ei = attn.edge_index.cpu()
        else:
            ei = graph.edge_index.cpu()
        n_edges = ei.shape[1]
        edge_labels = [f"{ei[0, i].item()}→{ei[1, i].item()}" for i in range(n_edges)]

        mean_attn = attn.mean(dim=1).numpy()  # (n_edges_actual,)

        fig, ax = plt.subplots(figsize=(max(4, n_edges * 0.5), 4))
        x_pos = range(n_edges)
        ax.bar(x_pos, mean_attn, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean attention coefficient", fontsize=9)
        ax.set_xlabel("Edge (src→dst)", fontsize=9)
        ax.set_title(f"GAT Edge Attention — Layer {layer_idx}", fontsize=11)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fig.savefig(save_dir / f"gat_attention_layer{layer_idx}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Inter-sample k-NN network visualization
# ---------------------------------------------------------------------------


def plot_knn_network(
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    k: int = 3,
    save_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> None:
    """Visualise inter-sample k-NN relationships in the feature space.

    Builds a k-NN graph over a set of samples using their 24-d statistical
    features, then plots the network with t-SNE coordinates as layout.

    Parameters
    ----------
    features : np.ndarray shape (n, d)
        Feature vectors for each sample (e.g. 24-d statistical features).
    labels : np.ndarray shape (n,)
        Ground-truth integer class labels.
    predictions : np.ndarray shape (n,)
        Predicted integer class labels.
    probabilities : np.ndarray shape (n, n_classes)
        Softmax probabilities per sample.
    class_names : List[str]
        Human-readable class names (length == n_classes).
    k : int
        Number of nearest neighbours per sample.
    save_path : str or Path or None
        If given, save the figure to this path.
    seed : int
        Random seed for t-SNE reproducibility.
    """
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors

    n = len(features)
    if n < 2:
        return

    # BUG-QA02 guard: constant features → std=0 → segfault in t-SNE C-extension
    if np.std(features) < 1e-8:
        return

    # BUG-QA01 guard: t-SNE requires perplexity < n_samples.
    # We clamp perplexity to max(min(30, n-1), 2), so n must be > 2 (i.e. n >= 3).
    if n < 3:
        return

    k_actual = min(k, n - 1)

    # Build inter-sample k-NN
    nn_model = NearestNeighbors(n_neighbors=k_actual + 1, algorithm="ball_tree")
    nn_model.fit(features)
    distances, indices = nn_model.kneighbors(features)

    # t-SNE layout — perplexity must be < n_samples (sklearn constraint)
    perp = min(30, n - 1)
    tsne = TSNE(n_components=2, random_state=seed, perplexity=max(perp, 2), max_iter=1000)
    coords = tsne.fit_transform(features)

    # Default class colours
    default_colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd",
                      "#1f77b4", "#8c564b", "#e377c2", "#7f7f7f"]
    n_classes = len(class_names)
    colors = default_colors[:n_classes]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges (from each node to its k neighbours, skip self at index 0)
    for i in range(n):
        for j_idx in range(1, k_actual + 1):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            lw = max(0.3, 2.0 / (1.0 + dist))
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color="gray", alpha=0.3, linewidth=lw, zorder=1,
            )

    # Draw nodes
    correct = labels == predictions
    confidence = probabilities.max(axis=1) if probabilities.ndim == 2 else np.ones(n)
    sizes = 60 + 200 * confidence

    for c in range(n_classes):
        mask_correct = (labels == c) & correct
        mask_wrong = (labels == c) & ~correct

        if mask_correct.any():
            ax.scatter(
                coords[mask_correct, 0], coords[mask_correct, 1],
                c=colors[c], s=sizes[mask_correct], marker="o",
                edgecolors="black", linewidths=0.5, label=class_names[c],
                zorder=2, alpha=0.85,
            )
        if mask_wrong.any():
            ax.scatter(
                coords[mask_wrong, 0], coords[mask_wrong, 1],
                c=colors[c], s=sizes[mask_wrong], marker="^",
                edgecolors="red", linewidths=1.5, label=f"{class_names[c]} (error)",
                zorder=3, alpha=0.95,
            )

    ax.set_title(f"Inter-sample k-NN Network (k={k_actual}, {n} samples)", fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
