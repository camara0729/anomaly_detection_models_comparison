"""GAT (Graph Attention Network) model for vibration-based bearing fault classification (US-04).

Architecture
------------
Input: List[Data] where each Data has:
  - x          : (k+1, 24) — central window node (index 0) + k nearest neighbours
  - edge_index : (2, k)    — local indices: central node (0) → neighbours (1..k)

Each graph is self-contained: edge_index uses local node indices so PyG Batch
collation does not produce out-of-bounds errors on the GPU.

Pipeline:
  N × GATConv(in → hidden, heads=H, concat=True) + ELU + Dropout
    → last layer concat=False (average heads) → (hidden,)
  → Global Mean Pooling over all k+1 nodes in each graph
  → Linear(hidden → n_classes)

Hyperparameter defaults (per spec US-04):
  hidden=128, heads=8, num_layers=3, dropout=0.1

Graph convention: each window has k+1 nodes with local edge indices (built by
  build_knn_graph in preprocessing.py).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# Re-export EarlyStopping so notebooks can import from a single module
from src.models.transformer import EarlyStopping  # noqa: F401


# ---------------------------------------------------------------------------
# GAT model
# ---------------------------------------------------------------------------


class VibrationGAT(nn.Module):
    """Graph Attention Network for vibration fault classification.

    Parameters
    ----------
    n_feat : int
        Node feature dimension (24 for EGB-250 statistical features).
    n_classes : int
        Number of output classes (4).
    hidden : int
        Hidden channel dimension per attention head (default 128).
    heads : int
        Number of attention heads (default 8).
    num_layers : int
        Number of GATConv layers (default 3).
    dropout : float
        Dropout rate applied after each GATConv layer (default 0.1).
    """

    def __init__(
        self,
        n_feat: int,
        n_classes: int,
        hidden: int = 128,
        heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Store spec attributes for inspection / tests
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.hidden = hidden
        self.heads = heads
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()

        # Build GATConv stack
        # - layers 0..num_layers-2: concat=True → output dim = hidden * heads
        # - last layer: concat=False → output dim = hidden (average over heads)
        in_dim = n_feat
        for i in range(num_layers):
            is_last = i == num_layers - 1
            concat = not is_last
            out_heads = 1 if is_last else heads
            self.convs.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=hidden,
                    heads=out_heads,
                    concat=concat,
                    dropout=dropout,
                )
            )
            self.acts.append(nn.ELU())
            self.drops.append(nn.Dropout(p=dropout))
            # next layer input: hidden * heads if concat else hidden
            in_dim = hidden * heads if concat else hidden

        # Classification head: global mean pool output → linear
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (total_nodes, n_feat)
        edge_index : Tensor shape (2, total_edges)
        batch : Tensor shape (total_nodes,) — graph assignment per node

        Returns
        -------
        Tensor shape (batch_size, n_classes) — raw logits
        """
        for conv, act, drop in zip(self.convs, self.acts, self.drops):
            x = conv(x, edge_index)
            x = act(x)
            x = drop(x)

        # Global mean pooling: aggregate node features per graph
        x = global_mean_pool(x, batch)  # (batch_size, hidden)

        return self.classifier(x)


# ---------------------------------------------------------------------------
# DataLoader factory (PyG)
# ---------------------------------------------------------------------------


def make_graph_dataloaders(
    train_graphs: List[Data],
    val_graphs: List[Data],
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[PyGDataLoader, PyGDataLoader]:
    """Create train and validation PyG DataLoaders from graph lists.

    Parameters
    ----------
    train_graphs : List[Data]
        Training graphs (each with .x, .edge_index, .y).
    val_graphs : List[Data]
        Validation graphs.
    batch_size : int
        Number of graphs per mini-batch.
    num_workers : int
        DataLoader workers (0 = main process, deterministic).
    seed : int
        Generator seed for reproducible train shuffling.

    Returns
    -------
    Tuple[PyGDataLoader, PyGDataLoader]
        (train_loader, val_loader)
    """
    g = torch.Generator()
    g.manual_seed(seed)

    train_dl = PyGDataLoader(
        train_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = PyGDataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# Epoch step (train or eval)
# ---------------------------------------------------------------------------


def epoch_step_gat(
    model: VibrationGAT,
    loader: PyGDataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    train: bool,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Run one epoch (train or eval) and return (mean_loss, accuracy).

    Parameters
    ----------
    model : VibrationGAT
    loader : PyGDataLoader
        PyTorch Geometric DataLoader yielding Batch objects.
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
        (mean_loss, accuracy) — accuracy in [0, 1].
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
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            y = batch.y

            loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += batch.num_graphs

    mean_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return mean_loss, accuracy
