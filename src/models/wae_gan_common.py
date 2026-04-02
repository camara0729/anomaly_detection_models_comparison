"""Shared building blocks used by the repository WAE-GAN implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    """Residual temporal block with causal cropping."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.batch_norm1 = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.batch_norm2 = nn.BatchNorm1d(n_outputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self._init_weights()

    def _init_weights(self) -> None:
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = x[:, :, :-self.padding].contiguous()
        x = self.dropout(self.batch_norm1(self.relu(x)))
        x = self.conv2(x)
        x = x[:, :, :-self.padding].contiguous()
        out = self.dropout(self.batch_norm2(self.relu(x)))
        return self.relu(out + residual)


class Encoder_TCN(nn.Module):
    """TCN encoder from (B, T, F) to latent sequences."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        dropout: float,
        num_layers: int,
        *,
        hidden_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        channels = num_layers * [hidden_dim]
        layers = []

        for i in range(num_layers):
            dilation = 2**i
            in_channels = input_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.tcn_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(channels[-1], embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        for layer in self.tcn_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        return self.output_layer(x)


class Decoder_TCN(nn.Module):
    """TCN decoder from latent sequences to reconstructions."""

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        dropout: float,
        num_layers: int,
        *,
        hidden_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        channels = num_layers * [hidden_dim]
        layers = []

        for i in range(num_layers):
            dilation = 2**i
            in_channels = embedding_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.tcn_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(channels[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        for layer in self.tcn_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        return self.output_layer(x)


class LSTMDiscriminator_TCN(nn.Module):
    """LSTM discriminator over latent sequences."""

    def __init__(
        self,
        input_dim: int,
        dropout: float,
        *,
        n_layers: int,
        disc_hidden: int,
        output_dim: int = 1,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=disc_hidden,
            dropout=dropout,
            batch_first=True,
            num_layers=n_layers,
        )
        self.output_layer = nn.Linear(disc_hidden, output_dim)
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        logits = self.output_layer(hidden[-1])
        return torch.sigmoid(logits) if self.apply_sigmoid else logits


__all__ = ["Decoder_TCN", "Encoder_TCN", "LSTMDiscriminator_TCN", "TemporalBlock"]
