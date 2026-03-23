"""Reference WAE-GAN implementation integrated into ``src/models``.

Based on the implementation from:
- https://github.com/schelotto/Wasserstein-AutoEncoders
- SILVA, Miguel EP; VELOSO, Bruno; GAMA, Joao. Predictive Maintenance,
  Adversarial Autoencoders and Explainability. In: Joint European Conference
  on Machine Learning and Knowledge Discovery in Databases. Springer, 2023.
  p. 260-275. - implemented in: https://github.com/migueleps/metro-anomaly-ECML2023

This variant follows the common WAE-GAN idea:

1. Reconstruct the input in data space.
2. Push the aggregated latent posterior ``q(z)`` towards a simple prior ``p(z)``.

In the original WAE view, this latent matching plays the role of the
distributional regularizer associated with the Wasserstein Autoencoder
objective. Here that matching is implemented with an adversarial surrogate in
latent space: a discriminator tries to distinguish samples from the prior and
encoded samples, while the encoder tries to fool it.

Important nuance: despite the WAE/Wasserstein motivation, this code does not
implement a WGAN critic. The discriminator uses a sigmoid output plus BCE loss,
which is a practical GAN-style proxy for latent distribution matching.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from src.models.wae_gan_common import Decoder_TCN, Encoder_TCN, LSTMDiscriminator_TCN


@dataclass
class WAEGANConfig:
    """Configuration for the repository WAE-GAN implementation."""

    n_features: int
    embedding_dim: int = 32
    encoder_layers: int = 3
    dropout: float = 0.1
    tcn_hidden: int = 64
    tcn_kernel: int = 3
    disc_hidden: int = 32
    disc_layers: int = 2
    lr: float = 1e-4
    disc_lr: float = 1e-4
    wae_regularization_term: float = 10.0
    batch_size: int = 64
    epochs: int = 50
    disc_steps: int = 1
    sigma_z: float = 1.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True
    device: str | None = None


class WAEGANCore(nn.Module):
    """TCN encoder/decoder plus latent discriminator."""

    def __init__(self, config: WAEGANConfig) -> None:
        super().__init__()
        self.encoder = Encoder_TCN(
            config.n_features,
            config.embedding_dim,
            config.dropout,
            config.encoder_layers,
            hidden_dim=config.tcn_hidden,
            kernel_size=config.tcn_kernel,
        )
        self.decoder = Decoder_TCN(
            config.embedding_dim,
            config.n_features,
            config.dropout,
            config.encoder_layers,
            hidden_dim=config.tcn_hidden,
            kernel_size=config.tcn_kernel,
        )
        self.discriminator = LSTMDiscriminator_TCN(
            config.embedding_dim,
            config.dropout,
            n_layers=config.disc_layers,
            disc_hidden=config.disc_hidden,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class WAEGAN:
    """WAE-GAN with batched DataLoader support."""

    def __init__(self, config: WAEGANConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = WAEGANCore(config).to(self.device)
        self.reconstruction_loss = nn.MSELoss(reduction="mean")
        self.adversarial_loss = nn.BCELoss(reduction="mean")
        self.history_: dict[str, list[float]] = {"discriminator": [], "generator": [], "monitor": []}
        self.best_epoch_: int | None = None
        self.best_monitor_value_: float | None = None
        self.stopped_epoch_: int | None = None

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_prior(self) -> MultivariateNormal:
        # The prior p(z) is the target latent distribution. Matching encoded
        # samples to this Gaussian is the regularization term that gives the WAE
        # its optimal-transport / Wasserstein-inspired interpretation.
        mean = torch.zeros(self.config.embedding_dim, device=self.device)
        cov = (self.config.sigma_z ** 2) * torch.eye(self.config.embedding_dim, device=self.device)
        return MultivariateNormal(mean, cov)

    @staticmethod
    def _to_tensor(data: Any) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            tensor = data.detach().clone().float()
        elif isinstance(data, (list, tuple)) and data and isinstance(data[0], torch.Tensor):
            stacked = []
            for item in data:
                current = item.detach().clone().float()
                if current.ndim == 3 and current.shape[0] == 1:
                    current = current.squeeze(0)
                stacked.append(current)
            tensor = torch.stack(stacked, dim=0)
        else:
            tensor = torch.as_tensor(np.asarray(data), dtype=torch.float32)

        if tensor.ndim == 4 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                "Expected data with shape (N, T, F), (N, 1, T, F) or (T, F). "
                f"Received shape {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _extract_batch(batch: Any) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("Received an empty batch.")
            return batch[0]
        return batch

    def make_dataloader(
        self,
        data: Any,
        batch_size: int | None = None,
        shuffle: bool = False,
    ) -> DataLoader:
        tensor = self._to_tensor(data)
        dataset = TensorDataset(tensor)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.config.batch_size,
            shuffle=shuffle,
        )

    def _sample_prior(self, latent_shape: torch.Size, prior: MultivariateNormal) -> torch.Tensor:
        return prior.sample(latent_shape[:2]).to(self.device)

    def _generator_step(
        self,
        batch: torch.Tensor,
        optimizer_encoder: optim.Optimizer,
        optimizer_decoder: optim.Optimizer,
    ) -> float:
        # Generator step = encoder + decoder update.
        # The discriminator is frozen here because we want gradients to push the
        # encoder towards latent codes that look like samples from p(z).
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.model.decoder.parameters():
            parameter.requires_grad = True
        for parameter in self.model.discriminator.parameters():
            parameter.requires_grad = False

        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        reconstruction, latent = self.model(batch)
        discriminator_fake = self.model.discriminator(latent)

        # Reconstruction keeps the autoencoder faithful to the input signal.
        recon_loss = self.reconstruction_loss(reconstruction, batch)
        # Adversarial loss acts as a latent matching surrogate:
        # if the discriminator cannot separate encoded z from prior samples,
        # q(z) is being pushed towards p(z).
        #
        # This is the "GAN" part of WAE-GAN. It is inspired by the WAE
        # distribution-matching penalty, but implemented with BCE instead of a
        # true Wasserstein critic.
        adv_loss = self.adversarial_loss(discriminator_fake, torch.ones_like(discriminator_fake))
        loss = recon_loss + self.config.wae_regularization_term * adv_loss
        loss.backward()

        # Gradient clipping keeps adversarial training more stable.
        nn.utils.clip_grad_norm_(self.model.encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.model.decoder.parameters(), 1.0)
        optimizer_encoder.step()
        optimizer_decoder.step()
        return float(loss.item())

    def _evaluate_generator_loss(self, data_loader: DataLoader) -> float:
        # Validation uses the same objective optimized by the generator:
        # reconstruction quality plus latent matching pressure.
        losses: list[float] = []
        was_training = self.model.training
        self.model.eval()

        with torch.no_grad():
            for batch in data_loader:
                batch = self._extract_batch(batch).to(self.device)
                reconstruction, latent = self.model(batch)
                discriminator_fake = self.model.discriminator(latent)
                recon_loss = self.reconstruction_loss(reconstruction, batch)
                adv_loss = self.adversarial_loss(
                    discriminator_fake,
                    torch.ones_like(discriminator_fake),
                )
                losses.append(float((recon_loss + self.config.wae_regularization_term * adv_loss).item()))

        if was_training:
            self.model.train()
        return float(np.mean(losses)) if losses else float("inf")

    def _discriminator_step(
        self,
        batch: torch.Tensor,
        optimizer_discriminator: optim.Optimizer,
        prior: MultivariateNormal,
    ) -> float:
        # Discriminator step = update only the latent-space classifier.
        # The encoder/decoder are frozen so the discriminator learns a cleaner
        # boundary between prior samples and encoded samples.
        for parameter in self.model.encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.model.decoder.parameters():
            parameter.requires_grad = False
        for parameter in self.model.discriminator.parameters():
            parameter.requires_grad = True

        optimizer_discriminator.zero_grad()
        latent_fake = self.model.encoder(batch).detach()
        # Draw "real" latent samples directly from the Gaussian prior p(z).
        latent_prior = self._sample_prior(latent_fake.shape, prior)

        discriminator_real = self.model.discriminator(latent_prior)
        discriminator_fake = self.model.discriminator(latent_fake)

        # BCE on prior-vs-encoded samples is the adversarial proxy that tries to
        # align q(z) with p(z). If the discriminator becomes unable to separate
        # them, latent matching has improved.
        loss_real = self.adversarial_loss(discriminator_real, torch.ones_like(discriminator_real))
        loss_fake = self.adversarial_loss(discriminator_fake, torch.zeros_like(discriminator_fake))
        loss = loss_real + loss_fake
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 1.0)
        optimizer_discriminator.step()
        return float(loss.item())

    def fit(
        self,
        train_data: Any | None = None,
        train_loader: DataLoader | None = None,
        validation_data: Any | None = None,
        validation_loader: DataLoader | None = None,
        verbose: bool = True,
    ) -> "WAEGAN":
        if train_loader is None:
            if train_data is None:
                raise ValueError("Provide train_data or train_loader.")
            train_loader = self.make_dataloader(train_data, shuffle=True)
        if validation_loader is None and validation_data is not None:
            validation_loader = self.make_dataloader(
                validation_data,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        optimizer_discriminator = optim.Adam(self.model.discriminator.parameters(), lr=self.config.disc_lr)
        optimizer_encoder = optim.Adam(self.model.encoder.parameters(), lr=self.config.lr)
        optimizer_decoder = optim.Adam(self.model.decoder.parameters(), lr=self.config.lr)
        prior = self._make_prior()

        patience = self.config.early_stopping_patience
        use_early_stopping = patience is not None and patience > 0
        best_state_dict = None
        epochs_without_improvement = 0
        self.history_ = {"discriminator": [], "generator": [], "monitor": []}
        self.best_epoch_ = None
        self.best_monitor_value_ = None
        self.stopped_epoch_ = None
        self.model.train()

        for epoch in range(self.config.epochs):
            discriminator_losses: list[float] = []
            generator_losses: list[float] = []

            for batch in train_loader:
                batch = self._extract_batch(batch).to(self.device)
                # Alternate the two-player game:
                # 1. teach the discriminator to separate prior and encoded z
                # 2. teach encoder/decoder to reconstruct well and fool it
                for _ in range(self.config.disc_steps):
                    discriminator_losses.append(
                        self._discriminator_step(batch, optimizer_discriminator, prior)
                    )
                generator_losses.append(
                    self._generator_step(batch, optimizer_encoder, optimizer_decoder)
                )

            disc_mean = float(np.mean(discriminator_losses)) if discriminator_losses else 0.0
            gen_mean = float(np.mean(generator_losses)) if generator_losses else 0.0
            monitor_value = (
                self._evaluate_generator_loss(validation_loader)
                if validation_loader is not None
                else gen_mean
            )
            # The monitor tracks the generator-side objective because that is
            # the quantity we ultimately want to minimize for reconstruction +
            # latent regularization quality.
            self.history_["discriminator"].append(disc_mean)
            self.history_["generator"].append(gen_mean)
            self.history_["monitor"].append(monitor_value)

            if verbose:
                monitor_name = "val_G" if validation_loader is not None else "loss_G"
                print(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"loss_D={disc_mean:.6f} | loss_G={gen_mean:.6f} | "
                    f"{monitor_name}={monitor_value:.6f}"
                )

            improved = self.best_monitor_value_ is None or (
                monitor_value < self.best_monitor_value_ - self.config.early_stopping_min_delta
            )
            if improved:
                self.best_monitor_value_ = monitor_value
                self.best_epoch_ = epoch + 1
                epochs_without_improvement = 0
                if self.config.restore_best_weights:
                    # Keep the best generator-side checkpoint according to the
                    # validation monitor, not simply the final epoch.
                    best_state_dict = deepcopy(self.model.state_dict())
            elif use_early_stopping:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    self.stopped_epoch_ = epoch + 1
                    if verbose:
                        print(
                            f"Early stopping at epoch {self.stopped_epoch_} "
                            f"(best_epoch={self.best_epoch_}, best_monitor={self.best_monitor_value_:.6f})."
                        )
                    break

        if self.config.restore_best_weights and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        return self

    def predict_details(
        self,
        data: Any | None = None,
        data_loader: DataLoader | None = None,
    ) -> dict[str, np.ndarray]:
        if data_loader is None:
            if data is None:
                raise ValueError("Provide data or data_loader.")
            data_loader = self.make_dataloader(data, batch_size=self.config.batch_size, shuffle=False)

        self.model.eval()
        reconstruction_errors: list[np.ndarray] = []
        critic_scores: list[np.ndarray] = []

        with torch.no_grad():
            for batch in data_loader:
                batch = self._extract_batch(batch).to(self.device)
                reconstruction, latent = self.model(batch)
                critic = self.model.discriminator(latent)

                # Reconstruction error is the anomaly score actually used by
                # this repository. The discriminator output is returned as
                # auxiliary information for inspection/calibration.
                batch_reconstruction = F.mse_loss(
                    reconstruction,
                    batch,
                    reduction="none",
                ).mean(dim=(1, 2))
                batch_critic = critic.reshape(critic.shape[0], -1).mean(dim=1)
                reconstruction_errors.append(batch_reconstruction.cpu().numpy())
                critic_scores.append(batch_critic.cpu().numpy())

        return {
            "reconstruction": np.concatenate(reconstruction_errors, axis=0),
            "critic": np.concatenate(critic_scores, axis=0),
        }

    def predict_anomaly_score(
        self,
        data: Any | None = None,
        data_loader: DataLoader | None = None,
    ) -> np.ndarray:
        return self.predict_details(data=data, data_loader=data_loader)["reconstruction"]

    def reconstruct(self, data: Any) -> np.ndarray:
        loader = self.make_dataloader(data, batch_size=self.config.batch_size, shuffle=False)
        reconstructions: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = self._extract_batch(batch).to(self.device)
                reconstruction, _ = self.model(batch)
                reconstructions.append(reconstruction.cpu().numpy())
        return np.concatenate(reconstructions, axis=0)

    @staticmethod
    def calculate_threshold(train_scores: np.ndarray, multiplier: float = 1.5) -> float:
        train_scores = np.asarray(train_scores, dtype=float)
        q25, q75 = np.quantile(train_scores, [0.25, 0.75])
        return float(q75 + multiplier * (q75 - q25))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "state_dict": self.model.state_dict(),
            "history": self.history_,
            "best_epoch": self.best_epoch_,
            "best_monitor_value": self.best_monitor_value_,
            "stopped_epoch": self.stopped_epoch_,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None) -> "WAEGAN":
        payload = torch.load(path, map_location=map_location or "cpu")
        config_data = dict(payload["config"])
        config_data["device"] = map_location or config_data.get("device")
        model = cls(WAEGANConfig(**config_data))
        model.model.load_state_dict(payload["state_dict"])
        history = payload.get("history", {})
        model.history_ = {
            "discriminator": list(history.get("discriminator", [])),
            "generator": list(history.get("generator", [])),
            "monitor": list(history.get("monitor", history.get("generator", []))),
        }
        model.best_epoch_ = payload.get("best_epoch")
        model.best_monitor_value_ = payload.get("best_monitor_value")
        model.stopped_epoch_ = payload.get("stopped_epoch")
        return model


__all__ = ["WAEGAN", "WAEGANConfig", "WAEGANCore"]
