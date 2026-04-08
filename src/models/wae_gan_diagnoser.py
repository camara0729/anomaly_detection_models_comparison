"""Two-stage fault diagnosis pipeline built on top of a trained WAE-GAN.

Concept
-------
A WAE-GAN trained exclusively on normal data learns a compact latent
distribution centred on an isotropic Gaussian prior.  When fault windows
pass through the frozen encoder they land in regions of the latent space
that are *outside* the normal cluster, and – crucially – different fault
types tend to occupy *distinct* off-normal regions.

This module exploits that structure with a two-stage pipeline:

Stage 1 – Anomaly Detection (WAE-GAN, unchanged)
    Reconstruction MSE is computed for every incoming window.  Windows
    below the threshold are labelled "Normal" and the pipeline terminates.

Stage 2 – Fault Classification (supervised sklearn model)
    For anomalous windows the encoder's latent sequence
    ``(B, T, embedding_dim)`` is compressed to a fixed-length vector via
    **Global Average Pooling** over the time axis → ``(B, embedding_dim)``.
    A supervised classifier (Random Forest, SVM, or XGBoost) trained on
    these compact vectors then identifies *which* fault type is present.

The WAE-GAN weights are always frozen after construction; only the
sklearn classifier is trained in :meth:`WAEGAN_FaultDiagnoser.fit`.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from src.models.wae_gan import WAEGAN


@dataclass
class FaultDiagnoserConfig:
    """Configuration for the classifier stage of :class:`WAEGAN_FaultDiagnoser`."""

    classifier: Literal["random_forest", "svm", "xgboost"] = "random_forest"

    # ── Random Forest / XGBoost ───────────────────────────────────────────
    n_estimators: int = 200
    max_depth: int | None = None

    # ── SVM ──────────────────────────────────────────────────────────────
    svm_C: float = 1.0
    svm_gamma: str | float = "scale"

    # ── Common ────────────────────────────────────────────────────────────
    random_state: int = 42

    # ── Anomaly detection threshold ────────────────────────────────────────
    # Populated automatically by fit() when not supplied by the caller.
    anomaly_threshold: float | None = None

    # ── Optional class names for reporting ───────────────────────────────
    # E.g. ["Normal", "Fault_A", "Fault_B", "Fault_C"]
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        valid = {"random_forest", "svm", "xgboost"}
        if self.classifier not in valid:
            raise ValueError(
                f"Unsupported classifier={self.classifier!r}. "
                f"Expected one of: {', '.join(sorted(valid))}."
            )


class WAEGAN_FaultDiagnoser:
    """Two-stage anomaly detection + fault classification system.

    Parameters
    ----------
    waegan:
        A trained :class:`~src.models.wae_gan.WAEGAN` instance.
        Its weights are frozen throughout; only the encoder is used
        for feature extraction.
    config:
        Classifier configuration.  Defaults to Random Forest with
        200 trees if omitted.

    Examples
    --------
    >>> # ── Stage 1+2 training ──────────────────────────────────────────
    >>> from src.models.wae_gan import WAEGAN
    >>> from src.models.wae_gan_diagnoser import WAEGAN_FaultDiagnoser, FaultDiagnoserConfig
    >>>
    >>> waegan = WAEGAN.load("models/wae_gan_best.pt")
    >>>
    >>> diagnoser = WAEGAN_FaultDiagnoser(
    ...     waegan,
    ...     FaultDiagnoserConfig(classifier="random_forest", n_estimators=300),
    ... )
    >>> diagnoser.fit(
    ...     data_by_label={0: X_normal, 1: X_faultA, 2: X_faultB, 3: X_faultC},
    ...     normal_label=0,
    ... )
    >>>
    >>> # ── Full inference pipeline ──────────────────────────────────────
    >>> results = diagnoser.predict(X_test)
    >>> # results["is_anomaly"]  → bool mask
    >>> # results["fault_label"] → -1 (normal) or 1/2/3 (fault type)
    """

    def __init__(
        self,
        waegan: WAEGAN,
        config: FaultDiagnoserConfig | None = None,
    ) -> None:
        self.waegan = waegan
        self.config = config or FaultDiagnoserConfig()
        self.classifier_: Any = None   # fitted sklearn estimator
        self.is_fitted_: bool = False

    # ── Latent Feature Extraction ─────────────────────────────────────────

    def extract_latent_features(
        self,
        data: Any,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode *data* and reduce the time axis via Global Average Pooling.

        Parameters
        ----------
        data:
            Array-like compatible with :meth:`WAEGAN._to_tensor`.
            Expected shape ``(N, T, F)`` (sequence-last format).
        batch_size:
            Inference mini-batch size.  Falls back to WAE-GAN config value.

        Returns
        -------
        np.ndarray of shape ``(N, embedding_dim)``
            One fixed-length latent vector per input window.
        """
        loader = self.waegan.make_dataloader(
            data,
            batch_size=batch_size or self.waegan.config.batch_size,
            shuffle=False,
        )

        device = self.waegan.device
        self.waegan.model.eval()
        latent_vectors: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                batch = self.waegan._extract_batch(batch).to(device)
                z = self.waegan.model.encoder(batch)   # (B, T, embedding_dim)
                z_gap = z.mean(dim=1)                  # GAP → (B, embedding_dim)
                latent_vectors.append(z_gap.cpu().numpy())

        return np.concatenate(latent_vectors, axis=0)

    # ── Classifier Construction ───────────────────────────────────────────

    def _build_classifier(self) -> Any:
        """Instantiate the sklearn estimator specified by *config.classifier*."""
        if self.config.classifier == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1,
            )

        if self.config.classifier == "svm":
            from sklearn.svm import SVC
            return SVC(
                C=self.config.svm_C,
                gamma=self.config.svm_gamma,
                kernel="rbf",
                probability=True,
                random_state=self.config.random_state,
            )

        # xgboost (optional dependency)
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            ) from exc
        return XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth or 6,
            random_state=self.config.random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
            use_label_encoder=False,
        )

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        data_by_label: dict[int, Any],
        normal_label: int = 0,
        threshold_multiplier: float = 1.5,
        batch_size: int | None = None,
        verbose: bool = True,
    ) -> "WAEGAN_FaultDiagnoser":
        """Build the latent dataset and train the fault classifier.

        The WAE-GAN encoder is run in *inference mode* (``torch.no_grad``)
        to extract compact latent vectors.  The sklearn classifier is then
        fitted on the resulting ``(N, embedding_dim)`` array.

        Parameters
        ----------
        data_by_label:
            Mapping ``{label: windows}`` where *windows* has shape
            ``(N_i, T, F)`` in sequence-last format.
            Example: ``{0: X_normal, 1: X_faultA, 2: X_faultB, 3: X_faultC}``.
        normal_label:
            Integer label representing normal operation.  Used to compute
            the anomaly threshold when :attr:`config.anomaly_threshold`
            is ``None``.
        threshold_multiplier:
            IQR multiplier: ``threshold = Q75 + k × (Q75 − Q25)``.
            Only applied when the threshold has not been set already.
        batch_size:
            Override the default batch size for feature extraction.
        verbose:
            Print extraction and training progress.

        Returns
        -------
        self
        """
        if verbose:
            print("── Stage 2: extracting latent features ─────────────────────")

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for label in sorted(data_by_label):
            windows = data_by_label[label]
            feats = self.extract_latent_features(windows, batch_size=batch_size)
            X_parts.append(feats)
            y_parts.append(np.full(len(feats), label, dtype=int))
            if verbose:
                class_name = (
                    self.config.class_names[label]
                    if self.config.class_names and label < len(self.config.class_names)
                    else f"class_{label}"
                )
                print(f"  [{label}] {class_name}: {len(feats)} samples")

        X_latent = np.vstack(X_parts)
        y_labels = np.concatenate(y_parts)

        # Compute the anomaly detection threshold from normal windows.
        if self.config.anomaly_threshold is None:
            if normal_label not in data_by_label:
                raise ValueError(
                    f"normal_label={normal_label} not found in data_by_label keys: "
                    f"{list(data_by_label.keys())}. "
                    "Either include normal data or set config.anomaly_threshold manually."
                )
            if verbose:
                print("\nComputing anomaly threshold from normal windows...")
            normal_scores = self.waegan.predict_anomaly_score(
                data=data_by_label[normal_label]
            )
            self.config.anomaly_threshold = WAEGAN.calculate_threshold(
                normal_scores, multiplier=threshold_multiplier
            )
            if verbose:
                print(f"  threshold = {self.config.anomaly_threshold:.6f}")

        if verbose:
            print(
                f"\nTraining {self.config.classifier} on "
                f"{len(X_latent)} samples × {X_latent.shape[1]} latent dims..."
            )

        self.classifier_ = self._build_classifier()
        self.classifier_.fit(X_latent, y_labels)
        self.is_fitted_ = True

        if verbose:
            from sklearn.metrics import accuracy_score
            train_acc = accuracy_score(y_labels, self.classifier_.predict(X_latent))
            print(f"  In-sample accuracy: {train_acc:.4f}")

        return self

    # ── Inference Pipeline ────────────────────────────────────────────────

    def predict(
        self,
        data: Any,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the full two-stage detection → diagnosis pipeline.

        For each input window the pipeline proceeds as follows:

        1. **Anomaly score** — reconstruction MSE from the frozen WAE-GAN.
        2. **Threshold check** — windows below the threshold are labelled
           *Normal* (``fault_label = -1``) and skipped in stage 2.
        3. **Fault classification** — anomalous windows pass through the
           encoder (again), are collapsed via GAP, then classified by the
           sklearn model.

        Parameters
        ----------
        data:
            Array-like ``(N, T, F)`` in sequence-last format.
        batch_size:
            Mini-batch size for both WAE-GAN scoring and feature extraction.

        Returns
        -------
        dict with keys:

        ``anomaly_score`` : ndarray of shape ``(N,)``
            Reconstruction MSE for every window.
        ``is_anomaly`` : ndarray of bool, shape ``(N,)``
            ``True`` for windows that exceed the threshold.
        ``fault_label`` : ndarray of int, shape ``(N,)``
            Classifier output for anomalous windows; ``-1`` for normal windows.
        ``fault_proba`` : ndarray of float, shape ``(N, n_classes)``
            Class probabilities for anomalous windows (zeros for normal).
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "The diagnoser has not been trained. Call fit() first."
            )
        if self.config.anomaly_threshold is None:
            raise RuntimeError(
                "anomaly_threshold is not set. Call fit() first or set "
                "config.anomaly_threshold manually."
            )

        # ── Stage 1: anomaly score for every window ───────────────────────
        anomaly_scores = self.waegan.predict_anomaly_score(data=data)
        is_anomaly = anomaly_scores > self.config.anomaly_threshold

        n_samples = len(anomaly_scores)
        n_classes = len(self.classifier_.classes_)

        fault_label = np.full(n_samples, -1, dtype=int)
        fault_proba = np.zeros((n_samples, n_classes), dtype=np.float32)

        # ── Stage 2: classify anomalous windows only ──────────────────────
        anomaly_idx = np.where(is_anomaly)[0]
        if len(anomaly_idx) > 0:
            # Slice out anomalous windows without re-running the full loader
            data_tensor = WAEGAN._to_tensor(data)
            anomaly_windows = data_tensor[anomaly_idx].numpy()
            z_gap = self.extract_latent_features(anomaly_windows, batch_size=batch_size)
            fault_label[anomaly_idx] = self.classifier_.predict(z_gap)
            if hasattr(self.classifier_, "predict_proba"):
                fault_proba[anomaly_idx] = self.classifier_.predict_proba(z_gap)

        return {
            "anomaly_score": anomaly_scores,
            "is_anomaly": is_anomaly,
            "fault_label": fault_label,
            "fault_proba": fault_proba,
        }

    def predict_fault_only(
        self,
        data: Any,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Classify fault type directly from latent features (no threshold gate).

        Useful when anomaly has already been confirmed upstream and a
        multiclass label is needed for every window regardless of score.

        Parameters
        ----------
        data:
            Array-like ``(N, T, F)`` in sequence-last format.
        batch_size:
            Mini-batch size for feature extraction.

        Returns
        -------
        np.ndarray of shape ``(N,)``
            Predicted class label for each window.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "The diagnoser has not been trained. Call fit() first."
            )
        z_gap = self.extract_latent_features(data, batch_size=batch_size)
        return self.classifier_.predict(z_gap)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(
        self,
        path: str | Path,
        waegan_path: str | Path | None = None,
    ) -> None:
        """Persist the classifier and config to a pickle file.

        Parameters
        ----------
        path:
            Destination path for the diagnoser checkpoint
            (e.g. ``'models/fault_diagnoser.pkl'``).
        waegan_path:
            If provided, the underlying WAE-GAN is also saved at this path
            via :meth:`WAEGAN.save`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": asdict(self.config),
            "classifier": self.classifier_,
            "is_fitted": self.is_fitted_,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        if waegan_path is not None:
            self.waegan.save(waegan_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        waegan: WAEGAN | None = None,
        waegan_path: str | Path | None = None,
        map_location: str | None = None,
    ) -> "WAEGAN_FaultDiagnoser":
        """Restore a :class:`WAEGAN_FaultDiagnoser` from disk.

        Parameters
        ----------
        path:
            Path to the diagnoser pickle produced by :meth:`save`.
        waegan:
            An already-loaded :class:`WAEGAN` instance.  Takes priority
            over *waegan_path* when both are supplied.
        waegan_path:
            Path to a WAE-GAN ``.pt`` checkpoint loaded via
            :meth:`WAEGAN.load` when *waegan* is ``None``.
        map_location:
            PyTorch device string forwarded to :meth:`WAEGAN.load`.

        Returns
        -------
        WAEGAN_FaultDiagnoser
        """
        if waegan is None and waegan_path is None:
            raise ValueError(
                "Provide either a loaded WAEGAN instance or a waegan_path."
            )

        if waegan is None:
            waegan = WAEGAN.load(waegan_path, map_location=map_location)  # type: ignore[arg-type]

        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        config = FaultDiagnoserConfig(**payload["config"])
        instance = cls(waegan=waegan, config=config)
        instance.classifier_ = payload["classifier"]
        instance.is_fitted_ = payload["is_fitted"]
        return instance


__all__ = ["WAEGAN_FaultDiagnoser", "FaultDiagnoserConfig"]
