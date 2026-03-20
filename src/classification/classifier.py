"""
CNN-based audio event classifier.

Operates on log-Mel spectrograms and supports multi-label
classification with BCEWithLogitsLoss.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class _AudioCNN(nn.Module):
    """Simple CNN backbone for spectrogram classification.

    Architecture:
        Conv2d → BN → ReLU → MaxPool  (repeated)
        → AdaptiveAvgPool → Flatten → FC → output
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = 10,
        hidden_channels: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = hidden_channels or [32, 64, 128]

        layers: list[nn.Module] = []
        in_ch = 1  # single-channel spectrogram
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, n_mels, T)``.

        Returns
        -------
        torch.Tensor
            Raw logits, shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════
# High-level wrapper
# ═══════════════════════════════════════════════════════════════

class AudioEventClassifier:
    """Train, evaluate, and predict audio event classes.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    labels : list[str], optional
        Human-readable class labels.
    n_mels : int
        Mel-band resolution of input spectrograms.
    hidden_channels : list[int]
        CNN channel widths per block.
    dropout : float
        Dropout probability before the final FC layer.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        num_classes: int = 10,
        labels: list[str] | None = None,
        n_mels: int = 128,
        hidden_channels: list[int] | None = None,
        dropout: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.labels = labels or [f"class_{i}" for i in range(num_classes)]
        self.device = torch.device(device)

        self.model = _AudioCNN(
            n_mels=n_mels,
            num_classes=num_classes,
            hidden_channels=hidden_channels,
            dropout=dropout,
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        logger.info(
            "AudioEventClassifier initialised — %d classes, device=%s",
            num_classes,
            self.device,
        )

    # ── Training ─────────────────────────────────────────────

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        patience: int = 5,
    ) -> dict[str, Any]:
        """Train the model with early stopping.

        Parameters
        ----------
        dataloader : DataLoader
            Yields ``(spectrogram_batch, label_batch)`` tensors.
        epochs : int
            Maximum number of epochs.
        learning_rate : float
            Optimizer learning rate.
        patience : int
            Early stopping patience.

        Returns
        -------
        dict
            Training history with loss per epoch.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()

        history: list[dict[str, float]] = []
        best_loss = float("inf")
        wait = 0

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for spectrograms, targets in dataloader:
                spectrograms = spectrograms.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(spectrograms)
                loss = self.criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history.append({"epoch": epoch, "loss": round(avg_loss, 6)})
            logger.info("Epoch %d/%d — loss: %.6f", epoch, epochs, avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return {"epochs_trained": len(history), "history": history}

    # ── Prediction ───────────────────────────────────────────

    def predict(
        self,
        spectrogram: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Predict labels for a single spectrogram.

        Parameters
        ----------
        spectrogram : np.ndarray
            Shape ``(n_mels, T)``.
        threshold : float
            Sigmoid threshold for positive labels.

        Returns
        -------
        dict
            ``{"labels": [...], "scores": [...], "latency_ms": float}``
        """
        self.model.eval()
        t0 = time.perf_counter()

        # (1, 1, n_mels, T)
        tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        positive_indices = np.where(probs >= threshold)[0]
        predicted_labels = [self.labels[i] for i in positive_indices]
        scores = {self.labels[i]: round(float(probs[i]), 4) for i in range(len(self.labels))}

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Classification (%.1f ms): %s", latency_ms, predicted_labels)

        return {
            "labels": predicted_labels,
            "scores": scores,
            "latency_ms": round(latency_ms, 2),
        }

    # ── Evaluation ───────────────────────────────────────────

    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate model on a dataset.

        Returns
        -------
        dict
            ``{"accuracy": float, "avg_loss": float, "f1_macro": float}``
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for spectrograms, targets in dataloader:
                spectrograms = spectrograms.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(spectrograms)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                preds = (torch.sigmoid(logits) >= threshold).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.cpu().numpy())

        all_preds_arr = np.vstack(all_preds)
        all_targets_arr = np.vstack(all_targets)

        # Exact-match accuracy
        accuracy = float(np.mean(np.all(all_preds_arr == all_targets_arr, axis=1)))

        # Per-class F1 → macro average
        eps = 1e-8
        f1_per_class = []
        for c in range(self.num_classes):
            tp = float(((all_preds_arr[:, c] == 1) & (all_targets_arr[:, c] == 1)).sum())
            fp = float(((all_preds_arr[:, c] == 1) & (all_targets_arr[:, c] == 0)).sum())
            fn = float(((all_preds_arr[:, c] == 0) & (all_targets_arr[:, c] == 1)).sum())
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1_per_class.append(f1)

        n_batches = max(len(all_preds), 1)
        result = {
            "accuracy": round(accuracy, 4),
            "avg_loss": round(total_loss / n_batches, 6),
            "f1_macro": round(float(np.mean(f1_per_class)), 4),
        }
        logger.info(
            "Evaluation — Accuracy: %.2f%%, F1: %.4f, Loss: %.6f",
            result["accuracy"] * 100,
            result["f1_macro"],
            result["avg_loss"],
        )
        return result

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved → %s", path)

    def load(self, path: str) -> None:
        """Load model weights."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("Model loaded ← %s", path)

    def __repr__(self) -> str:
        return f"AudioEventClassifier(classes={self.num_classes}, device={self.device})"
