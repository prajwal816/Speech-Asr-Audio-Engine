# src/classification — Audio event classification
"""CNN-based multi-label audio event classifier."""

from src.classification.classifier import AudioEventClassifier
from src.classification.dataset import AudioEventDataset

__all__ = ["AudioEventClassifier", "AudioEventDataset"]
