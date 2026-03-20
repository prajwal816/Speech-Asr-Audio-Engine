"""
Unified feature extraction pipeline.

Orchestrates MFCC, Mel spectrogram, and Chroma extraction with
optional normalisation and concatenation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.features.mfcc import MFCCExtractor
from src.features.mel_spectrogram import MelSpectrogramExtractor
from src.features.chroma import ChromaExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    """Orchestrate multiple audio feature extractors.

    Parameters
    ----------
    config : dict
        Feature configuration (mirrors the ``features`` key in
        ``configs/default.yaml``).  Falls back to sane defaults if
        keys are missing.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}

        mfcc_cfg = cfg.get("mfcc", {})
        mel_cfg = cfg.get("mel_spectrogram", {})
        chroma_cfg = cfg.get("chroma", {})

        self.mfcc = MFCCExtractor(
            n_mfcc=mfcc_cfg.get("n_mfcc", 13),
            n_fft=mfcc_cfg.get("n_fft", 2048),
            hop_length=mfcc_cfg.get("hop_length", 512),
            n_mels=mfcc_cfg.get("n_mels", 128),
        )

        self.mel = MelSpectrogramExtractor(
            n_mels=mel_cfg.get("n_mels", 128),
            n_fft=mel_cfg.get("n_fft", 2048),
            hop_length=mel_cfg.get("hop_length", 512),
            power=mel_cfg.get("power", 2.0),
        )

        self.chroma = ChromaExtractor(
            n_chroma=chroma_cfg.get("n_chroma", 12),
            n_fft=chroma_cfg.get("n_fft", 2048),
            hop_length=chroma_cfg.get("hop_length", 512),
        )

    # ── Public API ───────────────────────────────────────────

    def extract(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> dict[str, np.ndarray]:
        """Extract all features and return them in a dictionary.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 audio signal.
        sr : int
            Sample rate.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``"mfcc"``, ``"mel_spectrogram"``, ``"chroma"``.
            Each value has shape ``(n_features, T)``.
        """
        result = {
            "mfcc": self.mfcc.extract(waveform, sr),
            "mel_spectrogram": self.mel.extract(waveform, sr),
            "chroma": self.chroma.extract(waveform, sr),
        }
        logger.info(
            "Feature pipeline complete — MFCC %s, Mel %s, Chroma %s",
            result["mfcc"].shape,
            result["mel_spectrogram"].shape,
            result["chroma"].shape,
        )
        return result

    def extract_concatenated(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """Extract and vertically concatenate all features.

        The time axis is truncated to the minimum across extractors
        so that all feature matrices align.

        Returns
        -------
        np.ndarray
            Shape ``(total_features, T_min)``.
        """
        feats = self.extract(waveform, sr)
        min_t = min(v.shape[1] for v in feats.values())
        concatenated = np.vstack([v[:, :min_t] for v in feats.values()])
        logger.debug("Concatenated feature shape: %s", concatenated.shape)
        return concatenated.astype(np.float32)

    @staticmethod
    def normalize(features: np.ndarray) -> np.ndarray:
        """Per-feature (row) zero-mean, unit-variance normalization.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(n_features, T)``.

        Returns
        -------
        np.ndarray
            Normalised features (same shape).
        """
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True) + 1e-8
        return ((features - mean) / std).astype(np.float32)

    def __repr__(self) -> str:
        return f"FeaturePipeline(mfcc={self.mfcc}, mel={self.mel}, chroma={self.chroma})"
