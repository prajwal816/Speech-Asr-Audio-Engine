"""
Chroma feature extractor.

Extracts chromagram (pitch class profiles) useful for music and
tonal analysis.
"""

from __future__ import annotations

import librosa
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaExtractor:
    """Extract chroma features from audio waveforms.

    Parameters
    ----------
    n_chroma : int
        Number of chroma bins (typically 12 for the chromatic scale).
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length in samples.
    """

    def __init__(
        self,
        n_chroma: int = 12,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> None:
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """Compute chroma features (STFT-based).

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 audio signal.
        sr : int
            Sample rate.

        Returns
        -------
        np.ndarray
            Shape ``(n_chroma, T)``.
        """
        chroma = librosa.feature.chroma_stft(
            y=waveform,
            sr=sr,
            n_chroma=self.n_chroma,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        logger.debug("Chroma extracted — shape %s", chroma.shape)
        return chroma.astype(np.float32)

    def extract_cens(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """Compute CENS (Chroma Energy Normalized Statistics).

        CENS features are more robust to dynamics and timbre variations.

        Returns
        -------
        np.ndarray
            Shape ``(n_chroma, T)``.
        """
        chroma_cens = librosa.feature.chroma_cens(
            y=waveform,
            sr=sr,
            n_chroma=self.n_chroma,
            hop_length=self.hop_length,
        )

        logger.debug("Chroma CENS extracted — shape %s", chroma_cens.shape)
        return chroma_cens.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"ChromaExtractor(n_chroma={self.n_chroma}, "
            f"n_fft={self.n_fft}, hop={self.hop_length})"
        )
