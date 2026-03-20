"""
Mel spectrogram extractor.

Computes log-scaled Mel spectrograms suitable for classification
models and visualisation.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MelSpectrogramExtractor:
    """Extract log-Mel spectrograms.

    Parameters
    ----------
    n_mels : int
        Number of Mel bands.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length in samples.
    power : float
        Exponent for the magnitude spectrogram (2.0 = power spectrum).
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        power: float = 2.0,
    ) -> None:
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

    def extract(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """Compute log-Mel spectrogram.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 audio signal.
        sr : int
            Sample rate.

        Returns
        -------
        np.ndarray
            Shape ``(n_mels, T)`` — log-scaled Mel spectrogram.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=self.power,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        logger.debug("Mel spectrogram extracted — shape %s", log_mel.shape)
        return log_mel.astype(np.float32)

    def extract_normalized(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """Return min-max normalized log-Mel spectrogram in [0, 1].

        Returns
        -------
        np.ndarray
            Shape ``(n_mels, T)``.
        """
        log_mel = self.extract(waveform, sr)
        min_val = log_mel.min()
        max_val = log_mel.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(log_mel)
        return ((log_mel - min_val) / (max_val - min_val)).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"MelSpectrogramExtractor(n_mels={self.n_mels}, "
            f"n_fft={self.n_fft}, hop={self.hop_length})"
        )
