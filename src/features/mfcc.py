"""
MFCC (Mel-Frequency Cepstral Coefficients) extractor.

Extracts MFCCs with optional delta and delta-delta coefficients.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MFCCExtractor:
    """Extract MFCC features from audio waveforms.

    Parameters
    ----------
    n_mfcc : int
        Number of MFCC coefficients.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length in samples.
    n_mels : int
        Number of Mel bands used to compute MFCCs.
    include_deltas : bool
        If True, append delta and delta-delta coefficients (3× features).
    """

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        include_deltas: bool = True,
    ) -> None:
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.include_deltas = include_deltas

    def extract(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """Compute MFCCs from a waveform.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 audio signal.
        sr : int
            Sample rate.

        Returns
        -------
        np.ndarray
            Shape ``(n_features, T)`` where *n_features* is ``n_mfcc``
            (or ``3 * n_mfcc`` when deltas are included).
        """
        mfccs = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        if self.include_deltas:
            delta = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            mfccs = np.vstack([mfccs, delta, delta2])

        logger.debug(
            "MFCC extracted — shape %s (n_mfcc=%d, deltas=%s)",
            mfccs.shape,
            self.n_mfcc,
            self.include_deltas,
        )
        return mfccs.astype(np.float32)

    def extract_stats(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> dict[str, np.ndarray]:
        """Return mean and std across the time axis.

        Returns
        -------
        dict
            ``{"mean": ndarray, "std": ndarray}`` each of shape ``(n_features,)``.
        """
        feats = self.extract(waveform, sr)
        return {
            "mean": feats.mean(axis=1),
            "std": feats.std(axis=1),
        }

    def __repr__(self) -> str:
        return (
            f"MFCCExtractor(n_mfcc={self.n_mfcc}, n_fft={self.n_fft}, "
            f"hop={self.hop_length}, deltas={self.include_deltas})"
        )
