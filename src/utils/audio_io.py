"""
Audio I/O helpers.

Load, save, and resample audio waveforms using soundfile + librosa.
"""

import numpy as np
import soundfile as sf
import librosa

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_audio(
    path: str,
    sr: int = 16000,
    mono: bool = True,
    max_duration_sec: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load an audio file and return (waveform, sample_rate).

    Parameters
    ----------
    path : str
        Path to audio file (wav, flac, mp3, etc.).
    sr : int
        Target sample rate. Set to ``None`` to keep native rate.
    mono : bool
        Convert to mono if True.
    max_duration_sec : float, optional
        Truncate to this duration (seconds).

    Returns
    -------
    tuple[np.ndarray, int]
        (waveform as float32, actual sample rate)
    """
    logger.debug("Loading audio: %s (sr=%s, mono=%s)", path, sr, mono)
    waveform, native_sr = sf.read(path, dtype="float32", always_2d=False)

    # Mono conversion
    if mono and waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample
    if sr is not None and native_sr != sr:
        waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=sr)
        native_sr = sr

    # Truncate
    if max_duration_sec is not None:
        max_samples = int(max_duration_sec * native_sr)
        waveform = waveform[:max_samples]

    logger.info(
        "Loaded %s — %.2fs, %d Hz, %s",
        path,
        len(waveform) / native_sr,
        native_sr,
        "mono" if waveform.ndim == 1 else f"{waveform.shape[1]}ch",
    )
    return waveform.astype(np.float32), native_sr


def save_audio(path: str, waveform: np.ndarray, sr: int = 16000) -> None:
    """Save a waveform to a WAV file.

    Parameters
    ----------
    path : str
        Destination file path.
    waveform : np.ndarray
        Audio samples (float32).
    sr : int
        Sample rate.
    """
    sf.write(path, waveform, sr)
    logger.info("Saved audio → %s (%.2fs, %d Hz)", path, len(waveform) / sr, sr)


def resample_audio(
    waveform: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample a waveform from orig_sr to target_sr.

    Returns
    -------
    np.ndarray
        Resampled waveform (float32).
    """
    if orig_sr == target_sr:
        return waveform
    resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    logger.debug("Resampled %d Hz → %d Hz", orig_sr, target_sr)
    return resampled.astype(np.float32)
