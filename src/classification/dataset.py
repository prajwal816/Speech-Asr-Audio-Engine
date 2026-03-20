"""
Audio event dataset for PyTorch.

Loads audio files from a directory, extracts Mel spectrograms,
and returns (spectrogram, multi-hot label) pairs.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.mel_spectrogram import MelSpectrogramExtractor
from src.utils.audio_io import load_audio
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioEventDataset(Dataset):
    """PyTorch Dataset for multi-label audio event classification.

    Directory layout expected::

        root/
            file1.wav  →  labels in ``annotations``
            file2.wav
            ...

    Parameters
    ----------
    file_paths : list[str]
        Paths to audio files.
    annotations : list[list[int]]
        Multi-hot encoded labels for each file.
    sr : int
        Target sample rate for loading.
    max_duration_sec : float
        Maximum audio duration.
    n_mels : int
        Mel spectrogram resolution.
    target_length : int, optional
        If provided, pad or truncate spectrogram time axis to this
        length (for uniform batching).
    """

    def __init__(
        self,
        file_paths: list[str],
        annotations: list[list[int]],
        sr: int = 16000,
        max_duration_sec: float = 5.0,
        n_mels: int = 128,
        target_length: Optional[int] = None,
    ) -> None:
        if len(file_paths) != len(annotations):
            raise ValueError(
                f"Mismatch: {len(file_paths)} files vs {len(annotations)} labels"
            )
        self.file_paths = file_paths
        self.annotations = annotations
        self.sr = sr
        self.max_duration_sec = max_duration_sec
        self.target_length = target_length
        self._mel = MelSpectrogramExtractor(n_mels=n_mels)

        logger.info(
            "AudioEventDataset: %d samples, sr=%d, max_dur=%.1fs",
            len(file_paths),
            sr,
            max_duration_sec,
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[idx]
        waveform, sr = load_audio(
            path, sr=self.sr, mono=True, max_duration_sec=self.max_duration_sec
        )

        mel = self._mel.extract(waveform, sr)  # (n_mels, T)

        # Pad / truncate to target_length
        if self.target_length is not None:
            mel = self._pad_or_truncate(mel, self.target_length)

        spectrogram = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)
        label = torch.tensor(self.annotations[idx], dtype=torch.float32)

        return spectrogram, label

    @staticmethod
    def _pad_or_truncate(spec: np.ndarray, target_len: int) -> np.ndarray:
        """Pad with zeros or truncate the time axis."""
        _, t = spec.shape
        if t < target_len:
            pad = np.zeros((spec.shape[0], target_len - t), dtype=spec.dtype)
            return np.hstack([spec, pad])
        return spec[:, :target_len]

    @staticmethod
    def from_directory(
        root_dir: str,
        label_map: dict[str, list[int]],
        sr: int = 16000,
        max_duration_sec: float = 5.0,
        n_mels: int = 128,
        target_length: Optional[int] = None,
        extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
    ) -> "AudioEventDataset":
        """Create a dataset from a flat directory of audio files.

        Parameters
        ----------
        root_dir : str
            Directory containing audio files.
        label_map : dict[str, list[int]]
            Mapping from filename (basename) to multi-hot label list.
        """
        file_paths: list[str] = []
        annotations: list[list[int]] = []

        for fname in sorted(os.listdir(root_dir)):
            if not any(fname.lower().endswith(ext) for ext in extensions):
                continue
            if fname in label_map:
                file_paths.append(os.path.join(root_dir, fname))
                annotations.append(label_map[fname])

        logger.info("Found %d labelled files in %s", len(file_paths), root_dir)
        return AudioEventDataset(
            file_paths=file_paths,
            annotations=annotations,
            sr=sr,
            max_duration_sec=max_duration_sec,
            n_mels=n_mels,
            target_length=target_length,
        )
