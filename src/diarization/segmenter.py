"""
Speaker segmentation module.

Uses energy-based Voice Activity Detection (VAD) followed by
agglomerative clustering on MFCC embeddings to identify and
separate distinct speakers in an audio stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.features.mfcc import MFCCExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SpeakerSegment:
    """A contiguous speaker turn."""

    speaker_id: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
        }


class SpeakerSegmenter:
    """Segment an audio signal into per-speaker turns.

    Pipeline:
        1. Frame-level energy VAD → identify speech regions.
        2. Split speech regions into fixed-length windows.
        3. Extract MFCC embeddings per window.
        4. Agglomerative clustering to assign speaker labels.
        5. Merge adjacent windows with same label.

    Parameters
    ----------
    energy_threshold : float
        RMS energy below this is treated as silence.
    min_segment_sec : float
        Minimum segment duration; shorter segments are merged with
        the nearest neighbour.
    n_speakers : int
        Expected number of speakers (used by clustering).
    window_sec : float
        Window length for embedding extraction.
    hop_sec : float
        Hop between successive windows.
    """

    def __init__(
        self,
        energy_threshold: float = 0.01,
        min_segment_sec: float = 0.5,
        n_speakers: int = 2,
        window_sec: float = 1.5,
        hop_sec: float = 0.75,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.min_segment_sec = min_segment_sec
        self.n_speakers = n_speakers
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self._mfcc = MFCCExtractor(n_mfcc=13, include_deltas=True)

    # ── Public API ───────────────────────────────────────────

    def segment(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> list[SpeakerSegment]:
        """Run the full segmentation pipeline.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 audio.
        sr : int
            Sample rate.

        Returns
        -------
        list[SpeakerSegment]
        """
        logger.info("Starting speaker segmentation (%.2f s audio)", len(waveform) / sr)

        # Step 1 — VAD
        speech_mask = self._vad(waveform, sr)

        # Step 2 — windowed embeddings
        embeddings, window_starts = self._extract_embeddings(
            waveform, sr, speech_mask
        )

        if len(embeddings) == 0:
            logger.warning("No speech detected — returning empty segments.")
            return []

        # Step 3 — clustering
        labels = self._cluster(embeddings)

        # Step 4 — build segments
        segments = self._build_segments(labels, window_starts, sr)

        # Step 5 — merge short segments
        segments = self._merge_short(segments)

        logger.info("Segmentation complete — %d segments, %d speakers",
                     len(segments), len({s.speaker_id for s in segments}))
        return segments

    # ── Private helpers ──────────────────────────────────────

    def _vad(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Frame-level energy VAD → boolean mask (per sample)."""
        frame_len = int(0.025 * sr)  # 25 ms frames
        hop = int(0.010 * sr)        # 10 ms hop
        mask = np.zeros(len(waveform), dtype=bool)
        for start in range(0, len(waveform) - frame_len, hop):
            frame = waveform[start : start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms > self.energy_threshold:
                mask[start : start + frame_len] = True
        return mask

    def _extract_embeddings(
        self,
        waveform: np.ndarray,
        sr: int,
        speech_mask: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Extract MFCC-mean embeddings for each window."""
        win_samples = int(self.window_sec * sr)
        hop_samples = int(self.hop_sec * sr)
        embeddings: list[np.ndarray] = []
        starts: list[int] = []

        for start in range(0, len(waveform) - win_samples, hop_samples):
            end = start + win_samples
            # Only process windows with >50 % speech
            if speech_mask[start:end].mean() < 0.5:
                continue
            chunk = waveform[start:end]
            mfcc_feats = self._mfcc.extract(chunk, sr)
            embeddings.append(mfcc_feats.mean(axis=1))
            starts.append(start)

        if not embeddings:
            return np.empty((0, 0)), []
        return np.vstack(embeddings), starts

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Agglomerative clustering on embeddings."""
        n_clusters = min(self.n_speakers, len(embeddings))
        if n_clusters <= 1:
            return np.zeros(len(embeddings), dtype=int)
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(embeddings)
        return labels

    def _build_segments(
        self, labels: np.ndarray, starts: list[int], sr: int
    ) -> list[SpeakerSegment]:
        """Convert window-level labels to SpeakerSegment list."""
        win_samples = int(self.window_sec * sr)
        segments: list[SpeakerSegment] = []
        for label, start in zip(labels, starts):
            seg = SpeakerSegment(
                speaker_id=int(label),
                start_sec=start / sr,
                end_sec=(start + win_samples) / sr,
            )
            segments.append(seg)

        # Merge consecutive windows with same speaker
        merged: list[SpeakerSegment] = []
        for seg in segments:
            if merged and merged[-1].speaker_id == seg.speaker_id:
                merged[-1] = SpeakerSegment(
                    speaker_id=seg.speaker_id,
                    start_sec=merged[-1].start_sec,
                    end_sec=seg.end_sec,
                )
            else:
                merged.append(seg)
        return merged

    def _merge_short(self, segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        """Merge segments shorter than min_segment_sec with neighbours."""
        if not segments:
            return segments
        result = [segments[0]]
        for seg in segments[1:]:
            if seg.duration_sec < self.min_segment_sec:
                result[-1] = SpeakerSegment(
                    speaker_id=result[-1].speaker_id,
                    start_sec=result[-1].start_sec,
                    end_sec=seg.end_sec,
                )
            else:
                result.append(seg)
        return result

    def __repr__(self) -> str:
        return (
            f"SpeakerSegmenter(n_speakers={self.n_speakers}, "
            f"energy_thr={self.energy_threshold})"
        )
