"""
Transcript–speaker alignment module.

Aligns ASR word-level timestamps to diarization speaker segments
so that each transcribed word is attributed to a speaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.diarization.segmenter import SpeakerSegment
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlignedWord:
    """A single word with speaker attribution."""

    word: str
    start_sec: float
    end_sec: float
    speaker_id: int


@dataclass
class AlignedTranscript:
    """Full aligned transcript."""

    words: list[AlignedWord] = field(default_factory=list)

    def to_text(self, speaker_labels: bool = True) -> str:
        """Render as plain text, optionally prefixing speaker changes."""
        if not self.words:
            return ""

        parts: list[str] = []
        current_speaker: int | None = None
        for w in self.words:
            if speaker_labels and w.speaker_id != current_speaker:
                if parts:
                    parts.append("\n")
                parts.append(f"[Speaker {w.speaker_id}] ")
                current_speaker = w.speaker_id
            parts.append(w.word + " ")
        return "".join(parts).strip()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "word": w.word,
                "start_sec": round(w.start_sec, 3),
                "end_sec": round(w.end_sec, 3),
                "speaker_id": w.speaker_id,
            }
            for w in self.words
        ]


class TranscriptAligner:
    """Align ASR output to diarization speaker turns.

    For each word (with timestamps), the aligner finds the speaker
    segment with the greatest temporal overlap and assigns that
    speaker ID.

    Parameters
    ----------
    default_speaker : int
        Speaker ID to assign when no overlap is found.
    """

    def __init__(self, default_speaker: int = 0) -> None:
        self.default_speaker = default_speaker

    def align(
        self,
        words: list[dict[str, Any]],
        segments: list[SpeakerSegment],
    ) -> AlignedTranscript:
        """Align word-level ASR output to speaker segments.

        Parameters
        ----------
        words : list[dict]
            Each dict must have ``"word"``, ``"start"`` (sec), ``"end"`` (sec).
        segments : list[SpeakerSegment]
            Speaker segments from :class:`SpeakerSegmenter`.

        Returns
        -------
        AlignedTranscript
        """
        aligned: list[AlignedWord] = []
        for w in words:
            word_start = w["start"]
            word_end = w["end"]
            best_speaker = self.default_speaker
            best_overlap = 0.0

            for seg in segments:
                overlap = self._overlap(word_start, word_end, seg.start_sec, seg.end_sec)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = seg.speaker_id

            aligned.append(
                AlignedWord(
                    word=w["word"],
                    start_sec=word_start,
                    end_sec=word_end,
                    speaker_id=best_speaker,
                )
            )

        transcript = AlignedTranscript(words=aligned)
        n_speakers = len({w.speaker_id for w in aligned})
        logger.info(
            "Aligned %d words to %d speaker(s).", len(aligned), n_speakers
        )
        return transcript

    @staticmethod
    def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        """Compute temporal overlap between two intervals."""
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    @staticmethod
    def create_dummy_words(text: str, duration_sec: float) -> list[dict[str, Any]]:
        """Create evenly-spaced dummy word timestamps for testing.

        Parameters
        ----------
        text : str
            Transcript text.
        duration_sec : float
            Total audio duration in seconds.

        Returns
        -------
        list[dict]
        """
        words = text.split()
        if not words:
            return []
        word_dur = duration_sec / len(words)
        return [
            {"word": w, "start": i * word_dur, "end": (i + 1) * word_dur}
            for i, w in enumerate(words)
        ]
