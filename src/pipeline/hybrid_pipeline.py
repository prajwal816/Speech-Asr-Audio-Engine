"""
Hybrid end-to-end pipeline.

Audio → Feature Extraction → ASR → Speaker Diarization → Classification

Orchestrates every subsystem and returns a consolidated result dict.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import yaml

from src.features.feature_pipeline import FeaturePipeline
from src.diarization.segmenter import SpeakerSegmenter
from src.diarization.aligner import TranscriptAligner
from src.classification.classifier import AudioEventClassifier
from src.utils.logger import get_logger
from src.utils.audio_io import load_audio

logger = get_logger(__name__)


class HybridPipeline:
    """End-to-end speech + audio analysis pipeline.

    Parameters
    ----------
    config : dict
        Full project config (loaded from ``configs/default.yaml``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = config.get("project", {}).get("device", "cpu")

        pipe_cfg = config.get("pipeline", {})
        self.enable_features = pipe_cfg.get("enable_features", True)
        self.enable_asr = pipe_cfg.get("enable_asr", True)
        self.enable_diarization = pipe_cfg.get("enable_diarization", True)
        self.enable_classification = pipe_cfg.get("enable_classification", True)
        self.asr_backend = pipe_cfg.get("asr_backend", "whisper")

        # ── Lazy-init components ─────────────────────────────
        self._feature_pipeline: Optional[FeaturePipeline] = None
        self._asr = None  # WhisperASR | Wav2Vec2ASR
        self._segmenter: Optional[SpeakerSegmenter] = None
        self._aligner: Optional[TranscriptAligner] = None
        self._classifier: Optional[AudioEventClassifier] = None

        logger.info(
            "HybridPipeline initialised — ASR=%s, features=%s, diarization=%s, classification=%s",
            self.asr_backend if self.enable_asr else "off",
            self.enable_features,
            self.enable_diarization,
            self.enable_classification,
        )

    # ── Lazy loaders ─────────────────────────────────────────

    def _get_feature_pipeline(self) -> FeaturePipeline:
        if self._feature_pipeline is None:
            self._feature_pipeline = FeaturePipeline(
                self.config.get("features", {})
            )
        return self._feature_pipeline

    def _get_asr(self):
        if self._asr is None:
            if self.asr_backend == "whisper":
                from src.asr.whisper_asr import WhisperASR

                asr_cfg = self.config.get("asr", {}).get("whisper", {})
                self._asr = WhisperASR(
                    model_name=asr_cfg.get("model_name", "openai/whisper-small"),
                    language=asr_cfg.get("language", "en"),
                    task=asr_cfg.get("task", "transcribe"),
                    device=self.device,
                )
            else:
                from src.asr.wav2vec2_asr import Wav2Vec2ASR

                asr_cfg = self.config.get("asr", {}).get("wav2vec2", {})
                self._asr = Wav2Vec2ASR(
                    model_name=asr_cfg.get("model_name", "facebook/wav2vec2-base-960h"),
                    device=self.device,
                )
        return self._asr

    def _get_segmenter(self) -> SpeakerSegmenter:
        if self._segmenter is None:
            dia_cfg = self.config.get("diarization", {})
            clust_cfg = dia_cfg.get("clustering", {})
            self._segmenter = SpeakerSegmenter(
                energy_threshold=dia_cfg.get("energy_threshold", 0.01),
                min_segment_sec=dia_cfg.get("min_segment_sec", 0.5),
                n_speakers=clust_cfg.get("n_clusters", 2),
            )
        return self._segmenter

    def _get_aligner(self) -> TranscriptAligner:
        if self._aligner is None:
            self._aligner = TranscriptAligner()
        return self._aligner

    def _get_classifier(self) -> AudioEventClassifier:
        if self._classifier is None:
            cls_cfg = self.config.get("classification", {})
            model_cfg = cls_cfg.get("model", {})
            self._classifier = AudioEventClassifier(
                num_classes=cls_cfg.get("num_classes", 10),
                labels=cls_cfg.get("labels"),
                n_mels=model_cfg.get("n_mels", 128),
                hidden_channels=model_cfg.get("hidden_channels"),
                dropout=model_cfg.get("dropout", 0.3),
                device=self.device,
            )
        return self._classifier

    # ── Main entry point ─────────────────────────────────────

    def process(
        self,
        audio_path: Optional[str] = None,
        waveform: Optional[np.ndarray] = None,
        sr: int = 16000,
    ) -> dict[str, Any]:
        """Run the full pipeline on one audio input.

        Provide either ``audio_path`` or ``waveform`` (not both).

        Returns
        -------
        dict
            Consolidated results including features, ASR transcript,
            diarization segments, and classification labels.
        """
        t_start = time.perf_counter()

        # ── Load audio ───────────────────────────────────────
        if waveform is None and audio_path is not None:
            audio_cfg = self.config.get("audio", {})
            waveform, sr = load_audio(
                audio_path,
                sr=audio_cfg.get("sample_rate", 16000),
                mono=audio_cfg.get("mono", True),
                max_duration_sec=audio_cfg.get("max_duration_sec"),
            )
        elif waveform is None:
            raise ValueError("Provide either audio_path or waveform.")

        result: dict[str, Any] = {
            "duration_sec": round(len(waveform) / sr, 3),
            "sample_rate": sr,
        }

        # ── Features ─────────────────────────────────────────
        if self.enable_features:
            fp = self._get_feature_pipeline()
            features = fp.extract(waveform, sr)
            result["features"] = {
                k: {"shape": list(v.shape)} for k, v in features.items()
            }

        # ── ASR ──────────────────────────────────────────────
        if self.enable_asr:
            asr = self._get_asr()
            asr_result = asr.transcribe(waveform, sr=sr)
            result["asr"] = asr_result

        # ── Diarization ──────────────────────────────────────
        if self.enable_diarization:
            seg = self._get_segmenter()
            segments = seg.segment(waveform, sr)
            result["diarization"] = {
                "num_speakers": len({s.speaker_id for s in segments}),
                "segments": [s.to_dict() for s in segments],
            }

            # Align transcript to speakers (if ASR available)
            if self.enable_asr and "asr" in result:
                aligner = self._get_aligner()
                text = result["asr"]["text"]
                dummy_words = TranscriptAligner.create_dummy_words(
                    text, result["duration_sec"]
                )
                aligned = aligner.align(dummy_words, segments)
                result["aligned_transcript"] = aligned.to_text()

        # ── Classification ───────────────────────────────────
        if self.enable_classification:
            classifier = self._get_classifier()
            if self.enable_features and "features" in result:
                mel = self._get_feature_pipeline().mel.extract(waveform, sr)
            else:
                from src.features.mel_spectrogram import MelSpectrogramExtractor
                mel = MelSpectrogramExtractor().extract(waveform, sr)
            cls_result = classifier.predict(mel)
            result["classification"] = cls_result

        # ── Timing ───────────────────────────────────────────
        total_ms = (time.perf_counter() - t_start) * 1000
        result["total_latency_ms"] = round(total_ms, 2)

        logger.info("Pipeline complete — %.1f ms total", total_ms)
        return result

    # ── Config loader ────────────────────────────────────────

    @staticmethod
    def from_config(config_path: str) -> "HybridPipeline":
        """Create a HybridPipeline from a YAML config file."""
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return HybridPipeline(config)

    def __repr__(self) -> str:
        return (
            f"HybridPipeline(asr={self.asr_backend}, "
            f"features={self.enable_features}, "
            f"diarization={self.enable_diarization}, "
            f"classification={self.enable_classification})"
        )
