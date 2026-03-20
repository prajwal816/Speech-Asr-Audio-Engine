"""
Experiment runner.

Runs a full experiment lifecycle from a YAML config:
    1. Feature extraction on sample data
    2. ASR evaluation (simulated)
    3. Classification training + evaluation (simulated)
    4. Metric logging via ExperimentTracker

Usage::

    python experiments/run_experiment.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.feature_pipeline import FeaturePipeline
from src.asr.evaluator import ASREvaluator
from src.classification.classifier import AudioEventClassifier
from src.diarization.segmenter import SpeakerSegmenter
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.logger import get_logger

logger = get_logger("experiment")


def _generate_synthetic_audio(sr: int = 16000, duration: float = 5.0) -> np.ndarray:
    """Create a synthetic audio signal for testing."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    signal = (
        0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.05 * np.random.randn(len(t)).astype(np.float32)
    )
    return signal


def run_experiment(config_path: str) -> dict:
    """Execute a complete experiment run."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg.get("experiment", {})
    tracker = ExperimentTracker(
        experiment_name=exp_cfg.get("name", "default"),
        output_dir=cfg.get("paths", {}).get("output_dir", "experiments/outputs"),
        tags=exp_cfg.get("tags", []),
    )
    tracker.log_params(cfg)

    sr = cfg.get("audio", {}).get("sample_rate", 16000)

    # ───────────────────────────────────────────────────────
    # 1. Feature Extraction
    # ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Feature Extraction")
    logger.info("=" * 60)

    waveform = _generate_synthetic_audio(sr=sr, duration=5.0)
    fp = FeaturePipeline(cfg.get("features", {}))
    t0 = time.perf_counter()
    features = fp.extract(waveform, sr)
    feat_ms = (time.perf_counter() - t0) * 1000

    tracker.log_metric("feature_extraction_ms", feat_ms)
    for name, feat in features.items():
        logger.info("  %s shape: %s", name, feat.shape)
        tracker.log_metric(f"feat_{name}_dim", feat.shape[0])

    # ───────────────────────────────────────────────────────
    # 2. Speaker Diarization
    # ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Speaker Diarization")
    logger.info("=" * 60)

    dia_cfg = cfg.get("diarization", {})
    clust_cfg = dia_cfg.get("clustering", {})
    segmenter = SpeakerSegmenter(
        energy_threshold=dia_cfg.get("energy_threshold", 0.01),
        n_speakers=clust_cfg.get("n_clusters", 2),
    )
    segments = segmenter.segment(waveform, sr)
    tracker.log_metric("num_diarization_segments", len(segments))
    tracker.log_metric("num_speakers_detected", len({s.speaker_id for s in segments}))
    for seg in segments[:5]:
        logger.info("  %s", seg.to_dict())

    # ───────────────────────────────────────────────────────
    # 3. ASR Evaluation (simulated references/hypotheses)
    # ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: ASR Evaluation (simulated)")
    logger.info("=" * 60)

    references = [
        "the quick brown fox jumps over the lazy dog",
        "speech recognition is a complex task",
        "multilingual models handle diverse accents",
        "audio features include mfcc and mel spectrogram",
        "deep learning has transformed speech technology",
    ]
    # Simulate slight errors → WER ~5-10 %
    hypotheses = [
        "the quick brown fox jumps over the lazy dog",
        "speech recognition is a complex task",
        "multilingual models handle divers accents",
        "audio features include mfcc and mel spectrogram",
        "deep learning has transformed speech technolgy",
    ]

    evaluator = ASREvaluator()
    asr_result = evaluator.evaluate(references, hypotheses)
    report = ASREvaluator.format_report(asr_result)
    logger.info("\n%s", report)

    tracker.log_metric("wer", asr_result["wer"])
    tracker.log_metric("cer", asr_result["cer"])

    # ───────────────────────────────────────────────────────
    # 4. Audio Classification (simulated training)
    # ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Audio Classification (simulated)")
    logger.info("=" * 60)

    cls_cfg = cfg.get("classification", {})
    model_cfg = cls_cfg.get("model", {})
    classifier = AudioEventClassifier(
        num_classes=cls_cfg.get("num_classes", 10),
        labels=cls_cfg.get("labels"),
        n_mels=model_cfg.get("n_mels", 128),
        hidden_channels=model_cfg.get("hidden_channels"),
        dropout=model_cfg.get("dropout", 0.3),
    )

    # Predict on extracted mel spectrogram (untrained model → random)
    mel_spec = features["mel_spectrogram"]
    cls_result = classifier.predict(mel_spec)
    logger.info("  Predicted labels: %s", cls_result["labels"])
    logger.info("  Scores: %s", json.dumps(cls_result["scores"], indent=4))

    tracker.log_metric("classification_latency_ms", cls_result["latency_ms"])

    # ───────────────────────────────────────────────────────
    # 5. Summary
    # ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    summary = tracker.summary()
    logger.info("  Run ID    : %s", summary["run_id"])
    logger.info("  WER       : %.2f%%", summary["metrics"].get("wer", 0) * 100)
    logger.info("  CER       : %.2f%%", summary["metrics"].get("cer", 0) * 100)
    logger.info("  Feat time : %.1f ms", summary["metrics"].get("feature_extraction_ms", 0))
    logger.info("  Cls time  : %.1f ms", summary["metrics"].get("classification_latency_ms", 0))

    # Save run
    run_path = tracker.save()
    logger.info("  Run saved → %s", run_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    run_experiment(args.config)


if __name__ == "__main__":
    main()
