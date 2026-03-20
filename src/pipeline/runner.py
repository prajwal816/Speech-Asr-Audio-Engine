"""
Pipeline CLI runner.

Usage::

    python -m src.pipeline.runner --config configs/default.yaml --audio path/to/audio.wav
    python -m src.pipeline.runner --config configs/default.yaml --demo
"""

from __future__ import annotations

import argparse
import json
import sys
import os

import numpy as np

from src.pipeline.hybrid_pipeline import HybridPipeline
from src.utils.logger import get_logger

logger = get_logger("runner")


def _run_demo(pipeline: HybridPipeline) -> dict:
    """Run the pipeline on a synthetic sine-wave demo signal."""
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # 440 Hz sine wave + noise
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t)).astype(np.float32)

    logger.info("Running demo with synthetic %.1f s sine wave", duration)

    # Disable ASR for demo (models may not be downloaded)
    pipeline.enable_asr = False

    result = pipeline.process(waveform=waveform, sr=sr)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Speech Recognition & Audio Classification Engine — Pipeline Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to input audio file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on a synthetic demo signal (no audio file needed).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results JSON.",
    )
    args = parser.parse_args(argv)

    # Validate
    if not args.demo and args.audio is None:
        parser.error("Provide --audio <path> or --demo")

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    # Build pipeline
    pipeline = HybridPipeline.from_config(args.config)

    # Process
    if args.demo:
        result = _run_demo(pipeline)
    else:
        if not os.path.exists(args.audio):
            logger.error("Audio file not found: %s", args.audio)
            sys.exit(1)
        result = pipeline.process(audio_path=args.audio)

    # Output
    output_json = json.dumps(result, indent=2, default=str)
    print(output_json)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info("Results saved → %s", args.output)


if __name__ == "__main__":
    main()
