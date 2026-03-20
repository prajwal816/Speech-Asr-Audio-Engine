"""
Whisper-based ASR module.

Wraps HuggingFace ``openai/whisper-*`` for inference and simulated
fine-tuning with configurable parameters.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperASR:
    """OpenAI Whisper ASR wrapper.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``openai/whisper-small``).
    language : str
        Target language code.
    task : str
        ``"transcribe"`` or ``"translate"``.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "en",
        task: str = "transcribe",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.task = task
        self.device = torch.device(device)

        logger.info("Loading Whisper model: %s", model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Whisper model loaded on %s.", self.device)

    # ── Inference ────────────────────────────────────────────

    def transcribe(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
        return_timestamps: bool = False,
    ) -> dict[str, Any]:
        """Transcribe a single waveform.

        Parameters
        ----------
        waveform : np.ndarray
            Audio samples (float32, mono, 16 kHz recommended).
        sr : int
            Sample rate of the waveform.
        return_timestamps : bool
            If True, return word-level timestamps.

        Returns
        -------
        dict
            ``{"text": str, "language": str, "latency_ms": float}``
        """
        t0 = time.perf_counter()

        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features.to(self.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task=self.task
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
            )

        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Whisper transcription (%.1f ms): %s", latency_ms, transcription[:80]
        )

        return {
            "text": transcription,
            "language": self.language,
            "latency_ms": round(latency_ms, 2),
        }

    def transcribe_batch(
        self, waveforms: list[np.ndarray], sr: int = 16000
    ) -> list[dict[str, Any]]:
        """Transcribe a batch of waveforms sequentially.

        Parameters
        ----------
        waveforms : list[np.ndarray]
            List of audio waveforms.
        sr : int
            Common sample rate.

        Returns
        -------
        list[dict]
        """
        results = []
        for i, wav in enumerate(waveforms):
            logger.debug("Batch item %d / %d", i + 1, len(waveforms))
            results.append(self.transcribe(wav, sr=sr))
        return results

    # ── Simulated fine-tuning ────────────────────────────────

    def fine_tune(
        self,
        train_data: list[dict[str, Any]],
        epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 8,
        warmup_steps: int = 500,
    ) -> dict[str, Any]:
        """Simulated fine-tuning loop.

        In production this would use a real dataset and trainer; here we
        simulate the training loop to demonstrate the API surface and
        produce representative metrics.

        Parameters
        ----------
        train_data : list[dict]
            Each item: ``{"audio": np.ndarray, "text": str}``.
        epochs : int
            Number of training epochs.
        learning_rate : float
            Peak learning rate.
        batch_size : int
            Mini-batch size.
        warmup_steps : int
            Linear warmup steps.

        Returns
        -------
        dict
            Training summary with simulated loss curve.
        """
        logger.info(
            "Starting simulated Whisper fine-tuning: %d epochs, lr=%.1e, bs=%d",
            epochs,
            learning_rate,
            batch_size,
        )

        history: list[dict[str, float]] = []
        n_samples = max(len(train_data), 1)
        steps_per_epoch = max(n_samples // batch_size, 1)

        for epoch in range(1, epochs + 1):
            # Simulate decreasing loss
            simulated_loss = 1.0 / (epoch + 0.5) + np.random.uniform(0, 0.05)
            simulated_wer = max(0.05, 0.25 - 0.06 * epoch + np.random.uniform(0, 0.02))

            history.append(
                {"epoch": epoch, "loss": round(simulated_loss, 4), "wer": round(simulated_wer, 4)}
            )
            logger.info(
                "Epoch %d/%d — loss: %.4f, WER: %.2f%%",
                epoch,
                epochs,
                simulated_loss,
                simulated_wer * 100,
            )

        result = {
            "model": self.model_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "samples": n_samples,
            "steps_per_epoch": steps_per_epoch,
            "history": history,
            "final_loss": history[-1]["loss"],
            "final_wer": history[-1]["wer"],
        }
        logger.info("Fine-tuning complete — final WER: %.2f%%", result["final_wer"] * 100)
        return result

    def __repr__(self) -> str:
        return f"WhisperASR(model='{self.model_name}', lang='{self.language}', device={self.device})"
