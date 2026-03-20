"""
wav2vec2-based ASR module.

Wraps HuggingFace ``facebook/wav2vec2-base-960h`` for CTC-based
speech recognition with greedy decoding.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Wav2Vec2ASR:
    """Facebook wav2vec 2.0 ASR wrapper (CTC decoding).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)

        logger.info("Loading wav2vec2 model: %s", model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("wav2vec2 model loaded on %s.", self.device)

    # ── Inference ────────────────────────────────────────────

    def transcribe(
        self,
        waveform: np.ndarray,
        sr: int = 16000,
    ) -> dict[str, Any]:
        """Transcribe a single waveform using CTC greedy decoding.

        Parameters
        ----------
        waveform : np.ndarray
            Audio samples (float32, mono, 16 kHz).
        sr : int
            Sample rate.

        Returns
        -------
        dict
            ``{"text": str, "latency_ms": float, "logits_shape": tuple}``
        """
        t0 = time.perf_counter()

        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].strip()

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "wav2vec2 transcription (%.1f ms): %s", latency_ms, transcription[:80]
        )

        return {
            "text": transcription,
            "latency_ms": round(latency_ms, 2),
            "logits_shape": tuple(logits.shape),
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

    def __repr__(self) -> str:
        return f"Wav2Vec2ASR(model='{self.model_name}', device={self.device})"
