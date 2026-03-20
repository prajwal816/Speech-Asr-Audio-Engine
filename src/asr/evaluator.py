"""
ASR evaluation utilities.

Computes Word Error Rate (WER) and Character Error Rate (CER) using
the ``jiwer`` library, with per-sample and aggregate reporting.
"""

from __future__ import annotations

from typing import Any

import jiwer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ASREvaluator:
    """Evaluate ASR hypotheses against ground-truth references.

    Example
    -------
    >>> evaluator = ASREvaluator()
    >>> result = evaluator.evaluate(
    ...     references=["hello world"],
    ...     hypotheses=["hello word"],
    ... )
    >>> print(result["wer"])
    0.5
    """

    WER_TRANSFORM = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    def evaluate(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> dict[str, Any]:
        """Compute aggregate and per-sample WER / CER.

        Parameters
        ----------
        references : list[str]
            Ground-truth transcriptions.
        hypotheses : list[str]
            ASR-produced transcriptions.

        Returns
        -------
        dict
            ``{"wer": float, "cer": float, "samples": list[dict],
              "num_samples": int}``
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Length mismatch: {len(references)} refs vs {len(hypotheses)} hyps"
            )

        # Aggregate WER
        wer = jiwer.wer(
            references,
            hypotheses,
            truth_transform=self.WER_TRANSFORM,
            hypothesis_transform=self.WER_TRANSFORM,
        )

        # Aggregate CER
        cer = jiwer.cer(references, hypotheses)

        # Per-sample metrics
        samples: list[dict[str, Any]] = []
        for ref, hyp in zip(references, hypotheses):
            s_wer = jiwer.wer(
                ref,
                hyp,
                truth_transform=self.WER_TRANSFORM,
                hypothesis_transform=self.WER_TRANSFORM,
            )
            s_cer = jiwer.cer(ref, hyp)
            samples.append({
                "reference": ref,
                "hypothesis": hyp,
                "wer": round(s_wer, 4),
                "cer": round(s_cer, 4),
            })

        result = {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "num_samples": len(references),
            "samples": samples,
        }

        logger.info(
            "ASR Evaluation — WER: %.2f%%, CER: %.2f%% (%d samples)",
            wer * 100,
            cer * 100,
            len(references),
        )
        return result

    @staticmethod
    def format_report(result: dict[str, Any]) -> str:
        """Pretty-print an evaluation result.

        Parameters
        ----------
        result : dict
            Output of :meth:`evaluate`.

        Returns
        -------
        str
        """
        lines = [
            "=" * 60,
            "ASR Evaluation Report",
            "=" * 60,
            f"  Samples : {result['num_samples']}",
            f"  WER     : {result['wer'] * 100:6.2f} %",
            f"  CER     : {result['cer'] * 100:6.2f} %",
            "-" * 60,
        ]
        for i, s in enumerate(result["samples"]):
            lines.append(f"  [{i+1}] REF: {s['reference']}")
            lines.append(f"       HYP: {s['hypothesis']}")
            lines.append(f"       WER: {s['wer']*100:.2f}%  CER: {s['cer']*100:.2f}%")
        lines.append("=" * 60)
        return "\n".join(lines)
