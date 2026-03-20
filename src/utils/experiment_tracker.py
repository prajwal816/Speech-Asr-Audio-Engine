"""
JSON-based experiment tracker.

Logs parameters, metrics, and artifacts for each experiment run.
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Track experiment runs with params, metrics, and artifacts.

    Parameters
    ----------
    experiment_name : str
        Human-readable name for this experiment.
    output_dir : str
        Directory where run JSON files are stored.
    tags : list[str], optional
        Optional tags for filtering / grouping runs.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiments/outputs",
        tags: Optional[list[str]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.tags = tags or []

        self._run_id = f"{experiment_name}_{int(time.time())}"
        self._start_time = datetime.now(timezone.utc).isoformat()
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, list[float]] = {}
        self._artifacts: list[str] = []

        os.makedirs(output_dir, exist_ok=True)
        logger.info("Experiment '%s' — run %s started.", experiment_name, self._run_id)

    # ── Public API ───────────────────────────────────────────

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyper-parameter."""
        self._params[key] = value
        logger.debug("Param  %s = %s", key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple hyper-parameters at once."""
        self._params.update(params)
        logger.debug("Params logged: %s", list(params.keys()))

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value (appends to history)."""
        self._metrics.setdefault(key, []).append(value)
        step_info = f" (step {step})" if step is not None else ""
        logger.debug("Metric %s = %.6f%s", key, value, step_info)

    def log_artifact(self, path: str) -> None:
        """Register an artifact file path."""
        self._artifacts.append(path)
        logger.debug("Artifact registered: %s", path)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of the current run."""
        return {
            "run_id": self._run_id,
            "experiment": self.experiment_name,
            "tags": self.tags,
            "start_time": self._start_time,
            "params": self._params,
            "metrics": {k: v[-1] for k, v in self._metrics.items()},
            "metrics_history": self._metrics,
            "artifacts": self._artifacts,
        }

    def save(self, path: Optional[str] = None) -> str:
        """Persist the run to a JSON file and return the path."""
        if path is None:
            path = os.path.join(self.output_dir, f"{self._run_id}.json")
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.summary(), fp, indent=2, default=str)
        logger.info("Run saved → %s", path)
        return path

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        """Load a previously saved run from JSON."""
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)

    # ── Dunder ───────────────────────────────────────────────

    def __repr__(self) -> str:
        n_metrics = sum(len(v) for v in self._metrics.values())
        return (
            f"ExperimentTracker(run='{self._run_id}', "
            f"params={len(self._params)}, metric_entries={n_metrics})"
        )
