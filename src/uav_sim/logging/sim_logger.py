# Erwin Lejeune — 2026-02-15
"""Lightweight simulation data logger.

Collects per-timestep data, metadata, and summary statistics, then
serialises everything to a single JSON file alongside the simulation GIF.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


class SimLogger:
    """Accumulates simulation data and writes a JSON log.

    Parameters
    ----------
    sim_name:
        Short identifier used as file prefix, e.g. ``"lqr_tracking"``.
    out_dir:
        Directory where ``<sim_name>_log.json`` will be written.
    downsample:
        Keep every *n*-th timestep to limit file size.  ``1`` = keep all.
    """

    def __init__(
        self,
        sim_name: str,
        out_dir: str | Path,
        downsample: int = 1,
    ) -> None:
        self._sim_name = sim_name
        self._out_dir = Path(out_dir)
        self._downsample = max(1, downsample)

        self._metadata: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}
        self._timeseries: dict[str, list[Any]] = {}
        self._completion: dict[str, Any] = {}
        self._step_count = 0

    _REQUIRED_COMPLETION_KEYS = (
        "goal_reached_xy",
        "divergence",
        "stall",
        "timeout_reason",
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metadata(self, key: str, value: Any) -> None:
        """Store a simulation-level metadata entry (algorithm, dt, …)."""
        self._metadata[key] = value

    def log_step(self, **kwargs: Any) -> None:
        """Record one timestep of data.

        Only every *downsample*-th call actually stores data so that
        long simulations stay compact.
        """
        keep = (self._step_count % self._downsample) == 0
        self._step_count += 1
        if not keep:
            return
        for key, value in kwargs.items():
            self._timeseries.setdefault(key, []).append(value)

    def log_summary(self, key: str, value: Any) -> None:
        """Store a final summary metric (mean error, max speed, …)."""
        self._summary[key] = value

    def log_completion(self, **kwargs: Any) -> None:
        """Store standardized completion status fields for a simulation."""
        self._completion.update(kwargs)

    def _validate_completion_schema(self) -> None:
        """Validate mandatory completion fields for flight-coupled simulations."""
        if not bool(self._metadata.get("flight_coupled", False)):
            return
        missing = [k for k in self._REQUIRED_COMPLETION_KEYS if k not in self._completion]
        if missing:
            msg = ", ".join(missing)
            raise ValueError(
                f"Missing required completion keys for '{self._sim_name}': {msg}. "
                "Call log_completion(...) before save()."
            )

    def save(self) -> Path:
        """Write the accumulated data to ``<out_dir>/<sim_name>_log.json``."""
        self._validate_completion_schema()
        payload = {
            "simulation": self._sim_name,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": self._metadata,
            "summary": self._summary,
            "completion": self._completion,
            "timeseries": self._timeseries,
        }
        payload = _to_serializable(payload)

        self._out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._out_dir / f"{self._sim_name}_log.json"
        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        return out_path
