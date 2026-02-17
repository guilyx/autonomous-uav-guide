# Erwin Lejeune - 2026-02-17
"""Abstract sensor base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Sensor(ABC):
    """Abstract sensor that produces a measurement from state + environment."""

    def __init__(self, rate_hz: float = 100.0, seed: int | None = None) -> None:
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self._rng = np.random.default_rng(seed)
        self._last_measurement: NDArray[np.floating] | None = None

    @property
    def measurement(self) -> NDArray[np.floating] | None:
        return self._last_measurement

    @abstractmethod
    def sense(
        self,
        state: NDArray[np.floating],
        world=None,
    ) -> NDArray[np.floating]:
        """Produce a noisy measurement given vehicle *state* and optional *world*."""
        ...
