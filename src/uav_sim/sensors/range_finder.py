# Erwin Lejeune - 2026-02-17
"""Single-beam downward-facing range finder (altimeter)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.sensors.base import Sensor


class RangeFinder(Sensor):
    """Downward-facing range sensor for altitude estimation.

    Returns scalar altitude above ground (or max_range if too high).
    """

    def __init__(
        self,
        max_range: float = 40.0,
        noise_std: float = 0.05,
        rate_hz: float = 50.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(rate_hz, seed)
        self.max_range = max_range
        self.noise_std = noise_std

    def sense(self, state: NDArray[np.floating], world=None) -> NDArray[np.floating]:
        alt = max(0.0, state[2]) + self._rng.normal(0, self.noise_std)
        alt = np.clip(alt, 0, self.max_range)
        self._last_measurement = np.array([alt])
        return self._last_measurement
