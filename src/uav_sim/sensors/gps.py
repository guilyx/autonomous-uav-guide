# Erwin Lejeune - 2026-02-17
"""GPS sensor model with noise and optional dropout."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.sensors.base import Sensor


class GPS(Sensor):
    """GPS position sensor with Gaussian noise and dropout probability.

    Returns a 3-vector: [x, y, z] or NaN if dropped.
    """

    def __init__(
        self,
        noise_std: float = 0.5,
        dropout_prob: float = 0.0,
        rate_hz: float = 10.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(rate_hz, seed)
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def sense(self, state: NDArray[np.floating], world=None) -> NDArray[np.floating]:
        if self._rng.random() < self.dropout_prob:
            self._last_measurement = np.full(3, np.nan)
        else:
            self._last_measurement = state[:3] + self._rng.normal(0, self.noise_std, 3)
        return self._last_measurement
