# Erwin Lejeune - 2026-02-17
"""IMU sensor model with bias, noise, and saturation.

Reference: N. Trawny, S. I. Roumeliotis, "Indirect Kalman Filter for 3D
Attitude Estimation," TR, 2005.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.sensors.base import Sensor


class IMU(Sensor):
    """6-axis IMU: 3-axis accelerometer + 3-axis gyroscope.

    Returns a 6-vector: [ax, ay, az, gx, gy, gz].
    """

    def __init__(
        self,
        accel_noise_std: float = 0.1,
        gyro_noise_std: float = 0.01,
        accel_bias_std: float = 0.005,
        gyro_bias_std: float = 0.001,
        rate_hz: float = 200.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(rate_hz, seed)
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self._accel_bias = self._rng.normal(0, accel_bias_std, 3)
        self._gyro_bias = self._rng.normal(0, gyro_bias_std, 3)

    def sense(self, state: NDArray[np.floating], world=None) -> NDArray[np.floating]:
        accel_true = state[6:9] if len(state) >= 9 else np.zeros(3)
        gyro_true = state[9:12] if len(state) >= 12 else np.zeros(3)
        accel = accel_true + self._accel_bias + self._rng.normal(0, self.accel_noise_std, 3)
        gyro = gyro_true + self._gyro_bias + self._rng.normal(0, self.gyro_noise_std, 3)
        self._last_measurement = np.concatenate([accel, gyro])
        return self._last_measurement
