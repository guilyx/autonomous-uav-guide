# Erwin Lejeune - 2026-02-16
"""Complementary filter for attitude estimation from gyro + accelerometer.

Reference: R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear Complementary
Filters on the Special Orthogonal Group," IEEE TAC, 2008.
DOI: 10.1109/TAC.2008.923738
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ComplementaryFilter:
    """Lightweight attitude estimator fusing gyroscope and accelerometer.

    Gyro provides high-frequency attitude changes (integrated angular rates),
    while the accelerometer provides a low-frequency gravity reference.

    ``attitude = α * (attitude + gyro * dt) + (1 - α) * accel_attitude``

    Parameters:
        alpha: Blending parameter in (0, 1). Higher → more gyro trust.
    """

    def __init__(self, alpha: float = 0.98) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.roll: float = 0.0
        self.pitch: float = 0.0

    def reset(self, roll: float = 0.0, pitch: float = 0.0) -> None:
        self.roll = roll
        self.pitch = pitch

    def update(
        self,
        gyro: NDArray[np.floating],
        accel: NDArray[np.floating],
        dt: float,
    ) -> tuple[float, float]:
        """Fuse gyro and accelerometer to estimate roll and pitch.

        Args:
            gyro: ``[p, q, r]`` angular rates [rad/s] (body frame).
            accel: ``[ax, ay, az]`` specific force [m/s²] (body frame).
            dt: Time step [s].

        Returns:
            ``(roll, pitch)`` estimates [rad].
        """
        # Attitude from accelerometer (valid in quasi-static conditions).
        accel_roll = np.arctan2(accel[1], accel[2])
        accel_pitch = np.arctan2(-accel[0], np.sqrt(accel[1] ** 2 + accel[2] ** 2))

        # Integrate gyro rates.
        self.roll = self.alpha * (self.roll + gyro[0] * dt) + (1.0 - self.alpha) * accel_roll
        self.pitch = self.alpha * (self.pitch + gyro[1] * dt) + (1.0 - self.alpha) * accel_pitch

        return self.roll, self.pitch

    @property
    def attitude(self) -> NDArray[np.floating]:
        """Current ``[roll, pitch]`` estimate."""
        return np.array([self.roll, self.pitch])
