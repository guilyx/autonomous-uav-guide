# Erwin Lejeune - 2026-02-17
"""2-axis / 3-axis gimbal model for camera stabilisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GimbalParams:
    """Gimbal physical limits."""

    pitch_range: tuple[float, float] = (-np.pi / 2, np.pi / 4)
    yaw_range: tuple[float, float] = (-np.pi, np.pi)
    max_rate: float = 2.0  # rad/s


class Gimbal:
    """Simple 2-axis gimbal with rate-limited tracking.

    State: [pitch, yaw] relative to vehicle body.
    Command: desired [pitch, yaw].
    """

    def __init__(self, params: GimbalParams | None = None) -> None:
        self.params = params or GimbalParams()
        self.pitch = 0.0
        self.yaw = 0.0

    @property
    def orientation(self) -> NDArray[np.floating]:
        return np.array([self.pitch, self.yaw])

    def command(self, desired_pitch: float, desired_yaw: float, dt: float) -> None:
        """Move gimbal towards desired angles with rate limiting."""
        dp = np.clip(
            desired_pitch - self.pitch,
            -self.params.max_rate * dt,
            self.params.max_rate * dt,
        )
        dy = np.clip(
            desired_yaw - self.yaw,
            -self.params.max_rate * dt,
            self.params.max_rate * dt,
        )
        self.pitch = np.clip(self.pitch + dp, *self.params.pitch_range)
        self.yaw = np.clip(self.yaw + dy, *self.params.yaw_range)

    def get_rotation_matrix(self) -> NDArray[np.floating]:
        """Return the gimbal rotation matrix (body → gimbal frame)."""
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return Rp @ Ry
