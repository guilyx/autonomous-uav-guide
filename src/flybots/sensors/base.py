# Erwin Lejeune - 2026-02-18
"""Abstract sensor base class and sensor mount descriptor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class SensorMount:
    """Describes how a sensor is attached to a vehicle.

    Parameters
    ----------
    position : Body-frame offset [x, y, z] from vehicle CoG (metres).
    orientation : Body-frame Euler angles [roll, pitch, yaw] (radians).
    """

    position: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))
    orientation: NDArray[np.floating] = field(default_factory=lambda: np.zeros(3))

    def rotation_matrix(self) -> NDArray[np.floating]:
        """Return 3x3 rotation from sensor frame to body frame."""
        r, p, y = self.orientation
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx


class Sensor(ABC):
    """Abstract sensor that produces a measurement from state + environment.

    Parameters
    ----------
    rate_hz : Measurement rate.
    seed : Random seed for reproducible noise.
    mount : Where the sensor sits on the vehicle body.
    """

    def __init__(
        self,
        rate_hz: float = 100.0,
        seed: int | None = None,
        mount: SensorMount | None = None,
    ) -> None:
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self._rng = np.random.default_rng(seed)
        self._last_measurement: NDArray[np.floating] | None = None
        self.mount = mount or SensorMount()

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
