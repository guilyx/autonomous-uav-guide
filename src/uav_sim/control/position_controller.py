# Erwin Lejeune - 2026-02-15
"""Position controller — maps desired position to desired velocity.

The outermost control loop.  It produces a velocity setpoint that is
clamped to ``max_velocity``, giving smooth, predictable motion instead
of the old ``_limit_target`` approach.

Output: ``desired_velocity`` (world frame) for the :class:`VelocityController`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PositionControllerConfig:
    kp: float = 1.0
    kd: float = 0.4
    max_velocity: float = 3.0


class PositionController:
    """P(D) on position → desired velocity (clamped).

    Parameters
    ----------
    config : gains and velocity saturation limit.
    """

    def __init__(self, config: PositionControllerConfig | None = None) -> None:
        self.cfg = config or PositionControllerConfig()
        self._prev_error = np.zeros(3)
        self._first = True

    def reset(self) -> None:
        self._prev_error = np.zeros(3)
        self._first = True

    def compute(
        self,
        position: NDArray[np.floating],
        desired_position: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        """Return clamped ``desired_velocity`` [m/s] in world frame."""
        error = desired_position - position
        if self._first:
            deriv = np.zeros(3)
            self._first = False
        else:
            deriv = (error - self._prev_error) / dt if dt > 0 else np.zeros(3)
        self._prev_error = error.copy()

        vel = self.cfg.kp * error + self.cfg.kd * deriv
        speed = float(np.linalg.norm(vel))
        if speed > self.cfg.max_velocity:
            vel = vel * (self.cfg.max_velocity / speed)
        return vel
