# Erwin Lejeune - 2026-02-15
"""Position controller — maps desired position to desired velocity.

Uses measured velocity as the derivative term instead of
finite-differencing position error, which avoids 1/dt noise
amplification and setpoint kick.

Output: ``desired_velocity`` (world frame) for the :class:`VelocityController`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PositionControllerConfig:
    kp: float = 1.0
    kd: float = 0.6
    max_velocity: float = 2.0


class PositionController:
    """PD on position -> desired velocity (clamped).

    The D-term uses *measured velocity* (passed via :meth:`compute`)
    rather than differentiating the error signal.

    Parameters
    ----------
    config : gains and velocity saturation limit.
    """

    def __init__(self, config: PositionControllerConfig | None = None) -> None:
        self.cfg = config or PositionControllerConfig()

    def reset(self) -> None:
        pass

    def compute(
        self,
        position: NDArray[np.floating],
        desired_position: NDArray[np.floating],
        dt: float,  # noqa: ARG002 — kept for interface compat
        velocity: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Return clamped ``desired_velocity`` [m/s] in world frame."""
        error = desired_position - position
        vel_cmd = self.cfg.kp * error
        if velocity is not None:
            vel_cmd -= self.cfg.kd * velocity
        speed = float(np.linalg.norm(vel_cmd))
        if speed > self.cfg.max_velocity:
            vel_cmd = vel_cmd * (self.cfg.max_velocity / speed)
        return vel_cmd
