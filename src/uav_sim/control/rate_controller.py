# Erwin Lejeune - 2026-02-15
"""Body-rate PID controller (innermost loop).

Maps desired body rates ``[p_des, q_des, r_des]`` to torques
``[tau_x, tau_y, tau_z]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RateControllerConfig:
    kp: NDArray[np.floating] = None  # type: ignore[assignment]
    kd: NDArray[np.floating] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kp is None:
            self.kp = np.array([0.8, 0.8, 0.4])
        if self.kd is None:
            self.kd = np.array([0.05, 0.05, 0.02])


class RateController:
    """PID on body angular rates → body torques.

    Parameters
    ----------
    config : gains dataclass (default tuned for 250mm quad).
    """

    def __init__(self, config: RateControllerConfig | None = None) -> None:
        self.cfg = config or RateControllerConfig()
        self._prev_error = np.zeros(3)
        self._first = True

    def reset(self) -> None:
        self._prev_error = np.zeros(3)
        self._first = True

    def compute(
        self,
        angular_velocity: NDArray[np.floating],
        desired_rates: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        """Return ``[tau_x, tau_y, tau_z]``."""
        error = desired_rates - angular_velocity
        if self._first:
            deriv = np.zeros(3)
            self._first = False
        else:
            deriv = (error - self._prev_error) / dt if dt > 0 else np.zeros(3)
        self._prev_error = error.copy()
        return self.cfg.kp * error + self.cfg.kd * deriv
