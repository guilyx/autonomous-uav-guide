# Erwin Lejeune - 2026-02-15
"""Body-rate P controller (innermost loop).

Maps desired body rates ``[p_des, q_des, r_des]`` to torques
``[tau_x, tau_y, tau_z]`` using proportional gain scaled to the
vehicle inertia.

Gains are kept intentionally conservative to avoid the aggressive
angular accelerations that cause oscillation. For a 250 mm quad
(I_xx ~ 0.008 kg m^2), kp ~ 0.05 gives ~6 rad/s^2 per rad/s error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RateControllerConfig:
    kp: NDArray[np.floating] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kp is None:
            self.kp = np.array([0.055, 0.055, 0.10])


class RateController:
    """Proportional controller on body angular rates -> body torques.

    Parameters
    ----------
    config : gains dataclass (default tuned for 250 mm quad).
    """

    def __init__(self, config: RateControllerConfig | None = None) -> None:
        self.cfg = config or RateControllerConfig()

    def reset(self) -> None:
        pass

    def compute(
        self,
        angular_velocity: NDArray[np.floating],
        desired_rates: NDArray[np.floating],
        dt: float,  # noqa: ARG002 — kept for interface compat
    ) -> NDArray[np.floating]:
        """Return ``[tau_x, tau_y, tau_z]``."""
        error = desired_rates - angular_velocity
        return self.cfg.kp * error
