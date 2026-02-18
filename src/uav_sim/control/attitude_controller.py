# Erwin Lejeune - 2026-02-15
"""Attitude PID controller.

Maps desired Euler angles ``[phi_des, theta_des, psi_des]`` and a thrust
scalar to desired body rates ``[p_des, q_des, r_des]`` and passes them
through a :class:`RateController` to produce torques.

Output: ``[T, tau_x, tau_y, tau_z]`` wrench.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.control.rate_controller import RateController, RateControllerConfig


@dataclass
class AttitudeControllerConfig:
    kp: NDArray[np.floating] = None  # type: ignore[assignment]
    kd: NDArray[np.floating] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.kp is None:
            self.kp = np.array([6.0, 6.0, 3.0])
        if self.kd is None:
            self.kd = np.array([1.2, 1.2, 0.5])


class AttitudeController:
    """PID on Euler angles → desired body rates → torques via RateController.

    Parameters
    ----------
    config : attitude PID gains.
    rate_config : inner rate-loop gains.
    """

    def __init__(
        self,
        config: AttitudeControllerConfig | None = None,
        rate_config: RateControllerConfig | None = None,
    ) -> None:
        self.cfg = config or AttitudeControllerConfig()
        self.rate_ctrl = RateController(rate_config)
        self._prev_error = np.zeros(3)
        self._first = True

    def reset(self) -> None:
        self.rate_ctrl.reset()
        self._prev_error = np.zeros(3)
        self._first = True

    def compute(
        self,
        euler: NDArray[np.floating],
        angular_velocity: NDArray[np.floating],
        desired_euler: NDArray[np.floating],
        thrust: float,
        dt: float,
    ) -> NDArray[np.floating]:
        """Return ``[T, tau_x, tau_y, tau_z]`` wrench."""
        error = desired_euler - euler
        error[2] = _wrap(error[2])

        if self._first:
            deriv = np.zeros(3)
            self._first = False
        else:
            deriv = (error - self._prev_error) / dt if dt > 0 else np.zeros(3)
        self._prev_error = error.copy()

        desired_rates = self.cfg.kp * error + self.cfg.kd * deriv
        torques = self.rate_ctrl.compute(angular_velocity, desired_rates, dt)
        return np.array([thrust, torques[0], torques[1], torques[2]])


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
