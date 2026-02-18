# Erwin Lejeune - 2026-02-15
"""Attitude P controller.

Maps desired Euler angles ``[phi_des, theta_des, psi_des]`` and a thrust
scalar to desired body rates via proportional gain, then feeds into the
:class:`RateController` to produce torques.

The D-term is intentionally omitted: the RateController already provides
angular-rate damping, and error-derivative D-terms on attitude cause
catastrophic 1/dt noise amplification at typical sim rates (dt=0.005 s).

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

    def __post_init__(self) -> None:
        if self.kp is None:
            self.kp = np.array([4.5, 4.5, 2.0])


class AttitudeController:
    """P on Euler angles -> desired body rates -> torques via RateController.

    Parameters
    ----------
    config : attitude P gains.
    rate_config : inner rate-loop gains.
    """

    def __init__(
        self,
        config: AttitudeControllerConfig | None = None,
        rate_config: RateControllerConfig | None = None,
    ) -> None:
        self.cfg = config or AttitudeControllerConfig()
        self.rate_ctrl = RateController(rate_config)

    def reset(self) -> None:
        self.rate_ctrl.reset()

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

        desired_rates = self.cfg.kp * error
        torques = self.rate_ctrl.compute(angular_velocity, desired_rates, dt)
        return np.array([thrust, torques[0], torques[1], torques[2]])


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
