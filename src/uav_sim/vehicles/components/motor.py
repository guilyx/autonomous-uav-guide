# Erwin Lejeune - 2026-02-16
"""First-order motor model with RPM limits and thrust/torque curves.

Reference: R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles,"
IEEE RAM, 2012. DOI: 10.1109/MRA.2012.2206474
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Motor:
    """Single brushless motor with first-order lag dynamics.

    Parameters:
        k_thrust: Thrust coefficient ``T = k_thrust * omega^2`` [N/(rad/s)^2].
        k_torque: Reaction torque coefficient ``Q = k_torque * omega^2`` [Nm/(rad/s)^2].
        tau: Motor time constant [s].
        omega_min: Minimum angular velocity [rad/s].
        omega_max: Maximum angular velocity [rad/s].
        direction: +1 for CCW, -1 for CW (affects reaction torque sign).
    """

    def __init__(
        self,
        k_thrust: float = 8.55e-6,
        k_torque: float = 1.36e-7,
        tau: float = 0.02,
        omega_min: float = 0.0,
        omega_max: float = 1100.0,
        direction: int = 1,
    ) -> None:
        self.k_thrust = k_thrust
        self.k_torque = k_torque
        self.tau = tau
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.direction = direction

        self.omega: float = 0.0

    def reset(self, omega: float = 0.0) -> None:
        """Reset motor state to a given angular velocity."""
        self.omega = np.clip(omega, self.omega_min, self.omega_max)

    def step(self, omega_cmd: float, dt: float) -> None:
        """Advance the motor by one time step (first-order lag).

        Args:
            omega_cmd: Commanded angular velocity [rad/s].
            dt: Time step [s].
        """
        omega_cmd = np.clip(omega_cmd, self.omega_min, self.omega_max)
        alpha = dt / (self.tau + dt)
        self.omega = (1.0 - alpha) * self.omega + alpha * omega_cmd
        self.omega = np.clip(self.omega, self.omega_min, self.omega_max)

    @property
    def thrust(self) -> float:
        """Current thrust [N]."""
        return float(self.k_thrust * self.omega**2)

    @property
    def torque(self) -> float:
        """Current reaction torque [Nm] (sign follows motor direction)."""
        return float(self.direction * self.k_torque * self.omega**2)

    def thrust_to_omega(self, thrust: float) -> float:
        """Convert a desired thrust to the required angular velocity.

        Args:
            thrust: Desired thrust [N] (clamped to feasible range).

        Returns:
            Angular velocity [rad/s].
        """
        thrust = max(0.0, thrust)
        omega = np.sqrt(thrust / self.k_thrust)
        return float(np.clip(omega, self.omega_min, self.omega_max))

    def get_state(self) -> NDArray[np.floating]:
        """Return motor state as array ``[omega, thrust, torque]``."""
        return np.array([self.omega, self.thrust, self.torque])
