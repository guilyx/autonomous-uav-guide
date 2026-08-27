# Erwin Lejeune - 2026-02-15
"""Velocity controller — maps desired velocity to attitude + thrust.

This is where the conversion from "twist commands" to desired roll/pitch/yaw
and collective thrust happens, mirroring the velocity-control loop in PX4.

Output: ``(desired_euler, thrust)`` ready for the :class:`AttitudeController`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class VelocityControllerConfig:
    kp_xy: float = 1.2
    ki_xy: float = 0.03
    kp_z: float = 3.0
    ki_z: float = 0.1
    max_velocity: float = 5.0
    max_tilt: float = 0.25  # ~14 deg — matches CascadedPID tuning
    mass: float = 1.5
    gravity: float = 9.81
    integral_limit: float = 2.0


class VelocityController:
    """PID on world-frame velocity → desired attitude + thrust.

    Parameters
    ----------
    config : gains and physical params.
    """

    def __init__(self, config: VelocityControllerConfig | None = None) -> None:
        self.cfg = config or VelocityControllerConfig()
        self._integral_xy = np.zeros(2)
        self._integral_z = 0.0

    def reset(self) -> None:
        self._integral_xy = np.zeros(2)
        self._integral_z = 0.0

    def compute(
        self,
        velocity: NDArray[np.floating],
        yaw: float,
        desired_velocity: NDArray[np.floating],
        dt: float,
    ) -> tuple[NDArray[np.floating], float]:
        """Return ``(desired_euler [phi, theta, psi], thrust)``."""
        c = self.cfg
        error = desired_velocity - velocity

        self._integral_xy += error[:2] * dt
        self._integral_xy = np.clip(self._integral_xy, -c.integral_limit, c.integral_limit)
        self._integral_z += error[2] * dt
        self._integral_z = float(np.clip(self._integral_z, -c.integral_limit, c.integral_limit))

        ax = c.kp_xy * error[0] + c.ki_xy * self._integral_xy[0]
        ay = c.kp_xy * error[1] + c.ki_xy * self._integral_xy[1]
        az = c.kp_z * error[2] + c.ki_z * self._integral_z

        cy, sy = np.cos(yaw), np.sin(yaw)
        ax_rot = ax * cy + ay * sy
        ay_rot = ax * sy - ay * cy

        phi_des = float(np.clip(np.arctan2(ay_rot, c.gravity + az), -c.max_tilt, c.max_tilt))
        theta_des = float(np.clip(np.arctan2(ax_rot, c.gravity + az), -c.max_tilt, c.max_tilt))

        thrust = c.mass * (c.gravity + az) / (np.cos(phi_des) * np.cos(theta_des) + 1e-6)
        thrust = float(np.clip(thrust, 0.0, c.mass * c.gravity * 2.0))

        return np.array([phi_des, theta_des, yaw]), thrust
