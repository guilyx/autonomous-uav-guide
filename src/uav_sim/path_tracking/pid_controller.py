# Erwin Lejeune - 2026-02-16
"""Cascaded PID controller: outer position loop + inner attitude loop.

Reference: L. R. G. Carrillo et al., "Quad Rotorcraft Control,"
Springer, 2013, Chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class PIDGains:
    """PID gains for a single axis."""

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 1.0


class PIDAxis:
    """Single-axis PID controller with anti-windup clamping."""

    def __init__(self, gains: PIDGains) -> None:
        self.gains = gains
        self.integral: float = 0.0
        self.prev_error: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Compute PID output for one time step.

        Args:
            error: Current error (setpoint - measurement).
            dt: Time step [s].

        Returns:
            Control output.
        """
        g = self.gains
        self.integral += error * dt
        self.integral = np.clip(self.integral, -g.integral_limit, g.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return g.kp * error + g.ki * self.integral + g.kd * derivative


@dataclass
class CascadedPIDConfig:
    """Configuration for the cascaded position + attitude PID controller.

    The outer loop (position) outputs desired accelerations which are
    converted to desired angles and thrust. The inner loop (attitude)
    outputs body torques.
    """

    pos_x: PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=1.2, kd=3.5))
    pos_y: PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=1.2, kd=3.5))
    pos_z: PIDGains = field(default_factory=lambda: PIDGains(kp=10.0, ki=5.0, kd=5.0))
    att_phi: PIDGains = field(default_factory=lambda: PIDGains(kp=8.0, ki=0.0, kd=3.5))
    att_theta: PIDGains = field(default_factory=lambda: PIDGains(kp=8.0, ki=0.0, kd=3.5))
    att_psi: PIDGains = field(default_factory=lambda: PIDGains(kp=6.0, ki=1.0, kd=3.0))
    mass: float = 0.027
    gravity: float = 9.81


class CascadedPIDController:
    """Cascaded PID: position → desired attitude → torques.

    Outer loop: position PID → desired accelerations → thrust + desired φ, θ.
    Inner loop: attitude PID → body torques τx, τy, τz.

    Output: ``[T, τx, τy, τz]`` wrench for the mixer.
    """

    def __init__(self, config: CascadedPIDConfig | None = None) -> None:
        self.config = config or CascadedPIDConfig()
        c = self.config

        self.pid_x = PIDAxis(c.pos_x)
        self.pid_y = PIDAxis(c.pos_y)
        self.pid_z = PIDAxis(c.pos_z)
        self.pid_phi = PIDAxis(c.att_phi)
        self.pid_theta = PIDAxis(c.att_theta)
        self.pid_psi = PIDAxis(c.att_psi)

    def reset(self) -> None:
        """Reset all PID integrators."""
        for pid in [
            self.pid_x,
            self.pid_y,
            self.pid_z,
            self.pid_phi,
            self.pid_theta,
            self.pid_psi,
        ]:
            pid.reset()

    def compute(
        self,
        state: NDArray[np.floating],
        target_pos: NDArray[np.floating],
        target_yaw: float = 0.0,
        dt: float = 0.001,
    ) -> NDArray[np.floating]:
        """Compute control wrench given current state and target position.

        Args:
            state: 12-element state ``[x,y,z,φ,θ,ψ,vx,vy,vz,p,q,r]``.
            target_pos: Desired ``[x, y, z]``.
            target_yaw: Desired yaw angle [rad].
            dt: Time step [s].

        Returns:
            ``[T, τx, τy, τz]`` body-frame wrench.
        """
        c = self.config
        x, y, z, phi, theta, psi = state[:6]

        # --- Outer loop: position PID → desired accelerations ---
        ex = target_pos[0] - x
        ey = target_pos[1] - y
        ez = target_pos[2] - z

        ax_des = self.pid_x.compute(ex, dt)
        ay_des = self.pid_y.compute(ey, dt)
        az_des = self.pid_z.compute(ez, dt)

        # Total thrust (along body z-axis).
        T = c.mass * (az_des + c.gravity) / (np.cos(phi) * np.cos(theta) + 1e-6)
        T = max(0.0, T)

        # Desired roll and pitch from desired accelerations.
        phi_des = np.arcsin(
            np.clip(
                (ax_des * np.sin(psi) - ay_des * np.cos(psi)) * c.mass / (T + 1e-6),
                -1.0,
                1.0,
            )
        )
        theta_des = np.arcsin(
            np.clip(
                (ax_des * np.cos(psi) + ay_des * np.sin(psi))
                * c.mass
                / (T * np.cos(phi_des) + 1e-6),
                -1.0,
                1.0,
            )
        )

        # --- Inner loop: attitude PID → body torques ---
        e_phi = phi_des - phi
        e_theta = theta_des - theta
        e_psi = self._wrap_angle(target_yaw - psi)

        tau_x = self.pid_phi.compute(e_phi, dt)
        tau_y = self.pid_theta.compute(e_theta, dt)
        tau_z = self.pid_psi.compute(e_psi, dt)

        return np.array([T, tau_x, tau_y, tau_z])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi
