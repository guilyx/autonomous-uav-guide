# Erwin Lejeune - 2026-02-16
"""Geometric controller on SO(3) for quadrotor trajectory tracking.

Reference: T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control
of a Quadrotor UAV on SE(3)," CDC, 2010. DOI: 10.1109/CDC.2010.5717652
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


def _vee(M: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract vector from skew-symmetric matrix (vee map)."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


@dataclass
class GeometricControllerConfig:
    """Gains for the geometric SE(3) controller.

    Attitude gains (kR, kw) are scaled for the Crazyflie-class inertia
    (~1.66e-5 kg⋅m²).  Position output is clamped via ``max_acc`` to
    prevent extreme attitude commands.
    """

    kx: float = 4.0
    kv: float = 2.8
    kR: float = 8.0
    kw: float = 2.5
    mass: float = 1.5
    gravity: float = 9.81
    max_acc: float = 5.0
    inertia: NDArray[np.floating] | None = None

    def __post_init__(self):
        if self.inertia is None:
            self.inertia = np.diag([0.0082, 0.0082, 0.0148])


class GeometricController:
    """Geometric tracking controller on SE(3).

    Implements the controller from Lee et al. that avoids Euler angle
    singularities by working directly on the rotation group SO(3).

    The position controller produces a desired force vector, from which
    the desired rotation ``R_d`` is computed. The attitude controller
    then computes torques using the rotation error on SO(3).
    """

    def __init__(self, config: GeometricControllerConfig | None = None) -> None:
        self.config = config or GeometricControllerConfig()

    def compute(
        self,
        state: NDArray[np.floating],
        target_pos: NDArray[np.floating],
        target_vel: NDArray[np.floating] | None = None,
        target_acc: NDArray[np.floating] | None = None,
        target_yaw: float = 0.0,
    ) -> NDArray[np.floating]:
        """Compute control wrench using geometric control.

        Args:
            state: 12-element state ``[x,y,z,φ,θ,ψ,vx,vy,vz,p,q,r]``.
            target_pos: Desired ``[x, y, z]``.
            target_vel: Desired ``[vx, vy, vz]`` (default zeros).
            target_acc: Desired ``[ax, ay, az]`` feedforward (default zeros).
            target_yaw: Desired yaw angle [rad].

        Returns:
            ``[T, τx, τy, τz]`` body-frame wrench.
        """
        c = self.config
        if target_vel is None:
            target_vel = np.zeros(3)
        if target_acc is None:
            target_acc = np.zeros(3)

        pos = state[:3]
        euler = state[3:6]
        vel = state[6:9]
        omega = state[9:12]

        R = Quadrotor.rotation_matrix(*euler)
        e3 = np.array([0.0, 0.0, 1.0])

        # --- Position control ---
        e_pos = pos - target_pos
        e_vel = vel - target_vel
        a_cmd = -c.kx * e_pos - c.kv * e_vel + c.gravity * e3 + target_acc

        # Clamp acceleration to avoid extreme attitudes
        a_horiz = np.linalg.norm(a_cmd[:2])
        if a_horiz > c.max_acc:
            a_cmd[:2] *= c.max_acc / a_horiz

        F_des = c.mass * a_cmd

        # Thrust magnitude.
        T = float(F_des @ R @ e3)
        T = np.clip(T, 0.0, c.mass * c.gravity * 2.5)

        # --- Desired rotation from force vector ---
        b3d = F_des / (np.linalg.norm(F_des) + 1e-6)
        b1c = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        b2d = np.cross(b3d, b1c)
        b2d_norm = np.linalg.norm(b2d)
        b2d = np.array([0.0, 1.0, 0.0]) if b2d_norm < 1e-6 else b2d / b2d_norm
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack([b1d, b2d, b3d])

        # --- Attitude control on SO(3) ---
        eR_matrix = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = _vee(eR_matrix)
        e_omega = omega  # desired omega is zero for set-point control

        tau = -c.kR * eR - c.kw * e_omega + np.cross(omega, c.inertia @ omega)

        return np.array([T, tau[0], tau[1], tau[2]])
