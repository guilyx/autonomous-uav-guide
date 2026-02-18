# Erwin Lejeune - 2026-02-16
"""Feedback linearisation tracker exploiting quadrotor differential flatness.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011 (Sec. IV). DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from uav_sim.vehicles.multirotor.quadrotor import Quadrotor


class FeedbackLinearisationTracker:
    """Differential-flatness-based trajectory tracker.

    Exploits the flat outputs ``[x, y, z, ψ]`` to reduce the tracking
    problem to linear error dynamics. The desired acceleration is
    computed from the reference trajectory (up to 2nd derivative) plus
    PD feedback on position and velocity errors.

    Parameters:
        kp: Position gain (3-vector or scalar).
        kd: Velocity gain (3-vector or scalar).
        mass: Quadrotor mass [kg].
        gravity: Gravitational acceleration [m/s²].
        inertia: 3x3 inertia tensor.
    """

    def __init__(
        self,
        kp: float | NDArray[np.floating] = 4.0,
        kd: float | NDArray[np.floating] = 2.8,
        mass: float = 1.5,
        gravity: float = 9.81,
        inertia: NDArray[np.floating] | None = None,
        max_acc: float = 5.0,
    ) -> None:
        self.kp = np.atleast_1d(np.asarray(kp, dtype=np.float64))
        self.kd = np.atleast_1d(np.asarray(kd, dtype=np.float64))
        self.mass = mass
        self.gravity = gravity
        self.max_acc = max_acc
        self.inertia = inertia if inertia is not None else np.diag([0.0082, 0.0082, 0.0148])

    def compute(
        self,
        state: NDArray[np.floating],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating] | None = None,
        ref_acc: NDArray[np.floating] | None = None,
        ref_yaw: float = 0.0,
    ) -> NDArray[np.floating]:
        """Compute wrench to track a reference trajectory.

        Args:
            state: 12-element quadrotor state.
            ref_pos: Desired ``[x, y, z]``.
            ref_vel: Desired ``[vx, vy, vz]`` (default zeros).
            ref_acc: Desired ``[ax, ay, az]`` feedforward (default zeros).
            ref_yaw: Desired yaw [rad].

        Returns:
            ``[T, τx, τy, τz]`` wrench.
        """
        if ref_vel is None:
            ref_vel = np.zeros(3)
        if ref_acc is None:
            ref_acc = np.zeros(3)

        pos = state[:3]
        euler = state[3:6]
        vel = state[6:9]
        omega = state[9:12]

        e_pos = pos - ref_pos
        e_vel = vel - ref_vel

        # Desired acceleration via feedback linearisation.
        a_des = ref_acc - self.kp * e_pos - self.kd * e_vel
        a_des[2] += self.gravity

        # Clamp horizontal acceleration
        a_horiz = np.linalg.norm(a_des[:2])
        if a_horiz > self.max_acc:
            a_des[:2] *= self.max_acc / a_horiz

        # Thrust.
        R = Quadrotor.rotation_matrix(*euler)
        e3 = np.array([0.0, 0.0, 1.0])
        T = float(self.mass * a_des @ R @ e3)
        T = float(np.clip(T, 0.0, self.mass * self.gravity * 2.5))

        # Desired attitude from a_des.
        a_norm = np.linalg.norm(a_des)
        if a_norm < 1e-6:
            a_norm = 1e-6
        b3d = a_des / a_norm
        b1c = np.array([np.cos(ref_yaw), np.sin(ref_yaw), 0.0])
        b2d = np.cross(b3d, b1c)
        b2d_norm = np.linalg.norm(b2d)
        b2d = np.array([0.0, 1.0, 0.0]) if b2d_norm < 1e-6 else b2d / b2d_norm
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack([b1d, b2d, b3d])

        # SO(3) attitude error → torques (gains for Crazyflie inertia).
        eR = 0.5 * self._vee(Rd.T @ R - R.T @ Rd)
        kR = 0.005
        kw = 0.002
        tau = -kR * eR - kw * omega + np.cross(omega, self.inertia @ omega)

        return np.array([T, tau[0], tau[1], tau[2]])

    @staticmethod
    def _vee(M: NDArray[np.floating]) -> NDArray[np.floating]:
        return np.array([M[2, 1], M[0, 2], M[1, 0]])
