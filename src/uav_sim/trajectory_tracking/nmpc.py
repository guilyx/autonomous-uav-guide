# Erwin Lejeune - 2026-02-15
"""Nonlinear Model Predictive Controller for quadrotor trajectory tracking.

Unlike the linear MPC, this controller uses the full nonlinear quadrotor
dynamics in the prediction model via direct single-shooting and
``scipy.optimize.minimize``.

Reference: M. Diehl et al., "Real-Time Optimization and Nonlinear Model
Predictive Control of Processes Governed by Differential-Algebraic
Equations," J. Process Control, 2002. DOI: 10.1016/S0959-1524(02)00023-1
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


class NMPCTracker:
    """Nonlinear MPC using single-shooting with RK2 integration.

    Parameters
    ----------
    horizon : prediction steps.
    dt : control interval [s].
    mass, gravity : quadrotor parameters.
    inertia : 3x3 inertia tensor.
    Q : (12,) or (12,12) state cost (diagonal values if 1-D).
    R : (4,) or (4,4) input cost.
    max_thrust_ratio : maximum thrust as multiple of hover.
    max_torque : torque clamp [Nm].
    """

    def __init__(
        self,
        horizon: int = 8,
        dt: float = 0.02,
        mass: float = 1.5,
        gravity: float = 9.81,
        inertia: NDArray[np.floating] | None = None,
        Q: NDArray[np.floating] | None = None,
        R: NDArray[np.floating] | None = None,
        max_thrust_ratio: float = 2.0,
        max_torque: float = 2.0,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.mass = mass
        self.gravity = gravity
        self.inertia = inertia if inertia is not None else np.diag([0.0082, 0.0082, 0.0148])
        self.inv_inertia = np.linalg.inv(self.inertia)

        self.Q = (
            np.diag(Q)
            if Q is not None and Q.ndim == 1
            else (Q if Q is not None else np.diag([15, 15, 25, 1, 1, 1, 2, 2, 2, 0.1, 0.1, 0.1]))
        )
        self.R = (
            np.diag(R)
            if R is not None and R.ndim == 1
            else (R if R is not None else np.diag([0.01, 10, 10, 10]))
        )

        self.hover_wrench = np.array([mass * gravity, 0.0, 0.0, 0.0])
        self.max_thrust = mass * gravity * max_thrust_ratio
        self.max_torque = max_torque
        self._warm: NDArray[np.floating] | None = None

    def reset(self) -> None:
        self._warm = None

    def compute(
        self,
        state: NDArray[np.floating],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Compute NMPC wrench.

        Parameters
        ----------
        state : 12-element state.
        ref_pos : desired ``[x, y, z]``.
        ref_vel : desired ``[vx, vy, vz]`` (default zeros).

        Returns
        -------
        ``[T, τx, τy, τz]`` wrench.
        """
        ref = np.zeros(12)
        ref[:3] = ref_pos
        if ref_vel is not None:
            ref[6:9] = ref_vel

        H = self.horizon
        if self._warm is not None and len(self._warm) == H * 4:
            u0 = self._warm.copy()
        else:
            u0 = np.tile(self.hover_wrench, H)

        bounds = []
        for _ in range(H):
            bounds.append((0.0, self.max_thrust))
            for _ in range(3):
                bounds.append((-self.max_torque, self.max_torque))

        result = minimize(
            self._cost,
            u0,
            args=(state.copy(), ref),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 25, "ftol": 1e-5},
        )
        u_opt = result.x.reshape(H, 4)

        self._warm = np.zeros(H * 4)
        self._warm[: (H - 1) * 4] = result.x[4:]
        self._warm[(H - 1) * 4 :] = self.hover_wrench

        return u_opt[0]

    def _cost(self, u_flat: NDArray, x0: NDArray, ref: NDArray) -> float:
        H = self.horizon
        u = u_flat.reshape(H, 4)
        x = x0.copy()
        cost = 0.0
        for k in range(H):
            e = x - ref
            cost += float(e @ self.Q @ e + u[k] @ self.R @ u[k])
            x = self._step(x, u[k])
        e = x - ref
        cost += float(e @ self.Q @ e)
        return cost

    def _step(self, state: NDArray, wrench: NDArray) -> NDArray:
        """RK2 integration of simplified quadrotor dynamics."""
        k1 = self._dynamics(state, wrench)
        k2 = self._dynamics(state + self.dt * k1, wrench)
        return state + 0.5 * self.dt * (k1 + k2)

    def _dynamics(self, state: NDArray, wrench: NDArray) -> NDArray:
        phi, theta, psi = state[3], state[4], state[5]
        vx, vy, vz = state[6], state[7], state[8]
        p, q, r = state[9], state[10], state[11]
        T = wrench[0]

        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)

        # Rotation matrix body z to world
        Rz = np.array(
            [
                cy * st * cp + sy * sp,
                sy * st * cp - cy * sp,
                ct * cp,
            ]
        )
        acc = T / self.mass * Rz - np.array([0, 0, self.gravity])

        tau = wrench[1:]
        omega = np.array([p, q, r])
        omega_dot = self.inv_inertia @ (tau - np.cross(omega, self.inertia @ omega))

        ct_safe = ct if abs(ct) > 1e-8 else 1e-8
        euler_dot = np.array(
            [
                p + (q * sp + r * cp) * np.tan(theta),
                q * cp - r * sp,
                (q * sp + r * cp) / ct_safe,
            ]
        )

        ds = np.zeros(12)
        ds[:3] = [vx, vy, vz]
        ds[3:6] = euler_dot
        ds[6:9] = acc
        ds[9:12] = omega_dot
        return ds
