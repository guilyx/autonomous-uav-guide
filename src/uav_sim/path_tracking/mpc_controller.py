# Erwin Lejeune - 2026-02-15
"""Linear Model Predictive Controller for quadrotor path tracking.

Uses the same hover-linearised model as the LQR controller, but solves
a finite-horizon optimal control problem at every step, applying only
the first control input (receding horizon).

Reference: J. B. Rawlings, D. Q. Mayne, M. M. Diehl, "Model Predictive
Control: Theory, Computation, and Design," 2nd ed., Nob Hill, 2017.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize


class MPCController:
    """Receding-horizon linear MPC for quadrotor position tracking.

    Parameters
    ----------
    horizon : prediction horizon (number of steps).
    dt : discretisation time step [s].
    mass, gravity : quadrotor parameters.
    inertia : 3x3 inertia tensor.
    Q : 12x12 state cost.
    R : 4x4 input cost.
    """

    def __init__(
        self,
        horizon: int = 10,
        dt: float = 0.02,
        mass: float = 1.5,
        gravity: float = 9.81,
        inertia: NDArray[np.floating] | None = None,
        Q: NDArray[np.floating] | None = None,
        R: NDArray[np.floating] | None = None,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.mass = mass
        self.gravity = gravity
        inertia = inertia if inertia is not None else np.diag([0.0082, 0.0082, 0.0148])

        A_c, B_c = self._linearise(mass, gravity, inertia)
        # Euler discretisation
        self.A_d = np.eye(12) + A_c * dt
        self.B_d = B_c * dt

        self.Q = Q if Q is not None else np.diag([15, 15, 25, 1, 1, 1, 2, 2, 2, 0.1, 0.1, 0.1])
        self.R = R if R is not None else np.diag([0.01, 5.0, 5.0, 5.0])

        # Terminal cost from DARE
        P = solve_continuous_are(A_c, B_c, self.Q, self.R)
        self.Qf = P

        self.hover_wrench = np.array([mass * gravity, 0.0, 0.0, 0.0])
        self._warm_start: NDArray[np.floating] | None = None

    def reset(self) -> None:
        self._warm_start = None

    def compute(
        self,
        state: NDArray[np.floating],
        target_pos: NDArray[np.floating],
        target_vel: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Compute MPC wrench for one step.

        Parameters
        ----------
        state : 12-element quadrotor state.
        target_pos : desired ``[x, y, z]``.
        target_vel : desired ``[vx, vy, vz]`` (default zeros).

        Returns
        -------
        ``[T, τx, τy, τz]`` wrench.
        """
        ref = np.zeros(12)
        ref[:3] = target_pos
        if target_vel is not None:
            ref[6:9] = target_vel

        x0 = state - ref
        H = self.horizon
        u_dim = 4

        if self._warm_start is not None and len(self._warm_start) == H * u_dim:
            u0 = self._warm_start.copy()
        else:
            u0 = np.zeros(H * u_dim)

        bounds = []
        for _ in range(H):
            bounds.append((0.0 - self.hover_wrench[0], self.mass * self.gravity * 1.5))
            for _ in range(3):
                bounds.append((-1e-4, 1e-4))

        result = minimize(
            self._cost,
            u0,
            args=(x0,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 30, "ftol": 1e-6},
        )
        u_opt = result.x.reshape(H, u_dim)

        # Warm-start: shift solution
        self._warm_start = np.zeros(H * u_dim)
        self._warm_start[: (H - 1) * u_dim] = result.x[u_dim:]

        return self.hover_wrench + u_opt[0]

    def _cost(self, u_flat: NDArray[np.floating], x0: NDArray[np.floating]) -> float:
        H = self.horizon
        u = u_flat.reshape(H, 4)
        x = x0.copy()
        cost = 0.0
        for k in range(H):
            cost += float(x @ self.Q @ x + u[k] @ self.R @ u[k])
            x = self.A_d @ x + self.B_d @ u[k]
        cost += float(x @ self.Qf @ x)
        return cost

    @staticmethod
    def _linearise(
        mass: float, gravity: float, inertia: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        Ix, Iy, Iz = np.diag(inertia)
        A = np.zeros((12, 12))
        A[0, 6] = 1.0
        A[1, 7] = 1.0
        A[2, 8] = 1.0
        A[6, 4] = gravity
        A[7, 3] = -gravity
        A[3, 9] = 1.0
        A[4, 10] = 1.0
        A[5, 11] = 1.0

        B = np.zeros((12, 4))
        B[8, 0] = 1.0 / mass
        B[9, 1] = 1.0 / Ix
        B[10, 2] = 1.0 / Iy
        B[11, 3] = 1.0 / Iz
        return A, B
