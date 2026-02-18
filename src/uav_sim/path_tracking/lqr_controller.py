# Erwin Lejeune - 2026-02-16
"""Linear Quadratic Regulator for quadrotor hover stabilisation.

Reference: B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear
Quadratic Methods," Prentice-Hall, 1990.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are


class LQRController:
    """LQR controller linearised about the hover equilibrium.

    The quadrotor dynamics are linearised to
    ``ẋ = A δx + B δu`` about hover, where:
    - δx = x - x_eq (12-state error)
    - δu = u - u_eq (wrench error, u_eq = [mg, 0, 0, 0])

    The gain ``K`` is computed by solving the continuous-time
    Algebraic Riccati Equation: ``A'P + PA - PBR⁻¹B'P + Q = 0``.

    Parameters:
        mass: Quadrotor mass [kg].
        gravity: Gravitational acceleration [m/s²].
        inertia: 3x3 inertia tensor.
        Q: 12x12 state cost matrix.
        R: 4x4 input cost matrix.
    """

    def __init__(
        self,
        mass: float = 0.027,
        gravity: float = 9.81,
        inertia: NDArray[np.floating] | None = None,
        Q: NDArray[np.floating] | None = None,
        R: NDArray[np.floating] | None = None,
    ) -> None:
        self.mass = mass
        self.gravity = gravity
        self.inertia = inertia if inertia is not None else np.diag([1.4e-5, 1.4e-5, 2.17e-5])

        # Build linearised A, B matrices.
        self.A, self.B = self._linearise()

        # Default cost matrices.
        if Q is None:
            Q = np.diag([10, 10, 20, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1])
        if R is None:
            R = np.diag([1, 10, 10, 10])

        self.Q = Q
        self.R = R

        # Solve Riccati and compute gain.
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ P
        self.hover_wrench = np.array([mass * gravity, 0.0, 0.0, 0.0])

    def _linearise(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Linearise quadrotor dynamics about hover.

        Returns:
            (A, B) matrices for ``ẋ = A x + B u``.
        """
        m = self.mass
        g = self.gravity
        Ix, Iy, Iz = np.diag(self.inertia)

        A = np.zeros((12, 12))
        # Position derivatives = velocity.
        A[0, 6] = 1.0
        A[1, 7] = 1.0
        A[2, 8] = 1.0
        # Velocity from angles (gravity coupling at hover).
        A[6, 4] = g  # dvx/dtheta = g
        A[7, 3] = -g  # dvy/dphi = -g
        # Euler rate ≈ body rate at hover (small angle).
        A[3, 9] = 1.0
        A[4, 10] = 1.0
        A[5, 11] = 1.0

        B = np.zeros((12, 4))
        B[8, 0] = 1.0 / m  # dvz/dT
        B[9, 1] = 1.0 / Ix  # dp/dtau_x
        B[10, 2] = 1.0 / Iy  # dq/dtau_y
        B[11, 3] = 1.0 / Iz  # dr/dtau_z

        return A, B

    def compute(
        self,
        state: NDArray[np.floating],
        target_state: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Compute LQR wrench from state error.

        Args:
            state: Current 12-element state.
            target_state: Desired 12-element state (defaults to origin hover).

        Returns:
            ``[T, τx, τy, τz]`` wrench.
        """
        if target_state is None:
            target_state = np.zeros(12)
        error = state - target_state
        wrench = self.hover_wrench - self.K @ error
        wrench[0] = max(0.0, wrench[0])
        return wrench
