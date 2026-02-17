# Erwin Lejeune - 2026-02-16
"""Extended Kalman Filter for nonlinear state estimation.

Reference: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics,"
MIT Press, 2005, Chapter 3.3.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class ExtendedKalmanFilter:
    """Generic Extended Kalman Filter.

    Operates on an n-dimensional state with a nonlinear process model
    and nonlinear measurement model. Jacobians are provided as callables.

    Parameters:
        state_dim: Dimension of the state vector.
        meas_dim: Dimension of the measurement vector.
        f: State transition function ``f(x, u, dt) → x_pred``.
        h: Measurement function ``h(x) → z_pred``.
        F_jac: Jacobian of f w.r.t. state ``F_jac(x, u, dt) → (n, n)``.
        H_jac: Jacobian of h w.r.t. state ``H_jac(x) → (m, n)``.
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        f: Callable[..., NDArray[np.floating]],
        h: Callable[..., NDArray[np.floating]],
        F_jac: Callable[..., NDArray[np.floating]],
        H_jac: Callable[..., NDArray[np.floating]],
    ) -> None:
        self.n = state_dim
        self.m = meas_dim
        self.f = f
        self.h = h
        self.F_jac = F_jac
        self.H_jac = H_jac

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)
        self.Q = np.eye(self.n) * 0.01
        self.R = np.eye(self.m) * 0.1

    def reset(
        self,
        x0: NDArray[np.floating],
        P0: NDArray[np.floating] | None = None,
    ) -> None:
        """Reset filter state and covariance."""
        self.x = np.array(x0, dtype=np.float64)
        self.P = np.array(P0, dtype=np.float64) if P0 is not None else np.eye(self.n)

    def predict(self, u: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Predict step: propagate state through process model.

        Args:
            u: Control input.
            dt: Time step [s].

        Returns:
            Predicted state.
        """
        F = self.F_jac(self.x, u, dt)
        self.x = self.f(self.x, u, dt)
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z: NDArray[np.floating]) -> NDArray[np.floating]:
        """Update step: incorporate measurement.

        Args:
            z: Measurement vector.

        Returns:
            Updated state estimate.
        """
        H = self.H_jac(self.x)
        z_pred = self.h(self.x)
        y = z - z_pred  # innovation
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ H) @ self.P
        return self.x.copy()

    @property
    def state(self) -> NDArray[np.floating]:
        return self.x.copy()

    @property
    def covariance(self) -> NDArray[np.floating]:
        return self.P.copy()
