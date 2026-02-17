# Erwin Lejeune - 2026-02-16
"""Unscented Kalman Filter using Van der Merwe scaled sigma points.

Reference: E. A. Wan, R. Van Der Merwe, "The Unscented Kalman Filter for
Nonlinear Estimation," AS-SPCC, 2000. DOI: 10.1109/ASSPCC.2000.882463
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class UnscentedKalmanFilter:
    """UKF with Van der Merwe scaled sigma point selection.

    Parameters:
        state_dim: Dimension of the state vector.
        meas_dim: Dimension of the measurement vector.
        f: State transition ``f(x, u, dt) → x_pred``.
        h: Measurement function ``h(x) → z_pred``.
        alpha: Spread of sigma points (typically 1e-3).
        beta: Prior knowledge of distribution (2.0 for Gaussian).
        kappa: Secondary scaling parameter (typically 0 or 3-n).
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        f: Callable[..., NDArray[np.floating]],
        h: Callable[..., NDArray[np.floating]],
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.n = state_dim
        self.m = meas_dim
        self.f = f
        self.h = h

        # Sigma point parameters.
        self.lam = alpha**2 * (self.n + kappa) - self.n
        self.num_sigma = 2 * self.n + 1

        # Weights.
        self.Wm = np.full(self.num_sigma, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wc = np.full(self.num_sigma, 1.0 / (2.0 * (self.n + self.lam)))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1.0 - alpha**2 + beta)

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n)
        self.Q = np.eye(self.n) * 0.01
        self.R = np.eye(self.m) * 0.1

    def reset(
        self,
        x0: NDArray[np.floating],
        P0: NDArray[np.floating] | None = None,
    ) -> None:
        self.x = np.array(x0, dtype=np.float64)
        self.P = np.array(P0, dtype=np.float64) if P0 is not None else np.eye(self.n)

    def _sigma_points(self) -> NDArray[np.floating]:
        """Generate 2n+1 sigma points around current mean and covariance."""
        S = np.linalg.cholesky((self.n + self.lam) * self.P)
        sigmas = np.zeros((self.num_sigma, self.n))
        sigmas[0] = self.x
        for i in range(self.n):
            sigmas[i + 1] = self.x + S[i]
            sigmas[self.n + i + 1] = self.x - S[i]
        return sigmas

    def predict(self, u: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
        """Predict step: propagate sigma points through process model."""
        sigmas = self._sigma_points()
        sigmas_pred = np.array([self.f(s, u, dt) for s in sigmas])

        # Weighted mean.
        self.x = np.dot(self.Wm, sigmas_pred)

        # Weighted covariance.
        self.P = self.Q.copy()
        for i in range(self.num_sigma):
            d = sigmas_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(d, d)

        self._sigmas_pred = sigmas_pred
        return self.x.copy()

    def update(self, z: NDArray[np.floating]) -> NDArray[np.floating]:
        """Update step: incorporate measurement via UKF equations."""
        sigmas_pred = self._sigmas_pred
        z_sigmas = np.array([self.h(s) for s in sigmas_pred])
        z_mean = np.dot(self.Wm, z_sigmas)

        # Innovation covariance.
        Pzz = self.R.copy()
        for i in range(self.num_sigma):
            dz = z_sigmas[i] - z_mean
            Pzz += self.Wc[i] * np.outer(dz, dz)

        # Cross covariance.
        Pxz = np.zeros((self.n, self.m))
        for i in range(self.num_sigma):
            dx = sigmas_pred[i] - self.x
            dz = z_sigmas[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(Pzz)
        self.x = self.x + K @ (z - z_mean)
        self.P = self.P - K @ Pzz @ K.T
        return self.x.copy()

    @property
    def state(self) -> NDArray[np.floating]:
        return self.x.copy()

    @property
    def covariance(self) -> NDArray[np.floating]:
        return self.P.copy()
