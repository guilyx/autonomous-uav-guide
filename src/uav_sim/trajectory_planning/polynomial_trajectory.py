# Erwin Lejeune - 2026-02-16
"""Polynomial trajectory generation through waypoints.

Reference: C. Richter, A. Bry, N. Roy, "Polynomial Trajectory Planning for
Aggressive Quadrotor Flight in Dense Indoor Environments," ISRR, 2013.
DOI: 10.1007/978-3-319-28872-7_37
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PolynomialTrajectory:
    """Generate smooth polynomial trajectories through waypoints.

    For each axis and each segment between consecutive waypoints,
    fits a polynomial of specified order satisfying boundary conditions
    on position, velocity, and acceleration.

    Parameters:
        order: Polynomial order (e.g. 5 for quintic, 7 for septic).
    """

    def __init__(self, order: int = 5) -> None:
        if order < 3:
            raise ValueError("Polynomial order must be >= 3 for smooth trajectories.")
        self.order = order

    def generate(
        self,
        waypoints: NDArray[np.floating],
        segment_times: NDArray[np.floating],
    ) -> list[NDArray[np.floating]]:
        """Compute polynomial coefficients for each axis and segment.

        Args:
            waypoints: (M+1, D) array of M+1 waypoints in D dimensions.
            segment_times: (M,) array of durations for each segment.

        Returns:
            List of D arrays, each of shape (M, order+1) with coefficients.
            Coefficients are ordered [c0, c1, ..., cn] where p(t) = Σ ci * t^i.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        segment_times = np.asarray(segment_times, dtype=np.float64)
        num_segments = len(segment_times)
        dim = waypoints.shape[1]

        if waypoints.shape[0] != num_segments + 1:
            raise ValueError("Number of waypoints must be segments + 1.")

        all_coeffs = []
        for d in range(dim):
            coeffs = np.zeros((num_segments, self.order + 1))
            for k in range(num_segments):
                T = segment_times[k]
                p0 = waypoints[k, d]
                p1 = waypoints[k + 1, d]

                if self.order == 5:
                    coeffs[k] = self._quintic(p0, p1, T)
                else:
                    coeffs[k] = self._general(p0, p1, T)
            all_coeffs.append(coeffs)
        return all_coeffs

    def evaluate(
        self,
        coeffs: list[NDArray[np.floating]],
        segment_times: NDArray[np.floating],
        dt: float = 0.01,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Evaluate trajectory at regular intervals.

        Args:
            coeffs: Output of ``generate()``.
            segment_times: Segment durations.
            dt: Sample period [s].

        Returns:
            (times, positions) where positions is (N, D).
        """
        dim = len(coeffs)
        times_list = []
        positions_list = []
        t_offset = 0.0

        for k in range(len(segment_times)):
            T = segment_times[k]
            ts = np.arange(0, T, dt)
            for t in ts:
                pos = np.array([self._poly_eval(coeffs[d][k], t) for d in range(dim)])
                times_list.append(t_offset + t)
                positions_list.append(pos)
            t_offset += T

        return np.array(times_list), np.array(positions_list)

    @staticmethod
    def _quintic(p0: float, p1: float, T: float) -> NDArray[np.floating]:
        """Quintic polynomial: p(0)=p0, p(T)=p1, zero v/a at endpoints."""
        c0 = p0
        c1 = 0.0
        c2 = 0.0
        c3 = 10 * (p1 - p0) / T**3
        c4 = -15 * (p1 - p0) / T**4
        c5 = 6 * (p1 - p0) / T**5
        return np.array([c0, c1, c2, c3, c4, c5])

    def _general(self, p0: float, p1: float, T: float) -> NDArray[np.floating]:
        """General polynomial: p(0)=p0, p(T)=p1, other coeffs zero."""
        coeffs = np.zeros(self.order + 1)
        coeffs[0] = p0
        coeffs[1] = (p1 - p0) / T if T > 0 else 0.0
        return coeffs

    @staticmethod
    def _poly_eval(coeffs: NDArray[np.floating], t: float) -> float:
        """Evaluate polynomial p(t) = Σ ci * t^i."""
        return float(sum(c * t**i for i, c in enumerate(coeffs)))
