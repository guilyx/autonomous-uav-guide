# Erwin Lejeune - 2026-02-15
"""Quintic polynomial planner with full boundary conditions.

Generates point-to-point trajectories satisfying position, velocity,
and acceleration constraints at both endpoints — the standard building
block for smooth UAV motion primitives.

Reference: K. Takahashi, T. Scheuer, "Motion Planning in a Plane Using
Generalized Voronoi Diagrams," IEEE T-RA, 1989.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class QuinticState:
    """Boundary state: position, velocity, acceleration per axis."""

    pos: NDArray[np.floating]
    vel: NDArray[np.floating]
    acc: NDArray[np.floating]


class QuinticPolynomialPlanner:
    """Quintic polynomial trajectory from *start* to *goal*.

    Solves the 6-coefficient polynomial for each axis independently::

        p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵

    subject to::

        p(0)=x0, p'(0)=v0, p''(0)=a0
        p(T)=x1, p'(T)=v1, p''(T)=a1
    """

    def generate(
        self,
        start: QuinticState,
        goal: QuinticState,
        T: float,
    ) -> NDArray[np.floating]:
        """Compute polynomial coefficients.

        Returns
        -------
        (D, 6) coefficient array, one row per dimension.
        """
        dim = len(start.pos)
        coeffs = np.zeros((dim, 6))
        for d in range(dim):
            coeffs[d] = self._solve_1d(
                start.pos[d],
                start.vel[d],
                start.acc[d],
                goal.pos[d],
                goal.vel[d],
                goal.acc[d],
                T,
            )
        return coeffs

    def evaluate(
        self,
        coeffs: NDArray[np.floating],
        T: float,
        dt: float = 0.01,
    ) -> tuple[
        NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
    ]:
        """Evaluate trajectory at regular intervals.

        Returns
        -------
        (times, positions, velocities, accelerations) each (N, D).
        """
        ts = np.arange(0, T + dt * 0.5, dt)
        dim = coeffs.shape[0]
        positions = np.zeros((len(ts), dim))
        velocities = np.zeros((len(ts), dim))
        accelerations = np.zeros((len(ts), dim))
        for i, t in enumerate(ts):
            for d in range(dim):
                c = coeffs[d]
                positions[i, d] = (
                    c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
                )
                velocities[i, d] = (
                    c[1] + 2 * c[2] * t + 3 * c[3] * t**2 + 4 * c[4] * t**3 + 5 * c[5] * t**4
                )
                accelerations[i, d] = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3
        return ts, positions, velocities, accelerations

    @staticmethod
    def _solve_1d(
        x0: float,
        v0: float,
        a0: float,
        x1: float,
        v1: float,
        a1: float,
        T: float,
    ) -> NDArray[np.floating]:
        """Solve the 6 coefficients for a single axis."""
        c0 = x0
        c1 = v0
        c2 = a0 / 2.0

        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        A = np.array(
            [
                [T3, T4, T5],
                [3 * T2, 4 * T3, 5 * T4],
                [6 * T, 12 * T2, 20 * T3],
            ]
        )
        b = np.array(
            [
                x1 - c0 - c1 * T - c2 * T2,
                v1 - c1 - 2 * c2 * T,
                a1 - 2 * c2,
            ]
        )
        abc = np.linalg.solve(A, b)
        return np.array([c0, c1, c2, abc[0], abc[1], abc[2]])
