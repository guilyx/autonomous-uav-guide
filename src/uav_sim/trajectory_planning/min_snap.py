# Erwin Lejeune - 2026-02-16
"""Minimum-snap trajectory generation for quadrotor flight.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from math import factorial

import numpy as np
from numpy.typing import NDArray


class MinSnapTrajectory:
    """Generate minimum-snap (4th derivative) polynomial trajectories.

    Each segment uses a 7th-order polynomial (8 coefficients).
    The objective minimises the integral of snap squared over all segments
    subject to position, velocity, acceleration, and jerk continuity
    at internal waypoints, with start/end at rest.
    """

    POLY_ORDER = 7
    NUM_COEFFS = 8

    def generate(
        self,
        waypoints: NDArray[np.floating],
        segment_times: NDArray[np.floating],
    ) -> list[NDArray[np.floating]]:
        """Compute min-snap polynomial coefficients per axis.

        Args:
            waypoints: (M+1, D) array of waypoints.
            segment_times: (M,) segment durations.

        Returns:
            List of D arrays, each (M, 8) -- coefficients per segment.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        segment_times = np.asarray(segment_times, dtype=np.float64)
        num_seg = len(segment_times)
        dim = waypoints.shape[1]

        all_coeffs = []
        for d in range(dim):
            positions = waypoints[:, d]
            coeffs = self._solve_axis(positions, segment_times, num_seg)
            all_coeffs.append(coeffs)
        return all_coeffs

    def _solve_axis(
        self,
        positions: NDArray[np.floating],
        times: NDArray[np.floating],
        num_seg: int,
    ) -> NDArray[np.floating]:
        """Solve the min-snap QP for a single axis."""
        n = self.NUM_COEFFS
        total_vars = n * num_seg

        # Build cost matrix (integral of snap squared).
        Q = np.zeros((total_vars, total_vars))
        for k in range(num_seg):
            T = times[k]
            Qk = self._snap_cost_matrix(T)
            idx = k * n
            Q[idx : idx + n, idx : idx + n] = Qk

        # Build constraint matrix.
        constraints = []
        b_vals = []

        for k in range(num_seg):
            T = times[k]
            idx = k * n

            # Position at start of segment.
            row = np.zeros(total_vars)
            row[idx] = 1.0  # p(0) = c0
            constraints.append(row)
            b_vals.append(positions[k])

            # Position at end of segment.
            row = np.zeros(total_vars)
            for i in range(n):
                row[idx + i] = T**i
            constraints.append(row)
            b_vals.append(positions[k + 1])

        # Start at rest: v(0)=0, a(0)=0, j(0)=0.
        for deriv in [1, 2, 3]:
            row = np.zeros(total_vars)
            if deriv < n:
                row[deriv] = factorial(deriv)
            constraints.append(row)
            b_vals.append(0.0)

        # End at rest: v(T)=0, a(T)=0, j(T)=0.
        last_idx = (num_seg - 1) * n
        T_last = times[-1]
        for deriv in [1, 2, 3]:
            row = np.zeros(total_vars)
            for i in range(deriv, n):
                coeff = factorial(i) / factorial(i - deriv)
                row[last_idx + i] = coeff * T_last ** (i - deriv)
            constraints.append(row)
            b_vals.append(0.0)

        # Continuity at internal waypoints (vel, acc, jerk, snap).
        for k in range(num_seg - 1):
            T = times[k]
            idx_k = k * n
            idx_k1 = (k + 1) * n
            for deriv in range(1, 5):
                row = np.zeros(total_vars)
                # End of segment k.
                for i in range(deriv, n):
                    coeff = factorial(i) / factorial(i - deriv)
                    row[idx_k + i] = coeff * T ** (i - deriv)
                # Start of segment k+1.
                if deriv < n:
                    row[idx_k1 + deriv] = -factorial(deriv)
                constraints.append(row)
                b_vals.append(0.0)

        A_eq = np.array(constraints)
        b_eq = np.array(b_vals)

        # Solve using constrained least-squares approach.
        # min 0.5 x'Qx s.t. A_eq x = b_eq
        # KKT: [Q A_eq' ; A_eq 0] [x; λ] = [0; b]
        m = A_eq.shape[0]
        KKT = np.zeros((total_vars + m, total_vars + m))
        KKT[:total_vars, :total_vars] = Q + np.eye(total_vars) * 1e-10
        KKT[:total_vars, total_vars:] = A_eq.T
        KKT[total_vars:, :total_vars] = A_eq

        rhs = np.zeros(total_vars + m)
        rhs[total_vars:] = b_eq

        sol = np.linalg.solve(KKT, rhs)
        x = sol[:total_vars]

        return x.reshape(num_seg, n)

    def _snap_cost_matrix(self, T: float) -> NDArray[np.floating]:
        """Build the 8x8 cost matrix for ∫₀ᵀ snap² dt."""
        n = self.NUM_COEFFS
        Q = np.zeros((n, n))
        for i in range(4, n):
            for j in range(4, n):
                ci = factorial(i) / factorial(i - 4)
                cj = factorial(j) / factorial(j - 4)
                Q[i, j] = ci * cj * T ** (i + j - 7) / (i + j - 7)
        return Q

    def evaluate(
        self,
        coeffs: list[NDArray[np.floating]],
        segment_times: NDArray[np.floating],
        dt: float = 0.01,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Evaluate the trajectory at regular intervals.

        Returns:
            (times, positions) arrays.
        """
        dim = len(coeffs)
        times_list = []
        positions_list = []
        t_offset = 0.0

        for k in range(len(segment_times)):
            T = segment_times[k]
            ts = np.arange(0, T, dt)
            for t in ts:
                pos = np.array(
                    [
                        float(sum(coeffs[d][k, i] * t**i for i in range(self.NUM_COEFFS)))
                        for d in range(dim)
                    ]
                )
                times_list.append(t_offset + t)
                positions_list.append(pos)
            t_offset += T

        return np.array(times_list), np.array(positions_list)
