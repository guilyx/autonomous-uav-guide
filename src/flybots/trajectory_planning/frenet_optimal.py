# Erwin Lejeune - 2026-02-15
"""Optimal trajectory planning in the Frenet frame.

Samples candidate trajectories as lateral–longitudinal quintic polynomial
pairs along a reference path, scores them by a weighted cost, and returns
the best collision-free option.

Reference: M. Werling, J. Ziegler, S. Kammel, S. Thrun, "Optimal Trajectory
Generation for Dynamic Street Scenarios in a Frenet Frame," ICRA, 2010.
DOI: 10.1109/ROBOT.2010.5509799
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class FrenetPath:
    """A single candidate trajectory in Frenet coordinates."""

    t: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    s: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    d: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    s_dot: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    d_dot: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    s_ddot: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    d_ddot: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    # Cartesian
    x: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    y: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    z: NDArray[np.floating] = field(default_factory=lambda: np.empty(0))
    cost: float = float("inf")


class FrenetOptimalPlanner:
    """Sample and rank Frenet-frame trajectories along a 3-D reference.

    The planner samples lateral offsets and longitudinal speeds, generates
    quintic/quartic polynomials for each, converts to Cartesian space, and
    returns the lowest-cost collision-free trajectory.

    Parameters
    ----------
    max_lat_offset : maximum lateral deviation [m].
    n_lat_samples : number of lateral offsets to sample.
    min_speed, max_speed : longitudinal speed range [m/s].
    n_speed_samples : number of speed samples.
    dt : time step for trajectory evaluation [s].
    horizon : planning horizon [s].
    w_smooth : weight on jerk (smoothness).
    w_lateral : weight on lateral deviation.
    w_speed : weight on speed deviation from target.
    """

    def __init__(
        self,
        max_lat_offset: float = 3.0,
        n_lat_samples: int = 7,
        min_speed: float = 0.5,
        max_speed: float = 3.0,
        n_speed_samples: int = 5,
        dt: float = 0.1,
        horizon: float = 5.0,
        w_smooth: float = 0.1,
        w_lateral: float = 1.0,
        w_speed: float = 0.5,
        target_speed: float = 2.0,
    ) -> None:
        self.max_lat_offset = max_lat_offset
        self.n_lat_samples = n_lat_samples
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.n_speed_samples = n_speed_samples
        self.dt = dt
        self.horizon = horizon
        self.w_smooth = w_smooth
        self.w_lateral = w_lateral
        self.w_speed = w_speed
        self.target_speed = target_speed

    def plan(
        self,
        ref_path: NDArray[np.floating],
        s0: float = 0.0,
        d0: float = 0.0,
        s_dot0: float = 1.0,
        d_dot0: float = 0.0,
        obstacles: list[tuple[NDArray[np.floating], float]] | None = None,
    ) -> tuple[FrenetPath | None, list[FrenetPath]]:
        """Generate optimal trajectory in Frenet frame.

        Parameters
        ----------
        ref_path : (N, 3) reference waypoints defining the centre-line.
        s0, d0 : initial longitudinal and lateral Frenet coordinates.
        s_dot0, d_dot0 : initial speeds.
        obstacles : sphere obstacles ``(centre, radius)`` for collision check.

        Returns
        -------
        (best_path, all_candidates) — best may be None if all collide.
        """
        obstacles = obstacles or []
        cum_s, ref_interp = self._build_reference(ref_path)
        total_ref_len = cum_s[-1]

        candidates: list[FrenetPath] = []
        lat_offsets = np.linspace(-self.max_lat_offset, self.max_lat_offset, self.n_lat_samples)
        speeds = np.linspace(self.min_speed, self.max_speed, self.n_speed_samples)
        ts = np.arange(0, self.horizon + self.dt * 0.5, self.dt)

        for d_target in lat_offsets:
            for spd in speeds:
                fp = FrenetPath()
                fp.t = ts
                n = len(ts)

                # Lateral: quintic from (d0, d_dot0, 0) to (d_target, 0, 0)
                d_coeffs = self._quintic_coeffs(d0, d_dot0, 0.0, d_target, 0.0, 0.0, self.horizon)
                fp.d = np.polyval(d_coeffs[::-1], ts)
                fp.d_dot = np.polyval(np.polyder(d_coeffs[::-1]), ts)
                fp.d_ddot = np.polyval(np.polyder(d_coeffs[::-1], 2), ts)

                # Longitudinal: quartic with constant target speed
                s_vals = s0 + np.cumsum(np.full(n, spd * self.dt))
                s_vals = np.insert(s_vals, 0, s0)[:n]
                fp.s = s_vals
                fp.s_dot = np.full(n, spd)
                fp.s_ddot = np.zeros(n)

                # Convert to Cartesian
                xs, ys, zs = [], [], []
                valid = True
                for i in range(n):
                    si = float(np.clip(fp.s[i], 0.0, total_ref_len - 1e-6))
                    rx, ry, rz, nx, ny = ref_interp(si)
                    xs.append(rx + fp.d[i] * nx)
                    ys.append(ry + fp.d[i] * ny)
                    zs.append(rz)
                fp.x = np.array(xs)
                fp.y = np.array(ys)
                fp.z = np.array(zs)

                # Collision check
                for cx, cy, cz, r in [(o[0][0], o[0][1], o[0][2], o[1]) for o in obstacles]:
                    dists = np.sqrt((fp.x - cx) ** 2 + (fp.y - cy) ** 2 + (fp.z - cz) ** 2)
                    if np.any(dists < r + 0.5):
                        valid = False
                        break
                if not valid:
                    fp.cost = float("inf")
                    candidates.append(fp)
                    continue

                # Cost
                jerk_lat = np.sum(fp.d_ddot**2)
                lat_dev = np.sum(fp.d**2)
                speed_dev = np.sum((fp.s_dot - self.target_speed) ** 2)
                fp.cost = (
                    self.w_smooth * jerk_lat + self.w_lateral * lat_dev + self.w_speed * speed_dev
                )
                candidates.append(fp)

        valid_paths = [p for p in candidates if p.cost < float("inf")]
        if not valid_paths:
            return None, candidates
        best = min(valid_paths, key=lambda p: p.cost)
        return best, candidates

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reference(
        ref_path: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], object]:
        """Build cumulative arc-length and interpolation function."""
        diffs = np.diff(ref_path, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])

        def interp(s: float):
            idx = int(np.searchsorted(cum_s, s, side="right")) - 1
            idx = max(0, min(idx, len(ref_path) - 2))
            seg_len = max(seg_lens[idx], 1e-8)
            alpha = (s - cum_s[idx]) / seg_len
            alpha = np.clip(alpha, 0.0, 1.0)
            pt = ref_path[idx] * (1 - alpha) + ref_path[idx + 1] * alpha
            tangent = diffs[idx] / seg_len
            # 2D normal (perpendicular in XY plane)
            nx, ny = -tangent[1], tangent[0]
            return pt[0], pt[1], pt[2], nx, ny

        return cum_s, interp

    @staticmethod
    def _quintic_coeffs(
        x0: float,
        v0: float,
        a0: float,
        x1: float,
        v1: float,
        a1: float,
        T: float,
    ) -> NDArray[np.floating]:
        c0, c1, c2 = x0, v0, a0 / 2.0
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5
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
