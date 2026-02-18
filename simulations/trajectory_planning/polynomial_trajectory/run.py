# Erwin Lejeune - 2026-02-15
"""Quintic polynomial trajectory through 3-D waypoints.

Reference: C. Richter, A. Bry, N. Roy, "Polynomial Trajectory Planning for
Aggressive Quadrotor Flight," ISRR, 2013. DOI: 10.1007/978-3-319-28872-7_37
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.trajectory_planning.polynomial_trajectory import PolynomialTrajectory
from uav_sim.visualization import SimAnimator


def main() -> None:
    wps = np.array([[0, 0, 0], [1, 2, 1], [3, 3, 2], [5, 1, 1.5], [7, 0, 0.0]])
    st = np.array([1.5, 1.5, 1.5, 1.5])
    poly = PolynomialTrajectory(order=5)
    coeffs = poly.generate(wps, st)
    _, pos = poly.evaluate(coeffs, st, dt=0.01)
    skip = max(1, len(pos) // 200)
    idx = list(range(0, len(pos), skip))
    anim = SimAnimator("polynomial_trajectory", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Polynomial Trajectory (quintic)")
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="red", s=60, marker="D", zorder=5, label="WP")
    (trail,) = ax.plot([], [], [], "b-", lw=1.5)
    (dot,) = ax.plot([], [], [], "ko", ms=5)
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 4)
    ax.set_zlim(-0.5, 3)
    ax.legend(fontsize=8)

    def update(f):
        k = idx[f]
        trail.set_data(pos[:k, 0], pos[:k, 1])
        trail.set_3d_properties(pos[:k, 2])
        dot.set_data([pos[k, 0]], [pos[k, 1]])
        dot.set_3d_properties([pos[k, 2]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
