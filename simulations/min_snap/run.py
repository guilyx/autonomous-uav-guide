# Erwin Lejeune - 2026-02-15
"""Minimum-snap trajectory through 3-D waypoints.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.planning.min_snap import MinSnapTrajectory
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    wps = np.array([[0, 0, 0], [2, 0, 1], [4, 2, 1.5], [6, 2, 0.5], [8, 0, 0.0]])
    st = np.array([1.5, 1.5, 1.5, 1.5])
    ms = MinSnapTrajectory()
    coeffs = ms.generate(wps, st)
    _, pos = ms.evaluate(coeffs, st, dt=0.01)
    skip = max(1, len(pos) // 200)
    idx = list(range(0, len(pos), skip))
    anim = SimAnimator("min_snap", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Minimum-Snap Trajectory")
    ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="red", s=60, marker="D", zorder=5, label="WP")
    (trail,) = ax.plot([], [], [], "b-", lw=1.5)
    (dot,) = ax.plot([], [], [], "ko", ms=5)
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(-0.5, 3)
    ax.set_zlim(-0.5, 2.5)
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
