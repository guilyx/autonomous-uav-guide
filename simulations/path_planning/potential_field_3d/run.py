# Erwin Lejeune - 2026-02-15
"""3-D potential field navigation around spherical obstacles.

Reference: O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and
Mobile Robots," IJRR, 1986. DOI: 10.1177/027836498600500106
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.path_planning.potential_field_3d import PotentialField3D
from uav_sim.visualization import SimAnimator


def main() -> None:
    obs = [(np.array([4, 4, 4.0]), 1.5), (np.array([7, 3, 6.0]), 1.0)]
    planner = PotentialField3D(zeta=1.0, eta=80.0, rho0=3.0, step_size=0.3, max_iter=500)
    start, goal = np.zeros(3), np.array([9, 9, 9.0])
    pts = np.array(planner.plan(start, goal, obs))
    skip = max(1, len(pts) // 150)
    idx = list(range(0, len(pts), skip))
    anim = SimAnimator("potential_field_3d", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Potential Field 3D")
    for c, r in obs:
        anim.draw_sphere(ax, c, r)
    ax.scatter(*start, c="green", s=100, marker="^", label="Start", zorder=5)
    ax.scatter(*goal, c="red", s=100, marker="v", label="Goal", zorder=5)
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_zlim(-1, 11)
    ax.legend(fontsize=7)
    (trail,) = ax.plot([], [], [], "b-", lw=2)
    (dot,) = ax.plot([], [], [], "ko", ms=5)

    def update(f):
        k = idx[f]
        trail.set_data(pts[:k, 0], pts[:k, 1])
        trail.set_3d_properties(pts[:k, 2])
        dot.set_data([pts[k, 0]], [pts[k, 1]])
        dot.set_3d_properties([pts[k, 2]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
