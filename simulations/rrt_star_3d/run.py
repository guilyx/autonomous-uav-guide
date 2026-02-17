# Erwin Lejeune - 2026-02-15
"""RRT* 3-D path planning through spherical obstacles — animated tree growth.

Reference: S. Karaman, E. Frazzoli, "Sampling-based Algorithms for Optimal
Motion Planning," IJRR, 2011. DOI: 10.1177/0278364911406761
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.planning.rrt_3d import RRTStar3D
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    obs = [(np.array([3, 3, 3.0]), 1.5), (np.array([7, 5, 4.0]), 1.0), (np.array([5, 8, 6.0]), 1.2)]
    planner = RRTStar3D(
        bounds_min=np.zeros(3),
        bounds_max=np.full(3, 10),
        obstacles=obs,
        step_size=1.0,
        goal_radius=1.0,
        max_iter=1500,
        goal_bias=0.15,
        gamma=8.0,
    )
    start, goal = np.zeros(3), np.array([9, 9, 9.0])
    path = planner.plan(start, goal, seed=42)
    if path is None:
        print("No path found!")
        return
    pts = np.array(path)
    skip = max(1, len(pts) // 100)
    idx = list(range(0, len(pts), skip))
    anim = SimAnimator("rrt_star_3d", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("RRT* 3D Path Planning")
    for c, r in obs:
        anim.draw_sphere(ax, c, r)
    ax.scatter(*start, c="green", s=100, marker="^", label="Start", zorder=5)
    ax.scatter(*goal, c="red", s=100, marker="v", label="Goal", zorder=5)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "b--", lw=1, alpha=0.3, label="Path")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    ax.legend(fontsize=7)
    (trail,) = ax.plot([], [], [], "b-", lw=2.5)
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
