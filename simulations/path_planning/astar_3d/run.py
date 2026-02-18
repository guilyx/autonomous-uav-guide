# Erwin Lejeune - 2026-02-15
"""3-D A* grid search through walled obstacles.

Reference: P. E. Hart et al., "A Formal Basis for the Heuristic Determination
of Minimum Cost Paths," IEEE TSSC, 1968. DOI: 10.1109/TSSC.1968.300136
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.path_planning.astar_3d import AStar3D
from uav_sim.visualization import SimAnimator


def main() -> None:
    size = 30
    grid = np.zeros((size, size, size), dtype=bool)
    grid[10:12, :20, :20] = True
    grid[20:22, 10:, 10:] = True
    path = AStar3D(grid).plan((0, 0, 0), (size - 1, size - 1, size - 1))
    if path is None:
        print("No path found!")
        return
    pts = np.array(path)
    skip = max(1, len(pts) // 150)
    idx = list(range(0, len(pts), skip))
    anim = SimAnimator("astar_3d", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("A* 3D Path Planning")
    obs = np.argwhere(grid)
    if len(obs) > 500:
        obs = obs[np.random.default_rng(0).choice(len(obs), 500, replace=False)]
    ax.scatter(obs[:, 0], obs[:, 1], obs[:, 2], c="red", alpha=0.08, s=8)
    ax.scatter(0, 0, 0, c="green", s=100, marker="^", label="Start", zorder=5)
    ax.scatter(size - 1, size - 1, size - 1, c="red", s=100, marker="v", label="Goal", zorder=5)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
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
