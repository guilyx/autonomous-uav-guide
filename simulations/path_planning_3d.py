# Erwin Lejeune - 2026-02-16
"""3D path planning demonstration: RRT* through obstacle field."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from quadrotor_sim.planning.rrt_3d import RRTStar3D


def main() -> None:
    """Plan a path using RRT* through a 3D obstacle field and visualise."""
    obstacles = [
        (np.array([3.0, 3.0, 3.0]), 1.5),
        (np.array([7.0, 5.0, 4.0]), 1.0),
        (np.array([5.0, 8.0, 6.0]), 1.2),
    ]

    planner = RRTStar3D(
        bounds_min=np.array([0, 0, 0]),
        bounds_max=np.array([10, 10, 10]),
        obstacles=obstacles,
        step_size=0.8,
        goal_radius=0.5,
        max_iter=5000,
        goal_bias=0.1,
        gamma=8.0,
    )

    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([9.0, 9.0, 9.0])
    path = planner.plan(start, goal, seed=42)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw obstacles.
    for centre, radius in obstacles:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = centre[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.3, color="red")

    # Draw tree.
    for i, node in enumerate(planner.nodes):
        parent = planner.parents[i]
        if parent >= 0:
            p = planner.nodes[parent]
            ax.plot(
                [p[0], node[0]], [p[1], node[1]], [p[2], node[2]], "g-", alpha=0.1, linewidth=0.5
            )

    # Draw path.
    if path is not None:
        pts = np.array(path)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "b-", linewidth=2, label="RRT* path")

    ax.scatter(*start, color="green", s=100, marker="^", label="Start")
    ax.scatter(*goal, color="red", s=100, marker="v", label="Goal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("RRT* 3D Path Planning")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
