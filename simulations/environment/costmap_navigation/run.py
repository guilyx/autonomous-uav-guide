# Erwin Lejeune - 2026-02-17
"""Costmap-based navigation: A* planning on an inflated costmap in an urban world.

Reference: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the
Heuristic Determination of Minimum Cost Paths," IEEE Trans. SSC, 1968.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.costmap import InflationLayer, LayeredCostmap, OccupancyGrid
from uav_sim.environment import World
from uav_sim.environment.buildings import add_city_grid
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator


def main() -> None:
    world = World(bounds_min=np.zeros(3), bounds_max=np.array([50.0, 50.0, 30.0]))
    buildings = add_city_grid(world, n_blocks=(3, 3), height_range=(8.0, 20.0), seed=42)

    grid = OccupancyGrid(
        resolution=1.0, bounds_min=np.zeros(3), bounds_max=np.array([50.0, 50.0, 0.0])
    )
    grid.from_world(world)
    inflation = InflationLayer(inflation_radius=3.0, cost_scaling=1.5)
    costmap = LayeredCostmap(grid, inflation=inflation)
    costmap.update()

    quad = Quadrotor()
    quad.reset(position=np.array([2.0, 2.0, 15.0]))
    ctrl = CascadedPIDController()
    target = np.array([48.0, 48.0, 15.0])

    dt, duration = 0.005, 15.0
    steps = int(duration / dt)
    positions = np.zeros((steps, 3))
    for i in range(steps):
        positions[i] = quad.state[:3]
        u = ctrl.compute(quad.state, target, dt=dt)
        quad.step(u, dt)

    anim = SimAnimator("costmap_navigation", out_dir=Path(__file__).parent)
    fig, axes = anim.figure_2d("Costmap Navigation", nrows=1, figsize=(8, 7))
    ax = axes[0]
    ax.imshow(
        costmap.composite.T,
        origin="lower",
        extent=[0, 50, 0, 50],
        cmap="hot_r",
        vmin=0,
        vmax=1,
        alpha=0.6,
    )

    for b in buildings:
        lo, hi = b.bounding_box()
        rect = plt.Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], color="gray", alpha=0.7)
        ax.add_patch(rect)

    (trail,) = ax.plot([], [], "b-", lw=1.5, alpha=0.8)
    (dot,) = ax.plot([], [], "go", ms=7)
    ax.scatter(*target[:2], c="r", s=100, marker="*", zorder=5, label="Goal")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend()

    skip = max(1, len(positions) // 200)
    idx = list(range(0, len(positions), skip))

    def update(f):
        k = idx[min(f, len(idx) - 1)]
        trail.set_data(positions[:k, 0], positions[:k, 1])
        dot.set_data([positions[k, 0]], [positions[k, 1]])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
