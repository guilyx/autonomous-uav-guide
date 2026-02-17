# Erwin Lejeune - 2026-02-15
"""Costmap-based navigation: A* planning on an inflated costmap in an urban world.

Left panel: 3D scene with buildings, costmap surface, and quadrotor flight.
Right panel: 2D top-down costmap heatmap with trajectory overlay.

Reference: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the
Heuristic Determination of Minimum Cost Paths," IEEE Trans. SSC, 1968.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from uav_sim.costmap import InflationLayer, LayeredCostmap, OccupancyGrid
from uav_sim.environment import World
from uav_sim.environment.buildings import add_city_grid
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


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
    states = np.zeros((steps, 12))
    for i in range(steps):
        states[i] = quad.state
        u = ctrl.compute(quad.state, target, dt=dt)
        quad.step(u, dt)

    pos = states[:, :3]

    # ── Animation setup ────────────────────────────────────────────────────
    anim = SimAnimator("costmap_navigation", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 6.5))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])
    fig.suptitle("Costmap Navigation — 3D Scene", fontsize=13)

    # ── 3D panel: buildings as boxes, costmap as floor surface ─────────────
    comp = costmap.composite
    xg = np.arange(comp.shape[0] + 1)
    yg = np.arange(comp.shape[1] + 1)
    xg, yg = np.meshgrid(xg, yg, indexing="ij")
    ax3d.plot_surface(
        xg,
        yg,
        np.zeros_like(xg),
        facecolors=plt.cm.hot_r(comp[..., np.newaxis].repeat(1, axis=2))[:, :, 0],
        alpha=0.35,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=False,
    )
    for b in buildings:
        lo, hi = b.bounding_box()
        xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
        ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
        ht = hi[2]
        ax3d.plot(xs, ys, [0] * 5, color="gray", lw=0.5)
        ax3d.plot(xs, ys, [ht] * 5, color="gray", lw=0.5)
        for x, y in zip(xs[:4], ys[:4], strict=True):
            ax3d.plot([x, x], [y, y], [0, ht], color="gray", lw=0.5, alpha=0.4)

    ax3d.scatter(*target, c="red", s=120, marker="*", label="Goal", zorder=5)
    ax3d.set_xlim(0, 50)
    ax3d.set_ylim(0, 50)
    ax3d.set_zlim(0, 30)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.5, alpha=0.7)
    (dot3d,) = ax3d.plot([], [], [], "go", ms=6)

    # ── 2D panel: costmap heatmap ──────────────────────────────────────────
    ax2d.imshow(
        comp.T,
        origin="lower",
        extent=[0, 50, 0, 50],
        cmap="hot_r",
        vmin=0,
        vmax=1,
        alpha=0.6,
    )
    for b in buildings:
        lo, hi = b.bounding_box()
        ax2d.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], color="gray", alpha=0.7))
    ax2d.scatter(*target[:2], c="r", s=100, marker="*", zorder=5, label="Goal")
    ax2d.set_xlim(0, 50)
    ax2d.set_ylim(0, 50)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.set_aspect("equal")
    ax2d.legend(fontsize=7)
    ax2d.set_title("Top-down Costmap", fontsize=10)
    (trail2d,) = ax2d.plot([], [], "b-", lw=1.5, alpha=0.8)
    (dot2d,) = ax2d.plot([], [], "go", ms=5)

    skip = max(1, len(pos) // 200)
    idx = list(range(0, len(pos), skip))
    veh3d: list = []
    veh2d: list = []

    def update(f):
        k = idx[min(f, len(idx) - 1)]
        # 3D
        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])
        dot3d.set_data([pos[k, 0]], [pos[k, 1]])
        dot3d.set_3d_properties([pos[k, 2]])
        clear_vehicle_artists(veh3d)
        R = Quadrotor.rotation_matrix(*states[k, 3:6])
        veh3d.extend(draw_quadrotor_3d(ax3d, pos[k], R, scale=40.0))
        # 2D
        trail2d.set_data(pos[:k, 0], pos[:k, 1])
        dot2d.set_data([pos[k, 0]], [pos[k, 1]])
        clear_vehicle_artists(veh2d)
        veh2d.extend(draw_quadrotor_2d(ax2d, pos[k, :2], states[k, 5], scale=80.0))

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
