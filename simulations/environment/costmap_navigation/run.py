# Erwin Lejeune - 2026-02-15
"""Costmap-based navigation: A* planning on an inflated costmap in an urban world.

The quadrotor performs a full mission: takeoff -> fly A* path via pure-pursuit ->
loiter at goal -> land.  Left panel shows the 3D scene with building wireframes,
costmap floor surface, and quadrotor model.  Right panel shows a 2D top-down
costmap heatmap with trajectory overlay.

Reference: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the
Heuristic Determination of Minimum Cost Paths," IEEE Trans. SSC, 1968.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from uav_sim.costmap import InflationLayer, LayeredCostmap, OccupancyGrid
from uav_sim.environment import World
from uav_sim.environment.buildings import add_city_grid
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


def _simple_astar_2d(
    costmap_grid: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Minimal 2-D A* on the costmap grid.  Returns cell path or None."""
    rows, cols = costmap_grid.shape
    open_set: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}

    dirs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_cell:
            path: list[tuple[int, int]] = []
            c: tuple[int, int] | None = current
            while c is not None:
                path.append(c)
                c = came_from[c]
            return path[::-1]
        for dx, dy in dirs:
            nb = (current[0] + dx, current[1] + dy)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            cell_cost = costmap_grid[nb[0], nb[1]]
            if cell_cost > 0.9:
                continue
            step = np.sqrt(dx**2 + dy**2)
            cost = g_score[current] + step + cell_cost * 5.0
            if nb not in g_score or cost < g_score[nb]:
                g_score[nb] = cost
                h = np.sqrt((nb[0] - goal_cell[0]) ** 2 + (nb[1] - goal_cell[1]) ** 2)
                heapq.heappush(open_set, (cost + h, nb))
                came_from[nb] = current
    return None


def main() -> None:
    world = World(bounds_min=np.zeros(3), bounds_max=np.array([50.0, 50.0, 30.0]))
    buildings = add_city_grid(world, n_blocks=(3, 3), height_range=(8.0, 20.0), seed=42)

    grid = OccupancyGrid(
        resolution=1.0,
        bounds_min=np.zeros(3),
        bounds_max=np.array([50.0, 50.0, 0.0]),
    )
    grid.from_world(world)
    inflation = InflationLayer(inflation_radius=3.0, cost_scaling=1.5)
    costmap = LayeredCostmap(grid, inflation=inflation)
    costmap.update()

    # ── A* path on the costmap ─────────────────────────────────────────────
    start_xy = np.array([2.0, 2.0])
    goal_xy = np.array([48.0, 48.0])
    comp = costmap.composite

    start_cell = (int(start_xy[0]), int(start_xy[1]))
    goal_cell = (int(goal_xy[0]), int(goal_xy[1]))
    cell_path = _simple_astar_2d(comp, start_cell, goal_cell)

    if cell_path is None:
        print("A* failed to find a path!")
        return

    cruise_alt = 15.0
    raw_path_3d = np.array([[c[0] + 0.5, c[1] + 0.5, cruise_alt] for c in cell_path])
    if np.linalg.norm(raw_path_3d[-1, :2] - goal_xy) > 1.0:
        raw_path_3d = np.vstack([raw_path_3d, [goal_xy[0], goal_xy[1], cruise_alt]])

    # Smooth the A* cell path: prune zigzags, resample for flyability
    path_3d = smooth_path_3d(raw_path_3d, epsilon=2.0, min_spacing=2.0)

    # ── Fly mission ────────────────────────────────────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([start_xy[0], start_xy[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    states = fly_mission(
        quad,
        ctrl,
        path_3d,
        cruise_alt=cruise_alt,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=3.0,
        landing_duration=4.0,
        loiter_duration=1.0,
    )
    pos = states[:, :3]

    # ── Animation setup ────────────────────────────────────────────────────
    anim = SimAnimator("costmap_navigation", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 6.5))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])
    fig.suptitle("Costmap Navigation — Takeoff → A* Path → Land", fontsize=13)

    # 3D panel
    grid.visualize_3d(ax3d, cmap="hot_r", alpha=0.3, stride=2)
    for b in buildings:
        lo, hi = b.bounding_box()
        xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
        ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
        ht = hi[2]
        ax3d.plot(xs, ys, [0] * 5, color="gray", lw=0.5)
        ax3d.plot(xs, ys, [ht] * 5, color="gray", lw=0.5)
        for x, y in zip(xs[:4], ys[:4], strict=True):
            ax3d.plot([x, x], [y, y], [0, ht], color="gray", lw=0.5, alpha=0.4)

    ax3d.plot(
        path_3d[:, 0],
        path_3d[:, 1],
        path_3d[:, 2],
        "c--",
        lw=1,
        alpha=0.5,
        label="A* path",
    )
    ax3d.scatter(*path_3d[-1], c="red", s=120, marker="*", label="Goal", zorder=5)
    ax3d.set_xlim(0, 50)
    ax3d.set_ylim(0, 50)
    ax3d.set_zlim(0, 30)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.5, alpha=0.7)

    # 2D panel
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
        ax2d.add_patch(
            Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], color="gray", alpha=0.7),
        )
    ax2d.plot(path_3d[:, 0], path_3d[:, 1], "c--", lw=1, alpha=0.5)
    ax2d.scatter(*goal_xy, c="r", s=100, marker="*", zorder=5, label="Goal")
    ax2d.set_xlim(0, 50)
    ax2d.set_ylim(0, 50)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.set_aspect("equal")
    ax2d.legend(fontsize=7)
    ax2d.set_title("Top-down Costmap", fontsize=10)
    (trail2d,) = ax2d.plot([], [], "b-", lw=1.5, alpha=0.8)

    skip = max(1, len(pos) // 200)
    idx = list(range(0, len(pos), skip))
    veh3d: list = []
    veh2d: list = []

    def update(f):
        k = idx[min(f, len(idx) - 1)]
        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])
        clear_vehicle_artists(veh3d)
        R = Quadrotor.rotation_matrix(*states[k, 3:6])
        veh3d.extend(draw_quadrotor_3d(ax3d, pos[k], R, size=1.5))

        trail2d.set_data(pos[:k, 0], pos[:k, 1])
        clear_vehicle_artists(veh2d)
        veh2d.extend(draw_quadrotor_2d(ax2d, pos[k, :2], states[k, 5], size=1.5))

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
