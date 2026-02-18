# Erwin Lejeune - 2026-02-18
"""Costmap-based navigation: A* on an inflated costmap in a 30 m urban world.

Three panels (3D, top-down, side):
  - 3D scene with buildings and costmap floor surface.
  - Top-down costmap heatmap with trajectory overlay.
  - Side view with altitude profile.

The quadrotor performs takeoff -> A* path -> loiter -> land.

Reference: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the
Heuristic Determination of Minimum Cost Paths," IEEE Trans. SSC, 1968.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.costmap import InflationLayer, LayeredCostmap, OccupancyGrid
from uav_sim.environment import World
from uav_sim.environment.buildings import add_city_grid
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 15.0


def _simple_astar_2d(
    costmap_grid: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> list[tuple[int, int]] | None:
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
    world = World(bounds_min=np.zeros(3), bounds_max=np.full(3, WORLD_SIZE))
    add_city_grid(
        world,
        n_blocks=(2, 2),
        block_size=4.0,
        street_width=6.0,
        height_range=(8.0, 22.0),
        seed=42,
    )

    grid = OccupancyGrid(
        resolution=1.0, bounds_min=np.zeros(3), bounds_max=np.array([WORLD_SIZE, WORLD_SIZE, 0.0])
    )
    grid.from_world(world)
    inflation = InflationLayer(inflation_radius=3.0, cost_scaling=1.5)
    costmap = LayeredCostmap(grid, inflation=inflation)
    costmap.update()

    start_xy = np.array([2.0, 2.0])
    goal_xy = np.array([28.0, 28.0])
    comp = costmap.composite
    cell_path = _simple_astar_2d(
        comp, (int(start_xy[0]), int(start_xy[1])), (int(goal_xy[0]), int(goal_xy[1]))
    )
    if cell_path is None:
        print("A* failed to find a path!")
        return

    raw_path_3d = np.array([[c[0] + 0.5, c[1] + 0.5, CRUISE_ALT] for c in cell_path])
    if np.linalg.norm(raw_path_3d[-1, :2] - goal_xy) > 1.0:
        raw_path_3d = np.vstack([raw_path_3d, [goal_xy[0], goal_xy[1], CRUISE_ALT]])
    path_3d = smooth_path_3d(raw_path_3d, epsilon=2.0, min_spacing=2.0)

    quad = Quadrotor()
    quad.reset(position=np.array([start_xy[0], start_xy[1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    states = fly_mission(
        quad,
        ctrl,
        path_3d,
        cruise_alt=CRUISE_ALT,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=3.0,
        landing_duration=4.0,
        loiter_duration=1.0,
    )
    pos = states[:, :3]

    # ── 3-Panel viz ────────────────────────────────────────────────────
    viz = ThreePanelViz(
        title="Costmap Navigation — Takeoff → A* Path → Land", world_size=WORLD_SIZE
    )
    viz.draw_buildings(world.obstacles)

    # Costmap floor surface in 3D
    grid.visualize_3d(viz.ax3d, cmap="hot_r", alpha=0.25, stride=2)

    # Costmap heatmap underlay in top-down
    viz.ax_top.imshow(
        comp.T,
        origin="lower",
        extent=[0, WORLD_SIZE, 0, WORLD_SIZE],
        cmap="hot_r",
        vmin=0,
        vmax=1,
        alpha=0.4,
        zorder=0,
    )

    viz.draw_path(path_3d, color="cyan", lw=1.2, alpha=0.6, label="A* path")
    start_3d = np.array([start_xy[0], start_xy[1], 0.0])
    goal_3d = np.array([goal_xy[0], goal_xy[1], CRUISE_ALT])
    viz.mark_start_goal(start_3d, goal_3d)

    anim = SimAnimator("costmap_navigation", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="orange")

    skip = max(1, len(pos) // 200)
    idx = list(range(0, len(pos), skip))

    def update(f: int) -> None:
        k = idx[min(f, len(idx) - 1)]
        viz.update_trail(trail_arts, pos, k)
        viz.update_vehicle(pos[k], states[k, 3:6], size=1.5)

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
