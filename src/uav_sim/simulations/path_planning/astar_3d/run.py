# Erwin Lejeune - 2026-02-15
"""3-D A* path planning: 3-panel, two-phase visualisation.

Phase 1 — Algorithm: node-by-node exploration with the A* frontier.
           Path smoothing is visualised as a distinct intermediate step.
Phase 2 — Platform: quadrotor executing takeoff -> pure-pursuit along the
           smoothed A* path -> loiter -> landing.

Reference: P. E. Hart et al., "A Formal Basis for the Heuristic Determination
of Minimum Cost Paths," IEEE TSSC, 1968. DOI: 10.1109/TSSC.1968.300136
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30
GRID_RES = 1.0  # metres per voxel


def _build_occupancy(buildings, size: int) -> np.ndarray:
    """Convert box obstacles to a 3-D boolean grid (True = occupied)."""
    grid = np.zeros((size, size, size), dtype=bool)
    for b in buildings:
        lo = np.clip(np.floor(b.min_corner).astype(int), 0, size - 1)
        hi = np.clip(np.ceil(b.max_corner).astype(int), 0, size)
        grid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    return grid


def _astar(grid, start, goal):
    """A* with exploration recording."""
    import heapq

    size = grid.shape[0]
    explored_order: list[tuple[int, int, int]] = []
    open_set: list[tuple[float, tuple]] = []
    heapq.heappush(open_set, (0.0, start))
    came_from: dict[tuple, tuple | None] = {start: None}
    g_score: dict[tuple, float] = {start: 0.0}
    dirs = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    while open_set:
        _, current = heapq.heappop(open_set)
        explored_order.append(current)
        if current == goal:
            path: list[tuple] = []
            c: tuple | None = current
            while c is not None:
                path.append(c)
                c = came_from[c]
            return path[::-1], explored_order
        for dx, dy, dz in dirs:
            nb = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not all(0 <= nb[j] < size for j in range(3)):
                continue
            if grid[nb]:
                continue
            cost = g_score[current] + np.sqrt(dx**2 + dy**2 + dz**2)
            if nb not in g_score or cost < g_score[nb]:
                g_score[nb] = cost
                h = np.sqrt(sum((a - b) ** 2 for a, b in zip(nb, goal, strict=False)))
                heapq.heappush(open_set, (cost + h, nb))
                came_from[nb] = current
    return None, explored_order


def main() -> None:
    world, buildings = default_world()
    grid = _build_occupancy(buildings, WORLD_SIZE)

    start = (2, 2, 12)
    goal = (WORLD_SIZE - 3, WORLD_SIZE - 3, 12)
    path_nodes, explored = _astar(grid, start, goal)
    if path_nodes is None:
        print("No path found!")
        return
    raw_path = np.array(path_nodes, dtype=float)
    smooth_path = smooth_path_3d(raw_path, epsilon=2.0, min_spacing=1.5)

    # Phase 2: fly the smoothed path
    quad = Quadrotor()
    quad.reset(position=np.array([float(start[0]), float(start[1]), 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)

    flight_states = fly_mission(
        quad,
        ctrl,
        smooth_path,
        cruise_alt=float(start[2]),
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.5,
        landing_duration=2.5,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation frames ──────────────────────────────────────────────
    explore_step = max(1, len(explored) // 80)
    explore_frames = list(range(0, len(explored), explore_step))
    smooth_pause = 20  # frames to show smoothing transition
    fly_step = max(1, len(flight_pos) // 100)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_explore = len(explore_frames)
    n_fly = len(fly_frames)
    total = n_explore + smooth_pause + n_fly

    viz = ThreePanelViz(title="A* 3D — Search → Smooth → Flight", world_size=float(WORLD_SIZE))
    viz.draw_buildings(buildings)
    viz.mark_start_goal(np.array(start, dtype=float), np.array(goal, dtype=float))

    explored_arr = np.array(explored)
    explore_scat = viz.ax3d.scatter([], [], [], c="cyan", s=2, alpha=0.3)
    (raw_line_3d,) = viz.ax3d.plot([], [], [], "b-", lw=1.5, alpha=0.0)
    (smooth_line_3d,) = viz.ax3d.plot([], [], [], "lime", lw=2.5, alpha=0.0, label="Smoothed")
    (raw_line_top,) = viz.ax_top.plot([], [], "b-", lw=1.0, alpha=0.0)
    (smooth_line_top,) = viz.ax_top.plot([], [], "lime", lw=2.0, alpha=0.0)
    (raw_line_side,) = viz.ax_side.plot([], [], "b-", lw=1.0, alpha=0.0)
    (smooth_line_side,) = viz.ax_side.plot([], [], "lime", lw=2.0, alpha=0.0)

    fly_trail = viz.create_trail_artists()
    title = viz.ax3d.set_title("Phase 1: A* Exploration")

    anim = SimAnimator("astar_3d", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    def update(f: int) -> None:
        if f < n_explore:
            k = explore_frames[f]
            pts = explored_arr[: k + 1]
            explore_scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            pct = int(100 * (f + 1) / n_explore)
            title.set_text(f"Phase 1: A* Exploration — {pct}%")
        elif f < n_explore + smooth_pause:
            sf = f - n_explore
            # Show raw path, then smoothed
            if sf < smooth_pause // 2:
                raw_line_3d.set_alpha(1.0)
                raw_line_3d.set_data(raw_path[:, 0], raw_path[:, 1])
                raw_line_3d.set_3d_properties(raw_path[:, 2])
                raw_line_top.set_alpha(1.0)
                raw_line_top.set_data(raw_path[:, 0], raw_path[:, 1])
                raw_line_side.set_alpha(1.0)
                raw_line_side.set_data(raw_path[:, 0], raw_path[:, 2])
                title.set_text("Raw A* Path")
            else:
                smooth_line_3d.set_alpha(1.0)
                smooth_line_3d.set_data(smooth_path[:, 0], smooth_path[:, 1])
                smooth_line_3d.set_3d_properties(smooth_path[:, 2])
                smooth_line_top.set_alpha(1.0)
                smooth_line_top.set_data(smooth_path[:, 0], smooth_path[:, 1])
                smooth_line_side.set_alpha(1.0)
                smooth_line_side.set_data(smooth_path[:, 0], smooth_path[:, 2])
                raw_line_3d.set_alpha(0.2)
                raw_line_top.set_alpha(0.2)
                raw_line_side.set_alpha(0.2)
                title.set_text("Smoothed Path (RDP + Resample)")
        else:
            fi = f - n_explore - smooth_pause
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            viz.update_trail(fly_trail, flight_pos, k)
            viz.update_vehicle(flight_pos[k], flight_states[k, 3:6], size=1.5)
            title.set_text("Phase 2: Quadrotor Following Smoothed A* Path")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
