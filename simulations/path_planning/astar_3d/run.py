# Erwin Lejeune - 2026-02-17
"""3-D A* path planning: two-phase visualisation.

Phase 1 — Algorithm: node-by-node exploration with open/closed sets.
Phase 2 — Platform: quadrotor executing takeoff -> pure-pursuit along A* path
           -> loiter -> landing with full dynamics.

Reference: P. E. Hart et al., "A Formal Basis for the Heuristic Determination
of Minimum Cost Paths," IEEE TSSC, 1968. DOI: 10.1109/TSSC.1968.300136
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.path_smoothing import smooth_path_3d
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)


def main() -> None:
    # ── build grid ────────────────────────────────────────────────────────
    size = 20
    grid = np.zeros((size, size, size), dtype=bool)
    grid[7:9, :14, :14] = True
    grid[14:16, 6:, 6:] = True
    start, goal = (0, 0, 0), (size - 1, size - 1, size - 1)

    # ── run A* and capture exploration order ──────────────────────────────
    explored_order: list[tuple[int, int, int]] = []

    def instrumented_plan(s, g):
        """A* with exploration recording."""
        import heapq

        open_set: list[tuple[float, tuple]] = []
        heapq.heappush(open_set, (0.0, s))
        came_from: dict[tuple, tuple | None] = {s: None}
        g_score: dict[tuple, float] = {s: 0.0}
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
            if current == g:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            for dx, dy, dz in dirs:
                nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not all(0 <= nb[j] < size for j in range(3)):
                    continue
                if grid[nb]:
                    continue
                cost = g_score[current] + np.sqrt(dx**2 + dy**2 + dz**2)
                if nb not in g_score or cost < g_score[nb]:
                    g_score[nb] = cost
                    h = np.sqrt(sum((a - b) ** 2 for a, b in zip(nb, g, strict=False)))
                    heapq.heappush(open_set, (cost + h, nb))
                    came_from[nb] = current
        return None

    path_nodes = instrumented_plan(start, goal)
    if path_nodes is None:
        print("No path found!")
        return
    path_pts = np.array(path_nodes, dtype=float)

    # Smooth the raw A* path: prune zigzags, then resample evenly
    flight_path = smooth_path_3d(path_pts, epsilon=1.5, min_spacing=1.0)

    # ── Phase 2: fly mission (takeoff -> pure-pursuit -> loiter -> land) ──
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=2.0, waypoint_threshold=1.0, adaptive=True)

    states = fly_mission(
        quad,
        ctrl,
        flight_path,
        cruise_alt=float(flight_path[0, 2]),
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.0,
        landing_duration=2.0,
        loiter_duration=0.5,
    )
    flight_pos = states[:, :3]

    # ── Animation: Phase 1 (explore) + Phase 2 (fly) ─────────────────────
    explore_step = max(1, len(explored_order) // 100)
    explore_frames = list(range(0, len(explored_order), explore_step))
    fly_step = max(1, len(flight_pos) // 120)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_explore = len(explore_frames)
    n_fly = len(fly_frames)
    total_frames = n_explore + n_fly

    anim = SimAnimator("astar_3d", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("A* 3D — Search → Flight")
    obs_pts = np.argwhere(grid)
    rng = np.random.default_rng(0)
    if len(obs_pts) > 300:
        obs_pts = obs_pts[rng.choice(len(obs_pts), 300, replace=False)]
    ax.scatter(obs_pts[:, 0], obs_pts[:, 1], obs_pts[:, 2], c="gray", alpha=0.06, s=6)
    ax.scatter(*start, c="green", s=120, marker="^", label="Start", zorder=5)
    ax.scatter(*goal, c="red", s=120, marker="v", label="Goal", zorder=5)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
    ax.legend(fontsize=7, loc="upper left")

    explore_scat = ax.scatter([], [], [], c="cyan", s=2, alpha=0.3)
    (path_line,) = ax.plot([], [], [], "b-", lw=2.5, alpha=0.0)
    (flight_trail,) = ax.plot([], [], [], "orange", lw=1.8)
    title = ax.set_title("Phase 1: A* Exploration")

    explored_arr = np.array(explored_order)
    vehicle_arts: list = []

    def update(f):
        clear_vehicle_artists(vehicle_arts)
        if f < n_explore:
            k = explore_frames[f]
            pts = explored_arr[: k + 1]
            explore_scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            pct = int(100 * (f + 1) / n_explore)
            title.set_text(f"Phase 1: A* Exploration — {pct}% ({k + 1} nodes)")
        else:
            if f == n_explore:
                path_line.set_alpha(1.0)
                path_line.set_data(path_pts[:, 0], path_pts[:, 1])
                path_line.set_3d_properties(path_pts[:, 2])
            fi = f - n_explore
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            flight_trail.set_data(flight_pos[:k, 0], flight_pos[:k, 1])
            flight_trail.set_3d_properties(flight_pos[:k, 2])
            R = Quadrotor.rotation_matrix(*states[k, 3:6])
            vehicle_arts.extend(draw_quadrotor_3d(ax, flight_pos[k], R, size=0.8))
            title.set_text("Phase 2: Quadrotor Following A* Path")

    anim.animate(update, total_frames)
    anim.save()


if __name__ == "__main__":
    main()
