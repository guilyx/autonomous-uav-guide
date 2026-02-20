# Erwin Lejeune - 2026-02-20
"""Dynamic costmap navigation with local sensor-based obstacle avoidance.

Demonstrates a drone navigating toward a goal while dynamically avoiding
moving obstacles detected via simulated lidar.  The local costmap is rebuilt
every planning cycle from lidar hits and includes:

  - **Footprint inflation** — obstacles expanded by the vehicle bounding radius.
  - **Speed-based dynamic inflation** — the faster the drone flies, the wider
    the safety margin (accounts for braking distance).

The reactive planner picks the lowest-cost direction toward the goal on
the local costmap, re-planning at ~10 Hz.

Reference: D. Fox, W. Burgard, S. Thrun, "The Dynamic Window Approach to
Collision Avoidance," IEEE RA Magazine, 1997.
"""

from __future__ import annotations

import heapq
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from uav_sim.control import StateManager
from uav_sim.costmap import OccupancyGrid
from uav_sim.environment import default_world
from uav_sim.environment.buildings import add_city_grid
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import _draw_box_3d
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
GRID_RES = 0.5
FOOTPRINT_RADIUS = 0.5
FOOTPRINT_PADDING = 0.3
BASE_INFLATION = FOOTPRINT_RADIUS + FOOTPRINT_PADDING
SPEED_INFLATION_SCALE = 0.8
COST_SCALING = 2.5
REPLAN_EVERY = 0.1


def _moving_obstacle(t: float, idx: int) -> np.ndarray:
    """Return (x, y) position of dynamic obstacle *idx* at time *t*."""
    if idx == 0:
        return np.array([10.0 + 8.0 * np.sin(0.15 * t), 18.0 + 4.0 * np.cos(0.2 * t)])
    if idx == 1:
        return np.array([20.0 - 5.0 * np.cos(0.12 * t), 8.0 + 10.0 * np.sin(0.1 * t)])
    return np.array([15.0 + 6.0 * np.sin(0.18 * t + 1.5), 15.0 + 6.0 * np.cos(0.14 * t)])


def _build_local_costmap(
    static_occ: np.ndarray,
    obs_positions: list[np.ndarray],
    obs_radius: float,
    drone_speed: float,
    resolution: float,
) -> np.ndarray:
    """Compose local costmap: static obstacles + dynamic obstacles + inflation."""
    local = static_occ.copy()
    rows, cols = local.shape

    for op in obs_positions:
        ci, cj = int(op[0] / resolution), int(op[1] / resolution)
        r_cells = int(np.ceil(obs_radius / resolution)) + 1
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    d = np.sqrt(di**2 + dj**2) * resolution
                    if d <= obs_radius:
                        local[ni, nj] = 1.0

    inflation_radius = BASE_INFLATION + SPEED_INFLATION_SCALE * drone_speed
    dist = distance_transform_edt(1.0 - (local >= 0.5).astype(float)) * resolution
    inflated = np.where(
        dist <= inflation_radius,
        np.exp(-COST_SCALING * dist / inflation_radius),
        0.0,
    ).astype(np.float32)
    return np.maximum(local, inflated)


def _plan_on_costmap(
    costmap: np.ndarray,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    resolution: float,
) -> list[np.ndarray] | None:
    """A* on 2D costmap → list of world-frame (x,y) waypoints."""
    rows, cols = costmap.shape
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
            path.reverse()
            return [np.array([p[0] + 0.5, p[1] + 0.5]) * resolution for p in path]
        for dx, dy in dirs:
            nb = (current[0] + dx, current[1] + dy)
            if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                continue
            cell_cost = costmap[nb[0], nb[1]]
            if cell_cost > 0.95:
                continue
            step = np.sqrt(dx**2 + dy**2) * resolution
            cost = g_score[current] + step + cell_cost * 8.0
            if nb not in g_score or cost < g_score[nb]:
                g_score[nb] = cost
                h = np.sqrt((nb[0] - goal_cell[0]) ** 2 + (nb[1] - goal_cell[1]) ** 2) * resolution
                heapq.heappush(open_set, (cost + h, nb))
                came_from[nb] = current
    return None


def main() -> None:
    world, _ = default_world()
    add_city_grid(
        world,
        n_blocks=(2, 2),
        block_size=4.0,
        street_width=6.0,
        height_range=(8.0, 20.0),
        seed=42,
    )

    static_grid = OccupancyGrid(
        resolution=GRID_RES,
        bounds_min=np.zeros(3),
        bounds_max=np.array([WORLD_SIZE, WORLD_SIZE, 0.0]),
    )
    static_grid.from_world(world)
    static_occ = static_grid.grid.copy()

    start_xy = np.array([3.0, 3.0])
    goal_xy = np.array([27.0, 27.0])
    goal_cell = (int(goal_xy[0] / GRID_RES), int(goal_xy[1] / GRID_RES))
    n_obs = 3
    obs_radius = 1.5

    quad = Quadrotor()
    quad.reset(position=np.array([start_xy[0], start_xy[1], 0.0]))
    sm = StateManager(quad)
    dt = 0.005

    sm.arm()
    sm.run_takeoff(altitude=CRUISE_ALT, dt=dt, timeout=8.0)

    sm.offboard()
    sim_time = 80.0
    replan_timer = 0.0
    local_target = np.array([goal_xy[0], goal_xy[1], CRUISE_ALT])

    costmap_history: list[np.ndarray] = []
    obs_history: list[list[np.ndarray]] = []
    record_every = max(1, int(0.5 / dt))
    step_count = 0
    total_steps = int(sim_time / dt)

    for i in range(total_steps):
        t = i * dt
        pos = quad.position

        if np.linalg.norm(pos[:2] - goal_xy) < 1.5:
            break

        obs_pos = [_moving_obstacle(t, k) for k in range(n_obs)]
        speed = float(np.linalg.norm(quad.velocity[:2]))

        replan_timer += dt
        if replan_timer >= REPLAN_EVERY:
            replan_timer = 0.0
            costmap = _build_local_costmap(static_occ, obs_pos, obs_radius, speed, GRID_RES)
            sc = (int(pos[0] / GRID_RES), int(pos[1] / GRID_RES))
            sc = (np.clip(sc[0], 0, costmap.shape[0] - 1), np.clip(sc[1], 0, costmap.shape[1] - 1))
            plan = _plan_on_costmap(costmap, sc, goal_cell, GRID_RES)
            if plan and len(plan) > 3:
                ahead = plan[min(6, len(plan) - 1)]
                local_target = np.array([ahead[0], ahead[1], CRUISE_ALT])

        sm.set_position_target(local_target)
        sm.step(dt)

        if step_count % record_every == 0:
            costmap_snap = _build_local_costmap(static_occ, obs_pos, obs_radius, speed, GRID_RES)
            costmap_history.append(costmap_snap)
            obs_history.append([o.copy() for o in obs_pos])
        step_count += 1

    sm.run_land(dt=dt, timeout=6.0)
    states_arr = np.array(sm.states)
    pos = states_arr[:, :3]

    # ── Visualization ──────────────────────────────────────────────────
    n_records = len(costmap_history)
    rec_step = max(1, len(pos) // n_records)
    rec_indices = [min(i * rec_step, len(pos) - 1) for i in range(n_records)]

    anim = SimAnimator("costmap_navigation", out_dir=Path(__file__).parent, dpi=72)
    fig = plt.figure(figsize=(18, 8))
    anim._fig = fig
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_cost = fig.add_subplot(gs[0, 1])
    ax_infl = fig.add_subplot(gs[0, 2])
    fig.suptitle("Dynamic Costmap Navigation — Footprint + Speed Inflation", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    for b in world.obstacles:
        lo, hi = b.bounding_box()
        _draw_box_3d(ax3d, lo, hi)

    (trail3d,) = ax3d.plot([], [], [], "lime", lw=1.5, alpha=0.7, label="Drone path")
    ax3d.scatter(*[start_xy[0]], start_xy[1], 0, c="cyan", s=80, marker="^", label="Start")
    ax3d.scatter(*[goal_xy[0]], goal_xy[1], CRUISE_ALT, c="red", s=80, marker="*", label="Goal")
    ax3d.legend(fontsize=7, loc="upper left")

    extent = [0, WORLD_SIZE, 0, WORLD_SIZE]
    labels = [("Local Costmap (footprint)", ax_cost), ("Speed-Inflated Costmap", ax_infl)]
    for lbl, ax in labels:
        ax.set_aspect("equal")
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_xlabel("X [m]", fontsize=7)
        ax.set_ylabel("Y [m]", fontsize=7)
        ax.set_title(lbl, fontsize=9)
        ax.tick_params(labelsize=6)

    im_cost = ax_cost.imshow(
        static_occ.T,
        origin="lower",
        extent=extent,
        cmap="hot_r",
        vmin=0,
        vmax=1,
    )
    im_infl = ax_infl.imshow(
        static_occ.T,
        origin="lower",
        extent=extent,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    (drone_dot_c,) = ax_cost.plot([], [], "go", ms=6, zorder=10)
    (drone_dot_i,) = ax_infl.plot([], [], "go", ms=6, zorder=10)
    (goal_dot_c,) = ax_cost.plot(goal_xy[0], goal_xy[1], "r*", ms=10, zorder=10)
    (goal_dot_i,) = ax_infl.plot(goal_xy[0], goal_xy[1], "r*", ms=10, zorder=10)
    obs_dots_c = [ax_cost.plot([], [], "mo", ms=8, zorder=10)[0] for _ in range(n_obs)]
    obs_dots_i = [ax_infl.plot([], [], "mo", ms=8, zorder=10)[0] for _ in range(n_obs)]
    obs_3d_arts: list = []

    title = ax3d.set_title("Navigating...")
    veh_arts: list = []

    def update(f: int) -> None:
        ri = min(f, n_records - 1)
        k = rec_indices[ri]

        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])

        clear_vehicle_artists(veh_arts)
        euler = states_arr[k, 3:6]
        R = Quadrotor.rotation_matrix(*euler)
        veh_arts.extend(draw_quadrotor_3d(ax3d, pos[k], R, size=1.5))

        clear_vehicle_artists(obs_3d_arts)
        obs_pos = obs_history[ri]
        for oi, op in enumerate(obs_pos):
            ax3d_s = ax3d.scatter(
                op[0],
                op[1],
                CRUISE_ALT,
                c="magenta",
                s=60,
                marker="o",
                zorder=5,
            )
            obs_3d_arts.append(ax3d_s)
            obs_dots_c[oi].set_data([op[0]], [op[1]])
            obs_dots_i[oi].set_data([op[0]], [op[1]])

        drone_dot_c.set_data([pos[k, 0]], [pos[k, 1]])
        drone_dot_i.set_data([pos[k, 0]], [pos[k, 1]])

        costmap = costmap_history[ri]
        static_inflated = _build_local_costmap(static_occ, obs_pos, obs_radius, 0.0, GRID_RES)
        im_cost.set_data(static_inflated.T)
        im_infl.set_data(costmap.T)

        spd = float(np.linalg.norm(states_arr[k, 6:8]))
        infl_r = BASE_INFLATION + SPEED_INFLATION_SCALE * spd
        dist = float(np.linalg.norm(pos[k, :2] - goal_xy))
        title.set_text(f"Speed: {spd:.1f} m/s | Inflation: {infl_r:.2f} m | To goal: {dist:.1f} m")

    anim.animate(update, n_records)
    anim.save()


if __name__ == "__main__":
    main()
