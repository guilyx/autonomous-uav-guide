# Erwin Lejeune - 2026-02-17
"""Grid-based occupancy mapping: quadrotor explores an environment with lidar.

The drone follows a lawnmower pattern via pure pursuit, building an occupancy
grid incrementally using log-odds inverse sensor model.  Left panel: 2D map
with obstacles, drone, and lidar rays.  Right panel: evolving occupancy grid
with colour map.

Reference: S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor
Models," Autonomous Robots, 2003.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.costmap.occupancy_grid import OccupancyGrid
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.perception.occupancy_mapping import OccupancyMapper
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
)

matplotlib.use("Agg")


def _lawnmower_path(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    spacing: float,
    altitude: float,
) -> np.ndarray:
    """Generate a lawnmower (boustrophedon) waypoint sequence."""
    wps = []
    y_vals = np.arange(y_min, y_max + spacing, spacing)
    for i, y in enumerate(y_vals):
        if i % 2 == 0:
            wps.append([x_min, y, altitude])
            wps.append([x_max, y, altitude])
        else:
            wps.append([x_max, y, altitude])
            wps.append([x_min, y, altitude])
    return np.array(wps)


def main() -> None:
    rng = np.random.default_rng(42)
    world_size = 10.0
    cruise_alt = 1.0

    # Ground-truth obstacles (circles in 2D)
    obstacles = [
        (np.array([3.0, 3.0]), 0.8),
        (np.array([6.0, 4.0]), 1.0),
        (np.array([4.0, 7.0]), 0.7),
        (np.array([7.5, 7.5]), 0.9),
    ]

    # Occupancy grid
    grid = OccupancyGrid(
        resolution=0.5,
        bounds_min=np.zeros(3),
        bounds_max=np.array([world_size, world_size, 0.0]),
    )
    mapper = OccupancyMapper(grid)
    lidar = Lidar2D(max_range=4.0, num_beams=36, noise_std=0.05, seed=42)

    # ── Generate flight path ──────────────────────────────────────────────
    path_3d = _lawnmower_path(
        0.5, world_size - 0.5, 0.5, world_size - 0.5, spacing=2.0, altitude=cruise_alt
    )

    quad = Quadrotor()
    quad.reset(position=np.array([0.5, 0.5, cruise_alt]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=0.8, waypoint_threshold=0.3, adaptive=True)

    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=60.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))

    # ── Build map incrementally and record snapshots ──────────────────────
    n_steps = len(flight_states)
    scan_interval = max(1, n_steps // 80)
    map_snapshots: list[np.ndarray] = []
    scan_positions: list[np.ndarray] = []
    scan_ranges_list: list[np.ndarray] = []

    for i in range(0, n_steps, scan_interval):
        s = flight_states[i]
        pos = s[:3]

        # Simulate lidar ranges against circular obstacles
        yaw = s[5]
        angles = np.linspace(0, 2 * np.pi, lidar.num_beams, endpoint=False) + yaw
        ranges = np.full(lidar.num_beams, lidar.max_range)
        for j, a in enumerate(angles):
            direction = np.array([np.cos(a), np.sin(a)])
            for oc, orad in obstacles:
                oc2 = oc[:2] if len(oc) > 2 else oc
                to_obs = oc2 - pos[:2]
                proj = float(np.dot(to_obs, direction))
                if proj < 0:
                    continue
                perp = float(np.linalg.norm(to_obs - proj * direction))
                if perp < orad:
                    hit_dist = proj - np.sqrt(max(0, orad**2 - perp**2))
                    if 0 < hit_dist < ranges[j]:
                        ranges[j] = hit_dist + rng.normal(0, lidar.noise_std)

        beam_angles = np.linspace(0, 2 * np.pi, lidar.num_beams, endpoint=False)
        mapper.update(pos, ranges, beam_angles + yaw, max_range=lidar.max_range)
        map_snapshots.append(grid.grid.copy())
        scan_positions.append(pos[:2].copy())
        scan_ranges_list.append(ranges.copy())

    # ── Animation ─────────────────────────────────────────────────────────
    n_snapshots = len(map_snapshots)
    pos = flight_states[:, :3]
    skip = max(1, n_steps // (n_snapshots))
    frame_idx = list(range(0, n_steps, skip))[:n_snapshots]
    n_frames = len(frame_idx)

    anim = SimAnimator("grid_mapping", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 6.5))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    ax_scene = fig.add_subplot(gs[0])
    ax_map = fig.add_subplot(gs[1])
    fig.suptitle("Grid Mapping — Lawnmower Exploration", fontsize=13)

    # Scene panel
    ax_scene.set_aspect("equal")
    ax_scene.set_xlim(-0.5, world_size + 0.5)
    ax_scene.set_ylim(-0.5, world_size + 0.5)
    ax_scene.set_xlabel("X [m]")
    ax_scene.set_ylabel("Y [m]")
    ax_scene.grid(True, alpha=0.15)
    for oc, orad in obstacles:
        circle = plt.Circle(oc, orad, color="red", alpha=0.3)
        ax_scene.add_patch(circle)
    ax_scene.plot(path_3d[:, 0], path_3d[:, 1], "c--", lw=0.5, alpha=0.3, label="Plan")
    (trail,) = ax_scene.plot([], [], "b-", lw=1.0, alpha=0.6)
    ray_lines: list = []
    ax_scene.legend(fontsize=7)

    # Map panel
    ax_map.set_aspect("equal")
    ax_map.set_xlim(-0.5, world_size + 0.5)
    ax_map.set_ylim(-0.5, world_size + 0.5)
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_title("Occupancy Grid", fontsize=10)
    g_shape = grid.grid.shape
    im = ax_map.imshow(
        np.full(g_shape[:2], 0.5).T,
        origin="lower",
        extent=[0, world_size, 0, world_size],
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )

    vehicle_arts: list = []

    def update(f):
        nonlocal ray_lines
        for rl in ray_lines:
            rl.remove()
        ray_lines.clear()

        k = frame_idx[min(f, len(frame_idx) - 1)]
        snap_i = min(f, n_snapshots - 1)

        # Trail
        trail.set_data(pos[:k, 0], pos[:k, 1])

        # Lidar rays
        sp = scan_positions[snap_i]
        sr = scan_ranges_list[snap_i]
        yaw_k = flight_states[k, 5]
        angles_k = np.linspace(0, 2 * np.pi, lidar.num_beams, endpoint=False) + yaw_k
        for j in range(lidar.num_beams):
            end = sp + sr[j] * np.array([np.cos(angles_k[j]), np.sin(angles_k[j])])
            (rl,) = ax_scene.plot([sp[0], end[0]], [sp[1], end[1]], "g-", lw=0.3, alpha=0.3)
            ray_lines.append(rl)

        # Vehicle
        clear_vehicle_artists(vehicle_arts)
        vehicle_arts.extend(draw_quadrotor_2d(ax_scene, pos[k, :2], yaw_k, size=0.5))

        # Update occupancy grid
        snap = map_snapshots[snap_i]
        if snap.ndim == 3:
            im.set_data(snap[:, :, 0].T)
        else:
            im.set_data(snap.T)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
