# Erwin Lejeune - 2026-02-18
"""Grid-based occupancy mapping: quadrotor explores with 2D lidar.

Three panels (3D, top-down, side) + inset occupancy grid:
  - 3D scene with buildings, quadrotor model, and lidar FOV disc.
  - Top-down with lidar FOV wedge, ray hits, and exploration path.
  - Side view with altitude profile.
  - Inset: evolving occupancy grid.

Reference: S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor
Models," CMU Tech Report, 2003.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.sensor_viz import (
    draw_lidar2d_fov_3d,
    draw_lidar2d_fov_top,
    draw_lidar2d_rays_top,
)
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
GRID_RES = 0.5


def _lawnmower_path(size: float, alt: float, n_rows: int = 7) -> np.ndarray:
    """Lawnmower exploration with rounded transitions between rows."""
    margin = 3.0
    xs = np.linspace(margin, size - margin, 40)
    ys = np.linspace(margin, size - margin, n_rows)
    pts: list[np.ndarray] = []
    for i, y in enumerate(ys):
        row_x = xs if i % 2 == 0 else xs[::-1]
        for x in row_x:
            pts.append(np.array([x, y, alt]))
        if i < n_rows - 1:
            mid_y = (ys[i] + ys[i + 1]) / 2.0
            end_x = row_x[-1]
            pts.append(np.array([end_x, mid_y, alt]))
    return np.array(pts)


def main() -> None:
    world, buildings = default_world()

    lidar = Lidar2D(num_beams=72, max_range=10.0, noise_std=0.1, seed=42)

    # Lawnmower exploration
    path_3d = _lawnmower_path(WORLD_SIZE, CRUISE_ALT, n_rows=5)

    quad = Quadrotor()
    quad.reset(position=path_3d[0].copy())
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=180.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)

    # Occupancy grid
    n_cells = int(WORLD_SIZE / GRID_RES)
    log_odds = np.zeros((n_cells, n_cells))
    l_occ = 0.85
    l_free = -0.4

    scan_every = max(1, n_steps // 300)
    scan_indices = list(range(0, n_steps, scan_every))
    n_frames = len(scan_indices)

    grid_snapshots: list[np.ndarray] = []
    scan_ranges: list[np.ndarray] = []
    coverage_pct: list[float] = []

    for si in scan_indices:
        s = flight_states[si]
        ranges = lidar.sense(s, world)
        scan_ranges.append(ranges.copy())
        pos = s[:3]
        yaw = s[5] if len(s) > 5 else 0.0

        for beam_idx, angle in enumerate(lidar.angles):
            r = ranges[beam_idx]
            cos_a = np.cos(yaw + angle)
            sin_a = np.sin(yaw + angle)
            n_ray_steps = int(r / GRID_RES)
            for d_step in range(n_ray_steps + 1):
                d = d_step * GRID_RES
                px = pos[0] + d * cos_a
                py = pos[1] + d * sin_a
                ci = int(px / GRID_RES)
                cj = int(py / GRID_RES)
                if 0 <= ci < n_cells and 0 <= cj < n_cells:
                    if d_step == n_ray_steps and r < lidar.max_range - 0.5:
                        log_odds[ci, cj] += l_occ
                    else:
                        log_odds[ci, cj] += l_free
                    log_odds[ci, cj] = np.clip(log_odds[ci, cj], -5, 5)

        prob = 1.0 / (1.0 + np.exp(-log_odds))
        grid_snapshots.append(prob.copy())
        known = np.sum((prob > 0.6) | (prob < 0.4))
        coverage_pct.append(100.0 * known / (n_cells * n_cells))

    pos = flight_states[:, :3]

    # ── 3-Panel viz + inset occupancy grid ─────────────────────────────
    viz = ThreePanelViz(
        title="Grid Mapping — Lidar Exploration", world_size=WORLD_SIZE, figsize=(18, 9)
    )
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.6, alpha=0.3, label="Explore path")

    anim = SimAnimator("grid_mapping", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="orange")

    # Inset: occupancy grid
    ax_grid = viz.fig.add_axes([0.60, 0.02, 0.36, 0.35])
    ax_grid.set_xlim(0, WORLD_SIZE)
    ax_grid.set_ylim(0, WORLD_SIZE)
    ax_grid.set_aspect("equal")
    ax_grid.set_title("Occupancy Grid", fontsize=8)
    ax_grid.tick_params(labelsize=5)
    im_grid = ax_grid.imshow(
        grid_snapshots[0].T,
        origin="lower",
        extent=[0, WORLD_SIZE, 0, WORLD_SIZE],
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im_grid, ax=ax_grid, fraction=0.046, pad=0.04)

    fov_arts: list = []

    def update(f: int) -> None:
        k = scan_indices[min(f, n_frames - 1)]
        s = flight_states[k]
        yaw = s[5] if len(s) > 5 else 0.0

        viz.update_trail(trail_arts, pos, k)
        viz.update_vehicle(pos[k], s[3:6], size=1.5)

        clear_vehicle_artists(fov_arts)

        fov_arts.extend(
            draw_lidar2d_fov_3d(
                viz.ax3d, s[:3], yaw, lidar.fov, lidar.max_range, alpha=0.03, color="lime"
            )
        )
        fov_arts.extend(
            draw_lidar2d_fov_top(
                viz.ax_top, s[:2], yaw, lidar.fov, lidar.max_range, alpha=0.04, color="lime"
            )
        )

        ranges = scan_ranges[min(f, len(scan_ranges) - 1)]
        fov_arts.extend(
            draw_lidar2d_rays_top(
                viz.ax_top,
                s[:2],
                yaw,
                ranges,
                lidar.angles,
                lidar.max_range,
                ray_alpha=0.15,
                every_n=3,
                hit_size=5,
            )
        )

        im_grid.set_data(grid_snapshots[min(f, len(grid_snapshots) - 1)].T)

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
