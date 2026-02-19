# Erwin Lejeune - 2026-02-19
"""Occupancy grid mapping: quadrotor explores with 2D lidar.

Four-panel layout (3D scene, top-down with lidar, occupancy grid, coverage data):
  - 3D scene with buildings, quadrotor model, and lidar FOV.
  - Top-down with lidar FOV wedge, ray hits, and exploration path.
  - Occupancy grid heatmap built from log-odds ray casting.
  - Coverage percentage over time.

Reference: S. Thrun, "Learning Occupancy Grid Maps with Forward Sensor
Models," CMU Tech Report, 2003.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.environment import default_world
from uav_sim.path_tracking.flight_ops import fly_path, init_hover, takeoff
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.sensor_viz import (
    draw_lidar2d_fov_3d,
    draw_lidar2d_fov_top,
    draw_lidar2d_rays_top,
)
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
GRID_RES = 0.25


def _lawnmower_path(size: float, alt: float, n_rows: int = 6) -> np.ndarray:
    margin = 3.0
    xs = np.linspace(margin, size - margin, 30)
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

    lidar = Lidar2D(num_beams=120, max_range=12.0, noise_std=0.08, seed=42)

    path_3d = _lawnmower_path(WORLD_SIZE, CRUISE_ALT, n_rows=6)

    quad = Quadrotor()
    quad.reset(position=np.array([path_3d[0, 0], path_3d[0, 1], 0.0]))
    ctrl = CascadedPIDController()

    states_list: list[np.ndarray] = []
    takeoff(quad, ctrl, target_alt=CRUISE_ALT, dt=0.005, duration=3.0, states=states_list)
    init_hover(quad)

    pursuit = PurePursuit3D(lookahead=2.5, waypoint_threshold=2.0, adaptive=True)
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=200.0, states=states_list)
    flight_states = np.array(states_list) if states_list else np.zeros((1, 12))
    n_steps = len(flight_states)

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

    # ── 2x2 gridspec layout ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_grid = fig.add_subplot(gs[1, 0])
    ax_data = fig.add_subplot(gs[1, 1])

    fig.suptitle("Occupancy Grid Mapping — Lidar Exploration", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.set_title("Top Down — Lidar Scan", fontsize=9)
    ax_top.set_aspect("equal")

    for b in buildings:
        sz = b.max_corner - b.min_corner
        ax_top.add_patch(
            matplotlib.patches.Rectangle(b.min_corner[:2], sz[0], sz[1], fc="gray", alpha=0.3)
        )
        verts_3d = [
            [b.min_corner[0], b.min_corner[1], 0],
            [b.max_corner[0], b.min_corner[1], 0],
            [b.max_corner[0], b.max_corner[1], 0],
            [b.min_corner[0], b.max_corner[1], 0],
        ]
        xs = [v[0] for v in verts_3d] + [verts_3d[0][0]]
        ys = [v[1] for v in verts_3d] + [verts_3d[0][1]]
        zs = [b.max_corner[2]] * 5
        ax3d.plot(xs, ys, zs, "gray", lw=0.5, alpha=0.3)

    ax_top.plot(path_3d[:, 0], path_3d[:, 1], "c-", lw=0.4, alpha=0.3)

    ax_grid.set_xlim(0, WORLD_SIZE)
    ax_grid.set_ylim(0, WORLD_SIZE)
    ax_grid.set_aspect("equal")
    ax_grid.set_title("Occupancy Grid", fontsize=9)
    ax_grid.tick_params(labelsize=6)
    im_grid = ax_grid.imshow(
        grid_snapshots[0].T,
        origin="lower",
        extent=[0, WORLD_SIZE, 0, WORLD_SIZE],
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im_grid, ax=ax_grid, fraction=0.046, pad=0.04)

    ax_data.set_xlim(0, n_frames)
    ax_data.set_ylim(0, 100)
    ax_data.set_xlabel("Scan Step", fontsize=8)
    ax_data.set_ylabel("Coverage %", fontsize=8)
    ax_data.set_title("Map Coverage", fontsize=9)
    ax_data.grid(True, alpha=0.3)
    (cov_line,) = ax_data.plot([], [], "g-", lw=1.0)

    (trail_top,) = ax_top.plot([], [], "orange", lw=0.8, alpha=0.6)
    fov_arts: list = []
    veh_arts: list = []

    anim = SimAnimator("occupancy_mapping", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        k = scan_indices[min(f, n_frames - 1)]
        s = flight_states[k]
        yaw = s[5] if len(s) > 5 else 0.0

        trail_top.set_data(pos[:k, 0], pos[:k, 1])

        clear_vehicle_artists(veh_arts)
        R = Quadrotor.rotation_matrix(*s[3:6])
        veh_arts.extend(draw_quadrotor_3d(ax3d, s[:3], R, size=1.5))
        (dt_t,) = ax_top.plot(s[0], s[1], "ko", ms=4, zorder=10)
        veh_arts.append(dt_t)

        clear_vehicle_artists(fov_arts)
        fov_arts.extend(
            draw_lidar2d_fov_3d(
                ax3d, s[:3], yaw, lidar.fov, lidar.max_range, alpha=0.03, color="lime"
            )
        )
        fov_arts.extend(
            draw_lidar2d_fov_top(
                ax_top, s[:2], yaw, lidar.fov, lidar.max_range, alpha=0.04, color="lime"
            )
        )
        ranges = scan_ranges[min(f, len(scan_ranges) - 1)]
        fov_arts.extend(
            draw_lidar2d_rays_top(
                ax_top,
                s[:2],
                yaw,
                ranges,
                lidar.angles,
                lidar.max_range,
                ray_alpha=0.15,
                every_n=4,
                hit_size=4,
            )
        )

        im_grid.set_data(grid_snapshots[min(f, len(grid_snapshots) - 1)].T)
        cov_line.set_data(range(f + 1), coverage_pct[: f + 1])
        pct = coverage_pct[min(f, len(coverage_pct) - 1)]
        ax3d.set_title(f"Occupancy Mapping — {pct:.0f}% covered")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
