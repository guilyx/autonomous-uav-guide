# Erwin Lejeune - 2026-02-18
"""Lidar-based occupancy mapping with FOV and point cloud visualisation.

Three panels:
  - **3D** scene with buildings, quadrotor, lidar FOV cone, and coloured
    point cloud hits.
  - **2D top-down** (XY) with lidar wedge, hit scatter, drone, and evolving
    occupancy grid underlay.
  - **2D side** (XZ) with drone altitude and lidar plane indicator.

Reference: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT Press,
2005, Ch. 6.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.costmap import OccupancyGrid
from uav_sim.environment import World, add_urban_buildings
from uav_sim.path_tracking.flight_ops import fly_path, init_hover
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.perception import OccupancyMapper
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.sensor_viz import (
    draw_lidar2d_fov_3d,
    draw_lidar2d_fov_top,
    draw_lidar2d_rays_3d,
    draw_lidar2d_rays_top,
)
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0


def main() -> None:
    world = World(
        bounds_min=np.zeros(3),
        bounds_max=np.array([WORLD_SIZE, WORLD_SIZE, WORLD_SIZE]),
    )
    add_urban_buildings(world, world_size=WORLD_SIZE, n_buildings=6, seed=42)

    quad = Quadrotor()
    quad.reset(position=np.array([2.0, 2.0, CRUISE_ALT]))
    init_hover(quad)
    ctrl = CascadedPIDController()
    lidar = Lidar2D(num_beams=72, max_range=12.0, noise_std=0.1, seed=42)

    grid = OccupancyGrid(
        resolution=0.5,
        bounds_min=np.zeros(3),
        bounds_max=np.array([WORLD_SIZE, WORLD_SIZE, 0.0]),
    )
    mapper = OccupancyMapper(grid)

    path_3d = np.array(
        [
            [2.0, 2.0, CRUISE_ALT],
            [15.0, 3.0, CRUISE_ALT],
            [28.0, 8.0, CRUISE_ALT],
            [28.0, 22.0, CRUISE_ALT],
            [15.0, 28.0, CRUISE_ALT],
            [3.0, 22.0, CRUISE_ALT],
            [3.0, 12.0, CRUISE_ALT],
        ]
    )
    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=80.0, states=states_list)
    states_arr = np.array(states_list) if states_list else np.zeros((1, 12))

    steps = len(states_arr)
    grids_record: list[np.ndarray] = []
    ranges_record: list[tuple[np.ndarray, np.ndarray, float]] = []
    scan_every = max(1, steps // 100)
    record_every = max(1, steps // 120)

    for i in range(steps):
        if i % scan_every == 0:
            ranges = lidar.sense(states_arr[i], world)
            mapper.update(states_arr[i, :3], ranges, lidar.angles, lidar.max_range)
            ranges_record.append(
                (states_arr[i, :3].copy(), ranges.copy(), float(states_arr[i, 5]))
            )
        if i % record_every == 0:
            grids_record.append(grid.grid.copy())

    pos = states_arr[:, :3]

    # ── 3-Panel layout ─────────────────────────────────────────────────────
    viz = ThreePanelViz(title="Lidar Occupancy Mapping", world_size=WORLD_SIZE)
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.8, alpha=0.3, label="Plan")

    anim = SimAnimator("lidar_mapping", out_dir=Path(__file__).parent)
    anim._fig = viz.fig

    trail_arts = viz.create_trail_artists(color="dodgerblue")

    im_data = grids_record[0]
    im_data_2d = im_data[:, :, 0].T if im_data.ndim == 3 else im_data.T
    im = viz.ax_top.imshow(
        im_data_2d,
        origin="lower",
        extent=[0, WORLD_SIZE, 0, WORLD_SIZE],
        cmap="gray_r",
        vmin=0,
        vmax=1,
        alpha=0.55,
        zorder=0,
    )

    skip = max(1, len(pos) // len(grids_record))
    idx_map = list(range(0, len(pos), skip))[: len(grids_record)]

    fov_arts_3d: list = []
    fov_arts_top: list = []
    ray_arts_3d: list = []
    ray_arts_top: list = []

    def update(f: int) -> None:
        gi = min(f, len(grids_record) - 1)
        snap = grids_record[gi]
        im.set_data(snap[:, :, 0].T if snap.ndim == 3 else snap.T)
        k = idx_map[min(f, len(idx_map) - 1)]

        viz.update_trail(trail_arts, pos, k)
        euler = states_arr[k, 3:6]
        viz.update_vehicle(pos[k], euler, size=1.5)

        clear_vehicle_artists(fov_arts_3d)
        clear_vehicle_artists(fov_arts_top)
        clear_vehicle_artists(ray_arts_3d)
        clear_vehicle_artists(ray_arts_top)

        scan_idx = min(k // scan_every, len(ranges_record) - 1)
        origin, rngs, yaw_s = ranges_record[scan_idx]

        fov_arts_3d.extend(
            draw_lidar2d_fov_3d(viz.ax3d, origin, yaw_s, lidar.fov, lidar.max_range, alpha=0.03)
        )
        fov_arts_top.extend(
            draw_lidar2d_fov_top(
                viz.ax_top, origin[:2], yaw_s, lidar.fov, lidar.max_range, alpha=0.05
            )
        )
        ray_arts_3d.extend(
            draw_lidar2d_rays_3d(
                viz.ax3d, origin, yaw_s, rngs, lidar.angles, lidar.max_range, every_n=3
            )
        )
        ray_arts_top.extend(
            draw_lidar2d_rays_top(
                viz.ax_top, origin[:2], yaw_s, rngs, lidar.angles, lidar.max_range, every_n=2
            )
        )

    anim.animate(update, len(grids_record))
    anim.save()


if __name__ == "__main__":
    main()
