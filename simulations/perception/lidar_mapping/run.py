# Erwin Lejeune - 2026-02-15
"""Lidar-based occupancy mapping with a quadrotor flying through obstacles.

Left panel: 3D scene with obstacle spheres, quadrotor model, and lidar beams.
Right panel: 2D top-down evolving occupancy grid with ground-truth obstacles.

Reference: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT Press,
2005, Ch. 6.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.costmap import OccupancyGrid
from uav_sim.environment import SphereObstacle, World
from uav_sim.path_tracking.flight_ops import fly_path
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.perception import OccupancyMapper
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_2d,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


def main() -> None:
    world_size = 10.0
    cruise_alt = 1.5
    world = World(bounds_min=np.zeros(3), bounds_max=np.array([world_size, world_size, 5.0]))
    obstacles_info = [(3, 5), (5, 3), (7, 6), (4, 8)]
    obs_radius = 0.8
    for cx, cy in obstacles_info:
        world.add_obstacle(
            SphereObstacle(centre=np.array([float(cx), float(cy), 0.0]), radius=obs_radius)
        )

    quad = Quadrotor()
    quad.reset(position=np.array([0.5, 0.5, cruise_alt]))
    ctrl = CascadedPIDController()
    lidar = Lidar2D(num_beams=36, max_range=5.0, noise_std=0.05, seed=42)

    grid = OccupancyGrid(
        resolution=0.5,
        bounds_min=np.zeros(3),
        bounds_max=np.array([world_size, world_size, 0.0]),
    )
    mapper = OccupancyMapper(grid)

    # Fly a smooth multi-waypoint path through the scene
    path_3d = np.array(
        [
            [0.5, 0.5, cruise_alt],
            [5.0, 1.0, cruise_alt],
            [9.0, 3.0, cruise_alt],
            [9.0, 7.0, cruise_alt],
            [5.0, 9.0, cruise_alt],
            [1.0, 7.0, cruise_alt],
            [1.0, 4.0, cruise_alt],
        ]
    )
    pursuit = PurePursuit3D(lookahead=0.8, waypoint_threshold=0.3, adaptive=True)
    states_list: list[np.ndarray] = []
    fly_path(quad, ctrl, path_3d, dt=0.005, pursuit=pursuit, timeout=50.0, states=states_list)
    states_arr = np.array(states_list) if states_list else np.zeros((1, 12))

    steps = len(states_arr)
    grids_record: list[np.ndarray] = []
    ranges_record: list[tuple[np.ndarray, np.ndarray]] = []
    scan_every = max(1, steps // 80)
    record_every = max(1, steps // 100)

    for i in range(steps):
        if i % scan_every == 0:
            ranges = lidar.sense(states_arr[i], world)
            mapper.update(states_arr[i, :3], ranges, lidar.angles, lidar.max_range)
            ranges_record.append((states_arr[i, :3].copy(), ranges.copy()))
        if i % record_every == 0:
            grids_record.append(grid.grid.copy())

    pos = states_arr[:, :3]

    # ── Animation setup ────────────────────────────────────────────────────
    anim = SimAnimator("lidar_mapping", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 6.5))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])
    fig.suptitle("Lidar Occupancy Mapping", fontsize=13)

    # ── 3D panel: obstacles + quadrotor + lidar beams ──────────────────────
    for obs in world.obstacles:
        anim.draw_sphere(ax3d, obs.centre, obs.radius, color="red", alpha=0.2)
    ax3d.set_xlim(0, world_size)
    ax3d.set_ylim(0, world_size)
    ax3d.set_zlim(0, 4)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.2, alpha=0.5)
    (dot3d,) = ax3d.plot([], [], [], "go", ms=5)

    # ── 2D panel: evolving occupancy grid ──────────────────────────────────
    grid_snap = grids_record[0]
    im_data = grid_snap[:, :, 0].T if grid_snap.ndim == 3 else grid_snap.T
    im = ax2d.imshow(
        im_data,
        origin="lower",
        extent=[0, world_size, 0, world_size],
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    for obs in world.obstacles:
        circle = plt.Circle(
            obs.centre[:2], obs.radius, color="red", alpha=0.25, lw=1.5, fill=False
        )
        ax2d.add_patch(circle)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.set_aspect("equal")
    ax2d.set_title("Occupancy Grid (evolving)", fontsize=10)
    (trail2d,) = ax2d.plot([], [], "b-", lw=1.0, alpha=0.7)
    (dot2d,) = ax2d.plot([], [], "ro", ms=5)

    skip = max(1, len(pos) // len(grids_record))
    idx_map = list(range(0, len(pos), skip))[: len(grids_record)]

    veh3d: list = []
    veh2d: list = []
    beam_arts: list = []

    def update(f):
        gi = min(f, len(grids_record) - 1)
        snap = grids_record[gi]
        im.set_data(snap[:, :, 0].T if snap.ndim == 3 else snap.T)
        k = idx_map[min(f, len(idx_map) - 1)]

        # 3D
        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])
        dot3d.set_data([pos[k, 0]], [pos[k, 1]])
        dot3d.set_3d_properties([pos[k, 2]])
        clear_vehicle_artists(veh3d)
        R = Quadrotor.rotation_matrix(*states_arr[k, 3:6])
        veh3d.extend(draw_quadrotor_3d(ax3d, pos[k], R, size=0.5))

        # Lidar beams in 3D (from most recent scan up to this time)
        clear_vehicle_artists(beam_arts)
        scan_idx = min(k // scan_every, len(ranges_record) - 1)
        origin, rngs = ranges_record[scan_idx]
        for bi, (angle, r) in enumerate(zip(lidar.angles, rngs, strict=False)):
            if r >= lidar.max_range:
                continue
            if bi % 3 != 0:
                continue
            ex = origin[0] + r * np.cos(angle)
            ey = origin[1] + r * np.sin(angle)
            (beam,) = ax3d.plot(
                [origin[0], ex],
                [origin[1], ey],
                [origin[2], origin[2]],
                "lime",
                lw=0.6,
                alpha=0.5,
            )
            beam_arts.append(beam)

        # 2D
        trail2d.set_data(pos[:k, 0], pos[:k, 1])
        dot2d.set_data([pos[k, 0]], [pos[k, 1]])
        clear_vehicle_artists(veh2d)
        veh2d.extend(draw_quadrotor_2d(ax2d, pos[k, :2], states_arr[k, 5], size=0.5))

    anim.animate(update, len(grids_record))
    anim.save()


if __name__ == "__main__":
    main()
