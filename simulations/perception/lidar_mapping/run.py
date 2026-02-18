# Erwin Lejeune - 2026-02-17
"""Lidar-based occupancy mapping with a quadrotor flying through obstacles.

Reference: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics," MIT Press, 2005, Ch. 6.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from uav_sim.costmap import OccupancyGrid
from uav_sim.environment import SphereObstacle, World
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.perception import OccupancyMapper
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator


def main() -> None:
    world = World(bounds_min=np.zeros(3), bounds_max=np.full(3, 20.0))
    for cx, cy in [(5, 10), (10, 5), (15, 12), (8, 16)]:
        world.add_obstacle(SphereObstacle(centre=np.array([float(cx), float(cy), 0.0]), radius=1.5))

    quad = Quadrotor()
    quad.reset(position=np.array([1.0, 1.0, 2.0]))
    ctrl = CascadedPIDController()
    lidar = Lidar2D(num_beams=36, max_range=10.0, noise_std=0.05, seed=42)

    grid = OccupancyGrid(resolution=1.0, bounds_min=np.zeros(3), bounds_max=np.full(3, 20.0))
    mapper = OccupancyMapper(grid)

    waypoints = [np.array([10.0, 10.0, 2.0]), np.array([18.0, 18.0, 2.0])]
    dt, duration = 0.01, 6.0
    steps = int(duration / dt)
    wp_idx = 0
    positions = np.zeros((steps, 3))
    grids_record = []
    scan_every = 20
    record_every = max(1, steps // 100)

    for i in range(steps):
        positions[i] = quad.state[:3]
        target = waypoints[wp_idx]
        if np.linalg.norm(quad.state[:3] - target) < 1.0 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
        u = ctrl.compute(quad.state, target, dt=dt)
        quad.step(u, dt)
        if i % scan_every == 0:
            ranges = lidar.sense(quad.state, world)
            mapper.update(quad.state[:3], ranges, lidar.angles, lidar.max_range)
        if i % record_every == 0:
            grids_record.append(grid.grid.copy())

    anim = SimAnimator("lidar_mapping", out_dir=Path(__file__).parent)
    fig, axes = anim.figure_2d("Lidar Occupancy Mapping", nrows=1, figsize=(7, 6))
    ax = axes[0]
    im = ax.imshow(
        grids_record[0].T,
        origin="lower",
        extent=[0, 20, 0, 20],
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    (trail,) = ax.plot([], [], "b-", lw=1.0, alpha=0.7)
    (dot,) = ax.plot([], [], "ro", ms=5)
    for obs in world.obstacles:
        circle = plt.Circle(obs.centre[:2], obs.radius, color="red", alpha=0.3)
        ax.add_patch(circle)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    skip = max(1, len(positions) // len(grids_record))
    idx_map = list(range(0, len(positions), skip))[: len(grids_record)]

    def update(f):
        gi = min(f, len(grids_record) - 1)
        im.set_data(grids_record[gi].T)
        k = idx_map[min(f, len(idx_map) - 1)]
        trail.set_data(positions[:k, 0], positions[:k, 1])
        dot.set_data([positions[k, 0]], [positions[k, 1]])

    anim.animate(update, len(grids_record))
    anim.save()


if __name__ == "__main__":
    main()
