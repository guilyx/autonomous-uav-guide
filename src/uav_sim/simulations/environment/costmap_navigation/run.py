"""Sensor-driven costmap variants during figure-8 flight.

The UAV flies a standard figure-8 mission while a 2D lidar updates an
occupancy map online. Three map variants are visualised:
  - base occupancy map from sensor hits,
  - footprint-inflated map,
  - velocity-augmented map (higher caution at higher speed).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.costmap import FootprintInflationLayer, OccupancyGrid, VelocityCostLayer
from uav_sim.environment import default_world
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.perception.occupancy_mapping import OccupancyMapper
from uav_sim.sensors.lidar import Lidar2D
from uav_sim.simulations.common import COSTMAP_CMAP, CRUISE_ALT
from uav_sim.simulations.mission_runner import run_standard_mission
from uav_sim.simulations.setup_helpers import (
    build_logger,
    default_figure8_path,
    select_standard,
)
from uav_sim.vehicles.footprint import CircularFootprint
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import _draw_box_3d
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")

WORLD_SIZE = 30.0
GRID_RES = 0.4
MAX_VEL_FOR_COST = 6.0


def main() -> None:
    world, _ = default_world()
    standard = select_standard("flight_coupled")
    path_3d = default_figure8_path(standard, alt=CRUISE_ALT, rx=8.0, ry=6.0)

    quad = Quadrotor()
    quad.reset(position=np.array([path_3d[0, 0], path_3d[0, 1], 0.0]))
    mission = run_standard_mission(
        quad,
        CascadedPIDController(),
        path_3d,
        standard=standard,
        obstacles=world.obstacles,
    )
    states = mission.states
    n_steps = len(states)
    if n_steps == 0:
        return

    lidar = Lidar2D(num_beams=120, max_range=12.0, noise_std=0.06, seed=42)
    occ_grid = OccupancyGrid(
        resolution=GRID_RES,
        bounds_min=np.zeros(3),
        bounds_max=np.array([WORLD_SIZE, WORLD_SIZE, 0.0]),
    )
    mapper = OccupancyMapper(occ_grid, l_occ=0.80, l_free=0.35)
    infl_layer = FootprintInflationLayer(
        CircularFootprint(radius=0.45),
        padding=0.20,
        cost_scaling=2.0,
    )
    vel_layer = VelocityCostLayer(max_speed=MAX_VEL_FOR_COST, max_penalty=0.35)

    scan_step = max(1, int(0.1 / standard.dt))
    scan_indices = list(range(0, n_steps, scan_step))
    if scan_indices[-1] != n_steps - 1:
        scan_indices.append(n_steps - 1)

    base_maps: list[np.ndarray] = []
    inflated_maps: list[np.ndarray] = []
    velocity_maps: list[np.ndarray] = []
    speed_hist: list[float] = []

    for idx in scan_indices:
        s = states[idx]
        ranges = lidar.sense(s, world)
        mapper.update(s[:3], ranges, lidar.angles, max_range=lidar.max_range)
        base = occ_grid.grid.copy()
        inflated = infl_layer.apply(occ_grid)
        speed = float(np.linalg.norm(s[6:9])) if len(s) >= 9 else 0.0
        speed_hist.append(speed)
        velocity = vel_layer.apply(inflated, speed)
        base_maps.append(base)
        inflated_maps.append(inflated)
        velocity_maps.append(velocity)

    pos = states[:, :3]
    times = np.arange(n_steps) * standard.dt
    vel = states[:, 6:9]
    speed = np.linalg.norm(vel, axis=1)

    logger = build_logger(
        "costmap_navigation",
        Path(__file__).parent,
        algorithm="Sensor Costmap Variants",
        standard=standard,
        flight_coupled=True,
    )
    logger.log_metadata("grid_res", GRID_RES)
    logger.log_metadata("scan_count", len(scan_indices))
    logger.log_metadata("tracking_fallback", mission.tracking_fallback)
    logger.log_metadata("tracking_fallback_reason", mission.fallback_reason)
    logger.log_metadata("path_min_clearance_m", mission.path_min_clearance_m)
    for i in range(n_steps):
        logger.log_step(
            t=times[i],
            position=pos[i].tolist(),
            velocity=vel[i].tolist(),
            speed=float(speed[i]),
        )
    logger.log_completion(**mission.completion.as_dict())
    logger.log_summary("mean_speed_mps", float(speed.mean()))
    logger.log_summary(
        "final_goal_xy_error_m",
        float(np.linalg.norm(pos[-1, :2] - path_3d[-1, :2])),
    )
    logger.save()

    anim = SimAnimator("costmap_navigation", out_dir=Path(__file__).parent, dpi=72)
    fig = plt.figure(figsize=(18, 8))
    anim._fig = fig
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_base = fig.add_subplot(gs[0, 1])
    ax_infl = fig.add_subplot(gs[0, 2])
    ax_vel = fig.add_subplot(gs[0, 3])
    fig.suptitle("Costmap Variants from Lidar During Figure-8 Flight", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    for b in world.obstacles:
        lo, hi = b.bounding_box()
        _draw_box_3d(ax3d, lo, hi)
    ax3d.plot(
        path_3d[:, 0],
        path_3d[:, 1],
        path_3d[:, 2],
        "c--",
        lw=0.8,
        alpha=0.4,
        label="Reference",
    )
    (trail3d,) = ax3d.plot([], [], [], "lime", lw=1.4, alpha=0.8, label="Flight")
    ax3d.legend(fontsize=7, loc="upper left")

    extent = [0, WORLD_SIZE, 0, WORLD_SIZE]
    for ax, title in [
        (ax_base, "Base Costmap"),
        (ax_infl, "Inflation Costmap"),
        (ax_vel, "Velocity Costmap"),
    ]:
        ax.set_aspect("equal")
        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X [m]", fontsize=7)
        ax.set_ylabel("Y [m]", fontsize=7)
        ax.tick_params(labelsize=6)

    im_base = ax_base.imshow(
        base_maps[0].T,
        origin="lower",
        extent=extent,
        cmap=COSTMAP_CMAP,
        vmin=0,
        vmax=1,
    )
    im_infl = ax_infl.imshow(
        inflated_maps[0].T,
        origin="lower",
        extent=extent,
        cmap=COSTMAP_CMAP,
        vmin=0,
        vmax=1,
    )
    im_vel = ax_vel.imshow(
        velocity_maps[0].T,
        origin="lower",
        extent=extent,
        cmap=COSTMAP_CMAP,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im_vel, ax=ax_vel, fraction=0.046, pad=0.04)

    (dot_base,) = ax_base.plot([], [], "wo", ms=4)
    (dot_infl,) = ax_infl.plot([], [], "wo", ms=4)
    (dot_vel,) = ax_vel.plot([], [], "wo", ms=4)

    veh_arts: list = []
    title = ax3d.set_title("t = 0.0 s")
    scan_to_state = scan_indices
    n_frames = len(scan_indices)

    def update(f: int) -> None:
        k = scan_to_state[f]
        trail3d.set_data(pos[: k + 1, 0], pos[: k + 1, 1])
        trail3d.set_3d_properties(pos[: k + 1, 2])

        clear_vehicle_artists(veh_arts)
        R = Quadrotor.rotation_matrix(*states[k, 3:6])
        veh_arts.extend(draw_quadrotor_3d(ax3d, pos[k], R, size=1.4))

        im_base.set_data(base_maps[f].T)
        im_infl.set_data(inflated_maps[f].T)
        im_vel.set_data(velocity_maps[f].T)
        dot_base.set_data([pos[k, 0]], [pos[k, 1]])
        dot_infl.set_data([pos[k, 0]], [pos[k, 1]])
        dot_vel.set_data([pos[k, 0]], [pos[k, 1]])

        title.set_text(
            f"t = {k * standard.dt:.1f} s | speed = {speed_hist[f]:.2f} m/s | "
            f"clearance = {mission.path_min_clearance_m:.2f} m"
        )

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
