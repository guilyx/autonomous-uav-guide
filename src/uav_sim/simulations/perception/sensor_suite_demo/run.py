"""Sensor-suite demo on the standardized mission runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.logging import SimLogger
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.sensors.base import SensorMount
from uav_sim.sensors.camera import Camera, CameraIntrinsics
from uav_sim.sensors.lidar import Lidar2D, Lidar3D
from uav_sim.simulations.common import figure_8_path
from uav_sim.simulations.mission_runner import run_standard_mission
from uav_sim.simulations.standards import SimulationStandard
from uav_sim.vehicles.multirotor import Quadrotor
from uav_sim.visualization import SimAnimator, ThreePanelViz
from uav_sim.visualization.sensor_viz import (
    draw_camera_fov_side,
    draw_camera_fov_top,
    draw_camera_frustum_3d,
    draw_lidar2d_fov_top,
    draw_lidar3d_fov_3d,
    draw_lidar3d_points_3d,
)
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists

matplotlib.use("Agg")

WORLD_SIZE = 30.0
CRUISE_ALT = 12.0
MAX_SENSOR_RECORDS = 80


def main() -> None:
    world, _ = default_world(n_buildings=4)
    standard = replace(
        SimulationStandard.flight_coupled(),
        lookahead=2.2,
        waypoint_threshold=1.2,
        stall_window_s=8.0,
        stall_min_progress_m=0.2,
        safety_clearance_m=0.1,
    )

    lidar_2d = Lidar2D(
        num_beams=72,
        max_range=12.0,
        noise_std=0.1,
        seed=42,
        mount=SensorMount(position=np.array([0.0, 0.0, -0.1])),
    )
    lidar_3d = Lidar3D(
        num_beams_h=72,
        num_beams_v=6,
        max_range=12.0,
        h_fov=2 * np.pi,
        v_fov=np.radians(30.0),
        noise_std=0.05,
        seed=42,
        mount=SensorMount(position=np.array([0.0, 0.0, 0.05])),
    )
    cam = Camera(
        intrinsics=CameraIntrinsics(fx=300, fy=300, cx=320, cy=240, width=640, height=480),
        max_depth=12.0,
        mount=SensorMount(
            position=np.array([0.15, 0.0, -0.05]),
            orientation=np.array([0.0, -np.radians(15.0), 0.0]),
        ),
    )

    planned_path = figure_8_path(
        duration=standard.duration,
        dt=0.15,
        alt=CRUISE_ALT,
        alt_amp=0.0,
        rx=8.0,
        ry=6.0,
    )
    quad = Quadrotor()
    quad.reset(position=np.array([planned_path[0, 0], planned_path[0, 1], 0.0]))
    mission = run_standard_mission(
        quad,
        CascadedPIDController(),
        planned_path,
        standard=standard,
        obstacles=world.obstacles,
    )
    states_arr = mission.states

    n_steps = len(states_arr)
    scan_every = max(1, n_steps // 64)
    pc_records: list[tuple[np.ndarray, np.ndarray]] = []

    sample_indices = list(range(0, n_steps, scan_every))
    for j, i in enumerate(sample_indices):
        s = states_arr[i]
        ranges_3d = lidar_3d.sense(s, world)
        pc = lidar_3d.to_point_cloud(s, ranges_3d)
        pc_records.append((s.copy(), pc.copy()))
        if j == 0 or j == len(sample_indices) - 1 or (j + 1) % 16 == 0:
            print(f"  Sensor scan {j + 1}/{len(sample_indices)}", flush=True)
    if len(pc_records) > MAX_SENSOR_RECORDS:
        sample = np.linspace(0, len(pc_records) - 1, MAX_SENSOR_RECORDS, dtype=int)
        pc_records = [pc_records[i] for i in sample]

    pos = states_arr[:, :3]
    speeds = np.linalg.norm(states_arr[:, 6:9], axis=1)
    logger = SimLogger("sensor_suite_demo", out_dir=Path(__file__).parent)
    logger.log_metadata("algorithm", "Sensor Suite Demo")
    logger.log_metadata("flight_coupled", True)
    logger.log_metadata("dt", standard.dt)
    logger.log_metadata("n_steps", n_steps)
    logger.log_metadata("tracking_fallback", mission.tracking_fallback)
    logger.log_metadata("tracking_fallback_reason", mission.fallback_reason)
    logger.log_metadata("path_min_clearance_m", mission.path_min_clearance_m)
    for i in range(0, n_steps, max(1, n_steps // 500)):
        logger.log_step(t=i * standard.dt, position=pos[i].tolist(), speed=float(speeds[i]))
    logger.log_completion(**mission.completion.as_dict())
    logger.log_summary("mean_speed_m_s", float(speeds.mean()))
    logger.log_summary("n_lidar_scans", len(pc_records))
    logger.save()

    n_records = len(pc_records)
    raw_indices = list(range(0, n_steps, scan_every))
    if raw_indices and raw_indices[-1] != n_steps - 1:
        raw_indices.append(n_steps - 1)
    if len(raw_indices) > n_records:
        sample = np.linspace(0, len(raw_indices) - 1, n_records, dtype=int)
        rec_indices = [raw_indices[i] for i in sample]
    else:
        rec_indices = raw_indices[:n_records]

    viz = ThreePanelViz(title="Sensor Suite - 3D Lidar + Camera FOV", world_size=WORLD_SIZE)
    viz.draw_buildings(world.obstacles)
    viz.draw_path(mission.tracking_path, color="cyan", lw=0.8, alpha=0.3, label="Plan")

    anim = SimAnimator("sensor_suite_demo", out_dir=Path(__file__).parent, dpi=60)
    anim._fig = viz.fig
    trail_arts = viz.create_trail_artists(color="dodgerblue")
    fov_arts: list = []
    pc_arts: list = []
    cam_arts: list = []

    def update(f: int) -> None:
        ri = min(f, n_records - 1)
        k = rec_indices[ri]

        viz.update_trail(trail_arts, pos, k)
        euler = states_arr[k, 3:6]
        viz.update_vehicle(pos[k], euler, size=1.5)

        clear_vehicle_artists(fov_arts)
        clear_vehicle_artists(pc_arts)
        clear_vehicle_artists(cam_arts)

        s_k, pc_k = pc_records[ri]
        yaw = float(s_k[5]) if len(s_k) > 5 else 0.0
        pitch = float(s_k[4]) if len(s_k) > 4 else 0.0

        fov_arts.extend(
            draw_lidar3d_fov_3d(
                viz.ax3d,
                s_k[:3],
                yaw,
                pitch,
                lidar_3d.h_fov,
                lidar_3d.v_fov,
                lidar_3d.max_range,
                alpha=0.02,
                color="cyan",
            )
        )
        fov_arts.extend(
            draw_lidar2d_fov_top(
                viz.ax_top,
                s_k[:2],
                yaw,
                lidar_2d.fov,
                lidar_2d.max_range,
                alpha=0.04,
            )
        )

        if len(pc_k) > 0:
            pc_arts.extend(
                draw_lidar3d_points_3d(viz.ax3d, pc_k, origin=s_k[:3], size=3, alpha=0.5)
            )
            sc_top = viz.ax_top.scatter(
                pc_k[:, 0], pc_k[:, 1], c="cyan", s=1.5, alpha=0.4, zorder=6
            )
            pc_arts.append(sc_top)
            sc_side = viz.ax_side.scatter(
                pc_k[:, 0], pc_k[:, 2], c="cyan", s=1.5, alpha=0.4, zorder=6
            )
            pc_arts.append(sc_side)

        R = Quadrotor.rotation_matrix(*euler)
        cam_arts.extend(
            draw_camera_frustum_3d(
                viz.ax3d,
                s_k[:3],
                R,
                cam.h_fov,
                cam.v_fov,
                cam.max_depth,
                color="gold",
                alpha=0.06,
                mount_R=cam.mount.rotation_matrix(),
            )
        )
        cam_arts.extend(
            draw_camera_fov_top(
                viz.ax_top,
                s_k[:2],
                yaw,
                cam.h_fov,
                cam.max_depth,
                mount_yaw=cam.mount.orientation[2],
            )
        )
        cam_arts.extend(
            draw_camera_fov_side(
                viz.ax_side,
                np.array([s_k[0], s_k[2]]),
                pitch,
                cam.v_fov,
                cam.max_depth,
                mount_pitch=cam.mount.orientation[1],
            )
        )

    anim.animate(update, n_records)
    anim.save()


if __name__ == "__main__":
    main()
