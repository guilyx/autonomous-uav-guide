# Erwin Lejeune - 2026-02-15
"""Sensor-suite demo: 3D lidar point cloud + forward camera frustum.

Demonstrates all sensor payload types on a single drone flying through the
default environment using StateManager:
  - 3D lidar with coloured point cloud and FOV cone
  - Forward camera with frustum visualisation
  - 2D lidar with planar FOV wedge and hit scatter
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.control import StateManager
from uav_sim.environment import default_world
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.sensors.base import SensorMount
from uav_sim.sensors.camera import Camera, CameraIntrinsics
from uav_sim.sensors.lidar import Lidar2D, Lidar3D
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


def main() -> None:
    world, buildings = default_world()

    lidar_2d = Lidar2D(
        num_beams=72,
        max_range=12.0,
        noise_std=0.1,
        seed=42,
        mount=SensorMount(position=np.array([0.0, 0.0, -0.1])),
    )
    lidar_3d = Lidar3D(
        num_beams_h=90,
        num_beams_v=8,
        max_range=15.0,
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

    path_3d = np.array(
        [
            [3.0, 3.0, CRUISE_ALT],
            [15.0, 5.0, CRUISE_ALT],
            [27.0, 10.0, CRUISE_ALT],
            [27.0, 20.0, CRUISE_ALT],
            [15.0, 27.0, CRUISE_ALT],
            [5.0, 20.0, CRUISE_ALT],
            [5.0, 10.0, CRUISE_ALT],
        ]
    )

    quad = Quadrotor()
    quad.reset(position=path_3d[0].copy())
    sm = StateManager(quad)
    dt = 0.005

    sm.arm()
    sm.run_takeoff(altitude=CRUISE_ALT, dt=dt, timeout=10.0)

    pursuit = PurePursuit3D(lookahead=3.0, waypoint_threshold=1.5, adaptive=True)
    sm.offboard()
    for _ in range(int(80.0 / dt)):
        target = pursuit.compute_target(quad.position, path_3d, velocity=quad.velocity)
        sm.set_position_target(target)
        sm.step(dt)
        if pursuit.is_path_complete(quad.position, path_3d):
            break

    states_arr = np.array(sm.states)

    n_steps = len(states_arr)
    scan_every = max(1, n_steps // 80)
    pc_records: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(0, n_steps, scan_every):
        s = states_arr[i]
        ranges_3d = lidar_3d.sense(s, world)
        pc = lidar_3d.to_point_cloud(s, ranges_3d)
        pc_records.append((s.copy(), pc.copy()))

    pos = states_arr[:, :3]
    n_records = len(pc_records)
    rec_indices = list(range(0, n_steps, scan_every))[:n_records]

    viz = ThreePanelViz(title="Sensor Suite — 3D Lidar + Camera FOV", world_size=WORLD_SIZE)
    viz.draw_buildings(world.obstacles)
    viz.draw_path(path_3d, color="cyan", lw=0.8, alpha=0.3, label="Plan")

    anim = SimAnimator("sensor_suite_demo", out_dir=Path(__file__).parent)
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
