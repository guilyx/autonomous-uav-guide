# Erwin Lejeune - 2026-02-18
"""Visual Servoing: drone autonomously follows a moving ground target.

A ``SimulatedDetector`` projects a known 3D target into the camera as a
bounding box, and a ``VisualServoController`` converts the image-space
error into world-frame velocity commands that keep the target centred
and at a desired distance.

Four-panel view: 3D scene, top-down, side, and a camera-image panel
showing the bounding box and crosshair.

Reference: F. Chaumette & S. Hutchinson, "Visual Servo Control — Part I:
Basic Approaches," IEEE Robotics & Automation Magazine, 2006.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from uav_sim.environment import default_world
from uav_sim.perception.bbox_tracker import (
    SimulatedDetector,
    VisualServoConfig,
    VisualServoController,
)
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import PointTracker
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
DT = 0.02
SIM_TIME = 60.0


def _moving_target(t: float) -> np.ndarray:
    """Elliptical ground-level path for the target."""
    cx, cy = 15.0, 15.0
    rx, ry = 8.0, 6.0
    omega = 0.15
    return np.array([cx + rx * np.cos(omega * t), cy + ry * np.sin(omega * t), 0.5])


def main() -> None:
    world, buildings = default_world()

    quad = Quadrotor()
    start_pos = np.array([15.0, 15.0, 12.0])
    quad.reset(position=start_pos)

    gimbal = Gimbal(max_rate=1.5)
    gimbal.reset(pan=0.0, tilt=-np.pi / 4)
    tracker = PointTracker(gimbal)

    detector = SimulatedDetector(target_radius=0.5)
    servo = VisualServoController(
        VisualServoConfig(
            kp_lateral=1.2,
            kp_forward=0.8,
            desired_size_ratio=0.20,
            max_velocity=2.0,
        )
    )

    h_fov, v_fov = 1.2, 0.9

    n_steps = int(SIM_TIME / DT)
    drone_pos = np.zeros((n_steps, 3))
    target_pos = np.zeros((n_steps, 3))
    bbox_cx = np.zeros(n_steps)
    bbox_cy = np.zeros(n_steps)
    bbox_size = np.zeros(n_steps)
    visible_hist = np.zeros(n_steps, dtype=bool)
    vel_hist = np.zeros((n_steps, 3))

    pos = start_pos.copy()
    vel = np.zeros(3)

    for i in range(n_steps):
        t = i * DT
        tgt = _moving_target(t)
        target_pos[i] = tgt
        yaw = 0.0

        tracker.step(pos, tgt, yaw, DT)

        det = detector.detect(tgt, pos, gimbal, h_fov, v_fov, yaw)
        vel_cmd = servo.compute(det, yaw)
        vel_hist[i] = vel_cmd

        bbox_cx[i] = det.center_ndc[0] if det.visible else 0.0
        bbox_cy[i] = det.center_ndc[1] if det.visible else 0.0
        bbox_size[i] = det.size_ratio
        visible_hist[i] = det.visible

        alpha = 0.3
        vel = (1 - alpha) * vel + alpha * vel_cmd
        pos = pos + vel * DT
        pos[2] = np.clip(pos[2], 3.0, 25.0)
        drone_pos[i] = pos

    # ── Animation ─────────────────────────────────────────────────────
    skip = max(1, n_steps // 200)
    frames = list(range(0, n_steps, skip))
    n_frames = len(frames)

    viz = ThreePanelViz(
        title="Visual Servoing — Bounding Box Follow",
        world_size=WORLD_SIZE,
        figsize=(18, 9),
    )
    viz.draw_buildings(world.obstacles)

    # Target ground track
    tgt_trail = np.array([_moving_target(i * DT) for i in range(n_steps)])
    viz.ax_top.plot(
        tgt_trail[:, 0], tgt_trail[:, 1], "r--", lw=0.5, alpha=0.3, label="Target track"
    )
    (tgt_top,) = viz.ax_top.plot([], [], "r*", ms=10, zorder=10)
    (tgt_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10)
    trail_arts = viz.create_trail_artists(color="dodgerblue")

    # Camera-image inset
    ax_cam = viz.fig.add_axes([0.58, 0.02, 0.38, 0.30])
    ax_cam.set_xlim(-1.2, 1.2)
    ax_cam.set_ylim(-1.0, 1.0)
    ax_cam.set_facecolor("black")
    ax_cam.set_title("Camera View", fontsize=8, color="white")
    ax_cam.tick_params(labelsize=5, colors="white")
    ax_cam.axhline(0, color="gray", lw=0.3, alpha=0.5)
    ax_cam.axvline(0, color="gray", lw=0.3, alpha=0.5)
    (bbox_rect_art,) = ax_cam.plot([], [], "lime", lw=2)
    (crosshair,) = ax_cam.plot([0], [0], "w+", ms=12, mew=1)
    (target_dot,) = ax_cam.plot([], [], "r.", ms=10)

    anim = SimAnimator("visual_servoing", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = frames[f]
        dp = drone_pos[k]
        tp = target_pos[k]

        viz.update_trail(trail_arts, drone_pos, k)
        viz.update_vehicle(dp, np.zeros(3), size=1.5)

        tgt_3d.set_data([tp[0]], [tp[1]])
        tgt_3d.set_3d_properties([tp[2]])
        tgt_top.set_data([tp[0]], [tp[1]])

        if visible_hist[k]:
            cx_k = bbox_cx[k]
            cy_k = bbox_cy[k]
            sr = bbox_size[k] * 2
            half = max(sr, 0.05)
            xs = [cx_k - half, cx_k + half, cx_k + half, cx_k - half, cx_k - half]
            ys = [cy_k - half, cy_k - half, cy_k + half, cy_k + half, cy_k - half]
            bbox_rect_art.set_data(xs, ys)
            target_dot.set_data([cx_k], [cy_k])
        else:
            bbox_rect_art.set_data([], [])
            target_dot.set_data([], [])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
