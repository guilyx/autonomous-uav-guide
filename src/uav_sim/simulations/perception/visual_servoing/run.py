# Erwin Lejeune - 2026-02-19
"""Visual Servoing: drone autonomously follows a moving ground target.

A ``SimulatedDetector`` projects a known 3D target into the camera as a
bounding box, and a ``VisualServoController`` converts the image-space
error into world-frame velocity commands that keep the target centred
and at a desired distance.

Four-panel view: 3D scene, top-down, camera-image panel, and error data.

Reference: F. Chaumette & S. Hutchinson, "Visual Servo Control — Part I:
Basic Approaches," IEEE Robotics & Automation Magazine, 2006.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 30.0
DT = 0.02
SIM_TIME = 60.0
H_FOV = 0.8
V_FOV = 0.6


def _moving_target(t: float) -> np.ndarray:
    """Slow elliptical ground-level path for the target."""
    cx, cy = 15.0, 15.0
    rx, ry = 7.0, 5.0
    omega = 0.06
    return np.array([cx + rx * np.cos(omega * t), cy + ry * np.sin(omega * t), 0.5])


def main() -> None:
    world, buildings = default_world()

    gimbal = Gimbal(max_rate=3.0)
    init_tgt = _moving_target(0.0)
    init_pos = np.array([15.0, 15.0, 10.0])
    init_pan, init_tilt = gimbal.look_at(init_pos, init_tgt, 0.0)
    gimbal.reset(pan=init_pan, tilt=init_tilt)
    tracker = PointTracker(gimbal)

    detector = SimulatedDetector(target_radius=0.5)
    servo = VisualServoController(
        VisualServoConfig(
            kp_lateral=3.0,
            kp_forward=2.0,
            desired_size_ratio=0.08,
            max_velocity=2.5,
        )
    )

    n_steps = int(SIM_TIME / DT)
    drone_pos = np.zeros((n_steps, 3))
    target_pos = np.zeros((n_steps, 3))
    bbox_cx = np.zeros(n_steps)
    bbox_cy = np.zeros(n_steps)
    bbox_size = np.zeros(n_steps)
    visible_hist = np.zeros(n_steps, dtype=bool)
    dist_hist = np.zeros(n_steps)

    pos = init_pos.copy()
    vel = np.zeros(3)

    for i in range(n_steps):
        t = i * DT
        tgt = _moving_target(t)
        target_pos[i] = tgt
        yaw = 0.0

        tracker.step(pos, tgt, yaw, DT)

        det = detector.detect(tgt, pos, gimbal, H_FOV, V_FOV, yaw)
        vel_cmd = servo.compute(det, yaw)

        bbox_cx[i] = det.center_ndc[0] if det.visible else 0.0
        bbox_cy[i] = det.center_ndc[1] if det.visible else 0.0
        bbox_size[i] = det.size_ratio
        visible_hist[i] = det.visible
        dist_hist[i] = np.linalg.norm(pos - tgt)

        alpha = 0.35
        vel = (1 - alpha) * vel + alpha * vel_cmd
        pos = pos + vel * DT
        pos[2] = np.clip(pos[2], 3.0, 20.0)
        drone_pos[i] = pos

    # ── 2x2 gridspec layout ──────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.30)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_cam = fig.add_subplot(gs[1, 0])
    ax_data = fig.add_subplot(gs[1, 1])

    fig.suptitle("Visual Servoing — Bounding Box Follow", fontsize=13)

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
    ax_top.set_title("Top Down", fontsize=9)
    ax_top.set_aspect("equal")

    for b in buildings:
        sz = b.max_corner - b.min_corner
        ax_top.add_patch(
            matplotlib.patches.Rectangle(b.min_corner[:2], sz[0], sz[1], fc="gray", alpha=0.3)
        )

    tgt_trail = np.array([_moving_target(i * DT) for i in range(n_steps)])
    ax_top.plot(tgt_trail[:, 0], tgt_trail[:, 1], "r--", lw=0.5, alpha=0.3)
    (tgt_top,) = ax_top.plot([], [], "r*", ms=10, zorder=10)
    (tgt_3d,) = ax3d.plot([], [], [], "r*", ms=10, zorder=10, label="Target")
    (drone_trail_top,) = ax_top.plot([], [], "dodgerblue", lw=0.8, alpha=0.5)
    ax3d.legend(fontsize=7, loc="upper left")

    ax_cam.set_xlim(-1.2, 1.2)
    ax_cam.set_ylim(-1.0, 1.0)
    ax_cam.set_facecolor("black")
    ax_cam.set_title("Camera View", fontsize=9, color="white")
    ax_cam.tick_params(labelsize=5, colors="white")
    ax_cam.axhline(0, color="gray", lw=0.3, alpha=0.5)
    ax_cam.axvline(0, color="gray", lw=0.3, alpha=0.5)
    (bbox_rect_art,) = ax_cam.plot([], [], "lime", lw=2)
    (crosshair,) = ax_cam.plot([0], [0], "w+", ms=12, mew=1)
    (target_dot,) = ax_cam.plot([], [], "r.", ms=10)

    times = np.arange(n_steps) * DT
    ax_data.set_xlim(0, SIM_TIME)
    ax_data.set_ylim(0, max(20, dist_hist.max() * 1.2))
    ax_data.set_xlabel("Time [s]", fontsize=8)
    ax_data.set_ylabel("Distance [m]", fontsize=8)
    ax_data.set_title("Drone-Target Distance", fontsize=9)
    ax_data.grid(True, alpha=0.3)
    (dist_line,) = ax_data.plot([], [], "b-", lw=0.8, label="Distance")
    ax_data.legend(fontsize=7)

    skip = max(1, n_steps // 200)
    frames = list(range(0, n_steps, skip))
    n_frames = len(frames)

    veh_arts: list = []

    anim = SimAnimator("visual_servoing", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        k = frames[f]
        dp = drone_pos[k]
        tp = target_pos[k]

        drone_trail_top.set_data(drone_pos[:k, 0], drone_pos[:k, 1])

        clear_vehicle_artists(veh_arts)
        R = Quadrotor.rotation_matrix(0, 0, 0)
        veh_arts.extend(draw_quadrotor_3d(ax3d, dp, R, size=1.5))
        (dt_t,) = ax_top.plot(dp[0], dp[1], "ko", ms=4, zorder=10)
        veh_arts.append(dt_t)

        tgt_3d.set_data([tp[0]], [tp[1]])
        tgt_3d.set_3d_properties([tp[2]])
        tgt_top.set_data([tp[0]], [tp[1]])

        if visible_hist[k]:
            cx_k = bbox_cx[k]
            cy_k = bbox_cy[k]
            sr = bbox_size[k] * 2
            half = max(sr, 0.05)
            xs = [
                cx_k - half,
                cx_k + half,
                cx_k + half,
                cx_k - half,
                cx_k - half,
            ]
            ys = [
                cy_k - half,
                cy_k - half,
                cy_k + half,
                cy_k + half,
                cy_k - half,
            ]
            bbox_rect_art.set_data(xs, ys)
            target_dot.set_data([cx_k], [cy_k])
        else:
            bbox_rect_art.set_data([], [])
            target_dot.set_data([], [])

        dist_line.set_data(times[:k], dist_hist[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
