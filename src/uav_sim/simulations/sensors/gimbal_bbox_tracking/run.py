# Erwin Lejeune - 2026-02-18
"""Gimbal Bounding Box Tracking: hovering drone tracks a moving target
with gimbal-only control, keeping the target centred in the camera frame.

Unlike visual servoing (which moves the drone), this demo keeps the drone
stationary and commands only the gimbal pan/tilt to centre the bounding
box.

Four-panel view: 3D scene (with frustum), top-down, side, and a
camera-image panel showing the bounding box centring error over time.

Reference: F. Chaumette & S. Hutchinson, "Visual Servo Control — Part II:
Advanced Approaches," IEEE Robotics & Automation Magazine, 2007.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from uav_sim.environment import default_world
from uav_sim.perception.bbox_tracker import SimulatedDetector
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import BBoxTracker, BBoxTrackerConfig
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists

matplotlib.use("Agg")

WORLD_SIZE = 30.0
DRONE_POS = np.array([15.0, 15.0, 15.0])
DT = 0.03
SIM_TIME = 50.0
H_FOV = 1.2
V_FOV = 0.9


def _moving_target(t: float) -> np.ndarray:
    """Figure-eight ground pattern."""
    cx, cy = 15.0, 15.0
    r = 10.0
    return np.array(
        [
            cx + r * np.sin(0.12 * t),
            cy + r * np.sin(0.12 * t) * np.cos(0.12 * t),
            1.0,
        ]
    )


def main() -> None:
    world, buildings = default_world()

    gimbal = Gimbal(max_rate=1.2)
    gimbal.reset(pan=0.0, tilt=-np.pi / 3)

    detector = SimulatedDetector(target_radius=0.8)
    bbox_ctrl = BBoxTracker(gimbal, BBoxTrackerConfig(kp_pan=2.5, kp_tilt=2.5))

    n_steps = int(SIM_TIME / DT)
    pan_hist = np.zeros(n_steps)
    tilt_hist = np.zeros(n_steps)
    target_hist = np.zeros((n_steps, 3))
    err_x_hist = np.zeros(n_steps)
    err_y_hist = np.zeros(n_steps)
    visible_hist = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        t = i * DT
        tgt = _moving_target(t)
        target_hist[i] = tgt

        det = detector.detect(tgt, DRONE_POS, gimbal, H_FOV, V_FOV, yaw=0.0)
        visible_hist[i] = det.visible

        if det.visible:
            bbox_ctrl.step(det.center_ndc, det.size_ratio, DT)
            err_x_hist[i] = det.center_ndc[0]
            err_y_hist[i] = det.center_ndc[1]
        else:
            des_p, des_t = gimbal.look_at(DRONE_POS, tgt, 0.0)
            gimbal.step(des_p, des_t, DT)

        pan_hist[i] = gimbal.pan
        tilt_hist[i] = gimbal.tilt

    # ── Animation ─────────────────────────────────────────────────────
    skip = max(1, n_steps // 180)
    frames = list(range(0, n_steps, skip))
    n_frames = len(frames)

    viz = ThreePanelViz(
        title="Gimbal BBox Tracking — Stationary Drone",
        world_size=WORLD_SIZE,
        figsize=(18, 9),
    )
    viz.draw_buildings(world.obstacles)

    # Static drone
    viz.ax3d.scatter(*DRONE_POS, c="black", s=80, marker="D", zorder=10, label="Drone")
    viz.ax_top.plot(DRONE_POS[0], DRONE_POS[1], "kD", ms=8, zorder=10)
    viz.ax_side.plot(DRONE_POS[0], DRONE_POS[2], "kD", ms=8, zorder=10)

    # Target ground track
    tgt_track = np.array([_moving_target(i * DT) for i in range(n_steps)])
    viz.ax_top.plot(tgt_track[:, 0], tgt_track[:, 1], "r--", lw=0.4, alpha=0.3)
    (tgt_top,) = viz.ax_top.plot([], [], "r*", ms=10, zorder=10)
    (tgt_3d,) = viz.ax3d.plot([], [], [], "r*", ms=10, zorder=10)

    # Camera-image inset
    ax_cam = viz.fig.add_axes([0.58, 0.02, 0.18, 0.25])
    ax_cam.set_xlim(-1.2, 1.2)
    ax_cam.set_ylim(-1.0, 1.0)
    ax_cam.set_facecolor("black")
    ax_cam.set_title("Camera View", fontsize=7, color="white")
    ax_cam.tick_params(labelsize=5, colors="white")
    ax_cam.axhline(0, color="gray", lw=0.3, alpha=0.5)
    ax_cam.axvline(0, color="gray", lw=0.3, alpha=0.5)
    (bbox_art,) = ax_cam.plot([], [], "lime", lw=2)
    (cam_dot,) = ax_cam.plot([], [], "r.", ms=10)

    # Error history inset
    ax_err = viz.fig.add_axes([0.78, 0.02, 0.18, 0.25])
    times = np.arange(n_steps) * DT
    ax_err.set_xlim(0, SIM_TIME)
    ax_err.set_ylim(-1.1, 1.1)
    ax_err.set_xlabel("Time [s]", fontsize=6)
    ax_err.set_ylabel("BBox error", fontsize=6)
    ax_err.tick_params(labelsize=5)
    ax_err.grid(True, alpha=0.2)
    (err_x_line,) = ax_err.plot([], [], "r-", lw=0.6, label="err_x")
    (err_y_line,) = ax_err.plot([], [], "b-", lw=0.6, label="err_y")
    ax_err.legend(fontsize=5, loc="upper right")

    frustum_arts: list = []

    anim = SimAnimator("gimbal_bbox_tracking", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    def update(f: int) -> None:
        k = frames[f]
        tgt = target_hist[k]

        tgt_3d.set_data([tgt[0]], [tgt[1]])
        tgt_3d.set_3d_properties([tgt[2]])
        tgt_top.set_data([tgt[0]], [tgt[1]])

        clear_vehicle_artists(frustum_arts)
        gimbal.pan = pan_hist[k]
        gimbal.tilt = tilt_hist[k]
        corners = gimbal.frustum_corners_world(DRONE_POS, H_FOV, V_FOV, 18.0, yaw=0.0)

        for i_c in range(4):
            j_c = (i_c + 1) % 4
            tri = [DRONE_POS.tolist(), corners[i_c].tolist(), corners[j_c].tolist()]
            poly = Poly3DCollection(
                [tri], alpha=0.06, facecolor="gold", edgecolor="orange", linewidth=0.3
            )
            viz.ax3d.add_collection3d(poly)
            frustum_arts.append(poly)

        for i_c in range(4):
            (ln,) = viz.ax_top.plot(
                [DRONE_POS[0], corners[i_c, 0]],
                [DRONE_POS[1], corners[i_c, 1]],
                "orange",
                lw=0.5,
                alpha=0.4,
            )
            frustum_arts.append(ln)

        if visible_hist[k]:
            cx_k = err_x_hist[k]
            cy_k = err_y_hist[k]
            half = 0.15
            xs = [cx_k - half, cx_k + half, cx_k + half, cx_k - half, cx_k - half]
            ys = [cy_k - half, cy_k - half, cy_k + half, cy_k + half, cy_k - half]
            bbox_art.set_data(xs, ys)
            cam_dot.set_data([cx_k], [cy_k])
        else:
            bbox_art.set_data([], [])
            cam_dot.set_data([], [])

        err_x_line.set_data(times[:k], err_x_hist[:k])
        err_y_line.set_data(times[:k], err_y_hist[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
