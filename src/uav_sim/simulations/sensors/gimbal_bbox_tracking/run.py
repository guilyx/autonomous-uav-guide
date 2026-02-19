# Erwin Lejeune - 2026-02-19
"""Gimbal Bounding Box Tracking: hovering drone tracks a moving target
with gimbal-only control, keeping the target centred in the camera frame.

Four-panel view: 3D scene (with frustum), top-down, side, and a split
panel with camera view + error history.

Reference: F. Chaumette & S. Hutchinson, "Visual Servo Control — Part II:
Advanced Approaches," IEEE Robotics & Automation Magazine, 2007.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from uav_sim.environment import default_world
from uav_sim.perception.bbox_tracker import SimulatedDetector
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import BBoxTracker, BBoxTrackerConfig
from uav_sim.visualization import SimAnimator
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
    _world, buildings = default_world()

    gimbal = Gimbal(max_rate=1.5)
    gimbal.reset(pan=0.0, tilt=-np.pi / 3)

    detector = SimulatedDetector(target_radius=0.8)
    bbox_ctrl = BBoxTracker(gimbal, BBoxTrackerConfig(kp_pan=0.6, kp_tilt=0.6))

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

    # ── Build custom figure layout ────────────────────────────────────
    fig = matplotlib.pyplot.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[0, 2])
    ax_cam = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1:])

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

    ax_side.set_xlim(0, WORLD_SIZE)
    ax_side.set_ylim(0, WORLD_SIZE)
    ax_side.set_xlabel("X")
    ax_side.set_ylabel("Z")
    ax_side.set_title("Side View", fontsize=9)

    # Camera view panel
    ax_cam.set_xlim(-1.2, 1.2)
    ax_cam.set_ylim(-1.0, 1.0)
    ax_cam.set_facecolor("black")
    ax_cam.set_title("Camera View", fontsize=9, color="white")
    ax_cam.axhline(0, color="gray", lw=0.3, alpha=0.5)
    ax_cam.axvline(0, color="gray", lw=0.3, alpha=0.5)
    (bbox_art,) = ax_cam.plot([], [], "lime", lw=2)
    (cam_dot,) = ax_cam.plot([], [], "r.", ms=10)

    # Error history panel
    times = np.arange(n_steps) * DT
    ax_err.set_xlim(0, SIM_TIME)
    ax_err.set_ylim(-1.1, 1.1)
    ax_err.set_xlabel("Time [s]", fontsize=9)
    ax_err.set_ylabel("BBox Centre Error (NDC)", fontsize=9)
    ax_err.set_title("Tracking Error History", fontsize=9)
    ax_err.grid(True, alpha=0.2)
    (err_x_line,) = ax_err.plot([], [], "r-", lw=0.8, label="err_x (pan)")
    (err_y_line,) = ax_err.plot([], [], "b-", lw=0.8, label="err_y (tilt)")
    ax_err.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax_err.legend(fontsize=8, loc="upper right")

    # Static elements
    for b in buildings:
        sz = b.max_corner - b.min_corner
        ax_top.add_patch(
            matplotlib.patches.Rectangle(
                b.min_corner[:2],
                sz[0],
                sz[1],
                fc="gray",
                alpha=0.3,
            )
        )

    ax3d.scatter(*DRONE_POS, c="black", s=80, marker="D", zorder=10, label="Drone")
    ax_top.plot(DRONE_POS[0], DRONE_POS[1], "kD", ms=8, zorder=10)
    ax_side.plot(DRONE_POS[0], DRONE_POS[2], "kD", ms=8, zorder=10)

    tgt_track = target_hist
    ax_top.plot(tgt_track[:, 0], tgt_track[:, 1], "r--", lw=0.4, alpha=0.3)
    (tgt_top,) = ax_top.plot([], [], "r*", ms=10, zorder=10)
    (tgt_3d,) = ax3d.plot([], [], [], "r*", ms=10, zorder=10, label="Target")
    ax3d.legend(fontsize=7, loc="upper left")

    frustum_arts: list = []
    title = ax3d.set_title("Gimbal BBox Tracking")

    skip = max(1, n_steps // 200)
    frames = list(range(0, n_steps, skip))
    n_frames = len(frames)

    anim = SimAnimator("gimbal_bbox_tracking", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

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
            ax3d.add_collection3d(poly)
            frustum_arts.append(poly)

        for i_c in range(4):
            (ln,) = ax_top.plot(
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

        title.set_text(
            f"Gimbal BBox Tracking — pan={np.degrees(gimbal.pan):.0f}° "
            f"tilt={np.degrees(gimbal.tilt):.0f}°"
        )

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
