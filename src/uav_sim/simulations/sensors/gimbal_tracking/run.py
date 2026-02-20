# Erwin Lejeune - 2026-02-19
"""Gimbal FOV tracking: static drone with pan-tilt gimbal following
a ground target that walks along a coverage (snake) path from start to end.

The quadrotor hovers stationary while the gimbal camera sweeps across
the coverage path waypoints, showing the frustum FOV in all views.

Uses:
  - CoveragePathPlanner to generate the survey path
  - PointTracker to slew the gimbal towards the current target
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from uav_sim.path_planning.coverage_planner import CoveragePathPlanner, CoverageRegion
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import PointTracker
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
DRONE_POS = np.array([15.0, 15.0, 18.0])
CRUISE_ALT = 0.5
TARGET_SPEED = 1.2
H_FOV = 0.6
V_FOV = 0.45


def _interpolate_along_path(path: np.ndarray, speed: float, dt: float) -> list[np.ndarray]:
    """Walk a point along *path* at constant *speed*, returning one position per dt."""
    positions: list[np.ndarray] = []
    pos = path[0].copy().astype(float)
    seg_idx = 0
    total_segs = len(path) - 1

    while seg_idx < total_segs:
        target = path[seg_idx + 1]
        direction = target - pos
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            seg_idx += 1
            continue
        step = min(speed * dt, dist)
        pos = pos + (direction / dist) * step
        positions.append(pos.copy())
        if float(np.linalg.norm(pos - target)) < 0.05:
            seg_idx += 1

    positions.append(path[-1].copy())
    return positions


def main() -> None:
    region = CoverageRegion(
        origin=np.array([3.0, 3.0]),
        width=24.0,
        height=24.0,
        altitude=CRUISE_ALT,
    )
    planner = CoveragePathPlanner(swath_width=6.0, overlap=0.1, margin=2.0, points_per_row=15)
    coverage_path = planner.plan(region)

    gimbal = Gimbal(max_rate=3.0)
    init_pan, init_tilt = gimbal.look_at(DRONE_POS, coverage_path[0], 0.0)
    gimbal.reset(pan=init_pan + 0.3, tilt=init_tilt + 0.2)

    tracker = PointTracker(gimbal)

    dt = 0.05
    target_positions = _interpolate_along_path(coverage_path, TARGET_SPEED, dt)
    n_steps = len(target_positions)

    pan_hist = np.zeros(n_steps)
    tilt_hist = np.zeros(n_steps)

    for i in range(n_steps):
        tgt = target_positions[i]
        tracker.step(DRONE_POS, tgt, yaw=0.0, dt=dt)
        pan_hist[i] = gimbal.pan
        tilt_hist[i] = gimbal.tilt

    # ── Animation ─────────────────────────────────────────────────────
    frame_skip = max(1, n_steps // 200)
    frame_idx = list(range(0, n_steps, frame_skip))
    n_frames = len(frame_idx)

    viz = ThreePanelViz(title="Gimbal FOV Tracking — Coverage Survey", world_size=WORLD_SIZE)

    times = np.arange(n_steps) * dt
    viz.ax_data.set_xlim(0, times[-1] if n_steps > 1 else 1.0)
    viz.ax_data.set_ylim(-200, 200)
    viz.ax_data.set_xlabel("Time [s]", fontsize=8)
    viz.ax_data.set_ylabel("Angle [deg]", fontsize=8)
    viz.ax_data.set_title("Gimbal Angles", fontsize=9)
    viz.ax_data.grid(True, alpha=0.3)
    (pan_line,) = viz.ax_data.plot([], [], "b-", lw=0.8, label="Pan")
    (tilt_line,) = viz.ax_data.plot([], [], "r-", lw=0.8, label="Tilt")
    viz.ax_data.legend(fontsize=7, loc="upper right")

    rx, ry = region.origin
    rect_top = mpatches.Rectangle(
        (rx, ry),
        region.width,
        region.height,
        fill=False,
        edgecolor="green",
        lw=2,
        linestyle="--",
    )
    viz.ax_top.add_patch(rect_top)
    viz.draw_path(coverage_path, color="blue", lw=0.8, alpha=0.3, label="Coverage Path")

    viz.ax3d.scatter(*DRONE_POS, c="black", s=80, marker="D", zorder=10, label="Drone")
    viz.ax_top.plot(DRONE_POS[0], DRONE_POS[1], "kD", ms=8, zorder=10)

    (tgt_3d,) = viz.ax3d.plot([], [], [], "r*", ms=12, zorder=8, label="Look-at Target")
    (tgt_top,) = viz.ax_top.plot([], [], "r*", ms=10, zorder=8)

    frustum_polys: list = []
    fov_lines_top: list = []

    (scanned_top,) = viz.ax_top.plot([], [], "c-", lw=0.5, alpha=0.3)
    scanned_x: list[float] = []
    scanned_y: list[float] = []

    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("Gimbal Tracking Coverage Path")

    anim = SimAnimator("gimbal_tracking", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    frustum_depth = 20.0

    def update(f: int) -> None:
        for p in frustum_polys:
            p.remove()
        frustum_polys.clear()
        for ln in fov_lines_top:
            ln.remove()
        fov_lines_top.clear()

        k = frame_idx[f]
        tgt = target_positions[k]

        tgt_3d.set_data([tgt[0]], [tgt[1]])
        tgt_3d.set_3d_properties([tgt[2]])
        tgt_top.set_data([tgt[0]], [tgt[1]])

        gimbal.pan = pan_hist[k]
        gimbal.tilt = tilt_hist[k]
        corners = gimbal.frustum_corners_world(DRONE_POS, H_FOV, V_FOV, frustum_depth, yaw=0.0)

        for i in range(4):
            j = (i + 1) % 4
            tri = [DRONE_POS.tolist(), corners[i].tolist(), corners[j].tolist()]
            poly = Poly3DCollection(
                [tri],
                alpha=0.08,
                facecolor="gold",
                edgecolor="orange",
                linewidth=0.4,
            )
            viz.ax3d.add_collection3d(poly)
            frustum_polys.append(poly)
        far_face = Poly3DCollection(
            [corners.tolist()],
            alpha=0.06,
            facecolor="yellow",
            edgecolor="orange",
            linewidth=0.3,
        )
        viz.ax3d.add_collection3d(far_face)
        frustum_polys.append(far_face)

        for i in range(4):
            (ln,) = viz.ax_top.plot(
                [DRONE_POS[0], corners[i, 0]],
                [DRONE_POS[1], corners[i, 1]],
                "orange",
                lw=0.6,
                alpha=0.5,
            )
            fov_lines_top.append(ln)
        (ln_far,) = viz.ax_top.plot(
            np.append(corners[:, 0], corners[0, 0]),
            np.append(corners[:, 1], corners[0, 1]),
            "orange",
            lw=0.8,
            alpha=0.4,
        )
        fov_lines_top.append(ln_far)

        scanned_x.append(tgt[0])
        scanned_y.append(tgt[1])
        scanned_top.set_data(scanned_x, scanned_y)

        pan_deg = [np.degrees(pan_hist[j]) for j in range(k + 1)]
        tilt_deg = [np.degrees(tilt_hist[j]) for j in range(k + 1)]
        pan_line.set_data(times[: k + 1], pan_deg)
        tilt_line.set_data(times[: k + 1], tilt_deg)

        pct = int(100 * (k + 1) / n_steps)
        title.set_text(
            f"Gimbal Tracking — pan={np.degrees(gimbal.pan):.0f}° "
            f"tilt={np.degrees(gimbal.tilt):.0f}° — {pct}%"
        )

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
