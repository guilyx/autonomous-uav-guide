# Erwin Lejeune - 2026-02-19
"""Gimbal FOV tracking: static drone with pan-tilt gimbal following
a coverage path via pure-pursuit targeting.

The quadrotor hovers stationary while the gimbal camera sweeps across
the coverage path waypoints, showing the frustum FOV in all views.

Uses:
  - CoveragePathPlanner to generate the survey path
  - PointTracker to slew the gimbal towards each target
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from uav_sim.path_planning.coverage_planner import CoveragePathPlanner, CoverageRegion
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.sensors.camera import Camera, CameraIntrinsics
from uav_sim.sensors.gimbal import Gimbal
from uav_sim.sensors.gimbal_controller import PointTracker
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.three_panel import ThreePanelViz

matplotlib.use("Agg")

WORLD_SIZE = 30.0
DRONE_POS = np.array([15.0, 15.0, 18.0])
CRUISE_ALT = 0.5


def main() -> None:
    region = CoverageRegion(
        origin=np.array([3.0, 3.0]),
        width=24.0,
        height=24.0,
        altitude=CRUISE_ALT,
    )
    planner = CoveragePathPlanner(swath_width=6.0, overlap=0.1, margin=2.0, points_per_row=15)
    coverage_path = planner.plan(region)

    cam = Camera(intrinsics=CameraIntrinsics(fx=200, fy=200, cx=320, cy=240))
    gimbal = Gimbal(max_rate=1.0)
    gimbal.reset(pan=0.0, tilt=-np.pi / 3)

    tracker = PointTracker(gimbal)
    pursuit = PurePursuit3D(lookahead=5.0, waypoint_threshold=3.0, adaptive=False)

    dt = 0.05
    timeout = 60.0
    max_steps = int(timeout / dt)
    pan_hist: list[float] = []
    tilt_hist: list[float] = []
    target_hist: list[np.ndarray] = []

    virtual_pos = coverage_path[0].copy()

    for _step in range(max_steps):
        target = pursuit.compute_target(virtual_pos, coverage_path)
        target_hist.append(target.copy())

        tracker.step(DRONE_POS, target, yaw=0.0, dt=dt)
        pan_hist.append(gimbal.pan)
        tilt_hist.append(gimbal.tilt)

        direction = target - virtual_pos
        dist = float(np.linalg.norm(direction))
        if dist > 0.1:
            virtual_pos += direction / dist * min(4.0 * dt, dist)

        if pursuit.is_path_complete(virtual_pos, coverage_path):
            break

    n_steps = len(pan_hist)

    # ── Animation ─────────────────────────────────────────────────────
    frame_skip = max(1, n_steps // 150)
    frame_idx = list(range(0, n_steps, frame_skip))
    n_frames = len(frame_idx)

    viz = ThreePanelViz(title="Gimbal FOV Tracking — Coverage Survey", world_size=WORLD_SIZE)

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
    viz.ax_side.plot(DRONE_POS[0], DRONE_POS[2], "kD", ms=8, zorder=10)

    (tgt_3d,) = viz.ax3d.plot([], [], [], "r*", ms=12, zorder=8, label="Look-at Target")
    (tgt_top,) = viz.ax_top.plot([], [], "r*", ms=10, zorder=8)
    (tgt_side,) = viz.ax_side.plot([], [], "r*", ms=10, zorder=8)

    frustum_polys: list = []
    fov_lines_top: list = []
    fov_lines_side: list = []

    (scanned_top,) = viz.ax_top.plot([], [], "c-", lw=0.5, alpha=0.3)
    scanned_x: list[float] = []
    scanned_y: list[float] = []

    viz.ax3d.legend(fontsize=7, loc="upper left")
    title = viz.ax3d.set_title("Gimbal Tracking Coverage Path")

    anim = SimAnimator("gimbal_tracking", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = viz.fig

    h_fov = cam.h_fov
    v_fov = cam.v_fov
    frustum_depth = 20.0

    def update(f: int) -> None:
        for p in frustum_polys:
            p.remove()
        frustum_polys.clear()
        for ln in fov_lines_top:
            ln.remove()
        fov_lines_top.clear()
        for ln in fov_lines_side:
            ln.remove()
        fov_lines_side.clear()

        k = frame_idx[f]
        tgt = target_hist[k]

        tgt_3d.set_data([tgt[0]], [tgt[1]])
        tgt_3d.set_3d_properties([tgt[2]])
        tgt_top.set_data([tgt[0]], [tgt[1]])
        tgt_side.set_data([tgt[0]], [tgt[2]])

        gimbal.pan = pan_hist[k]
        gimbal.tilt = tilt_hist[k]
        corners = gimbal.frustum_corners_world(DRONE_POS, h_fov, v_fov, frustum_depth, yaw=0.0)

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

        for i in [0, 3]:
            (ln,) = viz.ax_side.plot(
                [DRONE_POS[0], corners[i, 0]],
                [DRONE_POS[2], corners[i, 2]],
                "orange",
                lw=0.6,
                alpha=0.5,
            )
            fov_lines_side.append(ln)

        scanned_x.append(tgt[0])
        scanned_y.append(tgt[1])
        scanned_top.set_data(scanned_x, scanned_y)

        pct = int(100 * (k + 1) / n_steps)
        title.set_text(
            f"Gimbal Tracking — pan={np.degrees(gimbal.pan):.0f}° "
            f"tilt={np.degrees(gimbal.tilt):.0f}° — {pct}%"
        )

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
