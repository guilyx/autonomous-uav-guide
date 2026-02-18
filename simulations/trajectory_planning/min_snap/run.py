# Erwin Lejeune - 2026-02-17
"""Minimum-snap trajectory: two-phase visualisation.

Phase 1 — Algorithm: incremental spline construction through waypoints.
Phase 2 — Platform: quadrotor takeoff -> pure-pursuit trajectory -> land.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_tracking.flight_ops import fly_mission
from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.path_tracking.pure_pursuit_3d import PurePursuit3D
from uav_sim.trajectory_planning.min_snap import MinSnapTrajectory
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


def main() -> None:
    wps = np.array([[0, 0, 1], [2, 0, 1.5], [4, 2, 2.0], [6, 2, 1.5], [8, 0, 1.0]])
    seg_times = np.array([1.5, 1.5, 1.5, 1.5])
    ms = MinSnapTrajectory()
    coeffs = ms.generate(wps, seg_times)
    _, traj_pts = ms.evaluate(coeffs, seg_times, dt=0.02)

    # ── Phase 2: fly mission via pure pursuit ─────────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([wps[0, 0], wps[0, 1], 0.0]))
    ctrl = CascadedPIDController()
    pursuit = PurePursuit3D(lookahead=0.8, waypoint_threshold=0.3, adaptive=True)
    flight_states = fly_mission(
        quad,
        ctrl,
        traj_pts,
        dt=0.005,
        pursuit=pursuit,
        takeoff_duration=2.0,
        landing_duration=2.0,
        loiter_duration=0.5,
    )
    flight_pos = flight_states[:, :3]

    # ── Animation ─────────────────────────────────────────────────────────
    n_traj = len(traj_pts)
    traj_step = max(1, n_traj // 100)
    traj_frames = list(range(0, n_traj, traj_step))
    fly_step = max(1, len(flight_pos) // 100)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_tf = len(traj_frames)
    n_ff = len(fly_frames)
    total = n_tf + n_ff

    anim = SimAnimator("min_snap", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(12, 6))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])
    fig.suptitle("Minimum-Snap Trajectory", fontsize=13)

    ax3d.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="red", s=80, marker="D", zorder=5, label="WP")
    for i, wp in enumerate(wps):
        ax3d.text(wp[0], wp[1], wp[2] + 0.15, f"WP{i}", fontsize=7, ha="center")
    ax3d.set_xlim(-0.5, 9)
    ax3d.set_ylim(-0.5, 3)
    ax3d.set_zlim(-0.5, 3)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (traj_line,) = ax3d.plot([], [], [], "b-", lw=2, alpha=0.7)
    (traj_dot,) = ax3d.plot([], [], [], "bo", ms=5)
    (fly_trail,) = ax3d.plot([], [], [], "orange", lw=1.8)

    ax2d.set_aspect("equal")
    ax2d.scatter(wps[:, 0], wps[:, 1], c="red", s=60, marker="D", zorder=5)
    ax2d.set_xlim(-0.5, 9)
    ax2d.set_ylim(-1, 3.5)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.grid(True, alpha=0.2)
    ax2d.set_title("Top-down View", fontsize=10)
    (traj2d,) = ax2d.plot([], [], "b-", lw=1.5, alpha=0.7)
    (fly2d,) = ax2d.plot([], [], "orange", lw=1.5)

    title = ax3d.set_title("Phase 1: Trajectory Generation")
    vehicle_arts: list = []

    def update(f):
        clear_vehicle_artists(vehicle_arts)
        if f < n_tf:
            k = traj_frames[f]
            traj_line.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 1])
            traj_line.set_3d_properties(traj_pts[: k + 1, 2])
            traj_dot.set_data([traj_pts[k, 0]], [traj_pts[k, 1]])
            traj_dot.set_3d_properties([traj_pts[k, 2]])
            traj2d.set_data(traj_pts[: k + 1, 0], traj_pts[: k + 1, 1])
            seg = min(int(k / (n_traj / len(seg_times))), len(seg_times) - 1)
            title.set_text(f"Phase 1: Min-Snap Generation — segment {seg + 1}/{len(seg_times)}")
        else:
            traj_dot.set_data([], [])
            traj_dot.set_3d_properties([])
            fi = f - n_tf
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            fly_trail.set_data(flight_pos[:k, 0], flight_pos[:k, 1])
            fly_trail.set_3d_properties(flight_pos[:k, 2])
            fly2d.set_data(flight_pos[:k, 0], flight_pos[:k, 1])
            R = Quadrotor.rotation_matrix(*flight_states[k, 3:6])
            vehicle_arts.extend(draw_quadrotor_3d(ax3d, flight_pos[k], R, size=0.3))
            title.set_text("Phase 2: Quadrotor Flying Trajectory")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
