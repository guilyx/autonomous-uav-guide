# Erwin Lejeune - 2026-02-17
"""Quintic polynomial trajectory: two-phase visualisation.

Phase 1 — Algorithm: spline segments generated incrementally through waypoints.
Phase 2 — Platform: quadrotor flying the trajectory with full dynamics.

Reference: C. Richter, A. Bry, N. Roy, "Polynomial Trajectory Planning for
Aggressive Quadrotor Flight," ISRR, 2013. DOI: 10.1007/978-3-319-28872-7_37
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.trajectory_planning.polynomial_trajectory import PolynomialTrajectory
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")


def main() -> None:
    wps = np.array([[0, 0, 0], [1, 2, 1], [3, 3, 2], [5, 1, 1.5], [7, 0, 0.0]])
    st = np.array([1.5, 1.5, 1.5, 1.5])
    poly = PolynomialTrajectory(order=5)
    coeffs = poly.generate(wps, st)
    _, traj_pts = poly.evaluate(coeffs, st, dt=0.02)

    # Phase 2: quadrotor following
    quad = Quadrotor()
    quad.reset(position=wps[0].astype(float))
    ctrl = CascadedPIDController()
    dt_fly = 0.005
    flight_pos_list: list[np.ndarray] = []
    flight_euler_list: list[np.ndarray] = []
    wp_step = max(1, len(traj_pts) // 120)
    for wp in traj_pts[::wp_step]:
        for _ in range(30):
            s = quad.state
            p = s[:3].copy()
            if np.any(np.isnan(p)) or np.any(np.abs(p) > 50):
                break
            flight_pos_list.append(p)
            flight_euler_list.append(s[3:6].copy())
            quad.step(ctrl.compute(s, wp, dt=dt_fly), dt_fly)
    flight_pos = np.array(flight_pos_list) if flight_pos_list else traj_pts[:1]
    flight_euler = np.array(flight_euler_list) if flight_euler_list else np.zeros((1, 3))

    # Animation
    n_traj = len(traj_pts)
    traj_step = max(1, n_traj // 100)
    traj_frames = list(range(0, n_traj, traj_step))
    fly_step = max(1, len(flight_pos) // 100)
    fly_frames = list(range(0, len(flight_pos), fly_step))
    n_tf = len(traj_frames)
    n_ff = len(fly_frames)
    total = n_tf + n_ff

    anim = SimAnimator("polynomial_trajectory", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(12, 6))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])
    fig.suptitle("Polynomial Trajectory (Quintic)", fontsize=13)

    ax3d.scatter(wps[:, 0], wps[:, 1], wps[:, 2], c="red", s=80, marker="D", zorder=5, label="WP")
    for i, wp in enumerate(wps):
        ax3d.text(wp[0], wp[1], wp[2] + 0.15, f"WP{i}", fontsize=7, ha="center")
    ax3d.set_xlim(-0.5, 8)
    ax3d.set_ylim(-0.5, 4)
    ax3d.set_zlim(-0.5, 3)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (traj_line,) = ax3d.plot([], [], [], "b-", lw=2, alpha=0.7)
    (traj_dot,) = ax3d.plot([], [], [], "bo", ms=5)
    (fly_trail,) = ax3d.plot([], [], [], "orange", lw=1.8)
    (fly_dot,) = ax3d.plot([], [], [], "ko", ms=7)

    ax2d.set_aspect("equal")
    ax2d.scatter(wps[:, 0], wps[:, 1], c="red", s=60, marker="D", zorder=5)
    ax2d.set_xlim(-0.5, 8)
    ax2d.set_ylim(-1, 4.5)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.grid(True, alpha=0.2)
    ax2d.set_title("Top-down View", fontsize=10)
    (traj2d,) = ax2d.plot([], [], "b-", lw=1.5, alpha=0.7)
    (fly2d,) = ax2d.plot([], [], "orange", lw=1.5)
    (dot2d,) = ax2d.plot([], [], "ko", ms=5)

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
            dot2d.set_data([traj_pts[k, 0]], [traj_pts[k, 1]])
            seg = min(int(k / (n_traj / len(st))), len(st) - 1)
            title.set_text(f"Phase 1: Polynomial Generation — segment {seg + 1}/{len(st)}")
        else:
            traj_dot.set_data([], [])
            traj_dot.set_3d_properties([])
            fi = f - n_tf
            k = fly_frames[min(fi, len(fly_frames) - 1)]
            fly_trail.set_data(flight_pos[:k, 0], flight_pos[:k, 1])
            fly_trail.set_3d_properties(flight_pos[:k, 2])
            fly_dot.set_data([flight_pos[k, 0]], [flight_pos[k, 1]])
            fly_dot.set_3d_properties([flight_pos[k, 2]])
            fly2d.set_data(flight_pos[:k, 0], flight_pos[:k, 1])
            dot2d.set_data([flight_pos[k, 0]], [flight_pos[k, 1]])
            R = Quadrotor.rotation_matrix(*flight_euler[k])
            vehicle_arts.extend(draw_quadrotor_3d(ax3d, flight_pos[k], R, scale=30.0))
            title.set_text("Phase 2: Quadrotor Flying Trajectory")

    anim.animate(update, total)
    anim.save()


if __name__ == "__main__":
    main()
