# Erwin Lejeune - 2026-02-17
"""PID-controlled hover: multi-panel live simulation.

Left panel: 3D view with quadrotor frame and trajectory.
Right panels: position, velocity, and attitude time histories updated live.

Reference: L. R. G. Carrillo et al., "Quad Rotorcraft Control," Springer, 2013.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_tracking.pid_controller import CascadedPIDController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import clear_vehicle_artists, draw_quadrotor_3d

matplotlib.use("Agg")


def main() -> None:
    # ── simulate ──────────────────────────────────────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 0.0]))
    ctrl = CascadedPIDController()
    target = np.array([0.0, 0.0, 1.0])
    dt, duration = 0.002, 5.0
    steps = int(duration / dt)
    states = np.zeros((steps, 12))
    times = np.zeros(steps)
    for i in range(steps):
        states[i] = quad.state
        times[i] = i * dt
        quad.step(ctrl.compute(quad.state, target, dt=dt), dt)

    # ── animation setup ───────────────────────────────────────────────────
    skip = max(1, steps // 250)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    anim = SimAnimator("pid_hover", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 7))
    anim._fig = fig
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.4, wspace=0.35)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_pos = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_att = fig.add_subplot(gs[2, 1])
    fig.suptitle("PID Hover — Live Simulation", fontsize=13)

    # 3D view
    ax3d.scatter(*target, c="g", s=100, marker="*", label="Target", zorder=5)
    ax3d.set_xlim(-0.5, 0.5)
    ax3d.set_ylim(-0.5, 0.5)
    ax3d.set_zlim(-0.1, 1.3)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.2, alpha=0.5)
    (dot3d,) = ax3d.plot([], [], [], "ro", ms=6)

    # Position subplot
    ax_pos.set_xlim(0, duration)
    ax_pos.set_ylim(-0.15, 1.2)
    ax_pos.set_ylabel("Pos [m]", fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.axhline(1.0, color="g", ls="--", lw=0.8, alpha=0.5, label="Target z")
    (lx,) = ax_pos.plot([], [], "r-", lw=1, label="x")
    (ly,) = ax_pos.plot([], [], "g-", lw=1, label="y")
    (lz,) = ax_pos.plot([], [], "b-", lw=1, label="z")
    ax_pos.legend(fontsize=6, ncol=4, loc="upper right")
    ax_pos.tick_params(labelsize=7)

    # Velocity subplot
    ax_vel.set_xlim(0, duration)
    ax_vel.set_ylim(-1.5, 1.5)
    ax_vel.set_ylabel("Vel [m/s]", fontsize=8)
    ax_vel.grid(True, alpha=0.3)
    (lvx,) = ax_vel.plot([], [], "r-", lw=1, label="vx")
    (lvy,) = ax_vel.plot([], [], "g-", lw=1, label="vy")
    (lvz,) = ax_vel.plot([], [], "b-", lw=1, label="vz")
    ax_vel.legend(fontsize=6, ncol=3, loc="upper right")
    ax_vel.tick_params(labelsize=7)

    # Attitude subplot
    ax_att.set_xlim(0, duration)
    ax_att.set_ylim(-15, 15)
    ax_att.set_xlabel("Time [s]", fontsize=8)
    ax_att.set_ylabel("Angle [deg]", fontsize=8)
    ax_att.grid(True, alpha=0.3)
    (lphi,) = ax_att.plot([], [], "r-", lw=1, label="roll")
    (lth,) = ax_att.plot([], [], "g-", lw=1, label="pitch")
    (lpsi,) = ax_att.plot([], [], "b-", lw=1, label="yaw")
    ax_att.legend(fontsize=6, ncol=3, loc="upper right")
    ax_att.tick_params(labelsize=7)

    pos = states[:, :3]
    vel = states[:, 6:9]
    att_deg = np.rad2deg(states[:, 3:6])
    vehicle_arts: list = []

    def update(f):
        k = idx[f]
        # 3D trail
        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])
        dot3d.set_data([pos[k, 0]], [pos[k, 1]])
        dot3d.set_3d_properties([pos[k, 2]])
        # Vehicle model
        clear_vehicle_artists(vehicle_arts)
        R = Quadrotor.rotation_matrix(*states[k, 3:6])
        vehicle_arts.extend(draw_quadrotor_3d(ax3d, pos[k], R, scale=30.0))
        # Position
        lx.set_data(times[:k], pos[:k, 0])
        ly.set_data(times[:k], pos[:k, 1])
        lz.set_data(times[:k], pos[:k, 2])
        # Velocity
        lvx.set_data(times[:k], vel[:k, 0])
        lvy.set_data(times[:k], vel[:k, 1])
        lvz.set_data(times[:k], vel[:k, 2])
        # Attitude
        lphi.set_data(times[:k], att_deg[:k, 0])
        lth.set_data(times[:k], att_deg[:k, 1])
        lpsi.set_data(times[:k], att_deg[:k, 2])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
