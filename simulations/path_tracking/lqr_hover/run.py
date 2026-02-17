# Erwin Lejeune - 2026-02-17
"""LQR hover stabilisation: multi-panel live simulation.

Left panel: 3D trajectory with start/target markers.
Right panels: position error, velocity, and control effort time histories.

Reference: B. D. O. Anderson, J. B. Moore, "Optimal Control: Linear Quadratic
Methods," Prentice-Hall, 1990.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_tracking.lqr_controller import LQRController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


def main() -> None:
    # ── simulate ──────────────────────────────────────────────────────────
    quad = Quadrotor()
    quad.reset(position=np.array([0.5, -0.3, 0.2]))
    ctrl = LQRController(
        mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
    )
    target = np.zeros(12)
    target[:3] = [0.0, 0.0, 1.0]
    dt, duration = 0.002, 5.0
    steps = int(duration / dt)
    states = np.zeros((steps, 12))
    controls = np.zeros((steps, 4))
    times = np.zeros(steps)
    for i in range(steps):
        states[i] = quad.state
        times[i] = i * dt
        u = ctrl.compute(quad.state, target)
        controls[i] = u
        quad.step(u, dt)

    # ── animation setup ───────────────────────────────────────────────────
    skip = max(1, steps // 250)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    anim = SimAnimator("lqr_hover", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 7))
    anim._fig = fig
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.45, wspace=0.35)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_err = fig.add_subplot(gs[0, 1])
    ax_vel = fig.add_subplot(gs[1, 1])
    ax_ctrl = fig.add_subplot(gs[2, 1])
    fig.suptitle("LQR Hover Stabilisation — Live Simulation", fontsize=13)

    # 3D view
    ax3d.scatter(0.5, -0.3, 0.2, c="blue", s=60, marker="o", label="Start", zorder=5)
    ax3d.scatter(*target[:3], c="g", s=100, marker="*", label="Target", zorder=5)
    ax3d.set_xlim(-0.6, 0.8)
    ax3d.set_ylim(-0.6, 0.6)
    ax3d.set_zlim(-0.1, 1.3)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.2, alpha=0.5)
    (dot3d,) = ax3d.plot([], [], [], "ro", ms=6)

    pos = states[:, :3]
    vel = states[:, 6:9]
    pos_err = pos - target[:3]

    # Position error subplot
    ax_err.set_xlim(0, duration)
    ax_err.set_ylim(np.min(pos_err) - 0.1, np.max(pos_err) + 0.1)
    ax_err.set_ylabel("Pos Error [m]", fontsize=8)
    ax_err.grid(True, alpha=0.3)
    (lex,) = ax_err.plot([], [], "r-", lw=1, label="ex")
    (ley,) = ax_err.plot([], [], "g-", lw=1, label="ey")
    (lez,) = ax_err.plot([], [], "b-", lw=1, label="ez")
    ax_err.axhline(0, color="k", ls=":", lw=0.5, alpha=0.4)
    ax_err.legend(fontsize=6, ncol=3, loc="upper right")
    ax_err.tick_params(labelsize=7)

    # Velocity subplot
    ax_vel.set_xlim(0, duration)
    ax_vel.set_ylim(np.min(vel) - 0.3, np.max(vel) + 0.3)
    ax_vel.set_ylabel("Vel [m/s]", fontsize=8)
    ax_vel.grid(True, alpha=0.3)
    (lvx,) = ax_vel.plot([], [], "r-", lw=1, label="vx")
    (lvy,) = ax_vel.plot([], [], "g-", lw=1, label="vy")
    (lvz,) = ax_vel.plot([], [], "b-", lw=1, label="vz")
    ax_vel.legend(fontsize=6, ncol=3, loc="upper right")
    ax_vel.tick_params(labelsize=7)

    # Control effort subplot (total thrust)
    thrust = controls[:, 0]
    ax_ctrl.set_xlim(0, duration)
    ax_ctrl.set_ylim(max(0, thrust.min() - 1), thrust.max() + 1)
    ax_ctrl.set_xlabel("Time [s]", fontsize=8)
    ax_ctrl.set_ylabel("Thrust [N]", fontsize=8)
    ax_ctrl.grid(True, alpha=0.3)
    (lt,) = ax_ctrl.plot([], [], "m-", lw=1, label="T")
    hover_t = quad.params.mass * quad.params.gravity
    ax_ctrl.axhline(hover_t, color="k", ls="--", lw=0.8, alpha=0.4, label="mg")
    ax_ctrl.legend(fontsize=6, ncol=2, loc="upper right")
    ax_ctrl.tick_params(labelsize=7)

    vehicle_arts: list = []

    def update(f):
        k = idx[f]
        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])
        dot3d.set_data([pos[k, 0]], [pos[k, 1]])
        dot3d.set_3d_properties([pos[k, 2]])
        clear_vehicle_artists(vehicle_arts)
        R = Quadrotor.rotation_matrix(*states[k, 3:6])
        vehicle_arts.extend(draw_quadrotor_3d(ax3d, pos[k], R, size=0.15))
        lex.set_data(times[:k], pos_err[:k, 0])
        ley.set_data(times[:k], pos_err[:k, 1])
        lez.set_data(times[:k], pos_err[:k, 2])
        lvx.set_data(times[:k], vel[:k, 0])
        lvy.set_data(times[:k], vel[:k, 1])
        lvz.set_data(times[:k], vel[:k, 2])
        lt.set_data(times[:k], thrust[:k])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
