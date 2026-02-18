# Erwin Lejeune - 2026-02-17
"""Geometric SO(3) controller: multi-panel attitude recovery + 3D view.

Left panel: 3D trajectory showing attitude recovery path.
Right panels: Euler angles, position, and angular rates updated live.

Reference: T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of
a Quadrotor UAV on SE(3)," CDC, 2010. DOI: 10.1109/CDC.2010.5717652
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.path_tracking.geometric_controller import GeometricController
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    quad = Quadrotor()
    quad.reset(position=np.array([0.0, 0.0, 1.0]), euler=np.array([0.3, -0.2, 0.0]))
    hover_f = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hover_f))
    ctrl = GeometricController()
    target = np.array([0.0, 0.0, 1.0])
    dt, dur = 0.002, 4.0
    steps = int(dur / dt)
    times = np.zeros(steps)
    states = np.zeros((steps, 12))
    for i in range(steps):
        times[i] = quad.time
        states[i] = quad.state
        quad.step(ctrl.compute(quad.state, target), dt)

    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    anim = SimAnimator("geometric_control", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 7))
    anim._fig = fig
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.45, wspace=0.35)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_euler = fig.add_subplot(gs[0, 1])
    ax_pos = fig.add_subplot(gs[1, 1])
    ax_omega = fig.add_subplot(gs[2, 1])
    fig.suptitle("Geometric SO(3) — Attitude Recovery", fontsize=13)

    pos = states[:, :3]
    euler = states[:, 3:6]
    omega = states[:, 9:12]

    ax3d.scatter(*target, c="g", s=100, marker="*", label="Target", zorder=5)
    ax3d.set_xlim(-0.3, 0.3)
    ax3d.set_ylim(-0.3, 0.3)
    ax3d.set_zlim(0.5, 1.3)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.2, alpha=0.6)
    (dot3d,) = ax3d.plot([], [], [], "ro", ms=6)

    ax_euler.set_xlim(0, dur)
    ax_euler.set_ylim(-0.5, 0.5)
    ax_euler.set_ylabel("Euler [rad]", fontsize=8)
    ax_euler.grid(True, alpha=0.3)
    ax_euler.axhline(0, color="k", ls=":", lw=0.5, alpha=0.4)
    (lr,) = ax_euler.plot([], [], "r-", lw=1, label="roll")
    (lp,) = ax_euler.plot([], [], "g-", lw=1, label="pitch")
    (ly,) = ax_euler.plot([], [], "b-", lw=1, label="yaw")
    ax_euler.legend(fontsize=6, ncol=3, loc="upper right")
    ax_euler.tick_params(labelsize=7)

    ax_pos.set_xlim(0, dur)
    ax_pos.set_ylim(-0.2, 1.3)
    ax_pos.set_ylabel("Pos [m]", fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    (lpx,) = ax_pos.plot([], [], "r-", lw=1, label="x")
    (lpy,) = ax_pos.plot([], [], "g-", lw=1, label="y")
    (lpz,) = ax_pos.plot([], [], "b-", lw=1, label="z")
    ax_pos.legend(fontsize=6, ncol=3, loc="upper right")
    ax_pos.tick_params(labelsize=7)

    ax_omega.set_xlim(0, dur)
    omega_max = max(1.0, np.abs(omega).max() * 1.1)
    ax_omega.set_ylim(-omega_max, omega_max)
    ax_omega.set_xlabel("Time [s]", fontsize=8)
    ax_omega.set_ylabel("ω [rad/s]", fontsize=8)
    ax_omega.grid(True, alpha=0.3)
    (lop,) = ax_omega.plot([], [], "r-", lw=1, label="p")
    (loq,) = ax_omega.plot([], [], "g-", lw=1, label="q")
    (lor,) = ax_omega.plot([], [], "b-", lw=1, label="r")
    ax_omega.legend(fontsize=6, ncol=3, loc="upper right")
    ax_omega.tick_params(labelsize=7)

    def update(f):
        k = idx[f]
        trail3d.set_data(pos[:k, 0], pos[:k, 1])
        trail3d.set_3d_properties(pos[:k, 2])
        dot3d.set_data([pos[k, 0]], [pos[k, 1]])
        dot3d.set_3d_properties([pos[k, 2]])
        lr.set_data(times[:k], euler[:k, 0])
        lp.set_data(times[:k], euler[:k, 1])
        ly.set_data(times[:k], euler[:k, 2])
        lpx.set_data(times[:k], pos[:k, 0])
        lpy.set_data(times[:k], pos[:k, 1])
        lpz.set_data(times[:k], pos[:k, 2])
        lop.set_data(times[:k], omega[:k, 0])
        loq.set_data(times[:k], omega[:k, 1])
        lor.set_data(times[:k], omega[:k, 2])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
