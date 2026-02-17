# Erwin Lejeune - 2026-02-17
"""Feedback-linearisation tracker: multi-panel with 3D + tracking error.

Left panel: 3D view showing reference circle and actual quadrotor trajectory.
Right panels: per-axis tracking error and attitude evolution.

Reference: D. Mellinger, V. Kumar, "Minimum Snap Trajectory Generation and
Control for Quadrotors," ICRA, 2011, Sec. IV. DOI: 10.1109/ICRA.2011.5980409
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.trajectory_tracking.feedback_linearisation import (
    FeedbackLinearisationTracker,
)
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")


def main() -> None:
    quad = Quadrotor()
    quad.reset(position=np.array([1.0, 0.0, 1.0]))
    hf = quad.hover_wrench()[0] / 4.0
    for m in quad.motors:
        m.reset(m.thrust_to_omega(hf))
    tracker = FeedbackLinearisationTracker(
        mass=quad.params.mass, gravity=quad.params.gravity, inertia=quad.params.inertia
    )
    dt, dur = 0.002, 8.0
    steps = int(dur / dt)
    states = np.zeros((steps, 12))
    refs = np.zeros((steps, 3))
    times = np.zeros(steps)

    for i in range(steps):
        t = i * dt
        times[i] = t
        rp = np.array([np.cos(0.5 * t), np.sin(0.5 * t), 1 + 0.2 * np.sin(t)])
        rv = np.array([-0.5 * np.sin(0.5 * t), 0.5 * np.cos(0.5 * t), 0.2 * np.cos(t)])
        quad.step(tracker.compute(quad.state, rp, rv, np.zeros(3)), dt)
        states[i] = quad.state
        refs[i] = rp

    pos = states[:, :3]
    euler = states[:, 3:6]
    track_err = pos - refs

    skip = max(1, steps // 200)
    idx = list(range(0, steps, skip))
    n_frames = len(idx)

    anim = SimAnimator("feedback_linearisation", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(14, 7))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_err = fig.add_subplot(gs[0, 1])
    ax_att = fig.add_subplot(gs[1, 1])
    fig.suptitle("Feedback Linearisation — Circular Tracking", fontsize=13)

    ax3d.plot(refs[:, 0], refs[:, 1], refs[:, 2], "r--", lw=1, alpha=0.4, label="Ref")
    ax3d.set_xlim(-1.5, 1.5)
    ax3d.set_ylim(-1.5, 1.5)
    ax3d.set_zlim(0, 1.5)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.legend(fontsize=7, loc="upper left")
    (trail3d,) = ax3d.plot([], [], [], "b-", lw=1.2, alpha=0.7)
    (dot3d,) = ax3d.plot([], [], [], "ko", ms=5)
    (ref_dot,) = ax3d.plot([], [], [], "r*", ms=10)

    ax_err.set_xlim(0, dur)
    err_lim = max(0.1, np.abs(track_err).max() * 1.1)
    ax_err.set_ylim(-err_lim, err_lim)
    ax_err.set_ylabel("Track Error [m]", fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.axhline(0, color="k", ls=":", lw=0.5, alpha=0.4)
    (lex,) = ax_err.plot([], [], "r-", lw=1, label="ex")
    (ley,) = ax_err.plot([], [], "g-", lw=1, label="ey")
    (lez,) = ax_err.plot([], [], "b-", lw=1, label="ez")
    ax_err.legend(fontsize=6, ncol=3, loc="upper right")
    ax_err.tick_params(labelsize=7)

    ax_att.set_xlim(0, dur)
    att_lim = max(0.2, np.abs(euler).max() * 1.1)
    ax_att.set_ylim(-att_lim, att_lim)
    ax_att.set_xlabel("Time [s]", fontsize=8)
    ax_att.set_ylabel("Euler [rad]", fontsize=8)
    ax_att.grid(True, alpha=0.3)
    (lr,) = ax_att.plot([], [], "r-", lw=1, label="roll")
    (lp,) = ax_att.plot([], [], "g-", lw=1, label="pitch")
    (ly,) = ax_att.plot([], [], "b-", lw=1, label="yaw")
    ax_att.legend(fontsize=6, ncol=3, loc="upper right")
    ax_att.tick_params(labelsize=7)

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
        ref_dot.set_data([refs[k, 0]], [refs[k, 1]])
        ref_dot.set_3d_properties([refs[k, 2]])
        lex.set_data(times[:k], track_err[:k, 0])
        ley.set_data(times[:k], track_err[:k, 1])
        lez.set_data(times[:k], track_err[:k, 2])
        lr.set_data(times[:k], euler[:k, 0])
        lp.set_data(times[:k], euler[:k, 1])
        ly.set_data(times[:k], euler[:k, 2])

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
