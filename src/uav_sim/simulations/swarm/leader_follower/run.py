# Erwin Lejeune - 2026-02-17
"""Leader-follower: multi-panel with 3D view, trails, and tracking error.

Shows the leader on a circular path with followers maintaining offsets.
Coloured trails for each agent and per-follower distance error plots.

Reference: J. Desai et al., "Modeling and Control of Formations of
Nonholonomic Mobile Robots," IEEE T-RA, 2001. DOI: 10.1109/70.976023
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.leader_follower import LeaderFollower
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    offsets = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0.0]])
    ctrl = LeaderFollower(offsets=offsets, kp=4.0, kd=2.0)
    n_ag = 1 + ctrl.num_followers
    pos = np.zeros((n_ag, 3))
    pos[0] = [0, 0, 1]
    vel = np.zeros((n_ag, 3))
    dt, n_steps = 0.05, 300

    snap = [pos.copy()]
    follower_err = np.zeros((n_steps, ctrl.num_followers))
    for step in range(n_steps):
        t = step * dt
        new_leader = np.array([2 * np.cos(0.4 * t), 2 * np.sin(0.4 * t), 1.0])
        leader_vel = (new_leader - pos[0]) / dt
        pos[0] = new_leader
        forces = ctrl.compute_forces(pos[0], leader_vel, pos[1:], vel[1:])
        vel[1:] = vel[1:] + forces * dt
        pos[1:] = pos[1:] + vel[1:] * dt
        snap.append(pos.copy())
        for fi in range(ctrl.num_followers):
            desired = pos[0] + offsets[fi]
            follower_err[step, fi] = np.linalg.norm(pos[1 + fi] - desired)

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    anim = SimAnimator("leader_follower", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(13, 6))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax_err = fig.add_subplot(gs[1])
    fig.suptitle("Leader-Follower Formation", fontsize=13)

    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax3d, all_p, pad=2)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")

    agent_colors = ["red", "steelblue", "green", "orange"]
    sc_leader = ax3d.scatter([], [], [], c="red", s=100, marker="D", label="Leader", zorder=5)
    sc_followers = ax3d.scatter([], [], [], c="steelblue", s=50)
    trails_3d = []
    for ai in range(n_ag):
        (tr,) = ax3d.plot([], [], [], "-", color=agent_colors[ai], lw=0.8, alpha=0.5)
        trails_3d.append(tr)
    ax3d.legend(fontsize=7, loc="upper left")

    ax_err.set_xlim(0, n_steps * dt)
    ax_err.set_ylim(0, max(0.5, follower_err.max() * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=9)
    ax_err.set_ylabel("Follower Distance Error [m]", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    ferr_lines = []
    for fi in range(ctrl.num_followers):
        (ln,) = ax_err.plot([], [], color=agent_colors[1 + fi], lw=1, label=f"F{fi}")
        ferr_lines.append(ln)
    ax_err.legend(fontsize=7, loc="upper right")
    ax_err.tick_params(labelsize=7)

    title = ax3d.set_title("t = 0.0 s")
    all_snap = np.array(snap)

    def update(f):
        step = idx[f]
        p = snap[step]
        sc_leader._offsets3d = ([p[0, 0]], [p[0, 1]], [p[0, 2]])
        sc_followers._offsets3d = (p[1:, 0], p[1:, 1], p[1:, 2])
        for ai in range(n_ag):
            trail_data = all_snap[: step + 1, ai]
            trails_3d[ai].set_data(trail_data[:, 0], trail_data[:, 1])
            trails_3d[ai].set_3d_properties(trail_data[:, 2])
        for fi in range(ctrl.num_followers):
            ferr_lines[fi].set_data(times[:step], follower_err[:step, fi])
        title.set_text(f"Leader-Follower — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
