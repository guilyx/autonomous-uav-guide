# Erwin Lejeune - 2026-02-17
"""Potential-based swarm: multi-panel with 3D + distance-to-goal plot.

Shows Lennard-Jones-based swarm navigation with agent trails, inter-agent
distances, and convergence to the goal position.

Reference: W. M. Spears et al., "Distributed, Physics-Based Control of Swarms
of Vehicles," Autonomous Robots, 2004. DOI: 10.1023/B:AURO.0000033971.96584.f2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.potential_swarm import PotentialSwarm
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    n_ag = 8
    rng = np.random.default_rng(3)
    pos = rng.uniform(-2, 2, (n_ag, 3))
    vel = np.zeros_like(pos)
    goal = np.array([8.0, 8.0, 4.0])
    ctrl = PotentialSwarm(d_des=2.0, a=6, b=3, goal_gain=0.8)
    dt, n_steps = 0.05, 300

    snap = [pos.copy()]
    dist_to_goal = np.zeros((n_steps, n_ag))
    for step in range(n_steps):
        forces = ctrl.compute_forces(pos, goal=goal)
        vel = np.clip(vel + forces * dt, -1.5, 1.5)
        pos = pos + vel * dt
        snap.append(pos.copy())
        for i in range(n_ag):
            dist_to_goal[step, i] = np.linalg.norm(pos[i] - goal)

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.tab10(np.linspace(0, 1, n_ag))

    anim = SimAnimator("potential_swarm", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(13, 6))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_spread = fig.add_subplot(gs[1, 1])
    fig.suptitle("Potential-Based Swarm Navigation", fontsize=13)

    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax3d, all_p, pad=2)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.scatter(*goal, c="gold", s=150, marker="*", label="Goal", zorder=5)
    ax3d.legend(fontsize=7, loc="upper left")
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=40, depthshade=True)

    ax_dist.set_xlim(0, n_steps * dt)
    ax_dist.set_ylim(0, max(1.0, dist_to_goal.max() * 1.1))
    ax_dist.set_ylabel("Dist to Goal [m]", fontsize=8)
    ax_dist.grid(True, alpha=0.3)
    dist_lines = []
    for i in range(n_ag):
        (ln,) = ax_dist.plot([], [], color=colors[i], lw=0.8, alpha=0.7)
        dist_lines.append(ln)
    (mean_line,) = ax_dist.plot([], [], "k-", lw=1.5, label="Mean")
    ax_dist.legend(fontsize=6, loc="upper right")
    ax_dist.tick_params(labelsize=7)

    spread = np.std(dist_to_goal, axis=1)
    ax_spread.set_xlim(0, n_steps * dt)
    ax_spread.set_ylim(0, max(0.5, spread.max() * 1.1))
    ax_spread.set_xlabel("Time [s]", fontsize=8)
    ax_spread.set_ylabel("Spread (σ) [m]", fontsize=8)
    ax_spread.grid(True, alpha=0.3)
    (lspread,) = ax_spread.plot([], [], "b-", lw=1)
    ax_spread.tick_params(labelsize=7)

    title = ax3d.set_title("t = 0.0 s")

    def update(f):
        step = idx[f]
        p = snap[step]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        for i in range(n_ag):
            dist_lines[i].set_data(times[:step], dist_to_goal[:step, i])
        mean_line.set_data(times[:step], dist_to_goal[:step].mean(axis=1))
        lspread.set_data(times[:step], spread[:step])
        title.set_text(f"Potential Swarm — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
