# Erwin Lejeune - 2026-02-19
"""Potential-based swarm: 100m env with quad models, 3-panel + data.

8 agents navigate to a goal using Lennard-Jones inter-agent potentials.

Reference: W. M. Spears et al., "Distributed, Physics-Based Control of Swarms
of Vehicles," Autonomous Robots, 2004. DOI: 10.1023/B:AURO.0000033971.96584.f2
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.potential_swarm import PotentialSwarm
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 100.0


def main() -> None:
    n_ag = 8
    rng = np.random.default_rng(3)
    pos = rng.uniform(10, 40, (n_ag, 3))
    pos[:, 2] = rng.uniform(40, 60, n_ag)
    vel = np.zeros_like(pos)
    goal = np.array([80.0, 80.0, 50.0])
    ctrl = PotentialSwarm(d_des=10.0, a=6, b=3, goal_gain=0.5)
    dt, n_steps = 0.1, 500

    snap = [pos.copy()]
    dist_to_goal = np.zeros((n_steps, n_ag))
    mean_dist = np.zeros(n_steps)

    for step in range(n_steps):
        forces = ctrl.compute_forces(pos, goal=goal)
        vel = vel + forces * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(speed > 5.0, vel / speed * 5.0, vel)
        pos = pos + vel * dt
        pos = np.clip(pos, 2, WORLD_SIZE - 2)
        snap.append(pos.copy())
        for i in range(n_ag):
            dist_to_goal[step, i] = np.linalg.norm(pos[i] - goal)
        dists = [np.linalg.norm(pos[i] - pos[j]) for i in range(n_ag) for j in range(i + 1, n_ag)]
        mean_dist[step] = np.mean(dists)

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.tab10(np.linspace(0, 1, n_ag))
    c_rgb = [c[:3] for c in colors]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[0, 2])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_neigh = fig.add_subplot(gs[1, 1:])

    fig.suptitle("Potential-Based Swarm Navigation (100m env)", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.scatter(*goal, c="gold", s=150, marker="*", label="Goal", zorder=5)
    ax3d.legend(fontsize=7, loc="upper left")

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_aspect("equal")
    ax_top.set_title("Top Down", fontsize=9)
    ax_top.grid(True, alpha=0.15)
    ax_top.plot(goal[0], goal[1], "y*", ms=15, zorder=10)

    ax_side.set_xlim(0, WORLD_SIZE)
    ax_side.set_ylim(0, WORLD_SIZE)
    ax_side.set_title("Side View", fontsize=9)
    ax_side.grid(True, alpha=0.15)

    ax_dist.set_xlim(0, n_steps * dt)
    ax_dist.set_ylim(0, max(5, dist_to_goal.max() * 1.1))
    ax_dist.set_xlabel("Time [s]", fontsize=8)
    ax_dist.set_ylabel("Dist to Goal [m]", fontsize=8)
    ax_dist.set_title("Distance to Goal", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    dist_lines = [ax_dist.plot([], [], color=colors[i], lw=0.6, alpha=0.6)[0] for i in range(n_ag)]
    (mean_line,) = ax_dist.plot([], [], "k-", lw=1.5, label="Mean")
    ax_dist.legend(fontsize=6)

    ax_neigh.set_xlim(0, n_steps * dt)
    ax_neigh.set_ylim(0, max(10, mean_dist.max() * 1.2))
    ax_neigh.set_xlabel("Time [s]", fontsize=8)
    ax_neigh.set_ylabel("Mean Neighbor Distance [m]", fontsize=8)
    ax_neigh.set_title("Neighbor Distances", fontsize=9)
    ax_neigh.grid(True, alpha=0.3)
    (lneigh,) = ax_neigh.plot([], [], "b-", lw=0.8)

    sc_top = ax_top.scatter(pos[:, 0], pos[:, 1], c=colors, s=30, zorder=5)
    sc_side = ax_side.scatter(pos[:, 0], pos[:, 2], c=colors, s=30, zorder=5)
    trails_top = [ax_top.plot([], [], color=colors[i], lw=0.4, alpha=0.3)[0] for i in range(n_ag)]

    veh_arts: list = []
    title = ax3d.set_title("t = 0.0 s")
    all_snap = np.array(snap)

    anim = SimAnimator("potential_swarm", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        step = idx[f]
        p = snap[step]

        clear_vehicle_artists(veh_arts)
        for i in range(n_ag):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            veh_arts.extend(
                draw_quadrotor_3d(ax3d, p[i], R, size=2.5, arm_colors=(c_rgb[i], c_rgb[i]))
            )

        sc_top.set_offsets(p[:, :2])
        sc_side.set_offsets(np.column_stack([p[:, 0], p[:, 2]]))

        for i in range(n_ag):
            trail = all_snap[: step + 1 : max(1, step // 50), i]
            trails_top[i].set_data(trail[:, 0], trail[:, 1])

        for i in range(n_ag):
            dist_lines[i].set_data(times[:step], dist_to_goal[:step, i])
        mean_line.set_data(times[:step], dist_to_goal[:step].mean(axis=1))
        lneigh.set_data(times[:step], mean_dist[:step])
        title.set_text(f"Potential Swarm — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
