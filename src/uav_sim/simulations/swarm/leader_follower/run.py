# Erwin Lejeune - 2026-02-19
"""Leader-follower: 100m env with quad models, 3-panel + data.

Shows the leader on a circular path with 3 followers maintaining offsets.
Uses quad models, 3D/top/side panels plus follower error + distances.

Reference: J. Desai et al., "Modeling and Control of Formations of
Nonholonomic Mobile Robots," IEEE T-RA, 2001. DOI: 10.1109/70.976023
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.leader_follower import LeaderFollower
from uav_sim.vehicles.multirotor.quadrotor import Quadrotor
from uav_sim.visualization import SimAnimator
from uav_sim.visualization.vehicle_artists import (
    clear_vehicle_artists,
    draw_quadrotor_3d,
)

matplotlib.use("Agg")

WORLD_SIZE = 100.0
CRUISE_ALT = 50.0


def main() -> None:
    offsets = np.array([[8, 0, 0], [-8, 0, 0], [0, 8, 0.0]])
    ctrl = LeaderFollower(offsets=offsets, kp=3.0, kd=2.0)
    n_ag = 1 + ctrl.num_followers
    pos = np.zeros((n_ag, 3))
    pos[0] = [50, 50, CRUISE_ALT]
    for i in range(ctrl.num_followers):
        pos[1 + i] = pos[0] + offsets[i] + np.random.randn(3) * 3
    vel = np.zeros((n_ag, 3))
    dt, n_steps = 0.1, 500

    snap = [pos.copy()]
    follower_err = np.zeros((n_steps, ctrl.num_followers))
    mean_dist = np.zeros(n_steps)

    for step in range(n_steps):
        t = step * dt
        r = 25.0
        new_leader = np.array(
            [
                50 + r * np.cos(0.08 * t),
                50 + r * np.sin(0.08 * t),
                CRUISE_ALT + 5 * np.sin(0.04 * t),
            ]
        )
        leader_vel = (new_leader - pos[0]) / dt
        pos[0] = new_leader
        forces = ctrl.compute_forces(pos[0], leader_vel, pos[1:], vel[1:])
        vel[1:] = vel[1:] + forces * dt
        pos[1:] = pos[1:] + vel[1:] * dt
        snap.append(pos.copy())
        for fi in range(ctrl.num_followers):
            desired = pos[0] + offsets[fi]
            follower_err[step, fi] = np.linalg.norm(pos[1 + fi] - desired)

        dists = [np.linalg.norm(pos[i] - pos[j]) for i in range(n_ag) for j in range(i + 1, n_ag)]
        mean_dist[step] = np.mean(dists)

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)

    agent_colors = ["red", "steelblue", "green", "orange"]
    cmap_colors = [np.array(matplotlib.colors.to_rgba(c)[:3]) for c in agent_colors]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[0, 2])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1:])

    fig.suptitle("Leader-Follower Formation (100m env)", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_aspect("equal")
    ax_top.set_title("Top Down", fontsize=9)
    ax_top.grid(True, alpha=0.15)

    ax_side.set_xlim(0, WORLD_SIZE)
    ax_side.set_ylim(0, WORLD_SIZE)
    ax_side.set_title("Side View", fontsize=9)
    ax_side.grid(True, alpha=0.15)

    ax_err.set_xlim(0, n_steps * dt)
    ax_err.set_ylim(0, max(1.0, follower_err.max() * 1.1))
    ax_err.set_xlabel("Time [s]", fontsize=8)
    ax_err.set_ylabel("Follower Error [m]", fontsize=8)
    ax_err.set_title("Formation Error", fontsize=9)
    ax_err.grid(True, alpha=0.3)
    ferr_lines = []
    for fi in range(ctrl.num_followers):
        (ln,) = ax_err.plot([], [], color=agent_colors[1 + fi], lw=1, label=f"F{fi}")
        ferr_lines.append(ln)
    ax_err.legend(fontsize=7)

    ax_dist.set_xlim(0, n_steps * dt)
    ax_dist.set_ylim(0, max(5, mean_dist.max() * 1.2))
    ax_dist.set_xlabel("Time [s]", fontsize=8)
    ax_dist.set_ylabel("Mean Neighbor Distance [m]", fontsize=8)
    ax_dist.set_title("Neighbor Distances", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    (ldist,) = ax_dist.plot([], [], "b-", lw=0.8)

    trails_top = [
        ax_top.plot([], [], color=agent_colors[i], lw=0.6, alpha=0.4)[0] for i in range(n_ag)
    ]
    trails_side = [
        ax_side.plot([], [], color=agent_colors[i], lw=0.6, alpha=0.4)[0] for i in range(n_ag)
    ]

    veh_arts: list = []
    title = ax3d.set_title("t = 0.0 s")
    all_snap = np.array(snap)

    anim = SimAnimator("leader_follower", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        step = idx[f]
        p = snap[step]

        clear_vehicle_artists(veh_arts)
        for ai in range(n_ag):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            veh_arts.extend(
                draw_quadrotor_3d(
                    ax3d, p[ai], R, size=3.0, arm_colors=(cmap_colors[ai], cmap_colors[ai])
                )
            )

        for ai in range(n_ag):
            trail = all_snap[: step + 1 : max(1, step // 50), ai]
            trails_top[ai].set_data(trail[:, 0], trail[:, 1])
            trails_side[ai].set_data(trail[:, 0], trail[:, 2])

        for fi in range(ctrl.num_followers):
            ferr_lines[fi].set_data(times[:step], follower_err[:step, fi])
        ldist.set_data(times[:step], mean_dist[:step])

        title.set_text(f"Leader-Follower — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
