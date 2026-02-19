# Erwin Lejeune - 2026-02-19
"""Consensus formation: 6 agents converging to a hexagon in 100 m env.

Slowed-down physics with quad models, 3-panel + data (formation error
+ neighbor distances).

Reference: R. Olfati-Saber, R. M. Murray, "Consensus Problems in Networks of
Agents with Switching Topology," IEEE TAC, 2004. DOI: 10.1109/TAC.2004.834113
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.consensus_formation import ConsensusFormation
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
    n_ag = 6
    rng = np.random.default_rng(1)
    pos = rng.uniform(15, 85, (n_ag, 3))
    pos[:, 2] = rng.uniform(35, 65, n_ag)

    r = 20.0
    angles = np.linspace(0, 2 * np.pi, n_ag, endpoint=False)
    offsets = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(n_ag)])

    adj = np.ones((n_ag, n_ag)) - np.eye(n_ag)
    ctrl = ConsensusFormation(adjacency=adj, offsets=offsets, gain=0.3)
    dt, n_steps = 0.1, 600

    vel = np.zeros_like(pos)
    damping = 0.8

    snap = [pos.copy()]
    form_err = np.zeros(n_steps)
    mean_dist = np.zeros(n_steps)

    for step in range(n_steps):
        forces = ctrl.compute_forces(pos)
        vel = vel * damping + forces * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        max_speed = 5.0
        vel = np.where(speed > max_speed, vel / speed * max_speed, vel)
        pos = pos + vel * dt
        snap.append(pos.copy())

        centroid = pos.mean(axis=0)
        form_err[step] = np.mean(
            [np.linalg.norm(pos[i] - centroid - offsets[i]) for i in range(n_ag)]
        )
        dists = [np.linalg.norm(pos[i] - pos[j]) for i in range(n_ag) for j in range(i + 1, n_ag)]
        mean_dist[step] = np.mean(dists)

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.Set2(np.linspace(0, 1, n_ag))

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[0, 2])
    ax_ferr = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1:])

    fig.suptitle("Consensus Formation Control (Hexagon, 100m env)", fontsize=13)

    ax3d.set_xlim(0, WORLD_SIZE)
    ax3d.set_ylim(0, WORLD_SIZE)
    ax3d.set_zlim(0, WORLD_SIZE)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    ax_top.set_xlim(0, WORLD_SIZE)
    ax_top.set_ylim(0, WORLD_SIZE)
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Y")
    ax_top.set_title("Top Down", fontsize=9)
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.15)

    ax_side.set_xlim(0, WORLD_SIZE)
    ax_side.set_ylim(0, WORLD_SIZE)
    ax_side.set_xlabel("X")
    ax_side.set_ylabel("Z")
    ax_side.set_title("Side View", fontsize=9)
    ax_side.grid(True, alpha=0.15)

    ax_ferr.set_xlim(0, n_steps * dt)
    ax_ferr.set_ylim(0, max(1.0, form_err.max() * 1.1))
    ax_ferr.set_xlabel("Time [s]", fontsize=8)
    ax_ferr.set_ylabel("Formation Error [m]", fontsize=8)
    ax_ferr.set_title("Formation Error", fontsize=9)
    ax_ferr.grid(True, alpha=0.3)
    (lferr,) = ax_ferr.plot([], [], "m-", lw=1)

    ax_dist.set_xlim(0, n_steps * dt)
    ax_dist.set_ylim(0, max(5, mean_dist.max() * 1.2))
    ax_dist.set_xlabel("Time [s]", fontsize=8)
    ax_dist.set_ylabel("Mean Neighbor Distance [m]", fontsize=8)
    ax_dist.set_title("Neighbor Distances", fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    (ldist,) = ax_dist.plot([], [], "b-", lw=1)
    ax_dist.axhline(
        2 * r * np.sin(np.pi / n_ag),
        color="gray",
        ls="--",
        lw=0.8,
        alpha=0.5,
        label="Target spacing",
    )
    ax_dist.legend(fontsize=7)

    trails_top = [ax_top.plot([], [], color=colors[i], lw=0.6, alpha=0.4)[0] for i in range(n_ag)]
    trails_side = [
        ax_side.plot([], [], color=colors[i], lw=0.6, alpha=0.4)[0] for i in range(n_ag)
    ]
    sc_top = ax_top.scatter(pos[:, 0], pos[:, 1], c=colors, s=40, zorder=5)
    sc_side = ax_side.scatter(pos[:, 0], pos[:, 2], c=colors, s=40, zorder=5)

    veh_arts: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("consensus_formation", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        k = idx[f]
        p = snap[k]

        clear_vehicle_artists(veh_arts)
        for i in range(n_ag):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            veh_arts.extend(
                draw_quadrotor_3d(
                    ax3d, p[i], R, size=3.0, arm_colors=(colors[i][:3], colors[i][:3])
                )
            )

        sc_top.set_offsets(p[:, :2])
        sc_side.set_offsets(np.column_stack([p[:, 0], p[:, 2]]))

        for i in range(n_ag):
            hist = np.array([snap[j][i] for j in range(0, k + 1, max(1, k // 50))])
            trails_top[i].set_data(hist[:, 0], hist[:, 1])
            trails_side[i].set_data(hist[:, 0], hist[:, 2])

        lferr.set_data(times[:k], form_err[:k])
        ldist.set_data(times[:k], mean_dist[:k])
        title.set_text(f"Consensus Formation — t = {k * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
