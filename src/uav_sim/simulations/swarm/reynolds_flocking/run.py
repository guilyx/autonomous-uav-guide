# Erwin Lejeune - 2026-02-19
"""Reynolds Flocking: 12 agents in 100m env with quad models + data panels.

Agents are drawn as quad models with velocity arrows. 3D + top-down view
plus data panels for mean speed and flock cohesion.

Reference: C. W. Reynolds, "Flocks, Herds, and Schools: A Distributed
Behavioral Model," SIGGRAPH, 1987.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.reynolds_flocking import ReynoldsFlocking
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
    n_agents = 12
    rng = np.random.default_rng(42)
    pos = rng.uniform(10, 90, (n_agents, 3))
    pos[:, 2] = rng.uniform(30, 70, n_agents)
    vel = rng.uniform(-1.0, 1.0, (n_agents, 3))

    flock = ReynoldsFlocking(r_percept=20.0, r_sep=8.0, w_sep=2.0, w_ali=1.0, w_coh=1.2)
    dt = 0.1
    n_steps = 400
    max_speed = 5.0

    positions_hist = [pos.copy()]
    velocities_hist = [vel.copy()]
    mean_speed_hist = np.zeros(n_steps)
    cohesion_hist = np.zeros(n_steps)

    for step in range(n_steps):
        forces = flock.compute_forces(pos, vel)
        vel = vel + forces * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(speed > max_speed, vel / speed * max_speed, vel)
        pos = pos + vel * dt
        pos = np.clip(pos, 5, WORLD_SIZE - 5)
        positions_hist.append(pos.copy())
        velocities_hist.append(vel.copy())
        mean_speed_hist[step] = np.mean(np.linalg.norm(vel, axis=1))
        centroid = pos.mean(axis=0)
        cohesion_hist[step] = np.mean(np.linalg.norm(pos - centroid, axis=1))

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 200)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    c_rgb = [c[:3] for c in colors]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_coh = fig.add_subplot(gs[1, 1])

    fig.suptitle("Reynolds Flocking (100m env)", fontsize=13)

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

    ax_speed.set_xlim(0, n_steps * dt)
    ax_speed.set_ylim(0, max(1, mean_speed_hist.max() * 1.2))
    ax_speed.set_xlabel("Time [s]", fontsize=8)
    ax_speed.set_ylabel("Mean Speed [m/s]", fontsize=8)
    ax_speed.set_title("Flock Speed", fontsize=9)
    ax_speed.grid(True, alpha=0.3)
    (lspd,) = ax_speed.plot([], [], "orange", lw=0.8)

    ax_coh.set_xlim(0, n_steps * dt)
    ax_coh.set_ylim(0, max(5, cohesion_hist.max() * 1.2))
    ax_coh.set_xlabel("Time [s]", fontsize=8)
    ax_coh.set_ylabel("Mean Distance to Centroid [m]", fontsize=8)
    ax_coh.set_title("Flock Cohesion", fontsize=9)
    ax_coh.grid(True, alpha=0.3)
    (lcoh,) = ax_coh.plot([], [], "m-", lw=0.8)

    sc_top = ax_top.scatter(pos[:, 0], pos[:, 1], c=colors, s=30, zorder=5)

    veh_arts: list = []
    quiver_top: list = []
    title = ax3d.set_title("t = 0.0 s")

    anim = SimAnimator("reynolds_flocking", out_dir=Path(__file__).parent, dpi=72)
    anim._fig = fig

    def update(f: int) -> None:
        step = idx[f]
        p = positions_hist[step]
        v = velocities_hist[step]

        clear_vehicle_artists(veh_arts)
        for i in range(n_agents):
            R = Quadrotor.rotation_matrix(0, 0, 0)
            veh_arts.extend(
                draw_quadrotor_3d(
                    ax3d,
                    p[i],
                    R,
                    size=2.5,
                    arm_colors=(c_rgb[i], c_rgb[i]),
                )
            )

        sc_top.set_offsets(p[:, :2])

        for q in quiver_top:
            q.remove()
        quiver_top.clear()
        for i in range(n_agents):
            s = np.linalg.norm(v[i, :2])
            if s > 0.1:
                vn = v[i, :2] / s * min(s, 3.0) * 2
                q2 = ax_top.quiver(
                    p[i, 0],
                    p[i, 1],
                    vn[0],
                    vn[1],
                    color=colors[i],
                    scale=30,
                    width=0.003,
                )
                quiver_top.append(q2)

        lspd.set_data(times[:step], mean_speed_hist[:step])
        lcoh.set_data(times[:step], cohesion_hist[:step])
        title.set_text(f"Reynolds Flocking — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
