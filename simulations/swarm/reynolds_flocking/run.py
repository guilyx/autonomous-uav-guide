# Erwin Lejeune - 2026-02-17
"""Reynolds Flocking: live multi-agent sim with velocity vectors and zones.

Each frame runs one simulation step. Agents are drawn with velocity arrows,
separation radii, and distinct colours. The flock evolves in real-time.

Reference: C. W. Reynolds, "Flocks, Herds, and Schools: A Distributed
Behavioral Model," SIGGRAPH, 1987.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.reynolds_flocking import ReynoldsFlocking
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    n_agents = 12
    rng = np.random.default_rng(42)
    pos = rng.uniform(-8, 8, (n_agents, 3))
    pos[:, 2] = rng.uniform(2, 6, n_agents)
    vel = rng.uniform(-0.5, 0.5, (n_agents, 3))

    flock = ReynoldsFlocking(r_percept=6.0, r_sep=2.0, w_sep=2.5, w_ali=1.0, w_coh=1.2)
    dt = 0.1
    n_steps = 200
    skip = 1
    n_frames = n_steps // skip

    anim = SimAnimator("reynolds_flocking", out_dir=Path(__file__).parent, fps=20)
    fig = plt.figure(figsize=(12, 6))
    anim._fig = fig
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], wspace=0.25)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax2d = fig.add_subplot(gs[1])
    fig.suptitle("Reynolds Flocking — Live Simulation", fontsize=13)

    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_xlim(-15, 15)
    ax3d.set_ylim(-15, 15)
    ax3d.set_zlim(0, 10)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.set_xlim(-15, 15)
    ax2d.set_ylim(-15, 15)
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.2)

    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    scat3d = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=40, depthshade=True)
    scat2d = ax2d.scatter(pos[:, 0], pos[:, 1], c=colors, s=30)

    quiver_artists_3d = []
    quiver_artists_2d = []
    circle_artists = []
    title = ax3d.set_title("t = 0.0 s")

    positions_hist = [pos.copy()]
    velocities_hist = [vel.copy()]
    for _ in range(n_steps):
        forces = flock.compute_forces(pos, vel)
        vel = vel + forces * dt
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = np.where(speed > 3.0, vel / speed * 3.0, vel)
        pos = pos + vel * dt
        positions_hist.append(pos.copy())
        velocities_hist.append(vel.copy())

    def update(f):
        nonlocal quiver_artists_3d, quiver_artists_2d, circle_artists
        for q in quiver_artists_3d:
            q.remove()
        for q in quiver_artists_2d:
            q.remove()
        for c in circle_artists:
            c.remove()
        quiver_artists_3d.clear()
        quiver_artists_2d.clear()
        circle_artists.clear()

        step = f * skip
        p = positions_hist[step]
        v = velocities_hist[step]
        t = step * dt

        scat3d._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        scat2d.set_offsets(p[:, :2])

        for i in range(n_agents):
            s = np.linalg.norm(v[i])
            if s > 0.05:
                vn = v[i] / s * min(s, 2.0)
                q3 = ax3d.quiver(
                    p[i, 0],
                    p[i, 1],
                    p[i, 2],
                    vn[0],
                    vn[1],
                    vn[2],
                    color=colors[i],
                    linewidth=1.2,
                    arrow_length_ratio=0.3,
                )
                quiver_artists_3d.append(q3)
                q2 = ax2d.quiver(
                    p[i, 0],
                    p[i, 1],
                    vn[0],
                    vn[1],
                    color=colors[i],
                    scale=15,
                    width=0.004,
                )
                quiver_artists_2d.append(q2)
            circle = plt.Circle(
                p[i, :2], flock.r_sep, fill=False, color=colors[i], alpha=0.15, lw=0.5
            )
            ax2d.add_patch(circle)
            circle_artists.append(circle)

        title.set_text(f"Reynolds Flocking — t = {t:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
