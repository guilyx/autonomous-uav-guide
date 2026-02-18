# Erwin Lejeune - 2026-02-17
"""Consensus formation: live multi-panel with 3D view + formation error.

Shows 6 agents converging to a hexagon formation with velocity vectors
and a formation-error metric converging to zero over time.

Reference: R. Olfati-Saber, R. M. Murray, "Consensus Problems in Networks of
Agents with Switching Topology," IEEE TAC, 2004. DOI: 10.1109/TAC.2004.834113
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.consensus_formation import ConsensusFormation
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    n_ag = 6
    rng = np.random.default_rng(1)
    pos = rng.uniform(5, 25, (n_ag, 3))
    r = 6.0
    angles = np.linspace(0, 2 * np.pi, n_ag, endpoint=False)
    offsets = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(n_ag)])
    adj = np.ones((n_ag, n_ag)) - np.eye(n_ag)
    ctrl = ConsensusFormation(adjacency=adj, offsets=offsets, gain=1.5)
    dt, n_steps = 0.05, 300

    snap = [pos.copy()]
    form_err = np.zeros(n_steps)
    for step in range(n_steps):
        forces = ctrl.compute_forces(pos)
        pos = pos + forces * dt
        snap.append(pos.copy())
        centroid = pos.mean(axis=0)
        form_err[step] = np.mean(
            [np.linalg.norm(pos[i] - centroid - offsets[i]) for i in range(n_ag)]
        )

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.Set2(np.linspace(0, 1, n_ag))

    anim = SimAnimator("consensus_formation", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(13, 6))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.3)
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])
    ax_ferr = fig.add_subplot(gs[1, 1])
    fig.suptitle("Consensus Formation Control (Hexagon)", fontsize=13)

    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax3d, all_p, pad=2)
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=50, depthshade=True)

    ax2d.set_aspect("equal")
    lo, hi = all_p[:, :2].min() - 2, all_p[:, :2].max() + 2
    ax2d.set_xlim(lo, hi)
    ax2d.set_ylim(lo, hi)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.grid(True, alpha=0.2)
    ax2d.set_title("Top-down", fontsize=10)
    sc2d = ax2d.scatter(pos[:, 0], pos[:, 1], c=colors, s=40)

    ax_ferr.set_xlim(0, n_steps * dt)
    ax_ferr.set_ylim(0, max(0.5, form_err.max() * 1.1))
    ax_ferr.set_xlabel("Time [s]", fontsize=8)
    ax_ferr.set_ylabel("Formation Error [m]", fontsize=8)
    ax_ferr.grid(True, alpha=0.3)
    (lferr,) = ax_ferr.plot([], [], "m-", lw=1)
    ax_ferr.tick_params(labelsize=7)

    title = ax3d.set_title("t = 0.0 s")

    def update(f):
        step = idx[f]
        p = snap[step]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        sc2d.set_offsets(p[:, :2])
        lferr.set_data(times[:step], form_err[:step])
        title.set_text(f"Consensus Formation — t = {step * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
