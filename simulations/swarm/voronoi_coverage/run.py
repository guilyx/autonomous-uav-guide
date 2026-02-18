# Erwin Lejeune - 2026-02-17
"""Voronoi coverage (Lloyd's algorithm): multi-panel with trails + cost.

Shows 6 agents converging to Voronoi centroids with coloured trails,
alongside a coverage cost metric decreasing over time.

Reference: J. Cortes et al., "Coverage Control for Mobile Sensing Networks,"
IEEE T-RA, 2004. DOI: 10.1109/TRA.2004.824698
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.swarm.coverage import CoverageController
from uav_sim.visualization import SimAnimator

matplotlib.use("Agg")


def main() -> None:
    n_ag = 6
    rng = np.random.default_rng(4)
    bounds = np.array([[0, 10], [0, 10]])
    pos_2d = rng.uniform(1, 9, (n_ag, 2))
    ctrl = CoverageController(bounds=bounds, resolution=0.5, gain=1.5)
    dt, n_steps = 0.05, 300

    snap = [pos_2d.copy()]
    coverage_cost = np.zeros(n_steps)
    for step in range(n_steps):
        forces = ctrl.compute_forces(pos_2d)
        pos_2d = pos_2d + forces * dt
        pos_2d = np.clip(pos_2d, 0, 10)
        snap.append(pos_2d.copy())
        coverage_cost[step] = np.sum(np.linalg.norm(forces, axis=1))

    times = np.arange(n_steps) * dt
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    n_frames = len(idx)
    colors = plt.cm.Set1(np.linspace(0, 1, n_ag))

    anim = SimAnimator("voronoi_coverage", out_dir=Path(__file__).parent)
    fig = plt.figure(figsize=(12, 6))
    anim._fig = fig
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_cost = fig.add_subplot(gs[0, 1])
    ax_disp = fig.add_subplot(gs[1, 1])
    fig.suptitle("Voronoi Coverage (Lloyd's Algorithm)", fontsize=13)

    ax_map.set_xlim(-0.5, 10.5)
    ax_map.set_ylim(-0.5, 10.5)
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.set_aspect("equal")
    ax_map.add_patch(plt.Rectangle((0, 0), 10, 10, fill=False, ec="gray", lw=1.5))
    ax_map.grid(True, alpha=0.15)
    sc = ax_map.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=80, zorder=5)
    trails = [ax_map.plot([], [], color=colors[i], lw=0.8, alpha=0.5)[0] for i in range(n_ag)]

    ax_cost.set_xlim(0, n_steps * dt)
    ax_cost.set_ylim(0, max(0.5, coverage_cost.max() * 1.1))
    ax_cost.set_ylabel("Force Magnitude", fontsize=8)
    ax_cost.grid(True, alpha=0.3)
    (lcost,) = ax_cost.plot([], [], "m-", lw=1)
    ax_cost.tick_params(labelsize=7)

    disp = np.zeros(n_steps)
    for step in range(1, n_steps):
        disp[step] = np.mean(np.linalg.norm(snap[step] - snap[step - 1], axis=1))
    ax_disp.set_xlim(0, n_steps * dt)
    ax_disp.set_ylim(0, max(0.01, disp.max() * 1.1))
    ax_disp.set_xlabel("Time [s]", fontsize=8)
    ax_disp.set_ylabel("Mean Displacement", fontsize=8)
    ax_disp.grid(True, alpha=0.3)
    (ldisp,) = ax_disp.plot([], [], "b-", lw=1)
    ax_disp.tick_params(labelsize=7)

    title = ax_map.set_title("t = 0.0 s")

    def update(f):
        k = idx[f]
        p = snap[k]
        sc.set_offsets(p)
        for i in range(n_ag):
            hist = np.array([snap[j][i] for j in range(k + 1)])
            trails[i].set_data(hist[:, 0], hist[:, 1])
        lcost.set_data(times[:k], coverage_cost[:k])
        ldisp.set_data(times[:k], disp[:k])
        title.set_text(f"Voronoi Coverage — t = {k * dt:.1f} s")

    anim.animate(update, n_frames)
    anim.save()


if __name__ == "__main__":
    main()
