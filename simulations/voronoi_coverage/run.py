# Erwin Lejeune - 2026-02-15
"""Voronoi-based area coverage (Lloyd's algorithm) — 6 agents.

Reference: J. Cortes et al., "Coverage Control for Mobile Sensing Networks,"
IEEE T-RA, 2004. DOI: 10.1109/TRA.2004.824698
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.swarm.coverage import CoverageController
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    N_ag = 6
    rng = np.random.default_rng(4)
    bounds = np.array([[0, 10], [0, 10]])
    pos_2d = rng.uniform(1, 9, (N_ag, 2))
    ctrl = CoverageController(bounds=bounds, resolution=0.5, gain=1.5)
    dt, n_steps = 0.05, 300
    snap = [pos_2d.copy()]
    for _ in range(n_steps):
        forces = ctrl.compute_forces(pos_2d)
        pos_2d = pos_2d + forces * dt
        pos_2d = np.clip(pos_2d, 0, 10)
        snap.append(pos_2d.copy())
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    anim = SimAnimator("voronoi_coverage", out_dir=Path(__file__).parent)
    fig, axes = anim.figure_2d("Voronoi Coverage (Lloyd)", nrows=1, figsize=(6, 6))
    ax = axes[0]
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), 10, 10, fill=False, ec="gray", lw=1.5))
    sc = ax.scatter([], [], c="darkgreen", s=60, zorder=5)
    trails = [ax.plot([], [], "darkgreen", lw=0.5, alpha=0.3)[0] for _ in range(N_ag)]

    def update(f):
        k = idx[f]
        p = snap[k]
        sc.set_offsets(p)
        for i, tr in enumerate(trails):
            hist = np.array([snap[j][i] for j in range(k + 1)])
            tr.set_data(hist[:, 0], hist[:, 1])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main()
