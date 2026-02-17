# Erwin Lejeune - 2026-02-15
"""Consensus-based formation control — 6 agents form a hexagon.

Reference: R. Olfati-Saber, R. M. Murray, "Consensus Problems in Networks of
Agents with Switching Topology," IEEE TAC, 2004. DOI: 10.1109/TAC.2004.834113
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.swarm.consensus_formation import ConsensusFormation
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    N_ag = 6
    rng = np.random.default_rng(1)
    pos = rng.uniform(-4, 4, (N_ag, 3))
    r = 2.0
    angles = np.linspace(0, 2 * np.pi, N_ag, endpoint=False)
    offsets = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.zeros(N_ag)])
    adj = np.ones((N_ag, N_ag)) - np.eye(N_ag)
    ctrl = ConsensusFormation(adjacency=adj, offsets=offsets, gain=1.5)
    dt, n_steps = 0.05, 300
    snap = [pos.copy()]
    for _ in range(n_steps):
        forces = ctrl.compute_forces(pos)
        pos = pos + forces * dt
        snap.append(pos.copy())
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    anim = SimAnimator("consensus_formation", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Consensus Formation (hexagon)")
    sc = ax.scatter([], [], [], c="darkorange", s=40)
    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax, all_p, pad=2)

    def update(f):
        p = snap[idx[f]]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
