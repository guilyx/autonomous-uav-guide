# Erwin Lejeune - 2026-02-15
"""Reynolds flocking — 12 agents with separation, alignment, cohesion.

Reference: C. W. Reynolds, "Flocks, Herds and Schools: A Distributed
Behavioral Model," SIGGRAPH '87, 1987. DOI: 10.1145/37402.37406
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.swarm.reynolds_flocking import ReynoldsFlocking
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    N_ag = 12
    rng = np.random.default_rng(0)
    pos = rng.uniform(-3, 3, (N_ag, 3))
    vel = rng.uniform(-0.2, 0.2, (N_ag, 3))
    flock = ReynoldsFlocking(r_percept=5.0, r_sep=1.5, w_sep=2.0, w_ali=1.0, w_coh=1.0)
    dt, n_steps = 0.05, 300
    snap = [pos.copy()]
    for _ in range(n_steps):
        forces = flock.compute_forces(pos, vel)
        vel = np.clip(vel + forces * dt, -1.5, 1.5)
        pos = pos + vel * dt
        snap.append(pos.copy())
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    anim = SimAnimator("reynolds_flocking", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Reynolds Flocking")
    sc = ax.scatter([], [], [], c="steelblue", s=30)
    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax, all_p, pad=2)

    def update(f):
        p = snap[idx[f]]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
