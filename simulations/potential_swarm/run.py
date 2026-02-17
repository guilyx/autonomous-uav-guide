# Erwin Lejeune - 2026-02-15
"""Potential-based (Lennard-Jones) swarm navigation — 8 agents to a target.

Reference: W. M. Spears et al., "Distributed, Physics-Based Control of Swarms
of Vehicles," Autonomous Robots, 2004. DOI: 10.1023/B:AURO.0000033971.96584.f2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadrotor_sim.swarm.potential_swarm import PotentialSwarm
from quadrotor_sim.visualization import SimAnimator


def main() -> None:
    N_ag = 8
    rng = np.random.default_rng(3)
    pos = rng.uniform(-2, 2, (N_ag, 3))
    vel = np.zeros_like(pos)
    goal = np.array([8, 8, 4.0])
    ctrl = PotentialSwarm(d_des=2.0, a=6, b=3, goal_gain=0.8)
    dt, n_steps = 0.05, 300
    snap = [pos.copy()]
    for _ in range(n_steps):
        forces = ctrl.compute_forces(pos, goal=goal)
        vel = np.clip(vel + forces * dt, -1.5, 1.5)
        pos = pos + vel * dt
        snap.append(pos.copy())
    skip = max(1, n_steps // 150)
    idx = list(range(0, n_steps, skip))
    anim = SimAnimator("potential_swarm", out_dir=Path(__file__).parent)
    _, ax = anim.figure_3d("Potential-Based Swarm Navigation")
    sc = ax.scatter([], [], [], c="teal", s=30)
    ax.scatter(*goal, c="gold", s=120, marker="*", label="Goal", zorder=5)
    all_p = np.concatenate(snap)
    anim.set_equal_3d(ax, all_p, pad=2)
    ax.legend(fontsize=8)

    def update(f):
        p = snap[idx[f]]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])

    anim.animate(update, len(idx))
    anim.save()


if __name__ == "__main__":
    main()
